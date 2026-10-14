"""Frozen-density temporal ablation and fresh all-mode frozen-policy probes."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import numpy as np
import torch

from scripts import probe_general_ood_v72_conditional_density as v72
from scripts.v73_recursive_mode_readout import MODES, NAMES, RecursiveModeReference, observe_likelihoods

v71, v70, v69, v68 = v72.v71, v72.v70, v72.v69, v72.v68
REVISION = "general_ood_v73_recursive_mode"
SPEC = dict(test=[135140, 135141], steps=6000, switch_step=3800)
MARGIN_UNITS = dict(window16="16-step mean log density", recursive="log posterior probability")


def test_cells(spec):
    return [dict(role="test", seed=seed, mode=mode) for seed in spec["test"] for mode in MODES]


def check_seeds():
    v72.check_seeds()
    previous = set(v70.SPEC["seeds"]) | set(v68.BASE.SPEC["test"]) | {132020}
    previous |= {v69.SPEC[k] for k in v69.SPEC if k.endswith("seed")}
    previous |= set(v71.SPEC["fit"]+v71.SPEC["density_calibration"]+v71.SPEC["test"]+v72.SPEC["test"])
    assert not set(SPEC["test"]) & previous


def freeze_references(args):
    saved = torch.load(args.v72_dir/"run_checkpoints/preflight/conditional_references.pt",
                       map_location="cpu", weights_only=False)
    assert saved["revision"] == v72.REVISION and saved["spec"] == v72.SPEC
    control = saved["references"]["conditional_density"]
    old = args.v69_dir/"run_checkpoints/reference/traces"
    nulls = [v68.load_trace(v69.trace_path(old, c)) for c in v69.reference_cells(v69.SPEC)
             if c["role"] == "normal_null"]
    candidate = RecursiveModeReference.calibrate(control.bank, nulls)
    assert candidate.bank is control.bank
    assert candidate.normal_null_calibration_transitions == control.normal_null_calibration_transitions
    return dict(window16=control, recursive=candidate)


def collect_cells(args, progress, extra):
    shared = torch.load(args.shared_dir/"shared.pt", map_location="cpu", weights_only=False)
    cells = test_cells(SPEC)
    stored = args.checkpoint_dir/"traces"
    checkpoint = args.checkpoint_dir/"preflight_latest.pt"
    for index in range(progress["next_cell"], len(cells)):
        cell = cells[index]
        env = gym.make("Hopper-v4")
        shadows = v69.make_shadows(env)
        agent = v69.ContinuationAgent(env, "fixed-minus2", args.device)
        agent.start_shared(args.source_dir/v69.SOURCE_NAME, shared)
        assert agent.training_steps == 99745 and agent.context_tracker._count == 100000 and agent.replay_buffer.size == 6144
        loop = None
        if extra and extra["loop"] is not None:
            agent.load(str(args.resume_from))
            loop = extra["loop"]
            v70.apply_mode(env, v70.active_mode(cell, max(0, loop["step"]-1), SPEC))
            env.reset(seed=cell["seed"])
            v69.restore_step_state(env, extra["environment"])
        trace, record = v70.collect(agent, env, shadows, cell, SPEC, loop,
            lambda active: v68.v53.save_progress(agent, env, progress, active, checkpoint))
        np.savez_compressed(v71.trace_path(stored, cell), **trace)
        if cell["mode"] != "normal":
            normal = v68.load_trace(v71.trace_path(stored, dict(cell, mode="normal")))
            assert v68.v53.paired_prefix_equal(trace, normal, SPEC["switch_step"])
        progress["records"].append(record)
        progress["next_cell"] = index+1
        v68.v53.save_progress(agent, env, progress, None, checkpoint)
        for physical in (env, *shadows.values()):
            physical.close()
        extra = None


def evaluation_inputs(args, stored):
    output = [(dict(group="v73_fresh_frozen", seed=c["seed"], mode=c["mode"], updating=False),
               v71.trace_path(stored, c)) for c in test_cells(SPEC)]
    for cell, path in v72.evaluation_inputs(args, args.v72_dir/"run_checkpoints/preflight/traces"):
        cell = dict(cell)
        if cell["group"] == "v72_fresh_frozen":
            cell["group"] = "v72_development_replay"
        elif cell["group"] == "v62_test":
            cell["group"] = "v62_development_replay"
        output.append((cell, path))
    return output


def window_statistics(trace, likelihoods, series, mode):
    n = len(trace["dones"])
    truth = np.where(np.arange(n) < SPEC["switch_step"], 0, MODES.index(mode))
    output = {}
    for window, (a, b) in v68.v66.v65.v64.window_ranges(n, SPEC["switch_step"]).items():
        scores, target = series["mode_scores"][a:b], truth[a:b]
        rows = np.arange(b-a)
        others = scores.copy()
        others[rows, target] = -np.inf
        raw = likelihoods[a:b]
        raw_others = raw.copy()
        raw_others[rows, target] = -np.inf
        predicted = scores.argmax(axis=1)
        output[window] = dict(mean_coefficient=float(series["risk"][a:b].mean()),
            mode_accuracy=float(np.mean(predicted == target)),
            mean_true_minus_best_other_score=float(np.mean(scores[rows, target]-others.max(axis=1))),
            instantaneous_mode_accuracy=float(np.mean(raw.argmax(axis=1) == target)),
            instantaneous_mean_true_minus_best_other_log_density=float(np.mean(raw[rows, target]-raw_others.max(axis=1))),
            predicted_mode_fractions={m: float(np.mean(predicted == i)) for i, m in enumerate(MODES)})
    return output


def evaluate(args, stored, references):
    records = []
    for cell, path in evaluation_inputs(args, stored):
        trace = v68.load_trace(path)
        likelihoods = references["window16"].bank.log_likelihoods(trace)
        observers = {name: window_statistics(trace, likelihoods,
            observe_likelihoods(likelihoods, trace["dones"], ref, name), cell["mode"])
            for name, ref in references.items()}
        record = dict(**cell, observers=observers)
        normal = record if cell["mode"] == "normal" else next(r for r in records
            if r["group"] == cell["group"] and r["seed"] == cell["seed"]
            and r["updating"] == cell["updating"] and r["mode"] == "normal")
        record["change_minus_nochange"] = {name: {w: d["mean_coefficient"]-normal["observers"][name][w]["mean_coefficient"]
            for w, d in windows.items()} for name, windows in observers.items()}
        records.append(record)
        print(f"[EVAL] {cell['group']} seed{cell['seed']} {cell['mode']} updating={cell['updating']}", flush=True)
    return records


def summarize(records):
    output = {}
    for group in dict.fromkeys(r["group"] for r in records):
        normal = [r for r in records if r["group"] == group and r["mode"] == "normal"]
        changed = [r for r in records if r["group"] == group and r["mode"] != "normal"]
        output[group] = {}
        for name in NAMES:
            maximum = max(r["observers"][name][w]["mean_coefficient"] for r in normal for w in v68.BASE.NORMAL_WINDOWS)
            contrasts = {w: sum(r["change_minus_nochange"][name][w] > 0 for r in changed) for w in v68.BASE.CHANGE_WINDOWS}
            margins = {w: sum(r["observers"][name][w]["mean_true_minus_best_other_score"] > 0
                              for r in changed) for w in v68.BASE.CHANGE_WINDOWS}
            output[group][name] = dict(changed_pairs=len(changed), largest_normal_mean=maximum,
                usable_normal_null=maximum <= .1, positive_risk_contrast_pairs=contrasts,
                positive_true_mode_margin_pairs=margins, margin_units=MARGIN_UNITS[name],
                screen_pass=maximum <= .1 and all(n == len(changed) for n in (*contrasts.values(), *margins.values())))
    return output


def run(args):
    torch.set_num_threads(2)
    check_seeds()
    stored = args.checkpoint_dir/"traces"
    stored.mkdir(parents=True, exist_ok=True)
    extra = torch.load(args.resume_from, map_location="cpu", weights_only=False)["extra_state"] if args.resume_from else None
    progress = extra["progress"] if extra else dict(revision=REVISION, spec=SPEC, next_cell=0, records=[])
    assert progress["revision"] == REVISION and progress["spec"] == SPEC
    saved_path = args.checkpoint_dir/"temporal_references.pt"
    if saved_path.is_file():
        saved = torch.load(saved_path, map_location="cpu", weights_only=False)
        assert saved["revision"] == REVISION and saved["spec"] == SPEC
        references = saved["references"]
    else:
        references = freeze_references(args)
        torch.save(dict(revision=REVISION, spec=SPEC, references=references), saved_path)
    assert references["recursive"].bank is references["window16"].bank
    print("[REFERENCE] Unchanged v72 density; both temporal readouts frozen BEFORE fresh tests", flush=True)
    metadata = {name: deepcopy(ref.metadata()) for name, ref in references.items()}
    collect_cells(args, progress, extra)
    records = evaluate(args, stored, references)
    assert metadata == {name: ref.metadata() for name, ref in references.items()}
    summary = summarize(records)
    totals = {k: sum(r[k] for r in progress["records"]) for k in ("actual_transitions", "counterfactual_step_calls",
                "environment_step_calls", "exact_true_mode_replays", "optimizer_updates")}
    assert totals["actual_transitions"] == len(test_cells(SPEC))*SPEC["steps"] and totals["optimizer_updates"] == 0
    passed = all(group["recursive"]["screen_pass"] for group in summary.values())
    # Curves, matrices and trajectory arrays stay in the server checkpoint.
    compact = {name: {k: value for k, value in m.items() if k != "phase"} for name, m in metadata.items()}
    output = dict(revision=REVISION, spec=SPEC, source_training_seed=1, source_updates=99745,
        reference_metadata=compact, margin_units=MARGIN_UNITS, server_reference_checkpoint=str(saved_path),
        reference_records=progress["records"], readout_controls_actor=False, forward_models_refitted=False,
        conditional_density_refitted=False, new_reference_labels=0, policy_weights_frozen=True,
        records=records, summary=summary, **totals,
        decision="recursive-screen-pass-not-RL-confirmation" if passed else "hold-recursive-mode",
        limitations=["One source training seed, mild known profiles and privileged offline simulator labels.",
            "A fixed-model posterior; conditional Gaussian approximation errors can produce confident wrong modes.",
            "Previously inspected updating and frozen traces are development replays; no closed-loop return evidence."])
    args.result_dir.mkdir(parents=True, exist_ok=True)
    path = args.result_dir/"recursive_mode.json"
    path.write_text(json.dumps(output, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] {output['decision']}: {path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ("source-dir", "shared-dir", "v72-dir", "v71-dir", "v69-dir", "v70-dir", "v61-dir", "v62-dir", "checkpoint-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
