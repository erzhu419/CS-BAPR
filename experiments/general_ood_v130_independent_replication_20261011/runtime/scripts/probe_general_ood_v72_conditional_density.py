"""Frozen-forward ablation of global and state-conditional residual densities."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import numpy as np
import torch

from scripts import probe_general_ood_v71_state_coverage as v71
from scripts.v72_conditional_likelihood import ConditionalMeanBank, ConditionalDensityBank, calibrate_bank

v70, v69, v68 = v71.v70, v71.v69, v71.v68
REVISION = "general_ood_v72_conditional_density"
SPEC = dict(test=[135130, 135131], steps=6000, switch_step=3800)
NAMES = ("global", "conditional_center", "conditional_density")


def test_cells(spec):
    return [dict(role="test", seed=seed, mode=mode) for seed in spec["test"]
            for mode in ("normal", "high_grav", "compound")]


def check_seeds():
    previous = set(v70.SPEC["seeds"]) | set(v68.BASE.SPEC["test"]) | {132020}
    previous |= {v69.SPEC[k] for k in v69.SPEC if k.endswith("seed")}
    previous |= set(v71.SPEC["fit"]+v71.SPEC["density_calibration"]+v71.SPEC["test"])
    assert not set(SPEC["test"]) & previous


def fit_references(args):
    source = torch.load(args.v71_dir/"run_checkpoints/preflight/coverage_references.pt",
                        map_location="cpu", weights_only=False)
    assert source["revision"] == v71.REVISION and source["spec"] == v71.SPEC
    global_reference = source["references"]["shifted_state"]
    old_directory = args.v69_dir/"run_checkpoints/reference/traces"
    data = v71.datasets(old_directory, args.v71_dir/"run_checkpoints/preflight/traces", v71.SPEC, "shifted_state")
    nulls = [v68.load_trace(v69.trace_path(old_directory, c)) for c in v69.reference_cells(v69.SPEC)
             if c["role"] == "normal_null"]
    references = {"global": global_reference}
    for name, cls in (("conditional_center", ConditionalMeanBank), ("conditional_density", ConditionalDensityBank)):
        bank = cls.fit_density(global_reference.bank, data["density_calibration"])
        for mode in v71.MODES:
            np.testing.assert_array_equal(bank.coefficients[mode], global_reference.bank.coefficients[mode])
        references[name] = calibrate_bank(bank, nulls)
        assert references[name].normal_null_calibration_transitions == global_reference.normal_null_calibration_transitions
        print(f"[DENSITY] {name}: frozen forward coefficients, same null calibration rules", flush=True)
    return references


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
    output = [(dict(group="v72_fresh_frozen", seed=c["seed"], mode=c["mode"], updating=False), v71.trace_path(stored, c))
              for c in test_cells(SPEC)]
    for cell, path in v71.evaluation_inputs(args, args.v71_dir/"run_checkpoints/preflight/traces"):
        if cell["group"] == "v71_fresh_frozen":
            cell["group"] = "v71_development_replay"
        output.append((cell, path))
    return output


def evaluate(args, stored, references):
    records = []
    for cell, path in evaluation_inputs(args, stored):
        trace = v68.load_trace(path)
        observers = {}
        for name, reference in references.items():
            windows = v68.window_statistics(trace, reference, cell["mode"], SPEC["switch_step"])
            observers[name] = {w: {k: windows[w][k] for k in ("mean_coefficient", "mode_accuracy",
                "mean_true_minus_best_other_log_likelihood", "predicted_mode_fractions")} for w in v71.WINDOWS}
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
            margins = {w: sum(r["observers"][name][w]["mean_true_minus_best_other_log_likelihood"] > 0
                              for r in changed) for w in v68.BASE.CHANGE_WINDOWS}
            output[group][name] = dict(changed_pairs=len(changed), largest_normal_mean=maximum,
                usable_normal_null=maximum <= .1, positive_risk_contrast_pairs=contrasts,
                positive_true_mode_margin_pairs=margins,
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
    saved_path = args.checkpoint_dir/"conditional_references.pt"
    if saved_path.is_file():
        saved = torch.load(saved_path, map_location="cpu", weights_only=False)
        assert saved["revision"] == REVISION and saved["spec"] == SPEC
        references = saved["references"]
    else:
        references = fit_references(args)
        torch.save(dict(revision=REVISION, spec=SPEC, references=references), saved_path)
    print("[REFERENCE] Three residual distributions frozen BEFORE fresh tests", flush=True)
    metadata = {name: deepcopy(ref.metadata()) for name, ref in references.items()}
    collect_cells(args, progress, extra)
    records = evaluate(args, stored, references)
    assert metadata == {name: ref.metadata() for name, ref in references.items()}
    summary = summarize(records)
    totals = {k: sum(r[k] for r in progress["records"]) for k in ("actual_transitions", "counterfactual_step_calls",
                "environment_step_calls", "exact_true_mode_replays", "optimizer_updates")}
    assert totals["actual_transitions"] == len(test_cells(SPEC))*SPEC["steps"] and totals["optimizer_updates"] == 0
    passed = all(group["conditional_density"]["screen_pass"] for group in summary.values())
    compact = {name: {k: m[k] for k in ("bank", "window_steps", "mode_weights", "settings",
        "normal_null_calibration_transitions", "mode_labels_used_online", "switch_step_used_online", "actor_feedback")}
        for name, m in metadata.items()}
    output = dict(revision=REVISION, spec=SPEC, source_training_seed=1, source_updates=99745,
        reference_metadata=compact, server_reference_checkpoint=str(saved_path), reference_records=progress["records"],
        readout_controls_actor=False, forward_models_refitted=False, new_reference_labels=0,
        policy_weights_frozen=True, records=records, summary=summary, **totals,
        decision="conditional-density-screen-pass-not-RL-confirmation" if passed else "hold-conditional-density",
        limitations=["One source training seed, mild known profiles and privileged offline simulator labels.",
            "Adds state-conditioned density capacity and online nearest-neighbor queries; not a new forward model or policy.",
            "Development replays remain separate from fresh frozen-policy tests; no closed-loop return evidence."])
    args.result_dir.mkdir(parents=True, exist_ok=True)
    path = args.result_dir/"conditional_density.json"
    path.write_text(json.dumps(output, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] {output['decision']}: {path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ("source-dir", "shared-dir", "v71-dir", "v69-dir", "v70-dir", "v61-dir", "v62-dir", "checkpoint-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
