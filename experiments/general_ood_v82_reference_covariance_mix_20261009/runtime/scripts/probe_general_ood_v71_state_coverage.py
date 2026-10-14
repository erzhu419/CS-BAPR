"""Equal-label normal-only versus shifted-state reference expansion."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import numpy as np
import torch

from scripts import probe_general_ood_v70_physical_audit as v70
from scripts.v68_mode_likelihood import MODES, ModeLikelihoodReference

v69 = v70.v69
v68 = v69.v68
REVISION = "general_ood_v71_state_coverage"
SPEC = dict(fit=[135080, 135081, 135082], density_calibration=[135090, 135091, 135092],
            test=[135120, 135121], steps=6000, switch_step=3800)
CONDITIONS = ("normal_only", "shifted_state")
REFERENCES = ("v69", *CONDITIONS)
KEYS = ("states", "executed_actions", "next_states")
WINDOWS = tuple(dict.fromkeys((*v68.BASE.NORMAL_WINDOWS, *v68.BASE.CHANGE_WINDOWS)))


def reference_cells(spec):
    cells = []
    for role in ("fit", "density_calibration"):
        for seed, mode in zip(spec[role], ("normal", "high_grav", "compound")):
            cells.append(dict(role=role, seed=seed, mode="normal"))
            if mode != "normal":
                cells.append(dict(role=role, seed=seed, mode=mode))
    return cells


def test_cells(spec):
    return [dict(role="test", seed=seed, mode=mode) for seed in spec["test"]
            for mode in ("normal", "high_grav", "compound")]


def selected_cells(spec, role, condition):
    assert condition in CONDITIONS
    return [dict(role=role, seed=seed, mode=mode if condition == "shifted_state" else "normal")
            for seed, mode in zip(spec[role], ("normal", "high_grav", "compound"))]


def trace_path(directory, cell):
    return directory/f"{cell['role']}_seed{cell['seed']}_{cell['mode']}.npz"


def check_seed_partition(spec):
    groups = [set(spec[k]) for k in ("fit", "density_calibration", "test")]
    assert len(spec["fit"]) == len(spec["density_calibration"]) == 3
    assert all(not a & b for i, a in enumerate(groups) for b in groups[i+1:])
    prior = {v69.SPEC[k] for k in v69.SPEC if k.endswith("seed")}
    prior |= set(v70.SPEC["seeds"]) | set(v68.BASE.SPEC["test"]) | {132020}
    assert not set.union(*groups) & prior, "reused-reference-or-evaluation-seed"


def datasets(old_directory, stored, spec, condition):
    output = {}
    old_cells = v69.reference_cells(v69.SPEC)
    for role in ("fit", "density_calibration"):
        extensions = [v68.load_trace(trace_path(stored, c)) for c in selected_cells(spec, role, condition)]
        output[role] = {}
        for i, mode in enumerate(MODES):
            originals = [v68.load_trace(v69.trace_path(old_directory, c, mode)) for c in old_cells if c["role"] == role]
            rows = [{k: t[k] for k in KEYS} for t in originals]
            for t in extensions:
                s = slice(spec["switch_step"], spec["steps"])
                rows.append(dict(states=t["states"][s], executed_actions=t["executed_actions"][s],
                                 next_states=t["oracle_observations"][s, i]))
            output[role][mode] = {k: np.concatenate([t[k] for t in rows]) for k in KEYS}
            assert len(output[role][mode]["states"]) == 2*v69.SPEC["steps"]+3*(spec["steps"]-spec["switch_step"])
        for mode in MODES[1:]:
            for key in ("states", "executed_actions"):
                np.testing.assert_array_equal(output[role][mode][key], output[role]["normal"][key])
    return output


def fit_references(args, stored):
    original = torch.load(args.v69_dir/"run_checkpoints/reference/counterfactual_reference.pt",
                          map_location="cpu", weights_only=False)
    assert original["revision"] == v69.REVISION and original["spec"] == v69.SPEC
    old_directory = args.v69_dir/"run_checkpoints/reference/traces"
    nulls = [v68.load_trace(v69.trace_path(old_directory, c)) for c in v69.reference_cells(v69.SPEC)
             if c["role"] == "normal_null"]
    references = {"v69": original["reference"]}
    for condition in CONDITIONS:
        data = datasets(old_directory, stored, SPEC, condition)
        references[condition] = ModeLikelihoodReference.fit(data["fit"], data["density_calibration"], nulls)
        assert references[condition].normal_null_calibration_transitions == 2*v69.SPEC["steps"]
    for reference in references.values():
        meta = reference.metadata()
        assert meta["bank"]["feature_count"] == 120 and meta["window_steps"] == 16
        assert meta["settings"] == original["reference"].metadata()["settings"]
    return references


def collect_cells(args, shared, progress, extra, cells, stop):
    stored = args.checkpoint_dir/"traces"
    checkpoint = args.checkpoint_dir/"preflight_latest.pt"
    for index in range(progress["next_cell"], stop):
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
        np.savez_compressed(trace_path(stored, cell), **trace)
        if cell["mode"] != "normal":
            normal = v68.load_trace(trace_path(stored, dict(cell, mode="normal")))
            assert v68.v53.paired_prefix_equal(trace, normal, SPEC["switch_step"]), "non-matched-prefix"
        progress["records"].append(record)
        progress["next_cell"] = index+1
        v68.v53.save_progress(agent, env, progress, None, checkpoint)
        for physical in (env, *shadows.values()):
            physical.close()
        extra = None
    return extra


def evaluation_inputs(args, stored):
    output = [(dict(group="v71_fresh_frozen", seed=c["seed"], mode=c["mode"], updating=False), trace_path(stored, c))
              for c in test_cells(SPEC)]
    output += [(dict(group="v70_development_replay", **c, updating=False),
                v70.trace_path(args.v70_dir/"run_checkpoints/preflight/traces", c)) for c in v70.audit_cells(v70.SPEC)]
    old = json.loads((args.v69_dir/"run_results/screen/counterfactual.json").read_text())
    assert old["revision"] == v69.REVISION
    for r in old["records"]:
        cell = {k: r[k] for k in ("group", "seed", "mode", "updating")}
        if cell["group"] == "v62_test":
            kind = "updating" if cell["updating"] else "frozen"
            path = args.v62_dir/f"run_checkpoints/preflight/traces/test_seed{cell['seed']}_{kind}_{cell['mode']}.npz"
        else:
            assert cell["group"] == "v61_development_replay"
            path = args.v61_dir/f"run_checkpoints/preflight/traces/seed{cell['seed']}_{cell['mode']}.npz"
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
                "mean_true_minus_best_other_log_likelihood", "predicted_mode_fractions")} for w in WINDOWS}
        record = dict(**cell, observers=observers)
        control = record if cell["mode"] == "normal" else next(r for r in records
            if r["group"] == cell["group"] and r["seed"] == cell["seed"]
            and r["updating"] == cell["updating"] and r["mode"] == "normal")
        record["change_minus_nochange"] = {name: {w: d["mean_coefficient"]-control["observers"][name][w]["mean_coefficient"]
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
        for name in REFERENCES:
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
    check_seed_partition(SPEC)
    stored = args.checkpoint_dir/"traces"
    stored.mkdir(parents=True, exist_ok=True)
    extra = torch.load(args.resume_from, map_location="cpu", weights_only=False)["extra_state"] if args.resume_from else None
    progress = extra["progress"] if extra else dict(revision=REVISION, spec=SPEC, next_cell=0, records=[])
    assert progress["revision"] == REVISION and progress["spec"] == SPEC
    shared = torch.load(args.shared_dir/"shared.pt", map_location="cpu", weights_only=False)
    ref_cells = reference_cells(SPEC)
    cells = ref_cells+test_cells(SPEC)
    extra = collect_cells(args, shared, progress, extra, cells, len(ref_cells))
    destination = args.checkpoint_dir/"coverage_references.pt"
    if destination.is_file():
        saved = torch.load(destination, map_location="cpu", weights_only=False)
        assert saved["revision"] == REVISION and saved["spec"] == SPEC
        references = saved["references"]
    else:
        references = fit_references(args, stored)
        torch.save(dict(revision=REVISION, spec=SPEC, references=references), destination)
    print("[REFERENCE] Equal-count normal-only and shifted-state banks frozen BEFORE fresh tests", flush=True)
    metadata = {name: deepcopy(ref.metadata()) for name, ref in references.items()}
    collect_cells(args, shared, progress, extra, cells, len(cells))
    records = evaluate(args, stored, references)
    assert metadata == {name: ref.metadata() for name, ref in references.items()}
    summary = summarize(records)
    totals = {k: sum(r[k] for r in progress["records"]) for k in ("actual_transitions", "counterfactual_step_calls",
                "environment_step_calls", "exact_true_mode_replays", "optimizer_updates")}
    assert totals["actual_transitions"] == len(cells)*SPEC["steps"] and totals["optimizer_updates"] == 0
    passed = all(group["shifted_state"]["screen_pass"] for group in summary.values())
    compact_metadata = {name: {k: m[k] for k in ("bank", "window_steps", "mode_weights", "settings",
        "normal_null_calibration_transitions", "mode_labels_used_online", "switch_step_used_online", "actor_feedback")}
        for name, m in metadata.items()}
    output = dict(revision=REVISION, spec=SPEC, source_training_seed=1, source_updates=99745,
        conditions=list(CONDITIONS), reference_metadata=compact_metadata, reference_checkpoint=str(destination),
        original_labels_per_mode_per_role=2*v69.SPEC["steps"],
        added_labels_per_mode_per_role=3*(SPEC["steps"]-SPEC["switch_step"]),
        normal_null_recalibrated_per_reference=True, policy_weights_frozen=True, readout_controls_actor=False,
        reference_records=progress["records"], records=records, summary=summary, **totals,
        decision="coverage-screen-pass-not-RL-confirmation" if passed else "hold-state-coverage",
        limitations=["One source training seed; new test seeds are frozen-policy diagnostics, not control performance evidence.",
            "Known mild profiles and privileged simulator-state counterfactual labels.",
            "Coverage expansion changes fitting and density-calibration state support together; their separate effects are not isolated.",
            "v70/v61/v62 are development replays, never reference-fitting inputs."])
    args.result_dir.mkdir(parents=True, exist_ok=True)
    path = args.result_dir/"state_coverage.json"
    path.write_text(json.dumps(output, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] {output['decision']}: {path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ("source-dir", "shared-dir", "v69-dir", "v70-dir", "v61-dir", "v62-dir", "checkpoint-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
