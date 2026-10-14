"""Equal-count normal/slippery density coverage with fixed temporal readouts."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from scripts import probe_general_ood_v73_recursive_mode as v73
from scripts.v72_conditional_likelihood import calibrate_bank
from scripts.v73_recursive_mode_readout import MODES, NAMES, RecursiveModeReference, observe_likelihoods
from scripts.v74_density_coverage import DensityCoverageBank

v72, v71, v69, v68 = v73.v72, v73.v71, v73.v69, v73.v68
REVISION = "general_ood_v74_density_coverage"
SPEC = dict(density_calibration=[135150, 135151], test=[135160, 135161], steps=6000, switch_step=3800)
CONDITIONS = ("v72", "normal_state", "slippery_state")
BASE_DENSITY_TRANSITIONS = 18600
KEYS = ("states", "executed_actions", "next_states")


def reference_cells(spec):
    return [dict(role="density_extension", seed=seed, mode=mode)
            for seed in spec["density_calibration"] for mode in ("normal", "slippery")]


def test_cells(spec):
    return [dict(role="test", seed=seed, mode=mode) for seed in spec["test"] for mode in MODES]


def check_seed_partition():
    calibration, test = (set(SPEC[k]) for k in ("density_calibration", "test"))
    assert len(calibration) == len(test) == 2 and not calibration & test
    previous = set(v73.SPEC["test"]+v72.SPEC["test"])
    previous |= set(v71.SPEC["fit"]+v71.SPEC["density_calibration"]+v71.SPEC["test"])
    previous |= set(v73.v70.SPEC["seeds"]) | set(v68.BASE.SPEC["test"]) | {132020}
    previous |= {spec[k] for spec in (v69.SPEC, v68.SPEC) for k in spec if k.endswith("seed")}
    assert not (calibration | test) & previous
    # Reuse the established collector and statistics at their unchanged time contract.
    assert all(SPEC[k] == v71.SPEC[k] == v73.SPEC[k] for k in ("steps", "switch_step"))


def extensions(stored, spec, condition):
    assert condition in CONDITIONS[1:]
    mode = "normal" if condition == "normal_state" else "slippery"
    traces = [v68.load_trace(v71.trace_path(stored, dict(role="density_extension", seed=s, mode=mode)))
              for s in spec["density_calibration"]]
    suffix = slice(spec["switch_step"], spec["steps"])
    return {m: {key: np.concatenate([t["oracle_observations"][suffix, i] if key == "next_states"
                                    else t[key][suffix] for t in traces]) for key in KEYS}
            for i, m in enumerate(MODES)}


def freeze_references(args, stored):
    saved = torch.load(args.v73_dir/"run_checkpoints/preflight/temporal_references.pt", map_location="cpu", weights_only=False)
    assert saved["revision"] == v73.REVISION and saved["spec"] == v73.SPEC
    baseline = saved["references"]
    original = baseline["window16"].bank
    assert original is baseline["recursive"].bank and original.density_transitions == BASE_DENSITY_TRANSITIONS
    old = args.v69_dir/"run_checkpoints/reference/traces"
    nulls = [v68.load_trace(v69.trace_path(old, c)) for c in v69.reference_cells(v69.SPEC) if c["role"] == "normal_null"]
    references = dict(v72=baseline)
    for condition in CONDITIONS[1:]:
        bank = DensityCoverageBank.extend(original, extensions(stored, SPEC, condition), condition)
        assert bank.added_density_transitions == len(SPEC["density_calibration"])*(SPEC["steps"]-SPEC["switch_step"])
        for mode in MODES:
            np.testing.assert_array_equal(bank.coefficients[mode], original.coefficients[mode])
        references[condition] = dict(window16=calibrate_bank(bank, nulls), recursive=RecursiveModeReference.calibrate(bank, nulls))
        assert all(r.normal_null_calibration_transitions == baseline["window16"].normal_null_calibration_transitions
                   for r in references[condition].values())
        print(f"[DENSITY] {condition}: +{bank.added_density_transitions}/mode, forward coefficients/scales frozen", flush=True)
    return references


def evaluation_inputs(args, stored):
    output = [(dict(group="v74_fresh_frozen", seed=c["seed"], mode=c["mode"], updating=False), v71.trace_path(stored, c))
              for c in test_cells(SPEC)]
    for cell, path in v73.evaluation_inputs(args, args.v73_dir/"run_checkpoints/preflight/traces"):
        if cell["group"] == "v73_fresh_frozen":
            cell = dict(cell, group="v73_development_replay")
        output.append((cell, path))
    return output


def mean_error_diagnostics(trace, bank, mode):
    output = {}
    for window, a, b in (("first_post50", SPEC["switch_step"], SPEC["switch_step"]+50),
                         ("last64", SPEC["steps"]-64, SPEC["steps"])):
        query = {k: trace[k][a:b] for k in KEYS}
        center = bank.local_centers(bank.features.inputs(query))[mode]
        prediction = query["states"]+(bank.design(query)@bank.coefficients[mode]+center)*bank.target_scale
        errors = np.sqrt(np.mean(((prediction-query["next_states"])/bank.target_scale)**2, axis=1))
        gaps = np.sqrt(np.mean(((trace["oracle_observations"][a:b]-query["next_states"][:, None])/bank.target_scale)**2, axis=2))
        gaps[:, MODES.index(mode)] = np.inf
        nearest = gaps.min(axis=1)
        output[window] = dict(mean_conditional_model_error=float(errors.mean()), mean_nearest_mode_observation_gap=float(nearest.mean()),
                             error_exceeds_gap_fraction=float(np.mean(errors > nearest)))
    return output


def evaluate(args, stored, references):
    records = []
    for cell, path in evaluation_inputs(args, stored):
        trace = v68.load_trace(path)
        observers, diagnostics = {}, {}
        for condition, readouts in references.items():
            bank = readouts["window16"].bank
            assert bank is readouts["recursive"].bank
            likelihoods = bank.log_likelihoods(trace)
            observers[condition] = {name: v73.window_statistics(trace, likelihoods,
                observe_likelihoods(likelihoods, trace["dones"], ref, name), cell["mode"]) for name, ref in readouts.items()}
            if cell["group"] == "v74_fresh_frozen":
                diagnostics[condition] = mean_error_diagnostics(trace, bank, cell["mode"])
        record = dict(**cell, observers=observers, density_diagnostics=diagnostics)
        normal = record if cell["mode"] == "normal" else next(r for r in records
            if r["group"] == cell["group"] and r["seed"] == cell["seed"] and r["updating"] == cell["updating"] and r["mode"] == "normal")
        record["change_minus_nochange"] = {condition: {name: {w: d["mean_coefficient"]-normal["observers"][condition][name][w]["mean_coefficient"]
            for w, d in windows.items()} for name, windows in readouts.items()} for condition, readouts in observers.items()}
        records.append(record)
        print(f"[EVAL] {cell['group']} seed{cell['seed']} {cell['mode']} updating={cell['updating']}", flush=True)
    return records


def summarize(records):
    output = {}
    for group in dict.fromkeys(r["group"] for r in records):
        normal = [r for r in records if r["group"] == group and r["mode"] == "normal"]
        changed = [r for r in records if r["group"] == group and r["mode"] != "normal"]
        output[group] = {}
        for condition in CONDITIONS:
            output[group][condition] = {}
            for name in NAMES:
                maximum = max(r["observers"][condition][name][w]["mean_coefficient"] for r in normal for w in v68.BASE.NORMAL_WINDOWS)
                contrasts = {w: sum(r["change_minus_nochange"][condition][name][w] > 0 for r in changed) for w in v68.BASE.CHANGE_WINDOWS}
                margins = {w: sum(r["observers"][condition][name][w]["mean_true_minus_best_other_score"] > 0 for r in changed)
                           for w in v68.BASE.CHANGE_WINDOWS}
                output[group][condition][name] = dict(changed_pairs=len(changed), largest_normal_mean=maximum,
                    positive_risk_contrast_pairs=contrasts, positive_true_mode_margin_pairs=margins,
                    mean_changed_accuracy={w: float(np.mean([r["observers"][condition][name][w]["mode_accuracy"] for r in changed]))
                                           for w in v68.BASE.CHANGE_WINDOWS},
                    screen_pass=maximum <= .1 and all(n == len(changed) for n in (*contrasts.values(), *margins.values())))
    return output


def run(args):
    torch.set_num_threads(2)
    check_seed_partition()
    stored = args.checkpoint_dir/"traces"
    stored.mkdir(parents=True, exist_ok=True)
    extra = torch.load(args.resume_from, map_location="cpu", weights_only=False)["extra_state"] if args.resume_from else None
    progress = extra["progress"] if extra else dict(revision=REVISION, spec=SPEC, next_cell=0, records=[])
    assert progress["revision"] == REVISION and progress["spec"] == SPEC
    shared = torch.load(args.shared_dir/"shared.pt", map_location="cpu", weights_only=False)
    ref_cells = reference_cells(SPEC)
    cells = ref_cells+test_cells(SPEC)
    extra = v71.collect_cells(args, shared, progress, extra, cells, len(ref_cells))
    destination = args.checkpoint_dir/"density_coverage_references.pt"
    if destination.is_file():
        saved = torch.load(destination, map_location="cpu", weights_only=False)
        assert saved["revision"] == REVISION and saved["spec"] == SPEC
        references = saved["references"]
    else:
        references = freeze_references(args, stored)
        torch.save(dict(revision=REVISION, spec=SPEC, references=references), destination)
    print("[REFERENCE] Three density banks and both fixed temporal readouts frozen BEFORE fresh tests", flush=True)
    metadata = {condition: {name: deepcopy(ref.metadata()) for name, ref in readouts.items()} for condition, readouts in references.items()}
    v71.collect_cells(args, shared, progress, extra, cells, len(cells))
    records = evaluate(args, stored, references)
    assert metadata == {condition: {name: ref.metadata() for name, ref in readouts.items()} for condition, readouts in references.items()}
    summary = summarize(records)
    totals = {k: sum(r[k] for r in progress["records"]) for k in ("actual_transitions", "counterfactual_step_calls",
                "environment_step_calls", "exact_true_mode_replays", "optimizer_updates")}
    assert totals["actual_transitions"] == len(cells)*SPEC["steps"] and totals["optimizer_updates"] == 0
    passed = all(group["slippery_state"]["recursive"]["screen_pass"] for group in summary.values())
    compact = {condition: {name: {k: v for k, v in m.items() if k != "phase"} for name, m in readouts.items()}
               for condition, readouts in metadata.items()}
    added = len(SPEC["density_calibration"])*(SPEC["steps"]-SPEC["switch_step"])
    output = dict(revision=REVISION, spec=SPEC, source_training_seed=1, source_updates=99745,
        reference_metadata=compact, margin_units=v73.MARGIN_UNITS, server_reference_checkpoint=str(destination),
        reference_records=progress["records"][:len(ref_cells)], test_records=progress["records"][len(ref_cells):],
        added_density_labels_per_mode_per_condition=added, density_extension_labels_used=added*len(MODES)*2,
        forward_models_refitted=False, scaling_refitted=False, conditional_density_refitted=True,
        temporal_rules_changed=False, normal_null_recalibrated_per_readout=True,
        readout_controls_actor=False, policy_weights_frozen=True, records=records, summary=summary, **totals,
        decision="density-coverage-sign-screen-pass-not-RL-confirmation" if passed else "hold-density-coverage",
        limitations=["One source training seed, mild known modes and privileged offline counterfactual labels.",
            "Density mean/covariance support changes together; forward models and temporal rules stay fixed.",
            "Sign-based screening does not establish a tail-accuracy improvement or adaptive-control benefit.",
            "Previously inspected frozen/updating traces remain development replays."])
    args.result_dir.mkdir(parents=True, exist_ok=True)
    path = args.result_dir/"density_coverage.json"
    path.write_text(json.dumps(output, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] {output['decision']}: {path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ("source-dir", "shared-dir", "v73-dir", "v72-dir", "v71-dir", "v69-dir", "v70-dir", "v61-dir", "v62-dir", "checkpoint-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
