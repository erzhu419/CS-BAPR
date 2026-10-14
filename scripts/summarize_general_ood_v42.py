"""Summarize calibrated dynamics residuals; no policy-improvement claim."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.summarize_general_ood_v41 import contrast_statistics

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "experiments/general_ood_v42_frozen_dynamics_20261006"
DESTINATION = ROOT / "results/general_ood_v42_frozen_dynamics"
KEYS = ("mean_score", "first_transition_score", "fraction_above_calibration95", "nonterminal_mean_score")


def validate_run(data, manifest, stage):
    policy = data["source_policy"]
    seeds = manifest["preflight_evaluation_seeds"] if stage == "preflight" else manifest["evaluation_seeds"]
    modes = manifest["preflight_target_modes"] if stage == "preflight" else manifest["target_modes"]
    assert data["revision"] == manifest["revision"] and data["stage"] == stage
    assert policy in manifest["source_policies"] and data["source_training_seed"] == 1
    assert data["evaluation_seeds"] == seeds and data["target_modes"] == modes
    for key in ("predictor_names", "fit_range", "calibration_range", "validation_range", "phases",
                "ridge_mean_loss_coefficient", "scale_floor", "probe_interactions_per_rollout", "switch_step",
                "probe_episode_time_limit", "optimizer_updates_during_probe", "replay_insertions",
                "auxiliary_signal_controls_actions", "predictors_updated_after_calibration"):
        assert data[key] == manifest[key], (policy, key)
    assert data["actor_critic_target_frozen"] and data["restore_training_context_before_each_rollout"]
    assert data["loaded_training_context_count"] == data["source_training_interactions"] == 100000
    assert data["loaded_training_update_steps"] == 99745
    source = json.loads((ROOT / f"results/general_ood_v40_strength/formal/Hopper-v4/seed1/general_ood_v40_strength_Hopper-v4_{policy}_seed1.json").read_text())
    assert data["config"] == source["config"] and data["runtime"] == source["runtime"]
    assert data["source_checkpoint"] == source["checkpoint"]
    records = data["records"]
    assert len(records) == len(seeds) * len(modes)
    assert data["total_probe_interactions"] == len(records) * 1536
    indexed = {(r["seed"], r["target_mode"]): r for r in records}
    assert set(indexed) == {(s, m) for s in seeds for m in modes}
    for row in records:
        control = indexed[row["seed"], "normal"]
        assert row["paired_prefix_equal"] and row["interactions"] == 1536
        assert row["actual_change"] == (row["target_mode"] != "normal")
        assert row["reward_means_by_256"][:4] == control["reward_means_by_256"][:4]
        for name in manifest["predictor_names"]:
            for phase, (start, end) in manifest["phases"].items():
                observed, baseline = row["predictors"][name][phase], control["predictors"][name][phase]
                assert observed["transitions"] == end - start
                assert all(v is None or math.isfinite(v) for v in observed.values())
                assert 0 <= observed["fraction_above_calibration95"] <= 1
                ended = end - start - observed["nonterminal_transitions"]
                assert 0 <= observed["terminated"] <= ended
                assert 0 <= observed["truncated"] <= ended
                assert ended <= observed["terminated"] + observed["truncated"]
                if phase == "validation256":
                    assert observed == baseline
                computed = {k: observed[k] - baseline[k] for k in
                            ("mean_score", "first_transition_score", "fraction_above_calibration95")}
                assert computed == row["change_minus_nochange"][name][phase]
    # Same source policies/seeds as v41: instrumenting actual next states must not change behavior.
    old = json.loads((ROOT / f"results/general_ood_v41_frozen_signal/{stage}/general_ood_v41_frozen_signal_{policy}.json").read_text())
    previous = {(r["seed"], r["target_mode"]): r for r in old["records"]}
    for key, row in indexed.items():
        assert row["reward_means_by_256"] == previous[key]["reward_means_by_256"]
        assert [a + b for a, b in zip(row["termination_counts_by_256"], row["truncation_counts_by_256"])] == previous[key]["episode_endings_by_256"]
        assert row["reward_reference_by_256"][4]["surprise_raw"] == previous[key]["observers"]["previous-chunk-reference"][4]["means"]["surprise_raw"]
    return data


def values_statistics(values):
    return {"mean": mean(values), "median": median(values), "range": [min(values), max(values)]}


def summarize(manifest, destination, stage):
    runs, missing = {}, []
    for policy in manifest["source_policies"]:
        path = destination / stage / f"{manifest['revision']}_{policy}.json"
        if path.exists():
            runs[policy] = validate_run(json.loads(path.read_text()), manifest, stage)
        else:
            missing.append(policy)
    result = {"revision": manifest["revision"], "stage": stage, "missing_policies": missing,
              "matrix_complete": not missing, "decision": "pending-missing-policies"}
    if missing:
        return result
    methods = {}
    for policy, data in runs.items():
        indexed = {(r["seed"], r["target_mode"]): r for r in data["records"]}
        controls = [indexed[s, "normal"] for s in data["evaluation_seeds"]]
        null = {name: {phase: {key: values_statistics([r["predictors"][name][phase][key] for r in controls])
                              for key in KEYS} for phase in manifest["phases"]}
                for name in manifest["predictor_names"]}
        profiles = {}
        for mode in data["target_modes"]:
            if mode == "normal":
                continue
            rows = [indexed[s, mode] for s in data["evaluation_seeds"]]
            profiles[mode] = {
                "predictors": {name: {phase: {key: contrast_statistics(
                    [r["predictors"][name][phase][key] for r in rows],
                    [r["predictors"][name][phase][key] for r in controls]) for key in KEYS}
                    for phase in manifest["phases"]} for name in manifest["predictor_names"]},
                "endings": {phase: {f"{group}_{kind}": sum(r["predictors"][manifest["predictor_names"][0]][phase][kind] for r in group_rows)
                                     for group, group_rows in (("actual", rows), ("nochange", controls))
                                     for kind in ("terminated", "truncated")} for phase in manifest["phases"]},
                "reward_reference": {phase: contrast_statistics(
                    [r["reward_reference_by_256"][i]["surprise_raw"] for r in rows],
                    [r["reward_reference_by_256"][i]["surprise_raw"] for r in controls])
                    for phase, i in (("first_post256", 4), ("second_post256", 5))}}
        methods[policy] = {"null": null, "profiles": profiles, "seconds": data["seconds"],
                           "source_checkpoint": data["source_checkpoint"]}
    result.update(methods=methods, decision="diagnostic-complete-not-RL-confirmation",
                  optimizer_updates=0, new_training_seeds=0,
                  statistics_unit="paired evaluation seeds conditional on two frozen seed1 policies",
                  limitations=["Per-policy/seed local predictor, not a trained general dynamics model.",
                               "Calibration and forecast validity are judged on untouched no-change data, not guaranteed by normalization.",
                               "Nonterminal windows can include post-reset states; first-transition and first16 results are also reported.",
                               "Frozen deterministic policies confound action with state; state-only control is not a causal action-identification test.",
                               "This does not test revised belief feedback or improved RL returns."])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "formal"), default="formal")
    args = parser.parse_args()
    result = summarize(json.loads((STUDY / "dispatch_manifest.json").read_text()), DESTINATION, args.stage)
    path = DESTINATION / ("preflight_summary.json" if args.stage == "preflight" else "summary.json")
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(result["decision"], "saved", path)


if __name__ == "__main__":
    main()
