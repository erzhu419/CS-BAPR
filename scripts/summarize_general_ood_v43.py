"""Coverage/calibration diagnostics on the same frozen-policy development traces."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.summarize_general_ood_v41 import contrast_statistics
from scripts.summarize_general_ood_v42 import values_statistics

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "experiments/general_ood_v43_normal_reference_20261006"
DESTINATION = ROOT / "results/general_ood_v43_normal_reference"
KEYS = ("mean_score", "first_transition_score", "fraction_above_calibration95", "nonterminal_mean_score")


def validate_run(data, manifest, stage):
    policy = data["source_policy"]
    split = manifest[stage]
    assert data["revision"] == manifest["revision"] and data["stage"] == stage
    assert policy in manifest["source_policies"]
    for key in ("fit_seeds", "calibration_seeds", "evaluation_seeds", "target_modes"):
        assert data[key] == split[key], (policy, key)
    assert len(set(data["fit_seeds"] + data["calibration_seeds"] + data["evaluation_seeds"])) == sum(len(data[k]) for k in ("fit_seeds", "calibration_seeds", "evaluation_seeds"))
    for key in ("predictor_names", "calibrator_names", "reference_interactions_per_rollout",
                "probe_interactions_per_rollout", "switch_step", "episode_time_limit", "phases",
                "episode_age_bin_width", "episode_age_groups", "ridge_mean_loss_coefficient", "scale_floor",
                "optimizer_updates_during_probe", "replay_insertions", "auxiliary_signal_controls_actions",
                "predictors_updated_after_calibration"):
        assert data[key] == manifest[key], (policy, key)
    assert data["actor_critic_target_frozen"] and data["restore_training_context_before_each_rollout"]
    assert data["normal_only_reference_modes"] and data["independent_reference_and_test_seeds"]
    assert data["loaded_training_update_steps"] == 99745
    assert data["loaded_training_context_count"] == data["source_training_interactions"] == 100000
    previous = json.loads((ROOT / f"results/general_ood_v42_frozen_dynamics/{stage}/general_ood_v42_frozen_dynamics_{policy}.json").read_text())
    assert data["config"] == previous["config"] and data["runtime"] == previous["runtime"]
    assert data["source_checkpoint"] == previous["source_checkpoint"] and data["source_training_seed"] == 1
    references = data["references"]
    expected = {(seed, split_name) for split_name, key in (("fit", "fit_seeds"), ("calibration", "calibration_seeds")) for seed in split[key]}
    assert len(references) == len(expected) and {(r["seed"], r["split"]) for r in references} == expected
    assert data["total_reference_interactions"] == len(references) * 3072
    for row in references:
        assert row["mode"] == "normal" and row["interactions"] == 3072
        assert sum(row["episode_age_group_counts"]) == 3072
        assert row["completed_episode_lengths"] and sum(row["completed_episode_lengths"]) <= 3072
    for name, model in data["model_metadata"].items():
        assert name in manifest["predictor_names"]
        for channel, calibration in model["calibrations"].items():
            expected_counts = ([sum(r["episode_age_group_counts"][i] for r in references if r["split"] == "calibration") for i in range(4)]
                               if channel == "episode-centered" else [len(split["calibration_seeds"]) * 3072])
            assert calibration["counts"] == expected_counts and all(expected_counts)
            assert all(math.isfinite(t) and t > 0 for t in calibration["threshold95"])
    records = data["records"]
    assert len(records) == len(split["evaluation_seeds"]) * len(split["target_modes"])
    indexed = {(r["seed"], r["target_mode"]): r for r in records}
    assert set(indexed) == {(s, m) for s in split["evaluation_seeds"] for m in split["target_modes"]}
    assert data["total_probe_interactions"] == len(records) * 1536
    old = {(r["seed"], r["target_mode"]): r for r in previous["records"]}
    for key, row in indexed.items():
        control = indexed[row["seed"], "normal"]
        assert row["paired_prefix_equal"] and row["paired_switch_inputs_equal"] and row["interactions"] == 1536
        assert row["actual_change"] == (row["target_mode"] != "normal")
        for field in ("reward_means_by_256", "termination_counts_by_256", "truncation_counts_by_256"):
            assert row[field] == old[key][field], (policy, key, "v42-trajectory-changed")
        for name in manifest["predictor_names"]:
            for channel in manifest["calibrator_names"]:
                for phase in manifest["phases"]:
                    observed, baseline = row["predictors"][name][channel][phase], control["predictors"][name][channel][phase]
                    assert all(v is None or math.isfinite(v) for v in observed.values())
                    assert 0 <= observed["fraction_above_calibration95"] <= 1
                    if phase == "validation256":
                        assert observed == baseline
                    assert {k: observed[k] - baseline[k] for k in KEYS[:3]} == row["change_minus_nochange"][name][channel][phase]
    return data, previous


def summarize(manifest, destination, stage):
    runs, missing = {}, []
    for policy in manifest["source_policies"]:
        path = destination / stage / f"{manifest['revision']}_{policy}.json"
        if not path.exists():
            missing.append(policy)
        else:
            runs[policy] = validate_run(json.loads(path.read_text()), manifest, stage)
    result = {"revision": manifest["revision"], "stage": stage, "missing_policies": missing,
              "matrix_complete": not missing, "decision": "pending-missing-policies"}
    if missing:
        return result
    methods = {}
    for policy, (data, previous) in runs.items():
        controls = [r for r in data["records"] if r["target_mode"] == "normal"]
        old_controls = [r for r in previous["records"] if r["target_mode"] == "normal"]
        null = {name: {channel: {phase: {key: values_statistics([r["predictors"][name][channel][phase][key] for r in controls])
                                        for key in KEYS} for phase in manifest["phases"]}
                       for channel in manifest["calibrator_names"]} for name in manifest["predictor_names"]}
        comparison = {name: {phase: {"v42_fraction_above_calibration95": mean(r["predictors"][name][phase]["fraction_above_calibration95"] for r in old_controls),
                                    "v43_global_rms_fraction_above_calibration95": null[name]["global-rms"][phase]["fraction_above_calibration95"]["mean"]}
                             for phase in manifest["phases"]} for name in manifest["predictor_names"]}
        profiles = {}
        for mode in data["target_modes"]:
            if mode == "normal":
                continue
            rows = [r for r in data["records"] if r["target_mode"] == mode]
            profiles[mode] = {
                "predictors": {name: {channel: {phase: {key: contrast_statistics(
                    [r["predictors"][name][channel][phase][key] for r in rows],
                    [r["predictors"][name][channel][phase][key] for r in controls]) for key in KEYS}
                    for phase in manifest["phases"]} for channel in manifest["calibrator_names"]}
                    for name in manifest["predictor_names"]},
                "endings": {phase: {f"{group}_{kind}": sum(r["phase_endings"][phase][kind] for r in group_rows)
                                     for group, group_rows in (("actual", rows), ("nochange", controls))
                                     for kind in ("terminated", "truncated")} for phase in manifest["phases"]}}
        methods[policy] = {"null": null, "profiles": profiles, "v42_coverage_comparison": comparison,
                           "model_metadata": data["model_metadata"], "reference_rollouts": len(data["references"]),
                           "source_checkpoint": data["source_checkpoint"], "seconds": data["seconds"]}
    result.update(methods=methods, decision="diagnostic-complete-not-RL-confirmation", optimizer_updates=0,
                  new_training_seeds=0, statistics_unit="paired reused development seeds conditional on frozen seed1 policies",
                  limitations=["Reference capacity/settings are fixed, but fitting and calibration coverage change together.",
                               "Episode age is observable history, not a latent-mode label or general state-conditioned variance model.",
                               "Calibration quantiles need not provide a 5% false-alarm rate on test trajectories.",
                               "Normal test seeds repeat v42 development cases; no independent RL/control confirmation."])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "formal"), default="formal")
    args = parser.parse_args()
    result = summarize(json.loads((STUDY / "dispatch_manifest.json").read_text()), DESTINATION, args.stage)
    path = DESTINATION / ("summary.json" if args.stage == "formal" else "preflight_summary.json")
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(result["decision"], "saved", path)


if __name__ == "__main__":
    main()
