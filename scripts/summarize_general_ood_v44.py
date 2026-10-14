"""Matched frozen belief diagnostics; never an RL performance gate."""
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
STUDY = ROOT / "experiments/general_ood_v44_frozen_belief_20261006"
DESTINATION = ROOT / "results/general_ood_v44_frozen_belief"
METRICS = ("reset_probability", "effective_age", "penalty", "hypothetical_beta", "entropy")


def validate_run(data, manifest, stage, root=ROOT):
    policy = data["source_policy"]
    assert data["revision"] == manifest["revision"] and data["stage"] == stage
    assert policy in manifest["source_policies"]
    for key in ("fit_seeds", "calibration_seeds", "evaluation_seeds", "target_modes"):
        assert data[key] == manifest[stage][key], (policy, key)
    for key in ("predictor", "calibrator", "filter_names", "filter_settings", "signal_transform", "phases",
                "post_bin_width", "reset_filters_at_episode_end", "optimizer_updates_during_probe", "replay_insertions",
                "auxiliary_signal_controls_actions", "predictors_updated_after_calibration", "actor_critic_target_frozen"):
        assert data[key] == manifest[key], (policy, key)
    assert data["loaded_training_update_steps"] == 99745 and data["loaded_training_context_count"] == 100000
    old = json.loads((root / f"results/general_ood_v43_normal_reference/{stage}/general_ood_v43_normal_reference_{policy}.json").read_text())
    for key in ("config", "runtime", "source_checkpoint", "source_training_seed", "references",
                "total_reference_interactions", "total_probe_interactions"):
        assert data[key] == old[key], (policy, key, "v43-source-changed")
    assert data["model_metadata"] == old["model_metadata"][manifest["predictor"]], "v43-model-changed"
    records = data["records"]
    indexed = {(r["seed"], r["target_mode"]): r for r in records}
    expected = {(s, m) for s in data["evaluation_seeds"] for m in data["target_modes"]}
    assert set(indexed) == expected and len(records) == len(expected)
    previous = {(r["seed"], r["target_mode"]): r for r in old["records"]}
    for key, row in indexed.items():
        assert row["paired_prefix_equal"] and row["paired_switch_inputs_equal"] and row["interactions"] == 1536
        assert row["actual_change"] == (row["target_mode"] != "normal")
        for field in ("reward_means_by_256", "termination_counts_by_256", "truncation_counts_by_256"):
            assert row[field] == previous[key][field], (policy, key, "v43-trajectory-changed")
        assert row["v43_score_check"] == previous[key]["predictors"][manifest["predictor"]][manifest["calibrator"]], "v43-score-changed"
        for name in data["filter_names"]:
            assert row["observers"][name]["validation256"] == indexed[row["seed"], "normal"]["observers"][name]["validation256"]
            assert len(row["post_bins32"][name]) == 16
            for phase in manifest["phases"]:
                obs = row["observers"][name][phase]
                assert all(math.isfinite(v) for stats in obs.values() for v in stats.values())
                for stat in ("mean", "peak", "end"):
                    assert 0 <= obs["reset_probability"][stat] <= 1
                    assert 0 <= obs["effective_age"][stat] <= 19
                    assert -3 <= obs["hypothetical_beta"][stat] <= -2
                assert abs(obs["hypothetical_beta"]["mean"] + 2 + obs["penalty"]["mean"]) < 1e-12
    return data


def summarize(manifest, destination, stage):
    runs, missing = {}, []
    for policy in manifest["source_policies"]:
        path = destination / stage / f"{manifest['revision']}_{policy}.json"
        if not path.exists():
            missing.append(policy)
        else:
            runs[policy] = validate_run(json.loads(path.read_text()), manifest, stage)
    result = dict(revision=manifest["revision"], stage=stage, missing_policies=missing,
                  matrix_complete=not missing, decision="pending-missing-policies")
    if missing:
        return result
    methods = {}
    for policy, data in runs.items():
        controls = [r for r in data["records"] if r["target_mode"] == "normal"]
        by_mode = {}
        for mode in data["target_modes"]:
            rows = [r for r in data["records"] if r["target_mode"] == mode]
            by_mode[mode] = {
                "signal": {phase: values_statistics([r["signal"][phase]["mean"] for r in rows]) for phase in manifest["phases"]},
                "endings": {phase: {kind: sum(r["phase_endings"][phase][kind] for r in rows) for kind in ("terminated", "truncated")} for phase in manifest["phases"]},
                "filters": {name: {
                    "phases": {phase: {metric: {stat: values_statistics([r["observers"][name][phase][metric][stat] for r in rows])
                                                   for stat in ("mean", "peak", "end")} for metric in METRICS} for phase in manifest["phases"]},
                    "paired_mean_contrasts": {phase: {metric: contrast_statistics(
                        [r["observers"][name][phase][metric]["mean"] for r in rows],
                        [r["observers"][name][phase][metric]["mean"] for r in controls]) for metric in METRICS} for phase in manifest["phases"]},
                    "first16_cases_peak_reset_above_half": sum(r["observers"][name]["first_post16"]["reset_probability"]["peak"] > .5 for r in rows),
                    "cases_peak_reset_above_half": {phase: sum(r["observers"][name][phase]["reset_probability"]["peak"] > .5 for r in rows)
                                                    for phase in manifest["phases"]},
                    "post_bins32_means": [{metric: mean(r["post_bins32"][name][i][metric] for r in rows) for metric in METRICS} for i in range(16)]
                } for name in data["filter_names"]}}
        methods[policy] = dict(modes=by_mode, seconds=data["seconds"])
    result.update(methods=methods, decision="frozen-belief-diagnostic-complete-not-RL-confirmation", optimizer_updates=0,
                  new_training_seeds=0, environment_interactions=sum(d["total_reference_interactions"] + d["total_probe_interactions"] for d in runs.values()),
                  limitations=["Reused seed1 policies and development trajectories, not independent confirmation.",
                               "Student predictive likelihood changes both residual location and variance; positive log scores use a real-valued approximation.",
                               "Capped adaptive bin uses at most 19 preceding samples, not full unbounded BOCD.",
                               "Hypothetical beta never affects actions, gradients, replay, or return here.",
                               "Later windows include physical failures; first16 is also reported separately."])
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
