"""Whole-trace false triggers and paired response for all crossed arms."""
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
STUDY = ROOT / "experiments/general_ood_v46_crossed_tails_20261006"
DESTINATION = ROOT / "results/general_ood_v46_crossed_tails"
METRICS = ("reset_probability", "effective_age", "penalty", "hypothetical_beta", "entropy")


def validate_run(data, manifest, stage, root=ROOT):
    policy = data["source_policy"]
    assert data["revision"] == manifest["revision"] and data["stage"] == stage
    assert policy in manifest["source_policies"]
    for key in ("fit_seeds", "calibration_seeds", "evaluation_seeds", "target_modes"):
        assert data[key] == manifest[stage][key], (policy, key)
    for key in ("predictor", "arm_calibrations", "arm_likelihoods", "filter_settings", "robust_student_degrees", "phases",
                "reset_filters_at_episode_end", "optimizer_updates_during_probe", "replay_insertions",
                "auxiliary_signal_controls_actions", "predictors_updated_after_calibration", "actor_critic_target_frozen"):
        assert data[key] == manifest[key], (policy, key)
    assert data["loaded_training_update_steps"] == 99745 and data["loaded_training_context_count"] == 100000
    old = json.loads((root / f"results/general_ood_v44_frozen_belief/{stage}/general_ood_v44_frozen_belief_{policy}.json").read_text())
    localized = json.loads((root / f"results/general_ood_v45_reset_localization/{stage}/general_ood_v45_reset_localization_{policy}.json").read_text())
    for key in ("config", "runtime", "source_checkpoint", "source_training_seed", "references", "model_metadata",
                "total_reference_interactions", "total_probe_interactions"):
        assert data[key] == old[key], (policy, key, "v44-source-changed")
    records = data["records"]
    indexed = {(r["seed"], r["target_mode"]): r for r in records}
    expected = {(s, m) for s in data["evaluation_seeds"] for m in data["target_modes"]}
    assert len(records) == len(expected) and set(indexed) == expected
    previous = {(r["seed"], r["target_mode"]): r for r in old["records"]}
    prior_global = {(r["seed"], r["target_mode"]): r for r in localized["records"]}
    for key, row in indexed.items():
        assert row["paired_prefix_equal"] and row["paired_switch_inputs_equal"] and row["interactions"] == 1536
        for field in ("reward_means_by_256", "termination_counts_by_256", "truncation_counts_by_256", "v43_score_check"):
            assert row[field] == previous[key][field], (policy, key, "v44-trajectory-or-score-changed")
        assert set(row["arms"]) == set(manifest["arm_calibrations"])
        for phase in old["phases"]:
            assert row["arms"]["episode-original"]["phases"][phase] == previous[key]["observers"]["student-adaptive"][phase], "v44-belief-changed"
            if key in prior_global:
                assert row["arms"]["global-original"]["phases"][phase] == prior_global[key]["channel_statistics"]["global-centered"][phase], "v45-global-belief-changed"
        for arm, observed in row["arms"].items():
            events = observed["major_reset_events"]
            assert len({e["step"] for e in events}) == len(events)
            assert all(0 <= e["step"] < 1536 and .5 < e["reset_probability"] <= 1 for e in events)
            for phase, (start, end) in manifest["phases"].items():
                obs = observed["phases"][phase]
                assert all(math.isfinite(v) for values in obs.values() for v in values.values())
                count = sum(start <= e["step"] < end for e in events)
                assert count == observed["major_reset_transition_counts"][phase], "major-reset-count-mismatch"
                assert (obs["reset_probability"]["peak"] > .5) == (count > 0)
                assert abs(obs["hypothetical_beta"]["mean"] + 2 + obs["penalty"]["mean"]) < 1e-12
                if phase in ("validation256", "prefix1024"):
                    assert obs == indexed[row["seed"], "normal"]["arms"][arm]["phases"][phase]
    return data


def summarize(manifest, destination, stage):
    runs, missing = {}, []
    for policy in manifest["source_policies"]:
        path = destination / stage / f"{manifest['revision']}_{policy}.json"
        if not path.exists():
            missing.append(policy)
        else:
            runs[policy] = validate_run(json.loads(path.read_text()), manifest, stage)
    result = dict(revision=manifest["revision"], stage=stage, matrix_complete=not missing, missing_policies=missing,
                  decision="pending-missing-policies")
    if missing:
        return result
    methods = {}
    for policy, data in runs.items():
        controls = [r for r in data["records"] if r["target_mode"] == "normal"]
        modes = {}
        for mode in data["target_modes"]:
            rows = [r for r in data["records"] if r["target_mode"] == mode]
            arms = {}
            for name in manifest["arm_calibrations"]:
                phases = {}
                for phase, (start, end) in manifest["phases"].items():
                    observations = [r["arms"][name]["phases"][phase] for r in rows]
                    events = [e for r in rows for e in r["arms"][name]["major_reset_events"] if start <= e["step"] < end]
                    phases[phase] = dict(
                        metrics={metric: {stat: values_statistics([o[metric][stat] for o in observations]) for stat in ("mean", "peak", "end")}
                                 for metric in METRICS},
                        cases_peak_reset_above_half=sum(o["reset_probability"]["peak"] > .5 for o in observations),
                        major_reset_transitions=len(events), major_resets_at_calibration_boundary=sum(e["calibration_changed"] for e in events),
                        major_resets_after_episode_end=sum(e["previous_transition_ended"] for e in events),
                        paired_penalty_contrast=contrast_statistics([o["penalty"]["mean"] for o in observations],
                                                                   [r["arms"][name]["phases"][phase]["penalty"]["mean"] for r in controls]))
                arms[name] = phases
            modes[mode] = dict(arms=arms, signal_means={channel: {phase: mean(r["signal"][channel][phase] for r in rows) for phase in manifest["phases"]}
                                                       for channel in set(manifest["arm_calibrations"].values())},
                               endings={phase: {kind: sum(r["phase_endings"][phase][kind] for r in rows) for kind in ("terminated", "truncated")}
                                        for phase in manifest["phases"]})
        methods[policy] = dict(modes=modes, seconds=data["seconds"])
    result.update(methods=methods, decision="crossed-frozen-replay-complete-not-RL-confirmation", optimizer_updates=0,
                  new_training_seeds=0, environment_interactions=sum(d["total_reference_interactions"] + d["total_probe_interactions"] for d in runs.values()),
                  limitations=["Reused frozen seed1 development policies and test trajectories, not independent performance evidence.",
                               "Majority-reset counts are descriptive; soft age/penalty changes are also retained.",
                               "Robust tails preserve conditional location/variance but change the predictive approximation.",
                               "No multiple-switch test or RL feedback is included in this round."])
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
