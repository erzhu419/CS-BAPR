"""Second-switch contrasts against the same seed's stay-current-mode branch."""
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
STUDY = ROOT / "experiments/general_ood_v47_multiple_switch_20261006"
DESTINATION = ROOT / "results/general_ood_v47_multiple_switch"
METRICS = ("reset_probability", "effective_age", "penalty", "hypothetical_beta", "entropy")
SECOND_PHASES = ("second_early16", "second_early64", "second_post256", "second_late256", "second_last64", "second_all512")


def validate_run(data, manifest, stage, root=ROOT):
    policy = data["source_policy"]
    assert data["revision"] == manifest["revision"] and data["stage"] == stage
    assert policy in manifest["source_policies"]
    for key in ("fit_seeds", "calibration_seeds", "evaluation_seeds"):
        assert data[key] == manifest[stage][key], (policy, key)
    for key in ("first_switch", "second_switch", "probe_steps", "episode_time_limit", "initial_modes", "arm_calibrations",
                "filter_settings", "robust_student_degrees", "phases", "optimizer_updates_during_probe", "replay_insertions",
                "auxiliary_signal_controls_actions", "reset_filters_at_episode_end", "predictors_updated_after_calibration", "actor_critic_target_frozen"):
        assert data[key] == manifest[key], (policy, key)
    old = json.loads((root / f"results/general_ood_v46_crossed_tails/{stage}/general_ood_v46_crossed_tails_{policy}.json").read_text())
    for key in ("source_checkpoint", "source_training_seed", "config", "runtime", "references", "model_metadata",
                "loaded_training_update_steps", "loaded_training_context_count", "total_reference_interactions"):
        assert data[key] == old[key], (policy, key, "v46-source-changed")
    previous = {(r["seed"], r["target_mode"]): r for r in old["records"]}
    indexed = {(r["seed"], r["initial_mode"], r["target_mode"]): r for r in data["records"]}
    expected = {(s, i, t) for s in data["evaluation_seeds"] for i in data["initial_modes"] for t in (*data["initial_modes"], "normal")}
    assert len(data["records"]) == len(expected) and set(indexed) == expected, "incomplete-or-duplicate-branches"
    assert data["total_probe_interactions"] == len(expected) * data["probe_steps"]
    for (seed, initial, target), row in indexed.items():
        control = indexed[seed, initial, initial]
        assert row["paired_prefix_equal"] and row["paired_switch_inputs_equal"] and row["paired_belief_prefix_equal"]
        assert row["actual_second_change"] == (initial != target) and row["interactions"] == data["probe_steps"]
        assert set(row["arms"]) == set(manifest["arm_calibrations"])
        for arm, observations in row["arms"].items():
            events = observations["major_reset_events"]
            assert len({e["step"] for e in events}) == len(events)
            assert all(0 <= e["step"] < data["probe_steps"] and .5 < e["reset_probability"] <= 1 for e in events)
            for phase, (start, end) in data["phases"].items():
                stats = observations["phases"][phase]
                assert all(math.isfinite(v) for values in stats.values() for v in values.values())
                count = sum(start <= e["step"] < end for e in events)
                assert count == observations["major_reset_transition_counts"][phase], "event-accounting-changed"
                assert (stats["reset_probability"]["peak"] > .5) == bool(count)
                if end <= data["second_switch"]:
                    assert stats == control["arms"][arm]["phases"][phase], "unpaired-belief-history"
                if end <= data["first_switch"]:
                    assert stats == previous[seed, "normal"]["arms"][arm]["phases"][phase], "v46-normal-prefix-changed"
                if initial == target and (seed, initial) in previous and phase in old["phases"]:
                    assert stats == previous[seed, initial]["arms"][arm]["phases"][phase], "v46-stay-belief-changed"
        if initial == target and (seed, initial) in previous:
            for key in ("reward_means_by_256", "termination_counts_by_256", "truncation_counts_by_256"):
                assert row[key][:6] == previous[seed, initial][key], "v46-stay-trajectory-changed"
            assert row["v43_score_check"] == previous[seed, initial]["v43_score_check"], "v46-stay-residual-changed"
    return data


def summarize(manifest, destination, stage):
    runs, missing = {}, []
    for policy in manifest["source_policies"]:
        path = destination / stage / f"{manifest['revision']}_{policy}.json"
        if path.exists():
            runs[policy] = validate_run(json.loads(path.read_text()), manifest, stage)
        else:
            missing.append(policy)
    result = dict(revision=manifest["revision"], stage=stage, matrix_complete=not missing,
                  missing_policies=missing, decision="pending-missing-policies")
    if missing:
        return result
    methods = {}
    for policy, data in runs.items():
        indexed = {(r["seed"], r["initial_mode"], r["target_mode"]): r for r in data["records"]}
        scenarios = {}
        for initial in data["initial_modes"]:
            controls = [indexed[s, initial, initial] for s in data["evaluation_seeds"]]
            for target in (*data["initial_modes"], "normal"):
                rows = [indexed[s, initial, target] for s in data["evaluation_seeds"]]
                arms = {}
                for arm in data["arm_calibrations"]:
                    phases = {}
                    for phase in SECOND_PHASES:
                        stats = [r["arms"][arm]["phases"][phase] for r in rows]
                        base = [r["arms"][arm]["phases"][phase] for r in controls]
                        phases[phase] = dict(
                            paired_metrics={m: contrast_statistics([o[m]["mean"] for o in stats], [b[m]["mean"] for b in base]) for m in METRICS},
                            end_metrics={m: values_statistics([o[m]["end"] for o in stats]) for m in METRICS},
                            cases_peak_reset_above_half=sum(o["reset_probability"]["peak"] > .5 for o in stats),
                            major_reset_transitions=sum(r["arms"][arm]["major_reset_transition_counts"][phase] for r in rows),
                            major_resets_after_observed_episode_end=sum(
                                e["previous_transition_ended"] for r in rows for e in r["arms"][arm]["major_reset_events"]
                                if data["phases"][phase][0] <= e["step"] < data["phases"][phase][1]))
                    arms[arm] = phases
                scenarios[f"{initial}->{target}"] = dict(actual_change=initial != target, arms=arms,
                    signal={c: {p: dict(actual=mean(r["signal"][c][p] for r in rows), stay=mean(r["signal"][c][p] for r in controls)) for p in SECOND_PHASES}
                            for c in set(data["arm_calibrations"].values())},
                    endings={p: {group: {k: sum(r["phase_endings"][p][k] for r in group_rows) for k in ("terminated", "truncated")}
                                 for group, group_rows in (("actual", rows), ("stay", controls))} for p in SECOND_PHASES})
        attribution = {}
        for arm in data["arm_calibrations"]:
            events = [e for r in data["records"] for e in r["arms"][arm]["major_reset_events"]
                      if e["step"] >= data["second_switch"] + 16]
            attribution[arm] = dict(events_after_second_first16=len(events),
                                    after_observed_episode_end=sum(e["previous_transition_ended"] for e in events),
                                    at_transition1536=sum(e["step"] == 1536 for e in events),
                                    at_calibration_change=sum(e["calibration_changed"] for e in events))
        methods[policy] = dict(scenarios=scenarios, seconds=data["seconds"], late_event_attribution=attribution)
    result.update(methods=methods, decision="multiple-switch-replay-complete-not-RL-confirmation", optimizer_updates=0,
                  environment_interactions=sum(d["total_reference_interactions"] + d["total_probe_interactions"] for d in runs.values()),
                  limitations=["Two frozen seed1 development policies, not independent RL performance confirmation.",
                               "Repeated-switch scalar residuals may not distinguish different OOD modes.",
                               "Majority resets are descriptive; signed soft responses and physical endings are retained."])
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
