"""Source-matched reset locations and pre-existing calibration comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "experiments/general_ood_v45_reset_localization_20261006"
DESTINATION = ROOT / "results/general_ood_v45_reset_localization"


def validate_run(data, manifest, stage, root=ROOT):
    policy = data["source_policy"]
    assert data["revision"] == manifest["revision"] and data["stage"] == stage
    assert policy in manifest["source_policies"]
    for key in ("fit_seeds", "calibration_seeds", "evaluation_seeds", "target_modes"):
        assert data[key] == manifest[stage][key], (policy, key)
    for key in ("predictor", "channel_names", "filter_settings", "event_radius", "phases",
                "optimizer_updates_during_probe", "replay_insertions", "auxiliary_signal_controls_actions",
                "predictors_updated_after_calibration", "actor_critic_target_frozen"):
        assert data[key] == manifest[key], (policy, key)
    assert data["loaded_training_update_steps"] == 99745 and data["loaded_training_context_count"] == 100000
    old = json.loads((root / f"results/general_ood_v44_frozen_belief/{stage}/general_ood_v44_frozen_belief_{policy}.json").read_text())
    for key in ("config", "runtime", "source_checkpoint", "source_training_seed", "references", "model_metadata", "total_reference_interactions"):
        assert data[key] == old[key], (policy, key, "v44-source-changed")
    records = data["records"]
    expected = {(s, m) for s in data["evaluation_seeds"] for m in data["target_modes"]}
    assert len(records) == len(expected) and {(r["seed"], r["target_mode"]) for r in records} == expected
    assert data["total_probe_interactions"] == len(records) * 1536
    previous = {(r["seed"], r["target_mode"]): r for r in old["records"]}
    for row in records:
        key = row["seed"], row["target_mode"]
        assert row["paired_prefix_equal"] and row["paired_switch_inputs_equal"] and row["interactions"] == 1536
        for field in ("reward_means_by_256", "termination_counts_by_256", "truncation_counts_by_256", "v43_score_check"):
            assert row[field] == previous[key][field], (policy, key, "v44-trajectory-or-score-changed")
        assert row["channel_statistics"]["episode-centered"] == previous[key]["observers"]["student-adaptive"], "v44-belief-changed"
        events = {r["step"]: r for r in row["event_window"]}
        assert len(events) == len(row["event_window"])
        assert all(events[t]["reset_probability"] > .5 for t in row["major_reset_steps"])
        assert all(events[t]["calibration_changed"] for t in row["calibration_boundary_steps"])
        assert all(not r["calibration_changed"] or r["steps_since_calibration_change"] == 0 for r in events.values())
        for phase, (start, end) in manifest["phases"].items():
            count = sum(start <= t < end for t in row["major_reset_steps"])
            assert count == row["major_reset_transition_counts"]["episode-centered"][phase]
            peak = row["phase_peak_steps"][phase]
            assert start <= peak < end
            assert events[peak]["reset_probability"] == row["channel_statistics"]["episode-centered"][phase]["reset_probability"]["peak"]
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
        modes = {}
        for mode in data["target_modes"]:
            rows = [r for r in data["records"] if r["target_mode"] == mode]
            phases = {}
            for phase, (start, end) in manifest["phases"].items():
                events = [event for row in rows for event in row["event_window"]
                          if start <= event["step"] < end and event["reset_probability"] > .5]
                phases[phase] = dict(
                    major_reset_transitions=len(events),
                    major_resets_at_calibration_boundary=sum(e["calibration_changed"] for e in events),
                    boundary_major_resets_removed_by_previous_group=sum(e["calibration_changed"] and e["previous_group_current_error_reset"] <= .5 for e in events),
                    major_resets_after_episode_end=sum(e["previous_transition_ended"] for e in events),
                    major_event_locations=[dict(seed=r["seed"], step=e["step"], episode_age=e["episode_age"],
                                                boundary=e["calibration_changed"], since_boundary=e["steps_since_calibration_change"])
                                           for r in rows for e in r["event_window"] if start <= e["step"] < end and e["reset_probability"] > .5],
                    channels={channel: dict(
                        cases_peak_reset_above_half=sum(r["channel_statistics"][channel][phase]["reset_probability"]["peak"] > .5 for r in rows),
                        major_reset_transitions=sum(r["major_reset_transition_counts"][channel][phase] for r in rows),
                        mean_reset=mean(r["channel_statistics"][channel][phase]["reset_probability"]["mean"] for r in rows),
                        mean_age=mean(r["channel_statistics"][channel][phase]["effective_age"]["mean"] for r in rows),
                        mean_penalty=mean(r["channel_statistics"][channel][phase]["penalty"]["mean"] for r in rows)) for channel in data["channel_names"]})
            modes[mode] = phases
        methods[policy] = dict(modes=modes, seconds=data["seconds"])
    result.update(methods=methods, decision="reset-localization-complete-not-RL-confirmation", optimizer_updates=0,
                  environment_interactions=sum(d["total_reference_interactions"] + d["total_probe_interactions"] for d in runs.values()),
                  limitations=["Same development trajectories; diagnostic previous-group intervention changes only one observation, not past history.",
                               "Global-centered is an existing normal-only calibration control, not a newly tuned repair.",
                               "No physical-mode labels enter belief; event locations do not themselves identify latent environmental causes."])
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
