"""Startup counterfactuals and exact first50 paired penalty curves."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.summarize_general_ood_v41 import contrast_statistics

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "experiments/general_ood_v48_reset_center_20261006"
DESTINATION = ROOT / "results/general_ood_v48_reset_center"
READOUT_PHASES = ("second_early16", "second_first50", "second_early64", "second_last64", "restart_first16", "full1792")


def validate_run(data, manifest, stage, root=ROOT):
    policy = data["source_policy"]
    assert data["revision"] == manifest["revision"] and data["stage"] == stage and policy in manifest["source_policies"]
    for key in ("fit_seeds", "calibration_seeds", "evaluation_seeds"):
        assert data[key] == manifest[stage][key]
    for key in ("arms", "phases", "startup_rule", "first_switch", "second_switch", "probe_steps", "episode_time_limit",
                "filter_settings", "robust_student_degrees", "optimizer_updates_during_probe", "replay_insertions",
                "auxiliary_signal_controls_actions", "reset_filters_at_episode_end", "predictors_updated_after_calibration", "actor_critic_target_frozen"):
        assert data[key] == manifest[key], (policy, key)
    old = json.loads((root / f"results/general_ood_v47_multiple_switch/{stage}/general_ood_v47_multiple_switch_{policy}.json").read_text())
    normal = json.loads((root / f"results/general_ood_v46_crossed_tails/{stage}/general_ood_v46_crossed_tails_{policy}.json").read_text())
    for key in ("config", "runtime", "source_checkpoint", "source_training_seed", "references", "model_metadata",
                "loaded_training_update_steps", "loaded_training_context_count", "total_reference_interactions"):
        assert data[key] == old[key], (policy, key, "v47-source-changed")
    assert data["startup_metadata"]["ongoing_scores_unchanged"] and data["startup_metadata"]["global_scales_and_thresholds_unchanged"]
    assert data["startup_metadata"]["normal_start_count"] >= len(data["calibration_seeds"])
    previous = {(r["seed"], r["initial_mode"], r["target_mode"]): r for r in old["records"]}
    normal_rows = {r["seed"]: r for r in normal["records"] if r["target_mode"] == "normal"}
    indexed = {(r["seed"], r["initial_mode"], r["target_mode"]): r for r in data["records"]}
    expected = set(previous) | {(s, "normal", "normal") for s in data["evaluation_seeds"]}
    assert len(data["records"]) == len(expected) and set(indexed) == expected, "branch-matrix-changed"
    assert data["total_probe_interactions"] == len(expected) * data["probe_steps"]
    for key, row in indexed.items():
        seed, initial, target = key
        control = indexed[seed, initial, initial]
        assert row["paired_prefix_equal"] and row["paired_switch_inputs_equal"] and row["paired_belief_prefix_equal"]
        assert row["interactions"] == data["probe_steps"] and set(row["arms"]) == set(data["arms"])
        for arm, observed in row["arms"].items():
            events = observed["major_reset_events"]
            assert len({e["step"] for e in events}) == len(events)
            for phase, (start, end) in data["phases"].items():
                stats = observed["phases"][phase]
                assert all(math.isfinite(v) for s in stats.values() for v in s.values())
                count = sum(start <= e["step"] < end for e in events)
                assert count == observed["major_reset_transition_counts"][phase]
                assert (stats["reset_probability"]["peak"] > .5) == bool(count)
                if end <= data["second_switch"]:
                    assert stats == control["arms"][arm]["phases"][phase], "unpaired-belief-prefix"
                if arm == "global-robust" and key in previous and phase in old["phases"]:
                    assert stats == previous[key]["arms"][arm]["phases"][phase], "v47-control-belief-changed"
                if initial == "normal" and arm == "global-robust" and phase in normal["phases"]:
                    assert stats == normal_rows[seed]["arms"][arm]["phases"][phase], "v46-normal-belief-changed"
            curve = row["paired_first50_penalty"][arm]
            assert len(curve) == 50 and all(math.isfinite(v) for v in curve)
            computed = observed["phases"]["second_first50"]["penalty"]["mean"] - control["arms"][arm]["phases"]["second_first50"]["penalty"]["mean"]
            assert abs(mean(curve) - computed) < 1e-12, "first50-curve-mismatch"
        if key in previous:
            for field in ("reward_means_by_256", "termination_counts_by_256", "truncation_counts_by_256", "v43_score_check"):
                assert row[field] == previous[key][field], "v47-trajectory-changed"
            assert row["arms"]["global-robust"]["major_reset_events"] == previous[key]["arms"]["global-robust"]["major_reset_events"]
    return data


def summarize(manifest, destination, stage):
    runs, missing = {}, []
    for policy in manifest["source_policies"]:
        path = destination / stage / f"{manifest['revision']}_{policy}.json"
        if path.exists():
            runs[policy] = validate_run(json.loads(path.read_text()), manifest, stage)
        else:
            missing.append(policy)
    result = dict(revision=manifest["revision"], stage=stage, matrix_complete=not missing, missing_policies=missing,
                  decision="pending-missing-policies")
    if missing:
        return result
    methods = {}
    for policy, data in runs.items():
        indexed = {(r["seed"], r["initial_mode"], r["target_mode"]): r for r in data["records"]}
        scenarios = {}
        combinations = [("normal", "normal")] + [(i, t) for i in ("heavy", "slippery") for t in (i, "normal", "slippery" if i == "heavy" else "heavy")]
        for initial, target in combinations:
            rows = [indexed[s, initial, target] for s in data["evaluation_seeds"]]
            controls = [indexed[s, initial, initial] for s in data["evaluation_seeds"]]
            arms = {}
            for arm in data["arms"]:
                phases = {}
                for p in READOUT_PHASES:
                    stats = [r["arms"][arm]["phases"][p] for r in rows]
                    base = [r["arms"][arm]["phases"][p] for r in controls]
                    phases[p] = dict(paired_metrics={m: contrast_statistics([r[m]["mean"] for r in stats], [b[m]["mean"] for b in base])
                                                      for m in ("penalty", "effective_age", "reset_probability")},
                        cases_peak_reset_above_half=sum(r["reset_probability"]["peak"] > .5 for r in stats),
                        major_reset_transitions=sum(r["arms"][arm]["major_reset_transition_counts"][p] for r in rows))
                curves = [r["paired_first50_penalty"][arm] for r in rows]
                events = [e for r in rows for e in r["arms"][arm]["major_reset_events"] if e["step"] >= 1296]
                arms[arm] = dict(phases=phases, first50_mean_paired_penalty=[mean(c[k] for c in curves) for k in range(50)],
                    first50_positive_pairs=[sum(c[k] > 0 for c in curves) for k in range(50)],
                    late_major_reset_transitions=len(events), late_major_resets_after_episode_end=sum(e["previous_transition_ended"] for e in events))
            startups = [s for r in rows for s in r["startup_observations"] if s["step"] > 0]
            scenarios[f"{initial}->{target}"] = dict(arms=arms,
                startup_counterfactual=dict(observations=len(startups), original_major=sum(s["original_reset_probability"] > .5 for s in startups),
                    same_past_corrected_major=sum(s["same_past_corrected_reset_probability"] > .5 for s in startups),
                    candidate_major=sum(s["corrected_reset_probability"] > .5 for s in startups)),
                endings={p: {g: {k: sum(r["phase_endings"][p][k] for r in group) for k in ("terminated", "truncated")}
                             for g, group in (("actual", rows), ("stay", controls))} for p in READOUT_PHASES})
        methods[policy] = dict(scenarios=scenarios, startup_metadata=data["startup_metadata"], seconds=data["seconds"])
    result.update(methods=methods, decision="reset-center-frozen-replay-complete-not-RL-confirmation", optimizer_updates=0,
        environment_interactions=sum(d["total_reference_interactions"] + d["total_probe_interactions"] for d in runs.values()),
        limitations=["Two reused frozen seed1 development policies, no RL performance confirmation.",
                     "Startup offset uses few normal episode starts; no new test-driven scale or threshold fitting.",
                     "Later paired differences include changed physical trajectories and episode endings."])
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
