"""Startup noise ablation, preserved switches, and same-reset true changes."""
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
STUDY = ROOT / "experiments/general_ood_v49_startup_noise_20261006"
DESTINATION = ROOT / "results/general_ood_v49_startup_noise"
SCENARIOS = ("normal->normal", "heavy->heavy", "heavy->normal", "heavy->slippery",
             "slippery->slippery", "slippery->normal", "slippery->heavy", "reset-normal->heavy", "reset-normal->slippery")


def validate_run(data, manifest, stage, root=ROOT):
    policy = data["source_policy"]
    assert data["revision"] == manifest["revision"] and data["stage"] == stage and policy in manifest["source_policies"]
    for k in ("fit_seeds", "calibration_seeds", "evaluation_seeds"):
        assert data[k] == manifest[stage][k]
    for k in ("arms", "phases", "assessment_offsets", "noise_rule", "first_switch", "second_switch", "probe_steps", "episode_time_limit",
              "filter_settings", "robust_student_degrees", "optimizer_updates_during_probe", "replay_insertions", "auxiliary_signal_controls_actions",
              "reset_filters_at_episode_end", "predictors_updated_after_calibration", "actor_critic_target_frozen"):
        assert data[k] == manifest[k], (policy, k)
    old = json.loads((root / f"results/general_ood_v48_reset_center/{stage}/general_ood_v48_reset_center_{policy}.json").read_text())
    for k in ("config", "runtime", "source_checkpoint", "source_training_seed", "references", "model_metadata",
              "loaded_training_update_steps", "loaded_training_context_count", "total_reference_interactions"):
        assert data[k] == old[k], (policy, k, "v48-source-changed")
    old_rows = {(r["seed"], f"{r['initial_mode']}->{r['target_mode']}"): r for r in old["records"]}
    rows = {(r["seed"], r["scenario"]): r for r in data["records"]}
    expected = {(s, c) for s in data["evaluation_seeds"] for c in SCENARIOS}
    assert len(data["records"]) == len(expected) and set(rows) == expected, "branch-matrix-changed"
    assert data["total_probe_interactions"] == len(expected) * data["probe_steps"]
    noise = data["noise_metadata"]
    assert noise["normal_start_count"] == old["startup_metadata"]["normal_start_count"]
    assert abs(noise["startup_mean_signal"] - old["startup_metadata"]["original_start_mean_signal"]) < 1e-12
    assert noise["startup_extra_variance"] == max(noise["startup_discrepancy_mse"] - noise["ongoing_signal_variance"], 0)
    assert 0 < noise["startup_history_precision"] <= 1
    for key, row in rows.items():
        seed, scenario = key
        normal = rows[seed, "normal->normal"]
        is_reset = scenario.startswith("reset-")
        control = normal if is_reset else rows[seed, f"{row['initial_mode']}->{row['initial_mode']}"]
        switch = row["assessed_switch_step"]
        assert row["paired_prefix_equal"] and row["paired_switch_inputs_equal"] and row["paired_belief_prefix_equal"]
        assert row["interactions"] == data["probe_steps"] and set(row["arms"]) == set(data["arms"])
        assert row["normal_first_restart_step"] == normal["normal_first_restart_step"]
        assert switch == (normal["normal_first_restart_step"] if is_reset else 1280)
        assert switch + 256 <= data["probe_steps"]
        if is_reset:
            startup = next(s for s in row["startup_observations"] if s["step"] == switch)
            assert startup["previous_terminated"] or startup["previous_truncated"], "reset-switch-not-observable-startup"
        for arm, obs in row["arms"].items():
            events = obs["major_reset_events"]
            assert len({e["step"] for e in events}) == len(events)
            for p, (start, end) in data["phases"].items():
                stats = obs["phases"][p]
                assert all(math.isfinite(v) for s in stats.values() for v in s.values())
                count = sum(start <= e["step"] < end for e in events)
                assert count == obs["major_reset_transition_counts"][p]
                assert (stats["reset_probability"]["peak"] > .5) == bool(count)
                if not is_reset and end <= switch:
                    assert stats == control["arms"][arm]["phases"][p], "unpaired-filter-prefix"
                if key in old_rows and arm == "global-robust":
                    assert stats == old_rows[key]["arms"][arm]["phases"][p], "v48-control-belief-changed"
            control_stats = normal["normal_restart_assessment"][arm] if is_reset else control["assessment"][arm]
            assert row["control_assessment"][arm] == control_stats, "unpaired-control-window"
            curve = row["paired_first50_penalty"][arm]
            assert len(curve) == 50 and all(math.isfinite(v) for v in curve)
            delta = row["assessment"][arm]["first50"]["penalty"]["mean"] - control_stats["first50"]["penalty"]["mean"]
            assert abs(mean(curve) - delta) < 1e-12, "first50-curve-mismatch"
        if key in old_rows:
            for k in ("reward_means_by_256", "termination_counts_by_256", "truncation_counts_by_256", "v43_score_check"):
                assert row[k] == old_rows[key][k], "v48-trajectory-changed"
            assert row["signal"] == old_rows[key]["signal"]["global-robust"], "v48-signal-changed"
            assert row["arms"]["global-robust"]["major_reset_events"] == old_rows[key]["arms"]["global-robust"]["major_reset_events"]
            assert row["paired_first50_penalty"]["global-robust"] == old_rows[key]["paired_first50_penalty"]["global-robust"]
    return data


def summarize(manifest, destination, stage):
    runs, missing = {}, []
    for policy in manifest["source_policies"]:
        p = destination / stage / f"{manifest['revision']}_{policy}.json"
        if p.exists():
            runs[policy] = validate_run(json.loads(p.read_text()), manifest, stage)
        else:
            missing.append(policy)
    result = dict(revision=manifest["revision"], stage=stage, matrix_complete=not missing, missing_policies=missing, decision="pending-missing-policies")
    if missing:
        return result
    methods = {}
    for policy, data in runs.items():
        scenarios = {}
        for scenario in SCENARIOS:
            rows = sorted((r for r in data["records"] if r["scenario"] == scenario), key=lambda r: r["seed"])
            arms = {}
            for arm in data["arms"]:
                phases = {}
                for p in data["assessment_offsets"]:
                    actual = [r["assessment"][arm][p] for r in rows]
                    controls = [r["control_assessment"][arm][p] for r in rows]
                    phases[p] = {m: contrast_statistics([s[m]["mean"] for s in actual], [s[m]["mean"] for s in controls])
                                 for m in ("penalty", "effective_age", "reset_probability")}
                events = [e for r in rows for e in r["arms"][arm]["major_reset_events"]]
                curves = [r["paired_first50_penalty"][arm] for r in rows]
                arms[arm] = dict(assessment=phases,
                    full1792_major_reset_cases=sum(r["arms"][arm]["phases"]["full1792"]["reset_probability"]["peak"] > .5 for r in rows),
                    full1792_major_reset_transitions=len(events), major_resets_after_episode_end=sum(e["previous_transition_ended"] for e in events),
                    full1792_mean_penalty=mean(r["arms"][arm]["phases"]["full1792"]["penalty"]["mean"] for r in rows),
                    first50_mean_paired_penalty=[mean(c[k] for c in curves) for k in range(50)],
                    first50_positive_pairs=[sum(c[k] > 0 for c in curves) for k in range(50)])
            starts = [s for r in rows for s in r["startup_observations"] if s["step"] > 0]
            scenarios[scenario] = dict(arms=arms, assessed_switch_steps=[r["assessed_switch_step"] for r in rows],
                startup_counterfactual=dict(observations=len(starts), original_major=sum(s["original_reset_probability"] > .5 for s in starts),
                    same_past_noisy_major=sum(s["same_past_noisy_reset_probability"] > .5 for s in starts),
                    arm_major={a: sum(s["arm_reset_probabilities"][a] > .5 for s in starts) for a in data["arms"]}),
                endings={p: {g: {k: sum(r[field][p][k] for r in rows) for k in ("terminated", "truncated")}
                    for g, field in (("actual", "assessment_endings"), ("stay", "control_assessment_endings"))} for p in data["assessment_offsets"]})
        normal_rows = [r for r in data["records"] if r["scenario"] == "normal->normal"]
        restart = {a: {p: {m: mean(r["normal_restart_assessment"][a][p][m]["mean"] for r in normal_rows)
                           for m in ("penalty", "effective_age", "reset_probability")}
                       for p in data["assessment_offsets"]} for a in data["arms"]}
        attribution = {}
        for arm in data["arms"]:
            events = [(r["scenario"], e) for r in data["records"] for e in r["arms"][arm]["major_reset_events"]]
            attribution[arm] = dict(total=len(events),
                after_episode_end=sum(e["previous_transition_ended"] for _, e in events),
                legacy_after_episode_end=sum(not c.startswith("reset-") and e["previous_transition_ended"] for c, e in events),
                nonstartup=sum(e["episode_age"] > 0 for _, e in events))
        methods[policy] = dict(scenarios=scenarios, noise_metadata=data["noise_metadata"], seconds=data["seconds"],
            normal_restart_assessment=restart, major_reset_attribution=attribution)
    result.update(methods=methods, decision="startup-noise-replay-complete-not-RL-confirmation", optimizer_updates=0,
        environment_interactions=sum(d["total_reference_interactions"] + d["total_probe_interactions"] for d in runs.values()),
        limitations=["Reused frozen seed1 development policies, not independent RL confirmation.",
                     "Noise budget includes normal startup bias and spread; fractional NIG moments and df4 variance addition are approximations.",
                     "Startup references are few; later windows include changed trajectories and episode endings."])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "formal"), default="formal")
    args = parser.parse_args()
    result = summarize(json.loads((STUDY / "dispatch_manifest.json").read_text()), DESTINATION, args.stage)
    p = DESTINATION / ("summary.json" if args.stage == "formal" else "preflight_summary.json")
    p.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(result["decision"], "saved", p)


if __name__ == "__main__":
    main()
