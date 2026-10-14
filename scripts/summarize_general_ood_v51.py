"""Matched departure coefficients and independent normal-null diagnostics."""
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
STUDY = ROOT / "experiments/general_ood_v51_departure_penalty_20261006"
DESTINATION = ROOT / "results/general_ood_v51_departure_penalty"


def validate_statistics(stats):
    assert all(math.isfinite(v) for v in stats.values())
    assert 0 <= stats["mean_risk"] <= stats["peak_risk"] <= 1
    assert 0 <= stats["positive_instant_fraction"] <= 1 and 0 <= stats["mean_instant_risk"] <= 1
    assert abs(stats["mean_beta"] + 2 + stats["mean_risk"]) < 1e-12


def validate_run(data, manifest, stage, root=ROOT):
    policy = data["source_policy"]
    assert data["revision"] == manifest["revision"] and data["stage"] == stage and policy in manifest["source_policies"]
    for k in ("fit_seeds", "calibration_seeds", "evaluation_seeds", "null_seeds"):
        assert data[k] == manifest[stage][k]
    for k in ("readout_settings", "readouts", "scenarios", "assessment_offsets", "probe_steps", "episode_time_limit", "filter_settings",
              "robust_student_degrees", "optimizer_updates_during_probe", "replay_insertions", "auxiliary_signal_controls_actions",
              "reset_filters_at_episode_end", "predictors_updated_after_calibration", "actor_critic_target_frozen",
              "risk_reference_updated_after_calibration", "risk_history_reset_at_episode_end"):
        assert data[k] == manifest[k], (policy, k)
    assert not (set(data["null_seeds"]) & set(data["fit_seeds"] + data["calibration_seeds"] + data["evaluation_seeds"]))
    old = json.loads((root / f"results/general_ood_v50_residual_readout/{stage}/general_ood_v50_residual_readout_{policy}.json").read_text())
    for k in ("config", "runtime", "source_checkpoint", "source_training_seed", "references", "model_metadata", "noise_metadata",
              "loaded_training_update_steps", "loaded_training_context_count", "total_reference_interactions"):
        assert data[k] == old[k], (policy, k, "v50-source-changed")
    previous = {(r["seed"], r["scenario"]): r for r in old["records"]}
    rows = {(r["source_v50"]["seed"], r["source_v50"]["scenario"]): r for r in data["records"]}
    assert set(rows) == set(previous) and len(rows) == len(data["records"]), "branch-matrix-changed"
    assert data["total_probe_interactions"] == (len(rows) + len(data["null_seeds"])) * data["probe_steps"]
    for key, row in rows.items():
        assert row["source_v50"] == previous[key], "v50-source-trajectory-signal-belief-changed"
        for timing, readouts in row["readouts"].items():
            for arm, phases in readouts.items():
                for stats in phases.values():
                    validate_statistics(stats)
                curve = row["first50"][timing][arm]
                assert len(curve) == 50 and all(0 <= v <= 1 for v in curve)
                assert abs(mean(curve) - phases["first50"]["mean_risk"]) < 1e-12, "readout-curve-mismatch"
    metadata = data["risk_metadata"]
    assert metadata["normal_start_count"] == data["noise_metadata"]["normal_start_count"]
    assert metadata["settings"] == manifest["readout_settings"] and all(n > 0 for n in metadata["counts"])
    assert len(metadata["ages"]) == len(metadata["upper_signal"]) == len(metadata["counts"])
    for group, seeds in (("reused_normal", data["evaluation_seeds"]), ("new_normal", data["null_seeds"])):
        assert [r["seed"] for r in data[group]] == seeds
        for row in data[group]:
            assert row["interactions"] == data["probe_steps"]
            for stats in row["full"].values():
                validate_statistics(stats)
            for windows in row["initial"].values():
                for stats in windows.values():
                    validate_statistics(stats)
            for restart in row["restarts"]:
                assert restart["previous_terminated"] or restart["previous_truncated"]
                for stats in restart["first50"].values():
                    validate_statistics(stats)
    return data


def normal_summary(rows, arm):
    starts = [s["first50"][arm] for r in rows for s in r["restarts"]]
    def aggregate(stats):
        return {k: dict(mean=mean(s[k] for s in stats), maximum=max(s[k] for s in stats))
                for k in ("mean_risk", "peak_risk", "positive_instant_fraction")}
    return dict(trajectories=len(rows), full=aggregate([r["full"][arm] for r in rows]),
        initial_first50=aggregate([r["initial"][arm]["first50"] for r in rows]),
        restart_windows=len(starts), restart_first50=aggregate(starts))


def summarize(manifest, destination, stage):
    runs, missing = {}, []
    for policy in manifest["source_policies"]:
        p = destination / stage / f"{manifest['revision']}_{policy}.json"
        if p.exists():
            runs[policy] = validate_run(json.loads(p.read_text()), manifest, stage)
        else:
            missing.append(policy)
    result = dict(revision=manifest["revision"], stage=stage, matrix_complete=not missing,
                  missing_policies=missing, decision="pending-missing-policies")
    if missing:
        return result
    methods = {}
    for policy, data in runs.items():
        indexed = {(r["source_v50"]["seed"], r["source_v50"]["scenario"]): r for r in data["records"]}
        scenarios = {}
        for scenario in manifest["scenarios"]:
            if scenario == "normal->normal":
                continue
            rows = [indexed[s, scenario] for s in data["evaluation_seeds"]]
            controls = [indexed[s, "normal->normal"] for s in data["evaluation_seeds"]]
            timing = rows[0]["source_v50"]["timing"]
            arms = {}
            for arm in data["readouts"]:
                assessment = {p: {k: contrast_statistics([r["readouts"][timing][arm][p][k] for r in rows],
                    [r["readouts"][timing][arm][p][k] for r in controls]) for k in ("mean_risk", "positive_instant_fraction", "mean_beta")}
                    for p in data["assessment_offsets"]}
                curves = [contrast_statistics([r["first50"][timing][arm][i] for r in rows],
                    [r["first50"][timing][arm][i] for r in controls]) for i in range(50)]
                arms[arm] = dict(assessment=assessment, first50_curves=curves)
            scenarios[scenario] = dict(readouts=arms,
                original_noise_likelihood_penalty={p: contrast_statistics(
                    [r["source_v50"]["diagnostics"][timing][p]["beliefs"]["noise-likelihood"]["penalty"]["mean"] for r in rows],
                    [r["source_v50"]["diagnostics"][timing][p]["beliefs"]["noise-likelihood"]["penalty"]["mean"] for r in controls]) for p in data["assessment_offsets"]})
        methods[policy] = dict(scenarios=scenarios, risk_metadata=data["risk_metadata"], seconds=data["seconds"],
            normal={g: {a: normal_summary(data[g], a) for a in data["readouts"]} for g in ("reused_normal", "new_normal")})
    result.update(methods=methods, decision="calibrated-departure-readout-complete-not-RL-confirmation", optimizer_updates=0,
        environment_interactions=sum(d["total_reference_interactions"] + d["total_probe_interactions"] for d in runs.values()),
        limitations=["Reused seed1 frozen policies; new normal seeds validate null behavior only, not independent OOD/RL confirmation.",
                     "Age neighborhoods and interpolated empirical quantiles do not guarantee 5% error; few reference starts and late-age samples.",
                     "Bounded departure coefficient is a candidate penalty mapping, not a calibrated OOD-mode probability or demonstrated beneficial conservatism."])
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
