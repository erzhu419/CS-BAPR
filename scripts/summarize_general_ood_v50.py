"""Signed residual contrasts and scalar/readout separation, not RL confirmation."""
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
STUDY = ROOT / "experiments/general_ood_v50_residual_readout_20261006"
DESTINATION = ROOT / "results/general_ood_v50_residual_readout"
SCALARS = ("mean_signal", "mean_score_ratio", "exceedance_fraction", "nonstartup_score_ratio", "signed_mean_energy")


def validate_run(data, manifest, stage, root=ROOT):
    policy = data["source_policy"]
    assert data["revision"] == manifest["revision"] and data["stage"] == stage and policy in manifest["source_policies"]
    for k in ("fit_seeds", "calibration_seeds", "evaluation_seeds"):
        assert data[k] == manifest[stage][k]
    for k in ("arms", "scenarios", "assessment_offsets", "clock_switch_step", "probe_steps", "episode_time_limit",
              "filter_settings", "robust_student_degrees", "optimizer_updates_during_probe", "replay_insertions", "auxiliary_signal_controls_actions",
              "reset_filters_at_episode_end", "predictors_updated_after_calibration", "actor_critic_target_frozen"):
        assert data[k] == manifest[k], (policy, k)
    old = json.loads((root / f"results/general_ood_v49_startup_noise/{stage}/general_ood_v49_startup_noise_{policy}.json").read_text())
    for k in ("config", "runtime", "source_checkpoint", "source_training_seed", "references", "model_metadata", "noise_metadata",
              "loaded_training_update_steps", "loaded_training_context_count", "total_reference_interactions"):
        assert data[k] == old[k], (policy, k, "v49-source-changed")
    old_rows = {(r["seed"], r["scenario"]): r for r in old["records"]}
    rows = {(r["seed"], r["scenario"]): r for r in data["records"]}
    expected = {(s, c) for s in data["evaluation_seeds"] for c in manifest["scenarios"]}
    assert len(data["records"]) == len(expected) and set(rows) == expected, "branch-matrix-changed"
    assert data["total_probe_interactions"] == len(expected) * data["probe_steps"]
    for (seed, scenario), row in rows.items():
        previous = old_rows[seed, manifest["scenarios"][scenario]]
        assert row["source_v49_scenario"] == manifest["scenarios"][scenario]
        normal = rows[seed, "normal->normal"]
        assert row["paired_prefix_equal"] and row["paired_switch_inputs_equal"] and row["paired_belief_prefix_equal"]
        assert row["interactions"] == data["probe_steps"]
        assert row["normal_first_restart_step"] == normal["normal_first_restart_step"] == previous["normal_first_restart_step"]
        assert row["restart_previous_terminated"] or row["restart_previous_truncated"]
        timing = "reset" if scenario.startswith("reset-") else "clock"
        assert row["timing"] == timing and row["assessed_switch_step"] == (row["normal_first_restart_step"] if timing == "reset" else data["clock_switch_step"])
        for k, value in row["source_v49"].items():
            if k == "arms":
                assert value == {a: previous[k][a] for a in data["arms"]}, "v49-control-belief-changed"
            else:
                assert value == previous[k], "v49-trajectory-or-signal-changed"
        for t, windows in row["diagnostics"].items():
            for phase, stats in windows.items():
                assert all(math.isfinite(stats[k]) for k in SCALARS)
                assert all(math.isfinite(v) for k in ("signed_mean", "squared_mean") for v in stats[k])
                threshold = data["model_metadata"]["calibrations"]["global-centered"]["threshold95"][0]
                assert math.isclose(mean(stats["squared_mean"]) / threshold, stats["mean_score_ratio"], rel_tol=1e-12, abs_tol=1e-10), "squared-residual-score-mismatch"
                assert math.isclose(mean(v * v for v in stats["signed_mean"]), stats["signed_mean_energy"], rel_tol=1e-12, abs_tol=1e-10)
                if phase == "first50":
                    curve = row["first50"][t]
                    for k, field in (("signal", "mean_signal"), ("score_ratio", "mean_score_ratio")):
                        assert len(curve[k]) == 50 and math.isclose(mean(curve[k]), stats[field], rel_tol=1e-12, abs_tol=1e-10), "first50-curve-mismatch"
                    for a in data["arms"]:
                        assert len(curve["penalty"][a]) == 50
                        assert abs(mean(curve["penalty"][a]) - stats["beliefs"][a]["penalty"]["mean"]) < 1e-12
            if t == "reset":
                baseline = normal["first50"][t]
                for a in data["arms"]:
                    delta = [x - y for x, y in zip(row["first50"][t]["penalty"][a], baseline["penalty"][a])]
                    if scenario.startswith("reset-"):
                        assert delta == previous["paired_first50_penalty"][a], "v49-reset-window-changed"
                    else:
                        assert all(v == 0 for v in delta)
    return data


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
        indexed = {(r["seed"], r["scenario"]): r for r in data["records"]}
        scenarios = {}
        for scenario in manifest["scenarios"]:
            if scenario == "normal->normal":
                continue
            rows = [indexed[s, scenario] for s in data["evaluation_seeds"]]
            controls = [indexed[s, "normal->normal"] for s in data["evaluation_seeds"]]
            timing = rows[0]["timing"]
            phases = {}
            for phase in data["assessment_offsets"]:
                actual = [r["diagnostics"][timing][phase] for r in rows]
                stay = [r["diagnostics"][timing][phase] for r in controls]
                vectors = {k: [contrast_statistics([a[k][i] for a in actual], [b[k][i] for b in stay])
                               for i in range(len(actual[0][k]))] for k in ("signed_mean", "squared_mean")}
                phases[phase] = dict(residuals={k: contrast_statistics([a[k] for a in actual], [b[k] for b in stay]) for k in SCALARS},
                    vectors=vectors, beliefs={arm: {k: contrast_statistics([a["beliefs"][arm][k]["mean"] for a in actual],
                        [b["beliefs"][arm][k]["mean"] for b in stay]) for k in ("penalty", "effective_age", "reset_probability")} for arm in data["arms"]},
                    endings={g: {k: sum(v[k] for v in group) for k in ("terminated", "truncated", "startup_count")}
                             for g, group in (("actual", actual), ("stay", stay))})
            curves = {}
            for k in ("signal", "score_ratio", *data["arms"]):
                def values(row):
                    c = row["first50"][timing]
                    return c[k] if k in ("signal", "score_ratio") else c["penalty"][k]
                curves[k] = [contrast_statistics([values(r)[i] for r in rows], [values(r)[i] for r in controls]) for i in range(50)]
            scenarios[scenario] = dict(assessment=phases, first50_curves=curves,
                assessed_switch_steps=[r["assessed_switch_step"] for r in rows])
        methods[policy] = dict(scenarios=scenarios, seconds=data["seconds"])
    result.update(methods=methods, decision="residual-readout-diagnostic-complete-not-RL-confirmation", optimizer_updates=0,
        environment_interactions=sum(d["total_reference_interactions"] + d["total_probe_interactions"] for d in runs.values()),
        limitations=["Two reused frozen seed1 policies and development probe seeds; no RL return confirmation.",
                     "Signed window moments diagnose directional information, not a fitted OOD-mode classifier.",
                     "Global normal calibration is not startup/state-conditioned; score95 exceedance is descriptive, not a guaranteed test error rate."])
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
