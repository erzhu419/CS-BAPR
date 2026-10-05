"""Apply the frozen coverage and added-value gates to the ten fresh runs."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from statistics import mean
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scripts.train_nonlinear_load_v15 import CANDIDATE, METHODS, RADII, REVISION, SEVERITIES, reset_radius

ROOT = Path(__file__).resolve().parents[1]


def actor_block(runs, condition, method, seed, radius, severity):
    return runs[(condition, method, seed)]["radius_results"][str(radius)][str(severity)]["controllers"]["actor"]


def summarize(runs, seeds):
    radius_summaries = {}
    for radius in RADII:
        methods = {m: {str(a): [{"training_seed": s, "mean_cost": actor_block(runs, "balanced", m, s, radius, a)["mean_cost"]}
                               for s in seeds] for a in SEVERITIES} for m in METHODS}
        learner_gates = {m: all(actor_block(runs, "balanced", CANDIDATE, s, radius, a)["mean_cost"] <
                                actor_block(runs, "balanced", m, s, radius, a)["mean_cost"]
                                for s in seeds for a in SEVERITIES) for m in METHODS[1:]}
        own_fitted = {str(a): [{"training_seed": s, **runs[("balanced", CANDIDATE, s)]["radius_results"][str(radius)][str(a)]
                               ["paired_excess_costs"]["fitted"]} for s in seeds] for a in SEVERITIES}
        own_gate = all(v["mean_total_cost"] < 0 for values in own_fitted.values() for v in values)
        radius_summaries[str(radius)] = {"methods": methods, "learner_comparison_gates": learner_gates,
                                        "own_fitted_excess_costs": own_fitted, "own_fitted_added_value_passed": own_gate}
    repair = {}
    for severity in SEVERITIES:
        values = []
        for seed in seeds:
            wide = actor_block(runs, "balanced", CANDIDATE, seed, 1.0, severity)
            narrow = actor_block(runs, "narrow", CANDIDATE, seed, 1.0, severity)
            values.append({"training_seed": seed, "total_cost_change": wide["mean_cost"] - narrow["mean_cost"],
                           "early_state_cost_change": wide["mean_early_state_cost"] - narrow["mean_early_state_cost"]})
        repair[str(severity)] = values
    coverage_repair = all(v["total_cost_change"] < 0 and v["early_state_cost_change"] < 0
                          for values in repair.values() for v in values)
    learners = all(all(v["learner_comparison_gates"].values()) for v in radius_summaries.values())
    own_fitted = all(v["own_fitted_added_value_passed"] for v in radius_summaries.values())
    return {"radius_summaries": radius_summaries, "balanced_minus_narrow_radius1": repair,
            "decision": {"coverage_repair_passed": coverage_repair, "all_learner_comparisons_passed": learners,
                         "all_own_fitted_added_value_passed": own_fitted,
                         "full_confirmation": coverage_repair and learners and own_fitted,
                         "historical_v13_hold_unchanged": True}}


def validate_run(data, manifest, task):
    condition, method, seed = (task[k] for k in ("condition", "method", "seed"))
    arm = next(a for a in manifest["arms"] if a["method"] == method)
    assert (data["revision"], data["condition"], data["method"], data["training_seed"]) == (REVISION, condition, method, seed)
    assert data["config"] == arm["config"] and data["environment_settings"] == manifest["environment_settings"]
    assert (data["training_episodes"], data["environment_steps"], data["training_updates"]) == (500, 100000, 99001)
    assert data["checkpoint_selection"] == "last" and not data["preflight_only"]
    assert data["reset_cycle"] == manifest["reset_cycles"][condition] and data["reset_index"] == "absolute_episode"
    assert data["reset_counts"] == dict(Counter(str(reset_radius(condition, e)) for e in range(500)))
    assert (data["identification_reset_radius"], data["training_load_severity"],
            data["optimizer_updates_during_evaluation"], data["early_steps"], data["sample_interval"]) == (0.2, 1, 0, 50, 10)
    assert len(data["training_returns"]) == 500 and np.isfinite(data["training_returns"]).all()
    assert (data["radii"], data["severities"], data["evaluation_seed_start"], data["evaluation_episodes_per_severity"]) == (
        list(RADII), list(SEVERITIES), 60000, 100)
    assert data["runtime"]["device"] == "cuda"
    for package, version in manifest["runtime_requirement"].items():
        if package == "python_minor_versions":
            assert ".".join(data["runtime"]["python"].split()[0].split(".")[:2]) in version
        else:
            assert data["runtime"][package] == version
    symbolic = method == CANDIDATE
    assert (data["symbolic_identification_interactions"], data["symbolic_validation_interactions"]) == (
        10000 if symbolic else 0, 1000 if symbolic else 0)
    assert (data["sindy"] is not None) == symbolic
    expected_modes = {"actor", "known_local", "known_inverse"} | ({"fitted"} if symbolic else set())
    assert data["evaluation_episodes_including_controls"] == 600 * len(expected_modes)
    for radius in RADII:
        for severity in SEVERITIES:
            block = data["radius_results"][str(radius)][str(severity)]
            assert set(block["controllers"]) == expected_modes
            for values in block["controllers"].values():
                assert values["evaluation_seeds"] == list(range(60000, 60100))
                arrays = {k: np.asarray(values[k]) for k in ("returns", "state_costs", "command_costs",
                          "early_state_costs", "early_command_costs", "late_state_costs", "late_command_costs")}
                assert all(v.shape == (100,) and np.isfinite(v).all() for v in arrays.values())
                np.testing.assert_allclose(-arrays["returns"], arrays["state_costs"] + arrays["command_costs"], rtol=1e-12, atol=1e-12)
                for kind in ("state", "command"):
                    np.testing.assert_allclose(arrays[f"{kind}_costs"], arrays[f"early_{kind}_costs"] + arrays[f"late_{kind}_costs"])
                assert np.isclose(values["mean_cost"], -mean(values["returns"]), rtol=1e-12, atol=1e-12)
                for key, array in arrays.items():
                    if key != "returns":
                        assert np.isclose(values[f"mean_{key[:-1]}"], array.mean(), rtol=1e-12, atol=1e-12)
            for mode, excess in block["paired_excess_costs"].items():
                differences = np.asarray(block["controllers"][mode]["returns"]) - block["controllers"]["actor"]["returns"]
                np.testing.assert_allclose([v["total_cost"] for v in excess["per_path"]], differences, rtol=1e-10, atol=1e-10)
                assert np.isclose(excess["mean_total_cost"], differences.mean(), rtol=1e-10, atol=1e-10)


def main():
    study = ROOT / "experiments/nonlinear_load_v15_coverage_20261002"
    destination = ROOT / "results" / REVISION
    manifest = json.loads((study / "dispatch_manifest.json").read_text())
    tasks, seeds = manifest["tasks"], manifest["training_seeds"]
    assert seeds == [9, 10] and len(tasks) == len({t["id"] for t in tasks}) == 10
    expected = {("balanced", m, s) for m in METHODS for s in seeds} | {("narrow", CANDIDATE, s) for s in seeds}
    assert {(t["condition"], t["method"], t["seed"]) for t in tasks} == expected
    runs = {}
    for task in tasks:
        condition, method, seed = (task[k] for k in ("condition", "method", "seed"))
        path = destination / condition / method / f"seed{seed}" / f"{REVISION}_{condition}_{method}_seed{seed}_evaluation.json"
        data = json.loads(path.read_text())
        validate_run(data, manifest, task)
        runs[(condition, method, seed)] = data
    for seed in seeds:
        assert runs[("balanced", CANDIDATE, seed)]["feedback_gains"] == runs[("narrow", CANDIDATE, seed)]["feedback_gains"]
    output = {"revision": REVISION, "completed_runs": len(runs), "training_seeds": seeds,
              "statistics_unit": "two fresh training-seed means, each evaluated on 100 reused paired paths",
              "radii": list(RADII), "severities": list(SEVERITIES), "reference_is_optimal": False,
              **summarize(runs, seeds)}
    (destination / "summary.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(output["decision"], indent=2))


if __name__ == "__main__":
    main()
