"""Frozen scope-intervention gates, with each training seed retained separately."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scripts.summarize_nonlinear_load_v19 import validate
from scripts.train_nonlinear_load_v20 import INPUT_COLUMNS, RADII, REDUCTION, REVISION, SCOPES, SEVERITIES, TRAINING_SEEDS

ROOT = Path(__file__).resolve().parents[1]


def summarize(runs, seeds):
    differences, fitted_gates = {}, {}
    for radius in RADII:
        differences[str(radius)], fitted_gates[str(radius)] = {}, True
        for severity in SEVERITIES:
            values = []
            for seed in seeds:
                modified = runs[("load-only", seed)]["radius_results"][str(radius)][str(severity)]
                baseline = runs[("full", seed)]["radius_results"][str(radius)][str(severity)]
                actor, control = modified["controllers"]["actor"], baseline["controllers"]["actor"]
                excess = modified["paired_excess_costs"]["fitted"]["mean_total_cost"]
                fitted_gates[str(radius)] &= excess < 0
                values.append({"training_seed": seed, "modified_cost": actor["mean_cost"], "control_cost": control["mean_cost"],
                               "total_cost_change": actor["mean_cost"] - control["mean_cost"],
                               "early_state_cost_change": actor["mean_early_state_cost"] - control["mean_early_state_cost"],
                               "own_fitted_excess_cost": excess})
            differences[str(radius)][str(severity)] = values
    repair = all(v["total_cost_change"] < 0 and v["early_state_cost_change"] < 0
                 for values in differences["1.0"].values() for v in values)
    retention = all(v["total_cost_change"] <= 0 for values in differences["0.2"].values() for v in values)
    added_value = all(fitted_gates.values())
    return {"modified_minus_control": differences, "own_fitted_added_value_per_radius": fitted_gates,
            "decision": {"radius1_repair_passed": repair, "small_radius_retention_passed": retention,
                         "all_radius_own_fitted_added_value_passed": added_value,
                         "ready_for_learner_replication": repair and retention and added_value,
                         "historical_v13_v15_v17_v19_hold_unchanged": True}}


def main():
    study = ROOT / "experiments/nonlinear_load_v20_jacobian_scope_20261003"
    destination = ROOT / "results" / REVISION
    manifest = json.loads((study / "dispatch_manifest.json").read_text())
    tasks = manifest["tasks"]
    assert len(tasks) == len({t["id"] for t in tasks}) == 4
    assert {(t["scope"], t["seed"]) for t in tasks} == {(a, s) for a in SCOPES for s in TRAINING_SEEDS}
    observations = json.loads((destination / "task_observations.json").read_text())
    assert {o["id"] for o in observations} == {t["id"] for t in tasks}
    assert all(o["status"] == "done" and o["exit_code"] == 0 for o in observations)
    runs, paths, known = {}, [], {}
    for task in tasks:
        scope, seed = task["scope"], task["seed"]
        path = destination / scope / f"seed{seed}" / f"{REVISION}_{scope}_seed{seed}_evaluation.json"
        data = json.loads(path.read_text())
        validate(data, manifest, {**task, "reference": "cubic-local"}, revision=REVISION)
        assert (data["jacobian_scope"], data["jacobian_input_columns"], data["jacobian_reduction"]) == (
            scope, INPUT_COLUMNS[scope], REDUCTION)
        runs[(scope, seed)] = data
        paths.append(path)
        for radius in RADII:
            for severity in SEVERITIES:
                for mode in ("known_local", "known_inverse"):
                    returns = data["radius_results"][str(radius)][str(severity)]["controllers"][mode]["returns"]
                    np.testing.assert_allclose(returns, known.setdefault((radius, severity, mode), returns), rtol=1e-6, atol=1e-7)
    for seed in TRAINING_SEEDS:
        a, b = [runs[(scope, seed)] for scope in SCOPES]
        left, right = a["reference_identification"], b["reference_identification"]
        for name in ("linear", "cubic-local"):
            np.testing.assert_allclose(left["gains"][name], right["gains"][name], rtol=1e-6, atol=1e-7)
        for model in ("linear", "cubic"):
            np.testing.assert_allclose(left["fits"][model]["coefficients"], right["fits"][model]["coefficients"], rtol=1e-6, atol=1e-8)
        for radius in RADII:
            for severity in SEVERITIES:
                left, right = [r["radius_results"][str(radius)][str(severity)]["controllers"]["fitted"]["returns"] for r in (a, b)]
                np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-7)
    output = {"revision": REVISION, "training_seeds": list(TRAINING_SEEDS), "completed_task_ids": [t["id"] for t in tasks],
              "completed_runs": len(runs), "formal_learner_interactions": 400000, "formal_training_updates": 396004,
              "formal_identification_interactions": 40000, "formal_validation_interactions": 4000,
              "formal_evaluation_episodes_including_controls": 9600, "evaluation_json_bytes": sum(p.stat().st_size for p in paths),
              "paired_known_controls_verified": True, "matched_identification_verified": True,
              "paired_fitted_reference_verified": True, "reference_is_optimal": False, **summarize(runs, TRAINING_SEEDS)}
    (destination / "summary.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(output["decision"], indent=2))


if __name__ == "__main__":
    main()
