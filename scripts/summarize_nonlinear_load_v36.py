"""Validate corrected v36 mappings and recompute the withdrawn own-reference gate."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

REVISION = "nonlinear_load_v36_reference_alignment"
SOURCE_REVISION = "nonlinear_load_v35_reference_intervention"
REFERENCES = ("cubic-local", "cost-linear")
MAPPINGS = {"cubic-local": "tanh", "cost-linear": "clip"}
SEEDS = (21, 22)
RADII = ("0.2", "0.5", "1.0")
SEVERITIES = ("1", "50")
EVAL_START = 66000


def main():
    root = Path(__file__).resolve().parents[1]
    destination = root / "results" / REVISION
    source_destination = root / "results" / SOURCE_REVISION
    runs, paths = {}, []
    for seed in SEEDS:
        for reference in REFERENCES:
            path = destination / f"seed{seed}" / f"{REVISION}_{reference}_seed{seed}_evaluation.json"
            data = json.loads(path.read_text())
            assert data["revision"] == REVISION and data["source_revision"] == SOURCE_REVISION
            assert data["source_training_seed"] == seed and data["source_reference"] == reference
            assert data["reference_mapping"] == MAPPINGS[reference]
            assert data["reference_mapping_verified"] and data["evaluation_only"]
            assert data["optimizer_updates_during_evaluation"] == 0
            assert data["evaluation_seed_start"] == EVAL_START
            runs[reference, seed] = data
            paths.append(path)
            old_path = source_destination / f"seed{seed}" / f"{SOURCE_REVISION}_{reference}_seed{seed}_evaluation.json"
            old = json.loads(old_path.read_text())
            for radius in RADII:
                for severity in SEVERITIES:
                    new_actor = np.asarray(data["radius_results"][radius][severity]["controllers"]["actor"]["returns"])
                    old_actor = np.asarray(old["radius_results"][radius][severity]["controllers"]["actor"]["returns"])
                    np.testing.assert_array_equal(new_actor, old_actor)
                    fitted = data["radius_results"][radius][severity]["controllers"]["fitted"]
                    assert fitted["evaluation_seeds"] == list(range(EVAL_START, EVAL_START + 100))
                    excess = data["radius_results"][radius][severity]["paired_excess_costs"]["fitted"]
                    delta = np.asarray(fitted["returns"]) - new_actor
                    per_path_total = np.asarray([row["total_cost"] for row in excess["per_path"]])
                    np.testing.assert_allclose(per_path_total, delta, rtol=1e-10, atol=1e-10)
                    np.testing.assert_allclose(excess["mean_total_cost"], delta.mean(), rtol=1e-10, atol=1e-10)
    own = {}
    for radius in RADII:
        own[radius] = {}
        for severity in SEVERITIES:
            rows = []
            for seed in SEEDS:
                block = runs["cost-linear", seed]["radius_results"][radius][severity]
                excess = block["paired_excess_costs"]["fitted"]
                rows.append({"training_seed": seed,
                             "mean_total_cost_excess": excess["mean_total_cost"],
                             "mean_state_cost_excess": excess["mean_state_cost"],
                             "mean_command_cost_excess": excess["mean_command_cost"],
                             "actor_better_paths": excess["actor_better_paths"]})
            own[radius][severity] = rows
    own_gate = all(row["mean_total_cost_excess"] < 0 for values in own.values() for rows in values.values() for row in rows)
    # These two gates compare the two actor arms and are unchanged by the old fitted-controller bug.
    old_summary = json.loads((source_destination / "summary.json").read_text())
    decision = {
        "corrected_own_fitted_added_value_passed": own_gate,
        "radius1_repair_passed": old_summary["decision"]["radius1_repair_passed"],
        "small_radius_retention_passed": old_summary["decision"]["small_radius_retention_passed"],
        "ready_for_learner_replication": own_gate and old_summary["decision"]["radius1_repair_passed"] and old_summary["decision"]["small_radius_retention_passed"],
        "historical_hold_unchanged": True,
    }
    output = {
        "revision": REVISION,
        "source_revision": SOURCE_REVISION,
        "completed_reference_evaluations": [f"{reference}/seed{seed}" for seed in SEEDS for reference in REFERENCES],
        "evaluation_only": True,
        "actor_returns_reproduced": True,
        "corrected_reference_mappings": {reference: MAPPINGS[reference] for reference in REFERENCES},
        "evaluation_json_bytes": sum(path.stat().st_size for path in paths),
        "own_fitted_excess": own,
        "decision": decision,
        "historical_v35_own_fitted_decision_withdrawn": True,
        "no_training_or_optimizer_updates": True,
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "summary.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
