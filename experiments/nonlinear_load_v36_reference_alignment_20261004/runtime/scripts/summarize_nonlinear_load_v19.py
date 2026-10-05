"""Frozen actor reference gates, without pooling seeds or historical cohorts."""
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scripts.train_nonlinear_load_v19 import CANDIDATE, RADII, REFERENCES, REVISION, SEVERITIES, TRAINING_SEEDS

ROOT = Path(__file__).resolve().parents[1]


def summarize(runs, seeds):
    differences, fitted_gates = {}, {}
    for radius in RADII:
        differences[str(radius)], fitted_gates[str(radius)] = {}, True
        for severity in SEVERITIES:
            values = []
            for seed in seeds:
                modified = runs[("cubic-local", seed)]["radius_results"][str(radius)][str(severity)]
                baseline = runs[("linear", seed)]["radius_results"][str(radius)][str(severity)]
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
            "decision": {"reference_repair_passed": repair, "small_radius_retention_passed": retention,
                         "all_radius_own_fitted_added_value_passed": added_value,
                         "ready_for_learner_replication": repair and retention and added_value,
                         "historical_v13_v15_v17_hold_unchanged": True}}


def validate(data, manifest, task, revision=REVISION, *, evaluation_seed_start=60000, method=CANDIDATE):
    reference, seed = task["reference"], task["seed"]
    assert (data["revision"], data["method"], data["reference"], data["training_seed"], data["sampling"]) == (
        revision, method, reference, seed, "reset-balanced")
    assert data["config"] == manifest["config"] and data["environment_settings"] == manifest["environment_settings"]
    assert not data["preflight_only"] and data["checkpoint_selection"] == "last"
    assert (data["training_episodes"], data["environment_steps"], data["training_updates"],
            data["symbolic_identification_interactions"], data["symbolic_validation_interactions"],
            data["optimizer_updates_during_evaluation"], data["training_load_severity"],
            data["identification_reset_radius"], data["early_steps"], data["sample_interval"]) == (
        500, 100000, 99001, 10000, 1000, 0, 1, .2, 50, 10)
    assert data["reset_cycle"] == list(RADII) and data["reset_index"] == "absolute_episode"
    assert data["reset_counts"] == dict(Counter(str(RADII[e % 3]) for e in range(500)))
    assert data["radii"] == list(RADII) and data["severities"] == list(SEVERITIES)
    assert data["evaluation_seed_start"] == evaluation_seed_start and data["evaluation_episodes_per_severity"] == 100
    assert len(data["training_returns"]) == 500 and np.isfinite(data["training_returns"]).all()
    assert data["sindy"] is not None and data["evaluation_episodes_including_controls"] == 2400
    assert data["runtime"]["device"] == "cuda"
    for package, version in manifest["runtime_requirement"].items():
        if package == "python_minor_versions":
            assert ".".join(data["runtime"]["python"].split()[0].split(".")[:2]) in version
        else:
            assert data["runtime"][package] == version
    sampler = data["jacobian_sampler"]
    assert (data["reset_sample_fraction"], data["sampler_seed"], sampler["episode_length"], sampler["batches"],
            sampler["reset_samples"], sampler["replay_samples"]) == (.5, 100000 + seed, 200, 99001, 12672128, 12672128)
    assert set(sampler["radius_samples"]) == {"0.2", "0.5", "1.0"}
    assert sum(sampler["radius_samples"].values()) == sampler["reset_samples"]
    identification = data["reference_identification"]
    assert (identification["selected_reference"], identification["identification_interactions"],
            identification["validation_interactions"], identification["auxiliary_world_model"]) == (reference, 10000, 1000, "linear-unchanged")
    assert identification["both_models_use_same_transitions"]
    assert data["feedback_gains"]["fitted"] == identification["gains"][reference]
    for radius in RADII:
        for severity in SEVERITIES:
            block = data["radius_results"][str(radius)][str(severity)]
            assert set(block["controllers"]) == {"actor", "fitted", "known_local", "known_inverse"}
            for result in block["controllers"].values():
                assert result["evaluation_seeds"] == list(range(evaluation_seed_start, evaluation_seed_start + 100))
                arrays = {key: np.asarray(result[key]) for key in ("returns", "state_costs", "command_costs",
                          "early_state_costs", "early_command_costs", "late_state_costs", "late_command_costs")}
                assert all(a.shape == (100,) and np.isfinite(a).all() for a in arrays.values())
                np.testing.assert_allclose(-arrays["returns"], arrays["state_costs"] + arrays["command_costs"], rtol=1e-12, atol=1e-12)
                assert np.isclose(result["mean_cost"], -arrays["returns"].mean(), rtol=1e-12, atol=1e-12)
                for kind in ("state", "command"):
                    np.testing.assert_allclose(arrays[f"{kind}_costs"], arrays[f"early_{kind}_costs"] + arrays[f"late_{kind}_costs"])
                for key, array in arrays.items():
                    if key != "returns":
                        assert np.isclose(result[f"mean_{key[:-1]}"], array.mean(), rtol=1e-12, atol=1e-12)
            for mode, excess in block["paired_excess_costs"].items():
                delta = np.asarray(block["controllers"][mode]["returns"]) - block["controllers"]["actor"]["returns"]
                np.testing.assert_allclose([v["total_cost"] for v in excess["per_path"]], delta, rtol=1e-10, atol=1e-10)
                assert np.isclose(excess["mean_total_cost"], delta.mean(), rtol=1e-10, atol=1e-10)


def main():
    study = ROOT / "experiments/nonlinear_load_v19_reference_20261002"
    destination = ROOT / "results" / REVISION
    manifest = json.loads((study / "dispatch_manifest.json").read_text())
    tasks = manifest["tasks"]
    assert len(tasks) == len({t["id"] for t in tasks}) == 4
    assert {(t["reference"], t["seed"]) for t in tasks} == {(a, s) for a in REFERENCES for s in TRAINING_SEEDS}
    observations = json.loads((destination / "task_observations.json").read_text())
    assert {o["id"] for o in observations} == {t["id"] for t in tasks}
    assert all(o["status"] == "done" and o["exit_code"] == 0 for o in observations)
    runs, paths, known = {}, [], {}
    for task in tasks:
        reference, seed = task["reference"], task["seed"]
        path = destination / reference / f"seed{seed}" / f"{REVISION}_{reference}_seed{seed}_evaluation.json"
        data = json.loads(path.read_text())
        validate(data, manifest, task)
        runs[(reference, seed)] = data
        paths.append(path)
        for radius in RADII:
            for severity in SEVERITIES:
                for mode in ("known_local", "known_inverse"):
                    returns = data["radius_results"][str(radius)][str(severity)]["controllers"][mode]["returns"]
                    np.testing.assert_allclose(returns, known.setdefault((radius, severity, mode), returns), rtol=1e-6, atol=1e-7)
    for seed in TRAINING_SEEDS:
        a, b = [runs[(reference, seed)]["reference_identification"] for reference in REFERENCES]
        for name in REFERENCES:
            np.testing.assert_allclose(a["gains"][name], b["gains"][name], rtol=1e-6, atol=1e-7)
        for model in ("linear", "cubic"):
            np.testing.assert_allclose(a["fits"][model]["coefficients"], b["fits"][model]["coefficients"], rtol=1e-6, atol=1e-8)
    output = {"revision": REVISION, "training_seeds": list(TRAINING_SEEDS), "completed_task_ids": [t["id"] for t in tasks],
              "completed_runs": len(runs), "formal_learner_interactions": 400000, "formal_training_updates": 396004,
              "formal_identification_interactions": 40000, "formal_validation_interactions": 4000,
              "formal_evaluation_episodes_including_controls": 9600, "evaluation_json_bytes": sum(p.stat().st_size for p in paths),
              "paired_known_controls_verified": True, "matched_identification_verified": True,
              "reference_is_optimal": False, **summarize(runs, TRAINING_SEEDS)}
    (destination / "summary.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(output["decision"], indent=2))


if __name__ == "__main__":
    main()
