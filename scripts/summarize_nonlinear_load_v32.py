"""Admit a reference intervention only on the frozen all-block diagnostic rule."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.diagnose_nonlinear_load_v32 import CONTROLLERS, RADII, REVISION, SEED_START, SEVERITIES, STUDY, TRAINING_SEEDS

ROOT = Path(__file__).resolve().parents[1]


def validate(data, manifest, seed, preflight=False):
    count = 2 if preflight else 100
    assert (data["revision"], data["source_revision"], data["training_seed"], data["source_symbolic_constraint"],
            data["preflight_only"]) == (REVISION, manifest["source_revision"], seed, "on", preflight)
    assert data["source_config"] == manifest["source_config"] and data["environment_settings"] == manifest["environment_settings"]
    assert data["evaluation_seeds"] == list(range(SEED_START, SEED_START + count))
    assert data["radii"] == list(RADII) and data["severities"] == list(SEVERITIES)
    assert data["origin_jacobian_preserved"] and data["source_controls_reproduced"]
    assert (data["optimizer_updates"], data["new_training_interactions"], data["new_identification_interactions"],
            data["checkpoint_access"], data["new_evaluation_episodes"], data["reproduction_episodes"],
            data["evaluation_interactions"]) == (0, 0, 0, False, 18 * count, 24, (18 * count + 24) * 200)
    assert data["runtime"]["numpy"] == manifest["runtime_requirement"]["numpy"]
    assert data["runtime"]["scipy"] == manifest["runtime_requirement"]["scipy"]
    assert '.'.join(data["runtime"]["python"].split()[0].split('.')[:2]) in manifest["runtime_requirement"]["python_minor_versions"]
    for radius in RADII:
        for severity in SEVERITIES:
            block = data["radius_results"][str(radius)][str(severity)]
            assert set(block["controllers"]) == set(CONTROLLERS)
            controls = block["controllers"]
            for control in controls.values():
                assert control["evaluation_seeds"] == data["evaluation_seeds"]
                returns = np.asarray(control["returns"])
                state, command = [np.asarray(control[f"{k}_costs_by_window"]) for k in ("state", "command")]
                assert returns.shape == (count,) and state.shape == command.shape == (count, 4)
                assert np.isfinite(returns).all() and np.isfinite(state).all() and np.isfinite(command).all()
                np.testing.assert_allclose(-returns, np.sum(state + command, axis=1), rtol=1e-12, atol=1e-12)
                assert np.isclose(control["mean_cost"], -returns.mean(), rtol=1e-12, atol=1e-12)
            delta = -np.asarray(controls["fitted_inverse"]["returns"]) + controls["fitted_local"]["returns"]
            np.testing.assert_allclose(delta, block["inverse_minus_local"]["per_path_cost_changes"])
            assert np.isclose(delta.mean(), block["inverse_minus_local"]["mean_total_cost_change"])
            for mode in CONTROLLERS[1:]:
                for key in ("force_rms", "maximum_load_components", "force_change_counts"):
                    np.testing.assert_array_equal(controls[mode][key], controls["fitted_local"][key])


def decide(runs):
    better = all(block["inverse_minus_local"]["mean_total_cost_change"] < 0
                 for run in runs for radius in run["radius_results"].values() for block in radius.values())
    return {"all_seed_radius_load_reference_improvement": better,
            "reference_intervention_admitted": better, "learner_replication_admitted": False,
            "historical_hold_unchanged": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "experiments" / STUDY / "dispatch_manifest.json").read_text())
    destination = ROOT / "results" / REVISION
    destination = destination / "preflight" if args.preflight else destination
    seeds = (21,) if args.preflight else TRAINING_SEEDS
    tasks = manifest["preflight_tasks" if args.preflight else "tasks"]
    assert {t["seed"] for t in tasks} == set(seeds) and len(tasks) == len(seeds)
    observations = json.loads((destination / "task_observations.json").read_text())
    assert {t["id"] for t in tasks} == {o["id"] for o in observations}
    assert all(o["status"] == "done" and o["exit_code"] == 0 for o in observations)
    runs, paths = [], []
    for seed in seeds:
        path = destination / f"seed{seed}" / f"{REVISION}_seed{seed}_diagnostic.json"
        data = json.loads(path.read_text())
        validate(data, manifest, seed, args.preflight)
        runs.append(data)
        paths.append(path)
    output = {"revision": REVISION, "preflight_only": args.preflight,
              "completed_task_ids": [t["id"] for t in tasks], "source_controls_reproduced": True,
              "evaluation_json_bytes": sum(p.stat().st_size for p in paths),
              "evaluation_interactions": sum(d["evaluation_interactions"] for d in runs),
              "new_training_interactions": 0, "optimizer_updates": 0, "checkpoint_access": False,
              "historical_hold_unchanged": True}
    if not args.preflight:
        output["decision"] = decide(runs)
    (destination / "summary.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(output.get("decision", {"preflight_validated": True}), indent=2))


if __name__ == "__main__":
    main()
