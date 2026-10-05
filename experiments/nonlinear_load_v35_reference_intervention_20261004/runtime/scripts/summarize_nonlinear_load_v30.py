"""Validate frozen environment-qualification evidence and retain HOLD."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scripts.diagnose_nonlinear_load_v30 import CONTROLLERS, PROFILES, RADII, REVISION, SEED_START, SEVERITIES, qualify_radius

ROOT = Path(__file__).resolve().parents[1]


def validate(data, radius, preflight=False):
    count = 2 if preflight else 100
    seeds = list(range(SEED_START, SEED_START + count))
    assert (data["revision"], data["radius"], data["preflight_only"]) == (REVISION, radius, preflight)
    assert data["environment_settings"] == {p:cls.specification() for p,cls in PROFILES.items()}
    assert data["evaluation_seeds"] == seeds and data["controllers"] == list(CONTROLLERS)
    assert data["severities"] == list(SEVERITIES) and data["window_steps"] == 50
    assert (data["controller_episodes"], data["evaluation_interactions"]) == (20 * count, 4000 * count)
    assert data["rl_training_runs"] == data["optimizer_updates"] == data["fitting_interactions"] == 0
    assert data["historical_hold_unchanged"] and not data["reference_is_optimal"]
    assert data["runtime"]["device"] == "cpu" and data["runtime"]["numpy"] == "2.2.6" and data["runtime"]["scipy"] == "1.15.3"
    for profile in PROFILES:
        for load in SEVERITIES:
            controllers = data["results"][profile][str(load)]
            assert set(controllers) == set(CONTROLLERS)
            shared_load = None
            for c in controllers.values():
                assert c["evaluation_seeds"] == seeds
                state, command = [np.asarray(c[f"{k}_costs_by_window"]) for k in ("state", "command")]
                assert state.shape == command.shape == (count, 4)
                assert np.isfinite(state).all() and np.isfinite(command).all() and np.min(state) >= 0 and np.min(command) >= 0
                returns = np.asarray(c["returns"])
                assert returns.shape == (count,) and np.isfinite(returns).all()
                np.testing.assert_allclose(-returns, (state + command).sum(1), rtol=1e-12, atol=1e-12)
                assert c["mean_cost"] == float(-returns.mean())
                np.testing.assert_allclose(c["mean_state_costs_by_window"], state.mean(0))
                np.testing.assert_allclose(c["mean_command_costs_by_window"], command.mean(0))
                assert c["late_cost_share"] == float((state[:, 1:] + command[:, 1:]).sum() / -returns.sum())
                for key in ("maximum_physical_state_norms", "saturated_component_fractions", "force_rms", "maximum_load_components", "force_change_counts"):
                    v = np.asarray(c[key])
                    assert v.shape == (count,) and np.isfinite(v).all() and np.min(v) >= 0
                assert max(c["saturated_component_fractions"]) <= 1
                assert max(c["maximum_load_components"]) <= data["environment_settings"][profile]["base_force_cap"] * load
                load_values = {k:c[k] for k in ("force_rms", "maximum_load_components", "force_change_counts")}
                assert shared_load is None or load_values == shared_load
                shared_load = load_values
    assert data["qualification"] == qualify_radius(data["results"], radius)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    study = ROOT / "experiments/nonlinear_load_v30_environment_20261003"
    destination = ROOT / "results" / REVISION
    manifest = json.loads((study / "dispatch_manifest.json").read_text())
    destination = destination / "preflight" if args.preflight else destination
    tasks = manifest["preflight_tasks" if args.preflight else "tasks"]
    radii = (0.2,) if args.preflight else RADII
    assert {t["radius"] for t in tasks} == set(radii) and len(tasks) == len(radii)
    observations = json.loads((destination / "task_observations.json").read_text())
    assert {o["id"] for o in observations} == {t["id"] for t in tasks}
    assert all(o["status"] == "done" and o["exit_code"] == 0 for o in observations)
    reports, paths = {}, []
    for radius in radii:
        path = destination / f"radius{radius}" / f"{REVISION}_radius{radius}.json"
        data = json.loads(path.read_text())
        validate(data, radius, args.preflight)
        reports[str(radius)] = data["qualification"]
        paths.append(path)
    output = {"revision": REVISION, "preflight_only": args.preflight,
              "completed_task_ids": [t["id"] for t in tasks], "evidence_validated": True,
              "evaluation_json_bytes": sum(p.stat().st_size for p in paths),
              "controller_episodes": len(radii) * (40 if args.preflight else 2000),
              "evaluation_interactions": len(radii) * (8000 if args.preflight else 400000),
              "rl_training_runs": 0, "optimizer_updates": 0, "qualification_per_radius": reports,
              "historical_hold_unchanged": True}
    if not args.preflight:
        output["environment_qualified"] = all(r["qualified"] for r in reports.values())
    (destination / "summary.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
