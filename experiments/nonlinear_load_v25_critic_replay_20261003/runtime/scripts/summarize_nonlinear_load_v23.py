"""Validate paired mean-direction measurements without training admission."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scripts.diagnose_nonlinear_load_v23 import BRANCHES, NOISE_SEEDS, RADII, REVISION, SEVERITIES, SOURCES, summarize_block

ROOT = Path(__file__).resolve().parents[1]


def validate(data, manifest, preflight=False):
    assert data["revision"] == manifest["revision"] and data["source_revision"] == manifest["source_revision"]
    assert data["source_ids"] == {mode: manifest["source_tasks"][mode]["id"] for mode in SOURCES}
    assert data["training_seed"] == 18 and data["checkpoint_selection"] == "last" and data["preflight_only"] == preflight
    assert data["configs"] == manifest["configs"] and data["environment_settings"] == manifest["environment_settings"]
    assert data["parameters_unchanged"] and data["optimizer_updates"] == data["fitting_interactions"] == 0
    assert data["gamma"] == .99 and data["finite_difference_steps"] == [.01, .005] and data["runtime"]["device"] == "cuda"
    for package, version in manifest["runtime_requirement"].items():
        if package == "python_minor_versions":
            assert '.'.join(data["runtime"]["python"].split()[0].split('.')[:2]) in version
        else:
            assert data["runtime"][package] == version
    count, radii, loads, noise = (2, (1.,), (50,), NOISE_SEEDS[:2]) if preflight else (100, RADII, SEVERITIES, NOISE_SEEDS)
    assert data["controller_episodes"] == len(radii) * len(loads) * count * len(BRANCHES) * 2
    assert set(data["radius_results"]) == {str(r) for r in radii}
    for radius in radii:
        blocks = data["radius_results"][str(radius)]
        assert set(blocks) == {str(s) for s in loads}
        for load in loads:
            block = blocks[str(load)]
            assert block["evaluation_seeds"] == list(range(60000, 60000 + count)) and block["disturbance_paths_matched"]
            assert np.asarray(block["initial_observations"]).shape == (count, 6) and set(block["models"]) == set(SOURCES)
            for model in block["models"].values():
                continuations, scores = model["continuations"], model["scores"]
                assert set(continuations) == set(BRANCHES) and scores["noise_seeds"] == list(noise)
                for costs in continuations.values():
                    assert all(np.asarray(v).shape == (count,) and np.isfinite(v).all() for v in costs.values())
                    np.testing.assert_allclose(costs["total_cost"], np.asarray(costs["state_cost"]) + costs["command_cost"])
                    np.testing.assert_allclose(costs["state_cost"], np.asarray(costs["early_state_cost"]) + costs["late_state_cost"])
                assert np.asarray(scores["latent_direction"]).shape == (count, 2)
                assert np.asarray(scores["latent_distance_to_fitted"]).shape == (count,)
                assert np.isfinite(scores["latent_direction"]).all() and np.isfinite(scores["latent_distance_to_fitted"]).all()
                for mode, draws in (("deterministic", 1), ("expected", len(noise))):
                    assert all(np.asarray(v).shape == (draws, count) and np.isfinite(v).all() for v in scores[mode].values())
                    c = {k: np.asarray(v) for k, v in scores[mode].items()}
                    np.testing.assert_allclose(c["risk_score"], c["q_mean"] + c["risk_penalty"])
                    np.testing.assert_allclose(c["soft_score"], c["risk_score"] + c["entropy_bonus"])
                assert model["summary"] == summarize_block(continuations, scores)


def main():
    study = ROOT / "experiments/nonlinear_load_v23_direction_20261003"
    result = ROOT / "results" / REVISION
    manifest = json.loads((study / "dispatch_manifest.json").read_text())
    observations = json.loads((result / "task_observations.json").read_text())
    assert len(manifest["tasks"]) == len(observations) == 1
    assert observations[0]["id"] == manifest["tasks"][0]["id"] and observations[0]["status"] == "done" and observations[0]["exit_code"] == 0
    path = result / f"{REVISION}_seed18.json"
    data = json.loads(path.read_text())
    validate(data, manifest)
    summary = {"revision": REVISION, "completed_task_id": observations[0]["id"], "controller_episodes": data["controller_episodes"],
               "result_json_bytes": path.stat().st_size, "optimizer_updates": 0, "fitting_interactions": 0,
               "diagnostic_only": True, "historical_hold_unchanged": True, "terminal": data["terminal"],
               "radius_results": {r: {s: {m: v["summary"] for m, v in b["models"].items()} for s, b in loads.items()}
                                  for r, loads in data["radius_results"].items()}}
    (result / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print("Validated same-GPU seed18 directional diagnostics; no training admission inferred.")


if __name__ == "__main__":
    main()
