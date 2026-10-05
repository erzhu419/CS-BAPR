"""Validate stochastic continuation diagnostics without training admission."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scripts.diagnose_nonlinear_load_v24 import (
    BRANCHES, COSTS, NOISE_SEEDS, RADII, RETURNS, REVISION, SCORES, SEVERITIES, SOURCES, STEPS, summarize_block,
)

ROOT = Path(__file__).resolve().parents[1]


def validate(data, manifest, preflight=False):
    assert data["revision"] == manifest["revision"] and data["input_revision"] == manifest["input_revision"]
    assert data["source_ids"] == {mode: manifest["source_tasks"][mode]["id"] for mode in SOURCES}
    assert data["training_seed"] == 18 and data["checkpoint_selection"] == "last" and data["preflight_only"] == preflight
    assert data["configs"] == manifest["configs"] and data["environment_settings"] == manifest["environment_settings"]
    assert data["parameters_unchanged"] and data["optimizer_updates"] == data["fitting_interactions"] == 0
    assert data["gamma"] == .99 and data["finite_difference_steps"] == list(STEPS) and data["runtime"]["device"] == "cuda"
    for package, version in manifest["runtime_requirement"].items():
        if package == "python_minor_versions":
            assert '.'.join(data["runtime"]["python"].split()[0].split('.')[:2]) in version
        else:
            assert data["runtime"][package] == version
    count, radii, loads, noise = (2, (1.,), (50,), NOISE_SEEDS[:2]) if preflight else (100, RADII, SEVERITIES, NOISE_SEEDS)
    assert data["controller_episodes"] == len(radii) * len(loads) * count * len(noise) * len(BRANCHES) * 2
    assert len(data["raw_files"]) == len(radii) * len(loads) and all(f["bytes"] > 0 for f in data["raw_files"])
    assert set(data["radius_results"]) == {str(r) for r in radii}
    for radius in radii:
        blocks = data["radius_results"][str(radius)]
        assert set(blocks) == {str(s) for s in loads}
        for load in loads:
            block = blocks[str(load)]
            assert block["evaluation_seeds"] == list(range(60000, 60000 + count)) and block["noise_seeds"] == list(noise)
            assert block["disturbance_paths_matched"] and block["action_noise_matched"]
            assert np.asarray(block["initial_observations"]).shape == (count, 6) and set(block["models"]) == set(SOURCES)
            for mode, model in block["models"].items():
                continuations, scores = model["continuations"], model["critic_values"]
                assert set(continuations) == set(scores) == set(BRANCHES)
                for branch, values in continuations.items():
                    assert set(values) == set(COSTS + RETURNS)
                    assert all(np.asarray(v).shape == (count,) and np.isfinite(v).all() for v in values.values())
                    c = {k: np.asarray(v) for k, v in values.items()}
                    np.testing.assert_allclose(c["total_cost"], c["state_cost"] + c["command_cost"])
                    np.testing.assert_allclose(c["state_cost"], c["early_state_cost"] + c["late_state_cost"])
                    np.testing.assert_allclose(c["discounted_cost"], c["discounted_state_cost"] + c["discounted_command_cost"])
                    np.testing.assert_allclose(c["normalized_reward_return"], -c["discounted_cost"] / data["terminal"][mode]["reward_std"])
                    np.testing.assert_allclose(c["soft_q_return"], c["normalized_reward_return"] + c["future_entropy_bonus"])
                    np.testing.assert_allclose(c["soft_actor_return"], c["soft_q_return"] + c["initial_entropy_bonus"])
                    q = {k: np.asarray(v) for k, v in scores[branch].items()}
                    assert set(q) == set(SCORES) and all(v.shape == (count,) and np.isfinite(v).all() for v in q.values())
                    np.testing.assert_allclose(q["risk_score"], q["q_mean"] + q["risk_penalty"])
                    np.testing.assert_allclose(q["unpenalized_soft_score"], q["q_mean"] + q["entropy_bonus"])
                    np.testing.assert_allclose(q["soft_score"], q["risk_score"] + q["entropy_bonus"])
                    np.testing.assert_allclose(q["entropy_bonus"], c["initial_entropy_bonus"])
                assert set(model["analytic_score_derivatives"]) == set(SCORES)
                assert all(np.asarray(v).shape == (count,) and np.isfinite(v).all() for v in model["analytic_score_derivatives"].values())
                assert model["summary"] == summarize_block(continuations, scores, model["analytic_score_derivatives"])
                for h in STEPS:
                    for group, keys in (("continuations", COSTS + RETURNS), ("critic_values", SCORES)):
                        draws = model["per_draw_mean_derivatives"][str(h)][group]
                        assert set(draws) == set(keys)
                        for key, values in draws.items():
                            assert np.asarray(values).shape == (len(noise),) and np.isfinite(values).all()
                            expected = model["summary"]["mean_directional_derivatives"][str(h)][group][key]
                            np.testing.assert_allclose(np.mean(values), expected, rtol=1e-8, atol=1e-9)


def main():
    study = ROOT / "experiments/nonlinear_load_v24_stochastic_20261003"
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
               "diagnostic_only": True, "historical_hold_unchanged": True, "terminal": data["terminal"], "raw_files": data["raw_files"],
               "radius_results": {r: {s: {m: {"summary": v["summary"], "per_draw_mean_derivatives": v["per_draw_mean_derivatives"]}
                                           for m, v in b["models"].items()} for s, b in loads.items()}
                                  for r, loads in data["radius_results"].items()}}
    (result / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print("Validated paired stochastic soft-return directions; historical HOLD unchanged.")


if __name__ == "__main__":
    main()
