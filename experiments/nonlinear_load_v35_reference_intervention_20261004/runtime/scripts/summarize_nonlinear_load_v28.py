"""Validate compact full-parameter diagnostics and preserve the historical gate."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from scripts.diagnose_nonlinear_load_v28 import (
    ARMS, BATCHES, COMPONENTS, EP_LEN, NOISE_SEEDS, PARAMETER_SCALES,
    RADII, REVISION, SEVERITIES, TRAINING_SEEDS, TRAIN_NOISE,
)

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "experiments/nonlinear_load_v28_actor_update_20261003"
RESULT = ROOT / "results" / REVISION


def validate(data, manifest, previous, preflight):
    seed = str(data["training_seed"])
    assert data["revision"] == manifest["revision"] == REVISION
    assert data["source_id"] == manifest["source_tasks"][seed]["training_id"]
    assert data["config"] == manifest["config"] and data["environment_settings"] == manifest["environment_settings"]
    assert data["terminal"] == previous["terminal"]["early"]
    assert data["preflight_only"] == preflight and data["parameter_scales"] == list(PARAMETER_SCALES)
    assert data["original_parameters_unchanged"] and data["retained_optimizer_updates"] == data["training_environment_interactions"] == 0
    assert data["shadow_adam_steps"] == (5 if preflight else 129)
    assert data["controller_episodes"] == (36 if preflight else 28800)
    assert data["evaluation_environment_interactions"] == data["controller_episodes"] * EP_LEN
    for key, value in manifest["runtime_requirement"].items():
        if key == "python_minor_versions":
            assert '.'.join(data["runtime"]["python"].split()[0].split('.')[:2]) in value
        else:
            assert data["runtime"][key] == value
    assert data["runtime"]["device"] == "cuda"
    assert len(data["batch_support"]) == BATCHES
    for bank in data["batch_support"]:
        for name in (*ARMS, "jacobian"):
            assert bank[name]["count"] == sum(bank[name]["episode_phase_counts"].values()) == 256
        assert bank["early"]["episode_phase_counts"]["reset"] >= 64
        assert bank["jacobian"]["episode_phase_counts"]["reset"] >= 128
    for arm in ARMS:
        draws = data["gradients"][arm]["draws"]
        assert len(draws) == (2 if preflight else BATCHES * len(TRAIN_NOISE))
        assert {(r["batch"], r["noise_seed"]) for r in draws} == {
            (batch, noise) for batch in range(1 if preflight else BATCHES)
            for noise in (TRAIN_NOISE[:2] if preflight else TRAIN_NOISE)}
        assert data["gradients"][arm]["mean_adam_delta_l2"] > 0
        for row in draws:
            assert set(row["losses"]) == set(COMPONENTS)
            assert 0 < row["clip_scale"] <= 1 and row["adam_delta_l2"] > 0
            assert np.isfinite(list(row["losses"].values()) + list(row["gradient_norms"].values()) + list(row["component_loss_derivatives_under_adam"].values())).all()
    radii, loads = ((1.,), (1,)) if preflight else (RADII, SEVERITIES)
    assert set(data["radius_results"]) == {str(r) for r in radii}
    for radius in radii:
        assert set(data["radius_results"][str(radius)]) == {str(s) for s in loads}
        for load in loads:
            block = data["radius_results"][str(radius)][str(load)]
            assert block["evaluation_seeds"] == list(range(60000, 60002 if preflight else 60100))
            assert block["noise_seeds"] == list(NOISE_SEEDS[:2] if preflight else NOISE_SEEDS)
            assert block["disturbance_paths_matched"] and block["action_noise_matched"]
            assert block["baseline_v26_reproduced"] == (not preflight)
            for arm in ARMS:
                for h in PARAMETER_SCALES:
                    result = block["arms"][arm][str(h)]
                    measured = result["continuations"]
                    for group in ("continuations", "critic_values"):
                        for key, value in result[group]["mean_derivatives"].items():
                            assert np.isfinite(value)
                            paths, draws = result[group]["per_path_mean_derivatives"][key], result[group]["per_draw_mean_derivatives"][key]
                            assert len(paths) == len(block["evaluation_seeds"]) and len(draws) == len(block["noise_seeds"])
                            np.testing.assert_allclose(value, np.mean(paths), rtol=1e-6, atol=1e-9)
                            np.testing.assert_allclose(value, np.mean(draws), rtol=1e-6, atol=1e-9)
                    values = measured["mean_derivatives"]
                    np.testing.assert_allclose(values["soft_q_return"], values["normalized_reward_return"] + values["future_entropy_bonus"], rtol=1e-5, atol=1e-8)
                    np.testing.assert_allclose(values["soft_actor_return"], values["soft_q_return"] + values["initial_entropy_bonus"], rtol=1e-5, atol=1e-8)
                    np.testing.assert_allclose(values["normalized_reward_return"], -values["discounted_cost"] / data["terminal"]["reward_std"], rtol=1e-5, atol=1e-8)
    assert len(data["raw_files"]) == len(radii) * len(loads)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    directory = RESULT / "preflight" if args.preflight_only else RESULT
    manifest = json.loads((STUDY / "dispatch_manifest.json").read_text())
    observations = json.loads((directory / "task_observations.json").read_text())
    tasks = manifest["preflight_tasks" if args.preflight_only else "tasks"]
    seeds = (19,) if args.preflight_only else TRAINING_SEEDS
    assert len(tasks) == len(observations) == len(seeds)
    assert {t["id"] for t in tasks} == {t["id"] for t in observations}
    assert all(t["status"] == "done" and t["exit_code"] == 0 for t in observations)
    rows, total_bytes = {}, 0
    for seed in seeds:
        path = directory / f"seed{seed}" / f"{REVISION}_seed{seed}.json"
        data = json.loads(path.read_text())
        previous = json.loads((STUDY / "inputs" / f"v26_seed{seed}.json").read_text())
        validate(data, manifest, previous, args.preflight_only)
        rows[str(seed)] = data
        total_bytes += path.stat().st_size
    (directory / "summary.json").write_text(json.dumps({"revision": REVISION, "completed_task_ids": [t["id"] for t in tasks],
        "result_json_bytes": total_bytes, "seeds": rows, "historical_hold_unchanged": True}, indent=2, allow_nan=False) + "\n")
    print(f"Validated {len(seeds)} full-parameter diagnostics; historical HOLD unchanged")


if __name__ == "__main__":
    main()
