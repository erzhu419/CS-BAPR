"""Validate critic-only calibration without changing policy-performance gates."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scripts.calibrate_nonlinear_load_v27 import (
    EP_LEN, NOISE_SEEDS, RADII, REVISION, SEVERITIES, SOURCE_REVISION, STEPS,
    TRAINING_SEEDS, calibration_spec, validate_source,
)
from scripts.summarize_nonlinear_load_v24 import validate_model
from scripts.summarize_nonlinear_load_v26 import direction_comparison

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "experiments/nonlinear_load_v27_critic_calibration_20261003"
RESULT = ROOT / "results" / REVISION


def shared_model(block, model):
    return {**model, "continuations": block["continuations"],
            "per_draw_mean_derivatives": {str(h): {
                "continuations": block["per_draw_return_mean_derivatives"][str(h)],
                "critic_values": model["per_draw_score_mean_derivatives"][str(h)]} for h in STEPS}}


def validate(data, manifest, payload, source, preflight=False):
    validate_source(payload, source, manifest, data["training_seed"])
    assert data["revision"] == manifest["revision"] == REVISION and data["source_revision"] == SOURCE_REVISION
    seed = str(data["training_seed"])
    assert data["source_training_id"] == manifest["source_tasks"][seed]["training_id"]
    assert data["source_diagnostic_id"] == manifest["source_tasks"][seed]["diagnostic_id"]
    assert data["config"] == manifest["config"] and data["environment_settings"] == manifest["environment_settings"]
    assert data["preflight_only"] == preflight and data["checkpoint_selection"] == "fixed-final-update"
    assert data["actor_unchanged"] and data["original_critic_unchanged"] and data["measured_returns_reused_from_v26"]
    assert data["online_training_interactions"] == data["new_evaluation_environment_interactions"] == 0
    spec = calibration_spec(preflight)
    assert data["calibration_spec"] == data["calibration"]["spec"] == spec
    assert data["optimizer_updates"] == data["fit"]["updates"] == spec["updates"]
    assert not set(spec["path_seeds"]).intersection(range(60000, 60100))
    assert not set(spec["noise_seeds"]).intersection(NOISE_SEEDS) and spec["load"] == 1
    for package, version in manifest["runtime_requirement"].items():
        if package == "python_minor_versions":
            assert '.'.join(data["runtime"]["python"].split()[0].split('.')[:2]) in version
        else:
            assert data["runtime"][package] == version
    assert data["runtime"]["device"] == "cuda" and data["terminal"] == source["terminal"]["early"]
    calibration = data["calibration"]
    states = len(spec["radii"]) * len(spec["path_seeds"])
    episodes = states * len(spec["noise_seeds"]) * 2
    assert calibration["states"] == states and calibration["controller_episodes"] == episodes
    assert calibration["environment_interactions"] == episodes * EP_LEN
    assert calibration["disturbance_paths_matched"] and calibration["action_noise_matched"]
    assert calibration["positive_direction_labels"] + calibration["negative_direction_labels"] <= states
    assert len(calibration["raw_files"]) == len(spec["radii"]) and all(f["bytes"] > 0 for f in calibration["raw_files"])
    history = data["fit"]["history"]
    assert history[0]["step"] == 0 and history[-1]["step"] == spec["updates"]
    assert all(np.isfinite([v[k] for k in ("total", "direction", "anchor")]).all() for v in history)
    for row in history:
        np.testing.assert_allclose(row["total"], row["direction"] + spec["anchor_weight"] * row["anchor"], rtol=1e-6)
    assert np.isfinite(data["fit"]["parameter_change_l2"]) and data["fit"]["parameter_change_l2"] > 0
    assert set(data["radius_results"]) == {str(r) for r in RADII}
    for radius in RADII:
        assert set(data["radius_results"][str(radius)]) == {str(s) for s in SEVERITIES}
        for load in SEVERITIES:
            block = data["radius_results"][str(radius)][str(load)]
            previous = source["radius_results"][str(radius)][str(load)]
            measured = previous["models"]["early"]
            assert block["evaluation_seeds"] == previous["evaluation_seeds"] == list(range(60000, 60100))
            assert block["noise_seeds"] == list(NOISE_SEEDS) and block["initial_observations"] == previous["initial_observations"]
            assert block["latent_distance_to_fitted"] == measured["latent_distance_to_fitted"]
            assert block["continuations"] == measured["continuations"] and set(block["models"]) == {"original", "calibrated"}
            for h in STEPS:
                assert block["per_draw_return_mean_derivatives"][str(h)] == measured["per_draw_mean_derivatives"][str(h)]["continuations"]
            for mode, model in block["models"].items():
                validate_model(shared_model(block, model), data["terminal"], 100, NOISE_SEEDS)
                if mode == "original":
                    for branch, values in model["critic_values"].items():
                        for key, value in values.items():
                            np.testing.assert_allclose(value, measured["critic_values"][branch][key], rtol=1e-4, atol=1e-5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    result = RESULT / "preflight" if args.preflight_only else RESULT
    manifest = json.loads((STUDY / "dispatch_manifest.json").read_text())
    observations = json.loads((result / "task_observations.json").read_text())
    tasks = manifest["preflight_tasks" if args.preflight_only else "tasks"]
    seeds = (19,) if args.preflight_only else TRAINING_SEEDS
    assert len(tasks) == len(observations) == len(seeds) and {t["seed"] for t in tasks} == set(seeds)
    assert {t["id"] for t in tasks} == {o["id"] for o in observations}
    assert all(o["status"] == "done" and o["exit_code"] == 0 for o in observations)
    outputs, total_bytes = {}, 0
    for seed in seeds:
        path = result / f"seed{seed}" / f"{REVISION}_seed{seed}.json"
        data = json.loads(path.read_text())
        payload = json.loads((STUDY / "inputs" / f"early_seed{seed}.json").read_text())
        source = json.loads((STUDY / "inputs" / f"v26_seed{seed}.json").read_text())
        validate(data, manifest, payload, source, args.preflight_only)
        total_bytes += path.stat().st_size
        outputs[str(seed)] = {"calibration": data["calibration"], "fit": data["fit"],
            "radius_results": {r: {s: direction_comparison({"models": {
                mode: shared_model(b, model) for mode, model in b["models"].items()}}) for s, b in loads.items()}
                for r, loads in data["radius_results"].items()}}
    summary = {"revision": REVISION, "completed_task_ids": [o["id"] for o in observations],
               "result_json_bytes": total_bytes, "seeds": outputs, "actor_unchanged": True,
               "policy_performance_gate_unchanged": True, "historical_hold_unchanged": True}
    (result / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"Validated {len(seeds)} frozen-actor critic calibrations; policy-performance gate/HOLD unchanged.")


if __name__ == "__main__":
    main()
