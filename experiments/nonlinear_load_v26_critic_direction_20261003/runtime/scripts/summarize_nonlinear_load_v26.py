"""Validate v26's frozen source identity and model-own return directions."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scripts.diagnose_nonlinear_load_v26 import (
    BASELINE_KEYS, BRANCHES, GAMMA, NOISE_SEEDS, RADII, REVISION, SEVERITIES,
    SOURCE_REVISION, SOURCES, STEPS, TRAINING_SEEDS, validate_inputs,
)
from scripts.summarize_nonlinear_load_v24 import validate_model

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "experiments/nonlinear_load_v26_critic_direction_20261003"
RESULT = ROOT / "results" / REVISION


def validate(data, manifest, payloads, preflight=False):
    seed = data["training_seed"]
    validate_inputs(payloads, manifest, seed)
    assert data["revision"] == manifest["revision"] == REVISION
    assert data["source_revision"] == manifest["source_revision"] == SOURCE_REVISION
    assert data["source_ids"] == {mode: payloads[mode]["source_id"] for mode in SOURCES}
    assert data["config"] == manifest["config"] and data["environment_settings"] == manifest["environment_settings"]
    assert data["checkpoint_selection"] == "last" and data["preflight_only"] == preflight
    assert data["parameters_unchanged"] and data["optimizer_updates"] == data["fitting_interactions"] == 0
    assert data["gamma"] == GAMMA and data["finite_difference_steps"] == list(STEPS) and data["runtime"]["device"] == "cuda"
    for package, version in manifest["runtime_requirement"].items():
        if package == "python_minor_versions":
            assert '.'.join(data["runtime"]["python"].split()[0].split('.')[:2]) in version
        else:
            assert data["runtime"][package] == version
    count, radii, loads, noise = (2, (1.,), (50,), NOISE_SEEDS[:2]) if preflight else (100, RADII, SEVERITIES, NOISE_SEEDS)
    baseline_count = len(radii) * len(loads) * count * len(SOURCES)
    assert data["deterministic_reproduction_episodes"] == baseline_count
    assert data["controller_episodes"] == baseline_count * len(noise) * len(BRANCHES)
    assert len(data["raw_files"]) == len(radii) * len(loads) and all(f["bytes"] > 0 for f in data["raw_files"])
    assert set(data["radius_results"]) == {str(r) for r in radii} and set(data["terminal"]) == set(SOURCES)
    for mode, terminal in data["terminal"].items():
        assert all(terminal[k] == payloads[mode][k] for k in ("alpha", "weighted_lambda", "reward_std"))
        assert terminal["beta"] == -2 - manifest["config"]["penalty_scale"] * terminal["weighted_lambda"]
    for radius in radii:
        blocks = data["radius_results"][str(radius)]
        assert set(blocks) == {str(s) for s in loads}
        for load in loads:
            block = blocks[str(load)]
            assert block["evaluation_seeds"] == list(range(60000, 60000 + count)) and block["noise_seeds"] == list(noise)
            assert block["disturbance_paths_matched"] and block["action_noise_matched"]
            assert np.asarray(block["initial_observations"]).shape == (count, 6) and np.isfinite(block["initial_observations"]).all()
            assert set(block["models"]) == set(block["deterministic_reproduction"]) == set(SOURCES)
            for mode, model in block["models"].items():
                reproduction = block["deterministic_reproduction"][mode]
                assert reproduction["passed"] and set(reproduction["maximum_absolute_cost_gaps"]) == {"total_cost", *BASELINE_KEYS}
                assert np.isfinite(list(reproduction["maximum_absolute_cost_gaps"].values())).all()
                distances = np.asarray(model["latent_distance_to_fitted"])
                assert distances.shape == (count,) and np.isfinite(distances).all() and (distances >= 0).all()
                validate_model(model, data["terminal"][mode], count, noise)


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
    assert {o["id"] for o in observations} == {t["id"] for t in tasks}
    assert all(o["status"] == "done" and o["exit_code"] == 0 for o in observations)
    outputs, total_bytes = {}, 0
    for seed in seeds:
        path = result / f"seed{seed}" / f"{REVISION}_seed{seed}.json"
        data = json.loads(path.read_text())
        payloads = {mode: json.loads((STUDY / "inputs" / f"{mode}_seed{seed}.json").read_text()) for mode in SOURCES}
        validate(data, manifest, payloads, args.preflight_only)
        total_bytes += path.stat().st_size
        outputs[str(seed)] = {"terminal": data["terminal"], "raw_files": data["raw_files"],
            "radius_results": {r: {s: {m: v["summary"] for m, v in b["models"].items()} for s, b in loads.items()}
                for r, loads in data["radius_results"].items()}}
    summary = {"revision": REVISION, "completed_task_ids": [o["id"] for o in observations],
               "result_json_bytes": total_bytes, "optimizer_updates": 0, "fitting_interactions": 0,
               "diagnostic_only": True, "historical_hold_unchanged": True, "seeds": outputs}
    (result / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"Validated {len(seeds)} paired v25 source seeds; historical HOLD unchanged.")


if __name__ == "__main__":
    main()
