"""Validate paired actor coverage without changing the existing cost gates."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from scripts.summarize_nonlinear_load_v19 import validate as validate_formal
from scripts.summarize_nonlinear_load_v22 import validate_temperature
from scripts.summarize_nonlinear_load_v25 import summarize as cost_gates
from scripts.train_nonlinear_load_v29 import ACTOR_SEED_OFFSET, MODES, PREFIX_EPISODES, RADII, REVISION, SEVERITIES, TRAINING_SEEDS

ROOT = Path(__file__).resolve().parents[1]


def validate_pair(runs, manifest, seed, preflight=False):
    episodes, count = (6, 2) if preflight else (500, 100)
    updates = episodes * 200 - 999
    for mode in MODES:
        data = runs[mode, seed]
        if not preflight:
            validate_formal(data, manifest, {"reference": "cubic-local", "seed": seed}, revision=REVISION)
        assert (data["revision"], data["training_seed"], data["actor_sampling"], data["critic_sampling"]) == (REVISION, seed, mode, "early")
        assert data["actor_sampler_seed"] == ACTOR_SEED_OFFSET + seed and data["actor_early_steps"] == 50
        assert data["config"] == {**manifest["config"], "max_episodes": episodes}
        assert data["environment_settings"] == manifest["environment_settings"] and data["preflight_only"] == preflight
        assert (data["training_episodes"], data["environment_steps"], data["training_updates"],
                data["optimizer_updates_during_evaluation"], data["shared_prefix_episodes"]) == (episodes, episodes * 200, updates, 0, PREFIX_EPISODES)
        for package, version in manifest["runtime_requirement"].items():
            if package == "python_minor_versions":
                assert '.'.join(data["runtime"]["python"].split()[0].split('.')[:2]) in version
            else:
                assert data["runtime"][package] == version
        assert data["runtime"]["device"] == "cuda" and data["checkpoint_selection"] == "last"
        validate_temperature(data)
        assert data["jacobian_scope"] == "full" and data["jacobian_input_columns"] == list(range(6))
        for name, expected_mode in (("critic", "early"), ("actor", mode)):
            sampler = data[f"{name}_sampler"]
            replaced = updates - 1 if expected_mode == "early" else 0
            assert sampler["mode"] == expected_mode and sampler["batches"] == updates
            assert sampler["counts"] == {"uniform": (updates - replaced) * 256 + replaced * 128,
                                         "reset": replaced * 64, "early": replaced * 64}
            assert all(sum(sampler["radius_counts"][k].values()) == sampler["counts"][k] for k in ("reset", "early"))
        jacobian = data["jacobian_sampler"]
        assert jacobian["batches"] == updates and jacobian["reset_samples"] == jacobian["replay_samples"] == updates * 128
        assert data["evaluation_episodes_including_controls"] == 3 * 2 * count * 4
        for radius in RADII:
            for load in SEVERITIES:
                block = data["radius_results"][str(radius)][str(load)]
                assert set(block["controllers"]) == {"actor", "fitted", "known_local", "known_inverse"}
                for controller in block["controllers"].values():
                    assert controller["evaluation_seeds"] == list(range(60000, 60000 + count))
                    values = {k: np.asarray(controller[k]) for k in ("returns", "state_costs", "command_costs",
                              "early_state_costs", "late_state_costs", "early_command_costs", "late_command_costs")}
                    assert all(v.shape == (count,) and np.isfinite(v).all() for v in values.values())
                    np.testing.assert_allclose(-values["returns"], values["state_costs"] + values["command_costs"])
                    for kind in ("state", "command"):
                        np.testing.assert_allclose(values[f"{kind}_costs"], values[f"early_{kind}_costs"] + values[f"late_{kind}_costs"])
    left, right = [runs[mode, seed] for mode in MODES]
    assert left["reference_identification"] == right["reference_identification"]
    assert left["runtime"] == right["runtime"]
    for key in ("training_returns", "temperature_trace"):
        assert left[key][:PREFIX_EPISODES] == right[key][:PREFIX_EPISODES]
    for radius in RADII:
        for load in SEVERITIES:
            a, b = [d["radius_results"][str(radius)][str(load)] for d in (left, right)]
            for controller in ("fitted", "known_local", "known_inverse"):
                np.testing.assert_array_equal(a["controllers"][controller]["returns"], b["controllers"][controller]["returns"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    study = ROOT / "experiments/nonlinear_load_v29_actor_replay_20261003"
    destination = ROOT / "results" / REVISION
    manifest = json.loads((study / "dispatch_manifest.json").read_text())
    tasks = manifest["preflight_tasks" if args.preflight else "tasks"]
    seeds = (19,) if args.preflight else TRAINING_SEEDS
    assert len(tasks) == len(seeds) and {t["seed"] for t in tasks} == set(seeds)
    destination = destination / "preflight" if args.preflight else destination
    observations = json.loads((destination / "task_observations.json").read_text())
    assert {o["id"] for o in observations} == {t["id"] for t in tasks}
    assert all(o["status"] == "done" and o["exit_code"] == 0 for o in observations)
    runs, paths = {}, []
    for seed in seeds:
        for mode in MODES:
            path = destination / f"seed{seed}" / f"{REVISION}_{mode}_seed{seed}_evaluation.json"
            runs[mode, seed] = json.loads(path.read_text())
            paths.append(path)
        validate_pair(runs, manifest, seed, args.preflight)
    output = {"revision": REVISION, "completed_task_ids": [t["id"] for t in tasks], "preflight_only": args.preflight,
              "evaluation_json_bytes": sum(p.stat().st_size for p in paths), "paired_full_prefix_verified": True,
              "actual_training_interactions": len(seeds) * (2 * (6 if args.preflight else 500) - 5) * 200,
              "actual_training_updates": len(seeds) * (2 * ((6 if args.preflight else 500) * 200 - 999) - 1),
              "shared_identification_and_validation_interactions": len(seeds) * 11000,
              "historical_hold_unchanged": True}
    if not args.preflight:
        output.update(cost_gates(runs, seeds))
    (destination / "summary.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(output.get("decision", {"preflight_validated": True}), indent=2))


if __name__ == "__main__":
    main()
