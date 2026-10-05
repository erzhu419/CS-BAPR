"""Validate the reference-only trial and preserve the existing learner gates."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.summarize_nonlinear_load_v19 import validate as validate_formal
from scripts.summarize_nonlinear_load_v22 import summarize as cost_gates, validate_temperature
from scripts.train_nonlinear_load_v35 import (
    EVALUATION_SEED_START, MAPPINGS, METHOD, PREFIX_EPISODES, PREFLIGHT_SEED_START,
    RADII, REFERENCES, REVISION, SEVERITIES, STUDY, TRAINING_SEEDS,
)

ROOT = Path(__file__).resolve().parents[1]


def summarize(runs, seeds):
    renamed = {(name, seed): runs[reference, seed] for name, reference in (
        ("floor", "cubic-local"), ("no-floor", "cost-linear")) for seed in seeds}
    return cost_gates(renamed, seeds)


def validate_pair(runs, manifest, seed, preflight=False):
    episodes, count = (6, 2) if preflight else (500, 100)
    start = PREFLIGHT_SEED_START if preflight else EVALUATION_SEED_START
    updates = episodes * 200 - 999
    for reference in REFERENCES:
        data = runs[reference, seed]
        if not preflight:
            validate_formal(data, manifest, {"reference": reference, "seed": seed},
                            revision=REVISION, evaluation_seed_start=start, method=METHOD)
        assert (data["revision"], data["method"], data["training_seed"], data["reference"],
                data["reference_mapping"], data["symbolic_constraint"], data["environment"]) == (
            REVISION, METHOD, seed, reference, MAPPINGS[reference], "on", "sustained_load")
        assert data["config"] == {**manifest["config"], "max_episodes": episodes}
        assert data["environment_settings"] == manifest["environment_settings"]
        assert data["preflight_only"] == preflight and data["config"]["jac_weight"] == .01
        assert data["runtime"]["device"] == "cuda" and data["checkpoint_selection"] == "last"
        for package, version in manifest["runtime_requirement"].items():
            if package == "python_minor_versions":
                assert '.'.join(data["runtime"]["python"].split()[0].split('.')[:2]) in version
            else:
                assert data["runtime"][package] == version
        assert (data["training_episodes"], data["environment_steps"], data["training_updates"],
                data["optimizer_updates_during_evaluation"], data["shared_prefix_episodes"],
                data["symbolic_identification_interactions"], data["symbolic_validation_interactions"]) == (
            episodes, episodes * 200, updates, 0, PREFIX_EPISODES, 10000, 1000)
        validate_temperature(data)
        assert data["jacobian_scope"] == "full" and data["jacobian_input_columns"] == list(range(6))
        jac = data["jacobian_sampler"]
        assert jac["batches"] == updates and jac["reset_samples"] == jac["replay_samples"] == updates * 128
        assert sum(jac["radius_samples"].values()) == jac["reset_samples"]
        for name, mode in (("critic", "early"), ("actor", "uniform")):
            sampler = data[f"{name}_sampler"]
            replaced = updates - 1 if name == "critic" else 0
            assert sampler["mode"] == mode and sampler["batches"] == updates
            assert sampler["counts"] == {"uniform": (updates - replaced) * 256 + replaced * 128,
                                         "reset": replaced * 64, "early": replaced * 64}
            assert all(sum(sampler["radius_counts"][k].values()) == sampler["counts"][k] for k in ("reset", "early"))
        assert (data["evaluation_seed_start"], data["evaluation_episodes_per_severity"],
                data["evaluation_episodes_including_controls"]) == (start, count, 24 * count)
        for radius in RADII:
            for load in SEVERITIES:
                block = data["radius_results"][str(radius)][str(load)]
                assert set(block["controllers"]) == {"actor", "fitted", "known_local", "known_inverse"}
                for control in block["controllers"].values():
                    assert control["evaluation_seeds"] == list(range(start, start + count))
                    arrays = {k: np.asarray(control[k]) for k in ("returns", "state_costs", "command_costs",
                              "early_state_costs", "early_command_costs", "late_state_costs", "late_command_costs")}
                    assert all(a.shape == (count,) and np.isfinite(a).all() for a in arrays.values())
                    np.testing.assert_allclose(-arrays["returns"], arrays["state_costs"] + arrays["command_costs"])
                    assert np.isclose(control["mean_cost"], -arrays["returns"].mean())
                    for kind in ("state", "command"):
                        np.testing.assert_allclose(arrays[f"{kind}_costs"], arrays[f"early_{kind}_costs"] + arrays[f"late_{kind}_costs"])
                own = data["reference_identification"]
                assert own["selected_reference"] == reference
                assert data["feedback_gains"]["fitted"] == own["gains"][reference]
        if reference == "cost-linear":
            np.testing.assert_array_equal(data["feedback_gains"]["fitted"],
                                          np.asarray(manifest["confirmed_gains"][str(seed)], dtype=np.float32))
    a, b = [runs[reference, seed] for reference in REFERENCES]
    assert a["runtime"] == b["runtime"]
    for key in ("fits", "gains", "validation"):
        assert a["reference_identification"][key] == b["reference_identification"][key]
    for key in ("jacobian_sampler", "critic_sampler", "actor_sampler"):
        assert a[key] == b[key]
    for key in ("training_returns", "temperature_trace"):
        assert a[key][:PREFIX_EPISODES] == b[key][:PREFIX_EPISODES]
    for radius in RADII:
        for load in SEVERITIES:
            for mode in ("known_local", "known_inverse"):
                np.testing.assert_array_equal(a["radius_results"][str(radius)][str(load)]["controllers"][mode]["returns"],
                                              b["radius_results"][str(radius)][str(load)]["controllers"][mode]["returns"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "experiments" / STUDY / "dispatch_manifest.json").read_text())
    tasks = manifest["preflight_tasks" if args.preflight else "tasks"]
    seeds = (21,) if args.preflight else TRAINING_SEEDS
    assert len(tasks) == len(seeds) and {t["seed"] for t in tasks} == set(seeds)
    destination = ROOT / "results" / REVISION
    destination = destination / "preflight" if args.preflight else destination
    observed = json.loads((destination / "task_observations.json").read_text())
    assert {t["id"] for t in tasks} == {t["id"] for t in observed}
    assert all(t["status"] == "done" and t["exit_code"] == 0 for t in observed)
    runs, paths = {}, []
    for seed in seeds:
        for reference in REFERENCES:
            path = destination / f"seed{seed}" / f"{REVISION}_{reference}_seed{seed}_evaluation.json"
            runs[reference, seed] = json.loads(path.read_text())
            paths.append(path)
        validate_pair(runs, manifest, seed, args.preflight)
    episodes = 6 if args.preflight else 500
    output = {"revision": REVISION, "preflight_only": args.preflight,
              "completed_task_ids": [t["id"] for t in tasks], "paired_full_prefix_verified": True,
              "evaluation_json_bytes": sum(p.stat().st_size for p in paths),
              "actual_training_interactions": len(seeds) * (2 * episodes - 5) * 200,
              "actual_training_updates": len(seeds) * (2 * (episodes * 200 - 999) - 1),
              "shared_identification_and_validation_interactions": len(seeds) * 11000,
              "historical_hold_unchanged": True}
    if not args.preflight:
        output.update(summarize(runs, seeds))
    (destination / "summary.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(output.get("decision", {"preflight_validated": True}), indent=2))


if __name__ == "__main__":
    main()
