"""Summarize v37 without pooling evaluation episodes as training replicates."""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
REVISION = "general_ood_v37_cross_env"
STUDY = ROOT / "experiments/general_ood_v37_cross_env_20261004"
DESTINATION = ROOT / "results" / REVISION
CANDIDATE = "csbapr-stable-context"
PROFILES = {
    "Pendulum-v1": {"low_load", "rush_load", "mega_event", "recovery_aftershock"},
    "Hopper-v4": {"easy_surface", "heavy_traffic", "mega_event", "slippery_rush"},
}


def seed_statistics(values):
    if not values:
        return None
    average = mean(values)
    sd = stdev(values) if len(values) > 1 else None
    critical = {2: 12.7062047364321, 3: 4.302652729911275}.get(len(values))
    half = critical * sd / math.sqrt(len(values)) if critical else None
    return {
        "seed_values": values, "n_training_seeds": len(values),
        "mean": average, "sample_sd": sd,
        "ci95": [average - half, average + half] if half is not None else None,
    }


def validate_run(data, manifest, environment, method, seed):
    expected = {
        "revision": REVISION, "environment": environment,
        "method": method, "training_seed": seed,
        "checkpoint": f"{method}_{environment}_{seed}_final.pt",
        "evaluation_only": True, "optimizer_updates_during_evaluation": 0,
        **{k: manifest[k] for k in (
            "evaluation_seed_start", "evaluation_episodes", "ood_intensity",
            "ood_profile_family", "ood_schedule", "ood_switch_probability",
        )},
        "ood_switch_steps": manifest["ood_switch_steps"][environment],
    }
    for key, value in expected.items():
        assert data[key] == value, (environment, method, seed, key)
    assert set(data["results"]) == {"id", "markov_50x"}
    assert set(data["fixed_profile_results"]) == PROFILES[environment]
    for blocks, count in ((data["results"], 20), (data["fixed_profile_results"], 5)):
        for name, result in blocks.items():
            values = result["rewards"]
            assert len(values) == len(result["lengths"]) == count, name
            assert all(math.isfinite(x) for x in values), name
            assert math.isfinite(result["std"]), name
            assert math.isclose(mean(values), result["mean"], rel_tol=1e-10, abs_tol=1e-10), name
    return data


def summarize(manifest, destination):
    runs, missing, artifacts = {}, [], []
    for env in manifest["environments"]:
        runs[env] = {}
        for method in manifest["methods"]:
            cells = runs[env][method] = {}
            for seed in manifest["training_seeds"]:
                path = destination / "formal" / env / method / f"seed{seed}" / f"{REVISION}_{env}_{method}_seed{seed}.json"
                if not path.exists():
                    missing.append({"environment": env, "method": method, "seed": seed})
                    continue
                cells[seed] = validate_run(json.loads(path.read_text()), manifest, env, method, seed)
                artifacts.append({"path": str(path.relative_to(destination)), "bytes": path.stat().st_size})

    methods, contrasts, gates = {}, {}, {}
    for env, environment_runs in runs.items():
        methods[env], contrasts[env] = {}, {}
        for method, cells in environment_runs.items():
            seeds = sorted(cells)
            methods[env][method] = {
                "training_seeds": seeds,
                "returns": {mode: seed_statistics([cells[s]["results"][mode]["mean"] for s in seeds])
                            for mode in ("id", "markov_50x")},
                "fixed_profiles": {profile: seed_statistics([cells[s]["fixed_profile_results"][profile]["mean"] for s in seeds])
                                   for profile in sorted(PROFILES[env])},
                "worst_fixed_profile": [
                    {"training_seed": s, "profile": profile, "mean_return": value}
                    for s in seeds
                    for value, profile in [min((r["mean"], p) for p, r in cells[s]["fixed_profile_results"].items())]
                ],
            }
        for comparator in manifest["methods"]:
            if comparator == CANDIDATE:
                continue
            shared = sorted(set(environment_runs[CANDIDATE]) & set(environment_runs[comparator]))
            modes = {}
            for mode in ("id", "markov_50x"):
                differences = [environment_runs[CANDIDATE][s]["results"][mode]["mean"]
                               - environment_runs[comparator][s]["results"][mode]["mean"] for s in shared]
                modes[mode] = {"statistics": seed_statistics(differences),
                               "matched_seed_wins": sum(x > 0 for x in differences)}
            contrasts[env][comparator] = {"training_seeds": shared, **modes}
        complete = all(len(cells) == 3 for cells in environment_runs.values())
        gates[env] = all(
            contrasts[env][method][mode]["matched_seed_wins"] == 3
            for method in ("bapr-stable-context", "resac")
            for mode in ("id", "markov_50x")
        ) if complete else None

    return {
        "revision": REVISION, "completed_runs": len(artifacts), "expected_runs": 24,
        "matrix_complete": not missing, "missing_cells": missing,
        "statistics_unit": "independent training-seed means; 20 evaluation episodes per primary mean",
        "ci95_method": "descriptive two-sided Student t intervals across training seeds; no multiplicity correction",
        "methods": methods, "candidate_minus_comparator": contrasts,
        "all_seed_baseline_superiority_by_environment": gates,
        "decision": "pending-missing-cells" if missing else "complete-matrix-requires-boundary-judgment",
        "artifacts": artifacts, "evaluation_json_bytes": sum(a["bytes"] for a in artifacts),
        "limitations": [
            "All four presets disable SINDy and Jacobian losses; this is not a full symbolic CS-BAPR test.",
            "No-context preset also changes beta_bc (0.005 to 0.001) and penalty_scale (1 to 2); not a single-factor context ablation.",
            "Equal episode limits do not imply equal interactions in early-terminating Hopper.",
            "Only rewards and episode lengths are recorded; Hopper reward is not a cost metric.",
        ],
    }


def main():
    manifest = json.loads((STUDY / "dispatch_manifest.json").read_text())
    summary = summarize(manifest, DESTINATION)
    path = DESTINATION / "summary.json"
    path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"{summary['completed_runs']}/24 valid results; saved {path}")
    for env, methods in summary["methods"].items():
        for method, values in methods.items():
            print(env, method, values["training_seeds"],
                  {mode: round(stats["mean"], 2) if stats else None for mode, stats in values["returns"].items()})
    print("baseline superiority", summary["all_seed_baseline_superiority_by_environment"])


if __name__ == "__main__":
    main()
