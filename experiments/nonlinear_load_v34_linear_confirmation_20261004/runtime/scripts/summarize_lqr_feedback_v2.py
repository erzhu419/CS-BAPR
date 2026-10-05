"""Summarize the frozen feedback_v2 study and a post-hoc control diagnostic."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from csbapr.envs.lqr_env import EP_LEN, LQREnv, get_optimal_gain


def control_diagnostic(amplitudes, seeds):
    gain = get_optimal_gain()
    results = {}
    for policy in ("zero_action", "true_model_tanh_feedback"):
        results[policy] = {}
        for amplitude in amplitudes:
            returns = []
            for seed in seeds:
                env = LQREnv(amplitude=amplitude, seed=seed)
                obs, _ = env.reset()
                total = 0.0
                for _ in range(EP_LEN):
                    action = (np.zeros(2) if policy == "zero_action"
                              else np.tanh(-gain @ obs))
                    obs, reward, terminated, truncated, _ = env.step(action)
                    total += reward
                    if terminated or truncated:
                        break
                returns.append(total)
            results[policy][str(amplitude)] = {
                "mean_return": mean(returns), "returns": returns,
                "evaluation_seeds": seeds,
            }
    return results


def main():
    study = ROOT / "experiments/lqr_feedback_v2_20260930"
    destination = ROOT / "results/lqr_feedback_v2"
    manifest = json.loads((study / "dispatch_manifest.json").read_text())
    amplitudes = manifest["amplitudes"]
    seeds = manifest["evaluation_seeds"]
    runs, csv_rows = {}, []
    for task in manifest["tasks"]:
        method, seed = task["method"], task["seed"]
        name = f"feedback_v2_{method}_seed{seed}_evaluation.json"
        data = json.loads((destination / f"{method}_seed{seed}" / name).read_text())
        assert (data["revision"], data["method"], data["training_seed"],
                data["training_episodes"], data["checkpoint_selection"]) == (
                    "feedback_v2", method, seed, 500, "last")
        assert (data["environment_steps"], data["training_updates"]) == (100000, 99001)
        assert len(data["training_returns"]) == 500
        assert all(math.isfinite(x) for x in data["training_returns"])
        assert set(data["amplitude_results"]) == {str(a) for a in amplitudes}
        runs.setdefault(method, {})[seed] = data
        for amplitude in amplitudes:
            result = data["amplitude_results"][str(amplitude)]
            assert result["evaluation_seeds"] == seeds
            assert len(result["returns"]) == len(seeds) == 20
            assert all(math.isfinite(x) for x in result["returns"])
            value = mean(result["returns"])
            assert math.isclose(value, result["mean"], rel_tol=1e-12)
            csv_rows.append((task["id"], method, seed, amplitude, value))

    assert len(runs) == 5 and all(set(rs) == {0, 1, 2} for rs in runs.values())
    summary = {"revision": "feedback_v2", "completed_runs": 15,
               "amplitudes": amplitudes, "evaluation_seeds": seeds,
               "checkpoint_selection": "last", "methods": {}, "comparisons": {}}
    for method, method_runs in runs.items():
        summary["methods"][method] = {}
        for amplitude in amplitudes:
            values = [method_runs[s]["amplitude_results"][str(amplitude)]["mean"]
                      for s in range(3)]
            summary["methods"][method][str(amplitude)] = {
                "seed_mean_returns": values, "mean_return": mean(values),
                "training_seed_sample_sd": stdev(values),
            }
    for comparator in runs:
        if comparator == "csbapr":
            continue
        summary["comparisons"][comparator] = {}
        for amplitude in amplitudes:
            full = summary["methods"]["csbapr"][str(amplitude)]
            other = summary["methods"][comparator][str(amplitude)]
            differences = [a - b for a, b in
                           zip(full["seed_mean_returns"], other["seed_mean_returns"])]
            summary["comparisons"][comparator][str(amplitude)] = {
                "mean_return_difference": mean(differences),
                "seed_return_differences": differences,
                "matched_seed_wins": sum(x > 0 for x in differences),
                "relative_mean_cost_reduction": 1 - full["mean_return"] / other["mean_return"],
            }
    summary["post_hoc_control_diagnostic"] = control_diagnostic(amplitudes, seeds)
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    with (destination / "seed_mean_returns.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("task_id", "method", "training_seed", "amplitude", "mean_return"))
        writer.writerows(csv_rows)
    print("15 complete runs; 90 seed/amplitude means; paired evaluation seeds verified")
    for method, values in summary["methods"].items():
        print(method, [round(values[str(a)]["mean_return"], 2) for a in amplitudes])
    for policy, values in summary["post_hoc_control_diagnostic"].items():
        print(policy, [round(values[str(a)]["mean_return"], 2) for a in amplitudes])


if __name__ == "__main__":
    main()
