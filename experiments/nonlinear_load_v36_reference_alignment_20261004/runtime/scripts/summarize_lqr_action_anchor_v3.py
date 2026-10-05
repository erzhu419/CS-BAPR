"""Aggregate the frozen action-anchor study over independent training seeds."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("csbapr-action-anchor", "csbapr", "bapr", "resac")


def seed_statistics(values):
    return {"seed_values": values, "mean": mean(values), "sample_sd": stdev(values)}


def main():
    study = ROOT / "experiments/lqr_action_anchor_v3_20260930"
    destination = ROOT / "results/lqr_action_anchor_v3"
    manifest = json.loads((study / "dispatch_manifest.json").read_text())
    amplitudes = manifest["amplitudes"]
    training_seeds = manifest["training_seeds"]
    evaluation_seeds = list(range(manifest["evaluation_seed_start"],
                                  manifest["evaluation_seed_start"] +
                                  manifest["evaluation_episodes_per_amplitude"]))
    assert amplitudes == [1, 2, 5, 10, 20, 50]
    assert training_seeds == [0, 1, 2] and evaluation_seeds == list(range(10000, 10020))
    assert manifest["training_episodes"] == 500 and manifest["checkpoint_selection"] == "last"
    assert (manifest["action_anchor_weight"], manifest["control_action_anchor_weight"]) == (1.0, 0.0)
    runs = {method: {} for method in METHODS}
    rows = []
    for task in manifest["tasks"]:
        method, seed = task["method"], task["seed"]
        path = destination / f"{method}_seed{seed}" / f"action_anchor_v3_{method}_seed{seed}_evaluation.json"
        data = json.loads(path.read_text())
        assert (data["revision"], data["method"], data["training_seed"],
                data["training_episodes"], data["checkpoint_selection"]) == (
                    "action_anchor_v3", method, seed, 500, "last"), path
        assert (data["environment_steps"], data["training_updates"]) == (100000, 99001), path
        assert len(data["training_returns"]) == 500, path
        assert all(math.isfinite(x) for x in data["training_returns"]), path
        assert seed not in runs[method], path
        runs[method][seed] = data
        alignment = data["policy_alignment"]
        expected_reference = "fitted_feedback" if method.startswith("csbapr") else "known_model_diagnostic"
        assert alignment["reference"] == expected_reference, path
        assert len(alignment["origin_action"]) == 2, path
        assert all(math.isfinite(x) for x in alignment["origin_action"]), path
        assert set(data["amplitude_results"]) == set(alignment["initial_states"]) == {str(a) for a in amplitudes}, path
        for amplitude in amplitudes:
            result = data["amplitude_results"][str(amplitude)]
            assert result["evaluation_seeds"] == evaluation_seeds, path
            assert len(result["returns"]) == 20 and all(math.isfinite(x) for x in result["returns"]), path
            value = mean(result["returns"])
            assert math.isclose(value, result["mean"], rel_tol=1e-12), path
            metrics = alignment["initial_states"][str(amplitude)]
            assert metrics["n_states"] == 20 and all(math.isfinite(x) for x in metrics.values()), path
            rows.append((task["id"], method, seed, amplitude, value))
    assert len(manifest["tasks"]) == 12 and all(set(rs) == set(training_seeds) for rs in runs.values())
    for seed in training_seeds:
        anchored = runs[METHODS[0]][seed]["config"]
        original = runs["csbapr"][seed]["config"]
        assert {key: (anchored.get(key), original.get(key))
                for key in anchored.keys() | original.keys()
                if anchored.get(key) != original.get(key)} == {"action_anchor_weight": (1.0, 0.0)}

    # Reuse the paired controls only while their frozen environment is identical.
    prior_environment = ROOT / "experiments/lqr_feedback_v2_20260930/runtime_r2/csbapr/envs/lqr_env.py"
    assert prior_environment.read_bytes() == (study / "runtime/csbapr/envs/lqr_env.py").read_bytes()
    prior_path = ROOT / "results/lqr_feedback_v2/summary.json"
    prior = json.loads(prior_path.read_text())
    assert prior["amplitudes"] == amplitudes and prior["evaluation_seeds"] == evaluation_seeds
    controls = prior["post_hoc_control_diagnostic"]
    for results in controls.values():
        for amplitude in amplitudes:
            result = results[str(amplitude)]
            assert result["evaluation_seeds"] == evaluation_seeds and len(result["returns"]) == 20
            assert math.isclose(mean(result["returns"]), result["mean_return"], rel_tol=1e-12)

    summary = {
        "revision": "action_anchor_v3", "completed_runs": 12,
        "training_seeds": training_seeds, "evaluation_seeds": evaluation_seeds,
        "amplitudes": amplitudes, "checkpoint_selection": "last",
        "primary_method": METHODS[0], "methods": {}, "comparisons": {},
        "post_hoc_control_source": str(prior_path.relative_to(ROOT)),
        "post_hoc_control_diagnostic": controls,
    }
    for method, method_runs in runs.items():
        ordered = [method_runs[s] for s in training_seeds]
        values = summary["methods"][method] = {
            "returns": {}, "initial_state_alignment": {},
            "origin_actions": [r["policy_alignment"]["origin_action"] for r in ordered],
            "runtime": [r["runtime"] for r in ordered],
            "final_25_training_return": seed_statistics([mean(r["training_returns"][-25:]) for r in ordered]),
        }
        for amplitude in amplitudes:
            key = str(amplitude)
            values["returns"][key] = seed_statistics([r["amplitude_results"][key]["mean"] for r in ordered])
            metrics = [r["policy_alignment"]["initial_states"][key] for r in ordered]
            values["initial_state_alignment"][key] = {
                name: seed_statistics([m[name] for m in metrics])
                for name in metrics[0] if name != "n_states"
            }
    for comparator in (*METHODS[1:], *controls):
        summary["comparisons"][comparator] = {}
        for amplitude in amplitudes:
            key = str(amplitude)
            anchor = summary["methods"][METHODS[0]]["returns"][key]["seed_values"]
            other = (summary["methods"][comparator]["returns"][key]["seed_values"]
                     if comparator in runs else [controls[comparator][key]["mean_return"]] * 3)
            differences = [a - b for a, b in zip(anchor, other)]
            summary["comparisons"][comparator][key] = {
                "seed_return_differences": differences,
                "mean_return_difference": mean(differences),
                "training_seed_wins": sum(x > 0 for x in differences),
                "relative_mean_cost_reduction": 1 - mean(anchor) / mean(other),
            }
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    with (destination / "seed_mean_returns.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("task_id", "method", "training_seed", "amplitude", "mean_return"))
        writer.writerows(rows)
    print("12 complete runs; 72 seed/amplitude means; paired evaluation and isolated configuration verified")
    for method in METHODS:
        print(method, [round(summary["methods"][method]["returns"][str(a)]["mean"], 2) for a in amplitudes])
    for comparator, results in summary["comparisons"].items():
        print("anchor vs", comparator, "wins", [results[str(a)]["training_seed_wins"] for a in amplitudes],
              "cost reduction %", [round(100 * results[str(a)]["relative_mean_cost_reduction"], 2) for a in amplitudes])


if __name__ == "__main__":
    main()
