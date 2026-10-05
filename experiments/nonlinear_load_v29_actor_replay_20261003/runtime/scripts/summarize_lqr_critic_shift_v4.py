"""Aggregate the frozen target-shift factorial over three training seeds."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]


def seed_statistics(values):
    return {"seed_values": values, "mean": mean(values), "sample_sd": stdev(values)}


def contrast(left, right):
    differences = [a - b for a, b in zip(left, right)]
    return {
        **seed_statistics(differences),
        "matched_training_seed_wins": sum(x > 0 for x in differences),
        "relative_mean_cost_reduction": 1 - mean(left) / mean(right),
    }


def main():
    study = ROOT / "experiments/lqr_critic_shift_v4_20261001"
    destination = ROOT / "results/lqr_critic_shift_v4"
    manifest = json.loads((study / "dispatch_manifest.json").read_text())
    methods = [arm["method"] for arm in manifest["arms"]]
    amplitudes, seeds = manifest["amplitudes"], manifest["training_seeds"]
    evaluation_seeds = list(range(manifest["evaluation_seed_start"],
                                  manifest["evaluation_seed_start"] +
                                  manifest["evaluation_episodes_per_amplitude"]))
    assert seeds == [0, 1, 2] and amplitudes == [1, 2, 5, 10, 20, 50]
    assert evaluation_seeds == list(range(10000, 10020))
    runs, rows = {method: {} for method in methods}, []
    for task in manifest["tasks"]:
        method, seed = task["method"], task["seed"]
        path = destination / f"{method}_seed{seed}" / f"critic_shift_v4_{method}_seed{seed}_evaluation.json"
        data = json.loads(path.read_text())
        assert (data["revision"], data["method"], data["training_seed"],
                data["training_episodes"], data["checkpoint_selection"]) == (
                    "critic_shift_v4", method, seed, 500, "last"), path
        assert (data["environment_steps"], data["training_updates"]) == (100000, 99001), path
        assert len(data["training_returns"]) == 500, path
        assert all(math.isfinite(x) for x in data["training_returns"]), path
        assert seed not in runs[method], path
        runs[method][seed] = data
        alignment = data["policy_alignment"]
        assert alignment["reference"] == "fitted_feedback", path
        assert set(data["amplitude_results"]) == set(alignment["initial_states"]) == {str(a) for a in amplitudes}, path
        for amplitude in amplitudes:
            result = data["amplitude_results"][str(amplitude)]
            assert result["evaluation_seeds"] == evaluation_seeds and len(result["returns"]) == 20, path
            assert all(math.isfinite(x) for x in result["returns"]), path
            value = mean(result["returns"])
            assert math.isclose(value, result["mean"], rel_tol=1e-12), path
            rows.append((task["id"], method, seed, amplitude, value))
    assert len(manifest["tasks"]) == 12 and all(set(rs) == set(seeds) for rs in runs.values())
    for seed in seeds:
        original = runs["csbapr"][seed]["config"]
        for arm in manifest["arms"]:
            expected = {**original, "action_anchor_weight": arm["action_anchor_weight"], "weight_reg": arm["weight_reg"]}
            assert runs[arm["method"]][seed]["config"] == expected

    summary = {
        "revision": "critic_shift_v4", "completed_runs": 12,
        "training_seeds": seeds, "evaluation_seeds": evaluation_seeds,
        "amplitudes": amplitudes, "checkpoint_selection": "last", "methods": {},
        "contrasts": {}, "removal_by_anchor_interaction": {},
    }
    for method, method_runs in runs.items():
        ordered = [method_runs[s] for s in seeds]
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
    pairs = {
        "remove_shift_without_anchor": ("csbapr-no-target-shift", "csbapr"),
        "remove_shift_with_anchor": ("csbapr-action-anchor-no-target-shift", "csbapr-action-anchor"),
        "add_anchor_without_shift": ("csbapr-action-anchor-no-target-shift", "csbapr-no-target-shift"),
        "add_anchor_with_shift": ("csbapr-action-anchor", "csbapr"),
    }
    for name, (left, right) in pairs.items():
        summary["contrasts"][name] = {
            str(a): contrast(summary["methods"][left]["returns"][str(a)]["seed_values"],
                             summary["methods"][right]["returns"][str(a)]["seed_values"])
            for a in amplitudes
        }
    for amplitude in amplitudes:
        key = str(amplitude)
        anchored = summary["contrasts"]["remove_shift_with_anchor"][key]["seed_values"]
        unanchored = summary["contrasts"]["remove_shift_without_anchor"][key]["seed_values"]
        summary["removal_by_anchor_interaction"][key] = seed_statistics([a - b for a, b in zip(anchored, unanchored)])

    old_environment = ROOT / "experiments/lqr_feedback_v2_20260930/runtime_r2/csbapr/envs/lqr_env.py"
    assert old_environment.read_bytes() == (study / "runtime/csbapr/envs/lqr_env.py").read_bytes()
    prior = json.loads((ROOT / "results/lqr_feedback_v2/summary.json").read_text())
    assert prior["amplitudes"] == amplitudes and prior["evaluation_seeds"] == evaluation_seeds
    summary["post_hoc_control_source"] = "results/lqr_feedback_v2/summary.json"
    summary["post_hoc_control_diagnostic"] = prior["post_hoc_control_diagnostic"]
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    with (destination / "seed_mean_returns.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("task_id", "method", "training_seed", "amplitude", "mean_return"))
        writer.writerows(rows)
    print("12 complete runs; 72 seed/amplitude means; factorial configs and paired evaluation verified")
    for method, values in summary["methods"].items():
        print(method, [round(values["returns"][str(a)]["mean"], 2) for a in amplitudes])
    print("interaction", [round(summary["removal_by_anchor_interaction"][str(a)]["mean"], 2) for a in amplitudes])


if __name__ == "__main__":
    main()
