"""Summarize final-checkpoint fair-target baselines and component ablations."""
from __future__ import annotations

import csv
from dataclasses import asdict
import importlib.util
import json
import math
from pathlib import Path
from statistics import mean

from summarize_lqr_critic_shift_v4 import contrast, seed_statistics

ROOT = Path(__file__).resolve().parents[1]


def main():
    study = ROOT / "experiments/lqr_fair_backbone_v5_20261001"
    destination = ROOT / "results/lqr_fair_backbone_v5"
    manifest = json.loads((study / "dispatch_manifest.json").read_text())
    spec = importlib.util.spec_from_file_location(
        "frozen_lqr_runner", study / "runtime/scripts/test_lqr_convergence.py",
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    methods = [a["method"] for a in manifest["arms"]]
    seeds, amplitudes = manifest["training_seeds"], manifest["amplitudes"]
    evaluation_seeds = list(range(manifest["evaluation_seed_start"],
                                  manifest["evaluation_seed_start"] +
                                  manifest["evaluation_episodes_per_amplitude"]))
    assert seeds == [0, 1, 2] and amplitudes == [1, 2, 5, 10, 20, 50]
    assert evaluation_seeds == list(range(10000, 10020))
    expected_configs = {}
    for method in methods:
        config = runner.make_config(method, "cuda")
        config.max_episodes = manifest["training_episodes"]
        expected_configs[method] = asdict(config)
    runs, rows = {m: {} for m in methods}, []
    for task in manifest["tasks"]:
        method, seed = task["method"], task["seed"]
        path = destination / f"{method}_seed{seed}" / f"fair_backbone_v5_{method}_seed{seed}_evaluation.json"
        data = json.loads(path.read_text())
        assert (data["revision"], data["method"], data["training_seed"],
                data["training_episodes"], data["checkpoint_selection"]) == (
                    "fair_backbone_v5", method, seed, 500, "last"), path
        assert (data["environment_steps"], data["training_updates"]) == (100000, 99001), path
        assert len(data["training_returns"]) == 500 and all(math.isfinite(x) for x in data["training_returns"]), path
        assert data["config"] == expected_configs[method], path
        assert data["config"]["weight_reg"] == data["config"]["action_anchor_weight"] == 0, path
        assert seed not in runs[method], path
        runs[method][seed] = data
        alignment = data["policy_alignment"]
        expected_reference = "fitted_feedback" if method in runner.SYMBOLIC_LQR_METHODS else "known_model_diagnostic"
        assert alignment["reference"] == expected_reference, path
        assert set(data["amplitude_results"]) == set(alignment["initial_states"]) == {str(a) for a in amplitudes}, path
        for amplitude in amplitudes:
            result = data["amplitude_results"][str(amplitude)]
            assert result["evaluation_seeds"] == evaluation_seeds and len(result["returns"]) == 20, path
            assert all(math.isfinite(x) for x in result["returns"]), path
            value = mean(result["returns"])
            assert math.isclose(value, result["mean"], rel_tol=1e-12), path
            rows.append((task["id"], method, seed, amplitude, value))
    assert len(manifest["tasks"]) == 15 and len(methods) == 5
    assert all(set(rs) == set(seeds) for rs in runs.values())

    summary = {
        "revision": "fair_backbone_v5", "completed_runs": 15,
        "training_seeds": seeds, "evaluation_seeds": evaluation_seeds,
        "amplitudes": amplitudes, "checkpoint_selection": "last",
        "methods": {}, "csbapr_minus_comparator": {},
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
    full = summary["methods"]["csbapr-no-target-shift"]
    for method in methods[1:]:
        summary["csbapr_minus_comparator"][method] = {
            str(a): contrast(full["returns"][str(a)]["seed_values"],
                             summary["methods"][method]["returns"][str(a)]["seed_values"])
            for a in amplitudes
        }
    previous = ROOT / "experiments/lqr_critic_shift_v4_20261001/runtime/csbapr/envs/lqr_env.py"
    assert previous.read_bytes() == (study / "runtime/csbapr/envs/lqr_env.py").read_bytes()
    prior = json.loads((ROOT / "results/lqr_critic_shift_v4/summary.json").read_text())
    assert prior["amplitudes"] == amplitudes and prior["evaluation_seeds"] == evaluation_seeds
    summary["post_hoc_control_source"] = prior["post_hoc_control_source"]
    summary["post_hoc_control_diagnostic"] = prior["post_hoc_control_diagnostic"]
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    with (destination / "seed_mean_returns.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("task_id", "method", "training_seed", "amplitude", "mean_return"))
        writer.writerows(rows)
    print("15 complete runs; 90 seed/amplitude means; frozen configs and paired evaluation verified")
    for method, values in summary["methods"].items():
        print(method, [round(values["returns"][str(a)]["mean"], 2) for a in amplitudes])
    for method, values in summary["csbapr_minus_comparator"].items():
        print("CS minus", method, "50x", values["50"])


if __name__ == "__main__":
    main()
