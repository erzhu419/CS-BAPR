"""Aggregate the frozen random-load study and paired non-learning controls."""
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


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def paired_controls(runner, pilot, severities, seeds):
    import numpy as np

    results = {}
    for severity in severities:
        policies = results[str(severity)] = {}
        for policy in ("zero", "force_cancel_only", "feedback_no_force", "feedback_with_force"):
            returns, saturated, state_maxima, storm_fractions, force_rms = [], [], [], [], []
            for seed in seeds:
                env = runner.make_env("random_load", severity, seed)
                obs, _ = env.reset()
                total, saturation, maximum, storms, force_square = 0.0, 0, 0.0, 0, 0.0
                for step in range(pilot.EP_LEN):
                    action = pilot.control(policy, obs)
                    saturation += int(np.sum(np.abs(action) > 0.95))
                    maximum = max(maximum, float(np.linalg.norm(obs[:4])))
                    obs, reward, terminated, truncated, info = env.step(action)
                    total += reward
                    storms += info["storm"]
                    force_square += float(np.mean(info["applied_disturbance"] ** 2))
                    assert not truncated and terminated == (step == pilot.EP_LEN - 1)
                returns.append(total)
                saturated.append(saturation / (2 * pilot.EP_LEN))
                state_maxima.append(max(maximum, float(np.linalg.norm(obs[:4]))))
                storm_fractions.append(storms / pilot.EP_LEN)
                force_rms.append(float(np.sqrt(force_square / pilot.EP_LEN)))
            policies[policy] = {
                "evaluation_seeds": seeds, "returns": returns, "mean_cost": -mean(returns),
                "mean_saturated_component_fraction": mean(saturated),
                "maximum_state_norm": max(state_maxima),
                "mean_storm_fraction": mean(storm_fractions), "mean_force_rms": mean(force_rms),
            }
        print("paired controls", severity, {p: round(v["mean_cost"], 4) for p, v in policies.items()}, flush=True)
    return results


def main():
    study = ROOT / "experiments/random_load_v6_20261001"
    destination = ROOT / "results/random_load_v6"
    manifest = json.loads((study / "dispatch_manifest.json").read_text())
    runner = load_module("frozen_random_load_runner", study / "runtime/scripts/test_lqr_convergence.py")
    pilot = load_module("frozen_random_load_pilot", study / "runtime/scripts/preflight_random_load_v6.py")
    methods = [a["method"] for a in manifest["arms"]]
    seeds, severities = manifest["training_seeds"], manifest["severities"]
    evaluation_seeds = list(range(manifest["evaluation_seed_start"],
                                  manifest["evaluation_seed_start"] +
                                  manifest["evaluation_episodes_per_severity"]))
    assert seeds == [0, 1, 2] and severities == [1, 2, 5, 10, 20, 50]
    assert evaluation_seeds == list(range(10000, 10020))
    observations = json.loads((destination / "task_observations.json").read_text())
    assert {r["id"] for r in observations} == {t["id"] for t in manifest["tasks"]}
    assert len(observations) == 15 and all(r["status"] == "done" and r["exit_code"] == 0 for r in observations)
    expected_configs = {}
    for method in methods:
        config = runner.make_config(method, "cuda", "random_load")
        config.max_episodes = manifest["training_episodes"]
        expected_configs[method] = asdict(config)
    runs, rows = {m: {} for m in methods}, []
    for task in manifest["tasks"]:
        method, seed = task["method"], task["seed"]
        path = destination / f"{method}_seed{seed}" / f"random_load_v6_{method}_seed{seed}_evaluation.json"
        data = json.loads(path.read_text())
        assert (data["revision"], data["method"], data["training_seed"],
                data["training_episodes"], data["checkpoint_selection"], data["environment"]) == (
                    "random_load_v6", method, seed, 500, "last", "random_load"), path
        assert (data["environment_steps"], data["training_updates"]) == (100000, 99001), path
        assert len(data["training_returns"]) == 500 and all(math.isfinite(x) for x in data["training_returns"]), path
        assert data["config"] == expected_configs[method], path
        assert data["environment_settings"] == manifest["environment_settings"] == pilot.RandomLoadEnv.specification(), path
        assert data["config"]["weight_reg"] == data["config"]["action_anchor_weight"] == 0, path
        assert seed not in runs[method], path
        runs[method][seed] = data
        alignment = data["policy_alignment"]
        expected_reference = "fitted_feedback" if method in runner.SYMBOLIC_LQR_METHODS else "known_model_diagnostic"
        assert alignment["reference"] == expected_reference, path
        assert set(data["amplitude_results"]) == set(alignment["initial_states"]) == {str(a) for a in severities}, path
        for severity in severities:
            result = data["amplitude_results"][str(severity)]
            assert result["evaluation_seeds"] == evaluation_seeds and len(result["returns"]) == 20, path
            assert all(math.isfinite(x) for x in result["returns"]), path
            value = mean(result["returns"])
            assert math.isclose(value, result["mean"], rel_tol=1e-12), path
            assert alignment["initial_states"][str(severity)]["n_states"] == 20, path
            rows.append((task["id"], method, seed, severity, value))
    assert len(manifest["tasks"]) == 15 and len(methods) == 5
    assert all(set(rs) == set(seeds) for rs in runs.values())

    summary = {
        "revision": "random_load_v6", "completed_runs": 15, "checkpoint_selection": "last",
        "training_seeds": seeds, "evaluation_seeds": evaluation_seeds, "severities": severities,
        "environment_settings": manifest["environment_settings"], "data_accounting": manifest["data_accounting"],
        "methods": {}, "csbapr_minus_comparator": {}, "no_nau_minus_baseline": {},
    }
    for method, method_runs in runs.items():
        ordered = [method_runs[s] for s in seeds]
        values = summary["methods"][method] = {
            "returns": {}, "cost_increase_from_id": {}, "initial_state_alignment": {},
            "origin_actions": [r["policy_alignment"]["origin_action"] for r in ordered],
            "runtime": [r["runtime"] for r in ordered], "sindy": [r["sindy"] for r in ordered],
            "policy_curvature_bounds": [r["policy_curvature_bound"] for r in ordered],
            "final_25_training_return": seed_statistics([mean(r["training_returns"][-25:]) for r in ordered]),
        }
        for severity in severities:
            key = str(severity)
            values["returns"][key] = seed_statistics([r["amplitude_results"][key]["mean"] for r in ordered])
            values["cost_increase_from_id"][key] = seed_statistics([
                r["amplitude_results"]["1"]["mean"] - r["amplitude_results"][key]["mean"] for r in ordered
            ])
            metrics = [r["policy_alignment"]["initial_states"][key] for r in ordered]
            values["initial_state_alignment"][key] = {
                name: seed_statistics([m[name] for m in metrics]) for name in metrics[0] if name != "n_states"
            }
    full = summary["methods"]["csbapr-no-target-shift"]
    for method in methods[1:]:
        summary["csbapr_minus_comparator"][method] = {
            str(a): contrast(full["returns"][str(a)]["seed_values"],
                             summary["methods"][method]["returns"][str(a)]["seed_values"])
            for a in severities
        }
    no_nau = summary["methods"]["csbapr-no-nau-no-target-shift"]
    for method in ("bapr-no-target-shift", "resac-no-target-shift"):
        summary["no_nau_minus_baseline"][method] = {
            str(a): contrast(no_nau["returns"][str(a)]["seed_values"],
                             summary["methods"][method]["returns"][str(a)]["seed_values"])
            for a in severities
        }
    summary["post_hoc_control_source"] = "frozen runtime, same formal evaluation seeds; no trained checkpoints"
    summary["post_hoc_control_diagnostic"] = paired_controls(runner, pilot, severities, evaluation_seeds)
    summary["reference_is_optimal"] = False
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    with (destination / "seed_mean_returns.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("task_id", "method", "training_seed", "load_severity", "mean_return"))
        writer.writerows(rows)
    print("15 complete runs; 90 seed/severity means; frozen configs and paired evaluations verified")
    for method, values in summary["csbapr_minus_comparator"].items():
        print("CS minus", method, "50x", values["50"])


if __name__ == "__main__":
    main()
