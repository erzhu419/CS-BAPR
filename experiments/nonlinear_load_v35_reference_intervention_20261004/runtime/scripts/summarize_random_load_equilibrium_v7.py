"""Aggregate the frozen no-NAU centering/symbolic factorial and its gates."""
from __future__ import annotations

import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
from statistics import mean

from summarize_lqr_critic_shift_v4 import contrast, seed_statistics
from summarize_random_load_v6 import load_module

ROOT = Path(__file__).resolve().parents[1]
REVISION = "random_load_equilibrium_v7"
METHODS = (
    "csbapr-no-nau-no-target-shift",
    "csbapr-no-nau-no-sindy-no-target-shift",
    "csbapr-centered-mlp-no-target-shift",
    "csbapr-centered-mlp-no-sindy-no-target-shift",
)


def validate_run(path, revision, method, seed, config, environment, evaluation_seeds, severities):
    data = json.loads(path.read_text())
    assert (data["revision"], data["method"], data["training_seed"],
            data["training_episodes"], data["checkpoint_selection"], data["environment"]) == (
                revision, method, seed, 500, "last", "random_load"), path
    assert (data["environment_steps"], data["training_updates"]) == (100000, 99001), path
    assert len(data["training_returns"]) == 500 and all(map(math.isfinite, data["training_returns"])), path
    assert data["config"] == config and data["environment_settings"] == environment, path
    assert data["actor_type"] == config["actor_type"], path
    alignment = data["policy_alignment"]
    symbolic = config["weight_sym"] > 0
    assert (data["sindy"] is not None) == symbolic, path
    assert alignment["reference"] == ("fitted_feedback" if symbolic else "known_model_diagnostic"), path
    keys = {str(a) for a in severities}
    assert set(data["amplitude_results"]) == set(alignment["initial_states"]) == keys, path
    for key in keys:
        result = data["amplitude_results"][key]
        assert result["evaluation_seeds"] == evaluation_seeds, path
        assert len(result["returns"]) == 20 and all(map(math.isfinite, result["returns"])), path
        assert math.isclose(mean(result["returns"]), result["mean"], rel_tol=1e-12), path
        assert alignment["initial_states"][key]["n_states"] == 20, path
    if config["actor_type"] == "centered-tanh-mlp":
        assert alignment["origin_action"] == [0.0, 0.0], path
    return data


def summarize_runs(ordered, severities):
    values = {
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
    return values


def main():
    study = ROOT / "experiments/random_load_equilibrium_v7_20261001"
    destination = ROOT / "results" / REVISION
    manifest = json.loads((study / "dispatch_manifest.json").read_text())
    runner = load_module("frozen_equilibrium_v7_runner", study / "runtime/scripts/test_lqr_convergence.py")
    seeds, severities = manifest["training_seeds"], manifest["severities"]
    evaluation_seeds = list(range(manifest["evaluation_seed_start"],
                                  manifest["evaluation_seed_start"] +
                                  manifest["evaluation_episodes_per_severity"]))
    assert seeds == [0, 1, 2] and severities == [1, 2, 5, 10, 20, 50]
    assert evaluation_seeds == list(range(10000, 10020))
    assert [a["method"] for a in manifest["arms"]] == list(METHODS)
    tasks = manifest["tasks"]
    assert len(tasks) == 12 and len({t["id"] for t in tasks}) == 12
    assert {(t["method"], t["seed"]) for t in tasks} == {(m, s) for m in METHODS for s in seeds}
    observations = json.loads((destination / "task_observations.json").read_text())
    assert len(observations) == 12 and {r["id"] for r in observations} == {t["id"] for t in tasks}
    assert all(r["status"] == "done" and r["exit_code"] == 0 for r in observations)
    environment = manifest["environment_settings"]
    assert environment == runner.make_env("random_load", 1, 0).specification()
    configs = {}
    for arm in manifest["arms"]:
        config = runner.make_config(arm["method"], "cuda", "random_load")
        config.max_episodes = 500
        configs[arm["method"]] = asdict(config)
        assert configs[arm["method"]] == arm["config"]
        assert not config.use_nau_actor and config.nau_reg_weight == 0
        assert config.weight_reg == config.action_anchor_weight == 0
        assert (config.actor_type == "centered-tanh-mlp") == arm["centered"]
        assert config.weight_sym == config.jac_weight == (0.01 if arm["symbolic"] else 0)
    runs, rows = {m: {} for m in METHODS}, []
    for task in tasks:
        method, seed = task["method"], task["seed"]
        path = destination / f"{method}_seed{seed}" / f"{REVISION}_{method}_seed{seed}_evaluation.json"
        data = validate_run(path, REVISION, method, seed, configs[method], environment, evaluation_seeds, severities)
        runs[method][seed] = data
        rows.extend((task["id"], method, seed, a, data["amplitude_results"][str(a)]["mean"]) for a in severities)

    summary = {
        "revision": REVISION, "completed_runs": 12, "checkpoint_selection": "last",
        "training_seeds": seeds, "evaluation_seeds": evaluation_seeds, "severities": severities,
        "environment_settings": environment, "data_accounting": manifest["data_accounting"],
        "configs": configs, "methods": {m: summarize_runs([runs[m][s] for s in seeds], severities) for m in METHODS},
        "contrasts": {}, "gates": {}, "centering_by_symbolic_interaction": {},
        "contrast_runtime_differences": {}, "historical_baselines": {}, "historical_baseline_comparisons": {},
        "statistics_unit": "three training-seed means, each averaging 20 paired evaluation episodes",
        "repeated_v6_arm": {"method": METHODS[0], "source": "results/random_load_v6", "pooled_with_v6": False},
    }
    plain_sym, plain, centered_sym, centered = METHODS
    pairs = {
        "centering_with_symbolic": (centered_sym, plain_sym),
        "centering_without_symbolic": (centered, plain),
        "remove_symbolic_without_centering": (plain, plain_sym),
        "remove_symbolic_with_centering": (centered, centered_sym),
    }
    for name, (left, right) in pairs.items():
        summary["contrasts"][name] = {
            str(a): contrast(summary["methods"][left]["returns"][str(a)]["seed_values"],
                             summary["methods"][right]["returns"][str(a)]["seed_values"])
            for a in severities
        }
        summary["gates"][name] = {
            "criterion": "lower mean cost and 3/3 matched-training-seed wins at both 1x and 50x",
            "passed": all(summary["contrasts"][name][a]["mean"] > 0 and
                          summary["contrasts"][name][a]["matched_training_seed_wins"] == 3 for a in ("1", "50")),
        }
        summary["contrast_runtime_differences"][name] = [
            s for s in seeds if runs[left][s]["runtime"] != runs[right][s]["runtime"]
        ]
    for a in severities:
        key = str(a)
        with_sym = summary["contrasts"]["centering_with_symbolic"][key]["seed_values"]
        without_sym = summary["contrasts"]["centering_without_symbolic"][key]["seed_values"]
        summary["centering_by_symbolic_interaction"][key] = seed_statistics([x - y for x, y in zip(with_sym, without_sym)])

    # Reuse historical evaluations only after checking their frozen physical task.
    historical = ROOT / manifest["reused_baselines"]["source"]
    old_runtime = ROOT / "experiments/random_load_v6_20261001/runtime"
    for filename in ("lqr_env.py", "random_load_env.py"):
        assert (study / "runtime/csbapr/envs" / filename).read_bytes() == (old_runtime / "csbapr/envs" / filename).read_bytes()
    for method in manifest["reused_baselines"]["methods"]:
        config = runner.make_config(method, "cuda", "random_load")
        config.max_episodes = 500
        ordered = [validate_run(
            historical / f"{method}_seed{s}" / f"random_load_v6_{method}_seed{s}_evaluation.json",
            "random_load_v6", method, s, asdict(config), environment, evaluation_seeds, severities,
        ) for s in seeds]
        summary["historical_baselines"][method] = {"source": str(historical.relative_to(ROOT)),
                                                 "contemporaneous": False, **summarize_runs(ordered, severities)}
    for method in (centered_sym, centered):
        summary["historical_baseline_comparisons"][method] = {
            baseline: {str(a): contrast(summary["methods"][method]["returns"][str(a)]["seed_values"],
                                       values["returns"][str(a)]["seed_values"]) for a in severities}
            for baseline, values in summary["historical_baselines"].items()
        }
    old_summary = json.loads((historical / "summary.json").read_text())
    controls = {str(a): old_summary["post_hoc_control_diagnostic"][str(a)]["feedback_with_force"] for a in severities}
    for control in controls.values():
        assert control["evaluation_seeds"] == evaluation_seeds and len(control["returns"]) == 20
        assert all(map(math.isfinite, control["returns"]))
        assert math.isclose(-mean(control["returns"]), control["mean_cost"], rel_tol=1e-12)
    summary["known_model_feedback_reference"] = {
        "source": "results/random_load_v6/summary.json", "frozen_environment_files_identical": True,
        "reference_is_optimal": False, "controls": controls,
        "best_cost_excess": {str(a): {
            **seed_statistics([-runs[centered_sym][s]["amplitude_results"][str(a)]["mean"] - controls[str(a)]["mean_cost"] for s in seeds]),
            "relative_mean_cost_excess": -summary["methods"][centered_sym]["returns"][str(a)]["mean"] / controls[str(a)]["mean_cost"] - 1,
        } for a in severities},
    }
    summary["decision"] = {
        "accuracy_first_candidate": centered_sym,
        "lightweight_alternative": centered,
        "remove_symbolic_gate_passed": summary["gates"]["remove_symbolic_with_centering"]["passed"],
    }
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    with (destination / "seed_mean_returns.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("task_id", "method", "training_seed", "load_severity", "mean_return"))
        writer.writerows(rows)
    print("12 complete runs; 72 seed/severity means; frozen configs, budgets and paired evaluations verified")
    for method, values in summary["methods"].items():
        print(method, "ID cost", -values["returns"]["1"]["mean"], "50x cost", -values["returns"]["50"]["mean"],
              "50x seed SD", values["returns"]["50"]["sample_sd"])
    print("gates", summary["gates"])
    print("50x interaction", summary["centering_by_symbolic_interaction"]["50"])
    print("runtime differences", summary["contrast_runtime_differences"])


if __name__ == "__main__":
    main()
