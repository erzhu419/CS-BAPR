"""Aggregate independent v8 seeds and their paired direct-feedback controls."""
from __future__ import annotations

import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
from statistics import mean

from summarize_lqr_critic_shift_v4 import contrast, seed_statistics
from summarize_random_load_equilibrium_v7 import summarize_runs, validate_run
from summarize_random_load_v6 import load_module

ROOT = Path(__file__).resolve().parents[1]
REVISION = "random_load_validation_v8"


def main():
    study = ROOT / "experiments/random_load_validation_v8_20261001"
    destination = ROOT / "results" / REVISION
    manifest = json.loads((study / "dispatch_manifest.json").read_text())
    runner = load_module("frozen_validation_v8_runner", study / "runtime/scripts/test_lqr_convergence.py")
    seeds, severities = manifest["training_seeds"], manifest["severities"]
    evaluation_seeds = list(range(manifest["evaluation_seed_start"],
                                  manifest["evaluation_seed_start"] +
                                  manifest["evaluation_episodes_per_severity"]))
    assert seeds == [3, 4] and severities == [1, 2, 5, 10, 20, 50]
    assert evaluation_seeds == list(range(20000, 20020))
    arms, tasks = manifest["arms"], manifest["tasks"]
    methods = [a["method"] for a in arms]
    candidate = "csbapr-centered-mlp-no-target-shift"
    assert methods[0] == candidate and len(set(methods)) == 4
    assert len(tasks) == len({t["id"] for t in tasks}) == 8
    assert {(t["method"], t["seed"]) for t in tasks} == {(m, s) for m in methods for s in seeds}
    observations = json.loads((destination / "task_observations.json").read_text())
    assert len(observations) == 8 and {r["id"] for r in observations} == {t["id"] for t in tasks}
    assert all(r["status"] == "done" and r["exit_code"] == 0 for r in observations)
    environment = manifest["environment_settings"]
    assert environment == runner.make_env("random_load", 1, 0).specification()
    configs = {}
    for arm in arms:
        config = runner.make_config(arm["method"], "cuda", "random_load")
        config.max_episodes = 500
        configs[arm["method"]] = asdict(config)
        assert configs[arm["method"]] == arm["config"]
    runs, rows, paths = {m: {} for m in methods}, [], []
    known_control = None
    evaluation_episodes = 0
    for task in tasks:
        method, seed = task["method"], task["seed"]
        path = destination / f"{method}_seed{seed}" / f"{REVISION}_{method}_seed{seed}_evaluation.json"
        data = validate_run(path, REVISION, method, seed, configs[method], environment, evaluation_seeds, severities)
        assert data["evaluation_seed_start"] == 20000, path
        for package, expected in manifest["runtime_requirement"].items():
            if package != "python_minor_versions":
                assert data["runtime"][package] == expected, path
        minor = ".".join(data["runtime"]["python"].split()[0].split(".")[:2])
        assert minor in manifest["runtime_requirement"]["python_minor_versions"], path
        controls = data["feedback_controls"]
        assert set(controls) == ({"known_model", "fitted_model"} if data["sindy"] else {"known_model"}), path
        for controller in controls.values():
            gain = controller["gain"]
            assert len(gain) == 2 and all(len(row) == 6 for row in gain), path
            assert all(math.isfinite(x) for row in gain for x in row), path
            assert set(controller["amplitude_results"]) == {str(a) for a in severities}, path
            for result in controller["amplitude_results"].values():
                assert result["evaluation_seeds"] == evaluation_seeds and len(result["returns"]) == 20, path
                assert all(map(math.isfinite, result["returns"])), path
                assert math.isclose(mean(result["returns"]), result["mean"], rel_tol=1e-12), path
                evaluation_episodes += len(result["returns"])
        known = controls["known_model"]
        if known_control is None:
            known_control = known
        else:
            for actual, expected in zip(sum(known["gain"], []), sum(known_control["gain"], [])):
                assert math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-8), path
            for a in severities:
                for actual, expected in zip(known["amplitude_results"][str(a)]["returns"],
                                            known_control["amplitude_results"][str(a)]["returns"]):
                    assert math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-7), path
        runs[method][seed] = data
        paths.append(path)
        rows.extend((task["id"], method, seed, a, data["amplitude_results"][str(a)]["mean"]) for a in severities)
        evaluation_episodes += sum(len(v["returns"]) for v in data["amplitude_results"].values())

    summary = {
        "revision": REVISION, "completed_runs": 8, "checkpoint_selection": "last",
        "training_seeds": seeds, "evaluation_seeds": evaluation_seeds, "severities": severities,
        "environment_settings": environment, "data_accounting": manifest["data_accounting"],
        "methods": {m: summarize_runs([runs[m][s] for s in seeds], severities) for m in methods},
        "candidate_minus_comparator": {}, "replication_gates": {},
        "direct_feedback": {}, "candidate_minus_feedback": {}, "paired_feedback_return_gaps": {},
        "known_feedback_reference": known_control, "reference_is_optimal": False,
        "statistics_unit": "two independent training-seed means, each averaging 20 paired evaluation episodes",
        "pooled_with_v7": False, "final_evaluation_json_bytes": sum(p.stat().st_size for p in paths),
        "formal_evaluation_episodes_including_controls": evaluation_episodes,
    }
    criterion = manifest["replication_criterion"]
    assert criterion["comparators"] == methods[1:] and criterion["severities"] == [1, 50]
    assert criterion["lower_mean_cost"] and criterion["required_matched_seed_wins"] == 2
    values = summary["methods"]
    for method in criterion["comparators"]:
        differences = summary["candidate_minus_comparator"][method] = {
            str(a): contrast(values[candidate]["returns"][str(a)]["seed_values"],
                             values[method]["returns"][str(a)]["seed_values"]) for a in severities
        }
        summary["replication_gates"][method] = all(
            differences[str(a)]["mean"] > 0 and differences[str(a)]["matched_training_seed_wins"] == 2
            for a in criterion["severities"]
        )
    for name in ("known_model", "fitted_model"):
        summary["direct_feedback"][name] = {
            "gains": [runs[candidate][s]["feedback_controls"][name]["gain"] for s in seeds],
            "returns": {str(a): seed_statistics([
                runs[candidate][s]["feedback_controls"][name]["amplitude_results"][str(a)]["mean"] for s in seeds
            ]) for a in severities},
        }
        summary["candidate_minus_feedback"][name] = {
            str(a): contrast(values[candidate]["returns"][str(a)]["seed_values"],
                             summary["direct_feedback"][name]["returns"][str(a)]["seed_values"]) for a in severities
        }
        summary["paired_feedback_return_gaps"][name] = {
            str(a): [[x - y for x, y in zip(
                runs[candidate][s]["amplitude_results"][str(a)]["returns"],
                runs[candidate][s]["feedback_controls"][name]["amplitude_results"][str(a)]["returns"],
            )] for s in seeds] for a in severities
        }
    added_value = manifest["learner_added_value_criterion"]
    assert added_value == {"comparator": "own_fitted_model_feedback", "severities": [1, 50], "required_matched_seed_wins": 2}
    fitted_gaps = summary["candidate_minus_feedback"]["fitted_model"]
    summary["learner_added_value_gate_passed"] = all(fitted_gaps[str(a)]["matched_training_seed_wins"] == 2 for a in (1, 50))
    development = json.loads((ROOT / "results/random_load_equilibrium_v7/summary.json").read_text())
    summary["preserved_v7_symbolic_removal_gates"] = {
        name: value for name, value in development["gates"].items() if name.startswith("remove_symbolic")
    }
    summary["decision"] = {
        "replication_passed": all(summary["replication_gates"].values()),
        "accuracy_first_candidate": candidate,
        "learner_added_value_over_fitted_feedback": summary["learner_added_value_gate_passed"],
        "next_step": "non-learning controllability pilot for a separately frozen nonlinear ongoing-load task",
    }
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    with (destination / "seed_mean_returns.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("task_id", "method", "training_seed", "load_severity", "mean_return"))
        writer.writerows(rows)
    print("8 complete runs; 48 seed/severity means; configs, budgets, runtime and paired controls verified")
    for method in methods:
        print(method, "ID cost", -values[method]["returns"]["1"]["mean"],
              "50x cost", -values[method]["returns"]["50"]["mean"])
    print("replication gates", summary["replication_gates"])
    print("learner added-value gate", summary["learner_added_value_gate_passed"])
    print("actor minus fitted 50x", fitted_gaps["50"])


if __name__ == "__main__":
    main()
