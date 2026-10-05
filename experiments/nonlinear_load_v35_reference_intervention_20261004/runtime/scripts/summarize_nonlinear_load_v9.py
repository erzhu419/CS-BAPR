"""Verify and aggregate the frozen v9 learner comparison and paired controls."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean, median, pstdev

from summarize_lqr_critic_shift_v4 import contrast, seed_statistics
from summarize_random_load_equilibrium_v7 import summarize_runs

ROOT = Path(__file__).resolve().parents[1]
REVISION = "nonlinear_load_v9_training"
CANDIDATE = "csbapr-centered-mlp-no-target-shift"


def validate_results(results, severities, evaluation_seeds, path):
    assert set(results) == {str(a) for a in severities}, path
    for result in results.values():
        assert result["evaluation_seeds"] == evaluation_seeds, path
        assert len(result["returns"]) == len(evaluation_seeds), path
        assert all(map(math.isfinite, result["returns"])), path
        assert math.isclose(mean(result["returns"]), result["mean"], rel_tol=1e-12), path
        assert math.isclose(pstdev(result["returns"]), result["std"], rel_tol=1e-12, abs_tol=1e-12), path


def main():
    study = ROOT / "experiments/nonlinear_load_v9_training_20261002"
    destination = ROOT / "results" / REVISION
    manifest = json.loads((study / "dispatch_manifest.json").read_text())
    observations = json.loads((destination / "task_observations.json").read_text())
    arms, tasks = manifest["arms"], manifest["tasks"]
    methods = [a["method"] for a in arms]
    seeds, severities = manifest["training_seeds"], manifest["severities"]
    evaluation_seeds = list(range(manifest["evaluation_seed_start"],
                                  manifest["evaluation_seed_start"] + manifest["evaluation_episodes_per_severity"]))
    assert seeds == [5, 6] and severities == [1, 2, 5, 10, 20, 50]
    assert evaluation_seeds == list(range(50000, 50100))
    assert methods[0] == CANDIDATE and len(set(methods)) == 4
    assert len(tasks) == len({t["id"] for t in tasks}) == 8
    assert {(t["method"], t["seed"]) for t in tasks} == {(m, s) for m in methods for s in seeds}
    assert len(observations) == 8 and {r["id"] for r in observations} == {t["id"] for t in tasks}
    assert all(r["status"] == "done" and r["exit_code"] == 0 for r in observations)
    pilot = json.loads((ROOT / "results/nonlinear_load_v9/pilot.json").read_text())
    assert manifest["environment_settings"] == pilot["environment_settings"]
    assert pilot["evaluation_seeds"] == evaluation_seeds
    configs = {a["method"]: a["config"] for a in arms}
    runs, rows, paths = {m: {} for m in methods}, [], []
    known_controls, episode_count = {}, 0
    for task in tasks:
        method, seed = task["method"], task["seed"]
        path = destination / method / f"seed{seed}" / f"{REVISION}_{method}_seed{seed}_evaluation.json"
        data = json.loads(path.read_text())
        assert (data["revision"], data["method"], data["training_seed"], data["training_episodes"],
                data["environment"], data["checkpoint_selection"]) == (REVISION, method, seed, 500, "nonlinear_load", "last"), path
        assert (data["environment_steps"], data["training_updates"]) == (100000, 99001), path
        assert len(data["training_returns"]) == 500 and all(map(math.isfinite, data["training_returns"])), path
        assert data["config"] == configs[method] and data["environment_settings"] == manifest["environment_settings"], path
        assert data["actor_type"] == configs[method]["actor_type"], path
        assert data["evaluation_seed_start"] == 50000 and data["evaluation_episodes_per_severity"] == 100, path
        assert data["runtime"]["device"] == "cuda", path
        for package, expected in manifest["runtime_requirement"].items():
            if package != "python_minor_versions":
                assert data["runtime"][package] == expected, path
        minor = ".".join(data["runtime"]["python"].split()[0].split(".")[:2])
        assert minor in manifest["runtime_requirement"]["python_minor_versions"], path
        symbolic = next(a["symbolic"] for a in arms if a["method"] == method)
        assert (data["sindy"] is not None) == symbolic, path
        alignment = data["policy_alignment"]
        assert alignment["reference"] == ("fitted_feedback" if symbolic else "known_model_diagnostic"), path
        assert set(alignment["initial_states"]) == {str(a) for a in severities}, path
        assert all(v["n_states"] == 100 for v in alignment["initial_states"].values()), path
        if configs[method]["actor_type"] == "centered-tanh-mlp":
            assert alignment["origin_action"] == [0.0, 0.0], path
        validate_results(data["amplitude_results"], severities, evaluation_seeds, path)
        controls = data["feedback_controls"]
        assert set(controls) == ({"known_local_linear", "known_inverse", "fitted_model"} if symbolic
                                 else {"known_local_linear", "known_inverse"}), path
        for name, controller in controls.items():
            gain = controller["physical_gain" if name == "known_inverse" else "gain"]
            assert len(gain) == 2 and all(len(row) == (4 if name == "known_inverse" else 6) for row in gain), path
            assert all(math.isfinite(x) for row in gain for x in row), path
            validate_results(controller["amplitude_results"], severities, evaluation_seeds, path)
            if name != "fitted_model":
                pilot_name = "local_linear_feedback" if name == "known_local_linear" else "inverse_feedback"
                for a in severities:
                    actual = controller["amplitude_results"][str(a)]["returns"]
                    expected = pilot["results"][str(a)][pilot_name]["returns"]
                    assert all(math.isclose(x, y, rel_tol=1e-6, abs_tol=1e-7) for x, y in zip(actual, expected)), path
                known_controls.setdefault(name, controller)
        episode_count += (1 + len(controls)) * len(severities) * len(evaluation_seeds)
        runs[method][seed] = data
        paths.append(path)
        rows.extend((task["id"], method, seed, a, data["amplitude_results"][str(a)]["mean"]) for a in severities)

    summary = {
        "revision": REVISION, "completed_runs": 8, "checkpoint_selection": "last",
        "training_seeds": seeds, "evaluation_seeds": evaluation_seeds, "severities": severities,
        "environment_settings": manifest["environment_settings"], "data_accounting": manifest["data_accounting"],
        "methods": {m: summarize_runs([runs[m][s] for s in seeds], severities) for m in methods},
        "candidate_minus_comparator": {}, "replication_gates": {}, "direct_feedback": {},
        "candidate_minus_feedback": {}, "paired_feedback_return_gaps": {}, "paired_feedback_diagnostics": {},
        "known_controls": known_controls, "reference_is_optimal": False,
        "statistics_unit": "two independent training-seed means, each averaging 100 paired evaluation episodes",
        "pooled_with_prior_rounds": False, "final_evaluation_json_bytes": sum(p.stat().st_size for p in paths),
        "formal_evaluation_episodes_including_controls": episode_count,
    }
    criterion = manifest["replication_criterion"]
    assert criterion == {"comparators": methods[1:], "severities": [1, 50],
                         "lower_mean_cost": True, "required_matched_seed_wins": 2}
    values = summary["methods"]
    for method in methods[1:]:
        differences = summary["candidate_minus_comparator"][method] = {
            str(a): contrast(values[CANDIDATE]["returns"][str(a)]["seed_values"],
                             values[method]["returns"][str(a)]["seed_values"]) for a in severities
        }
        summary["replication_gates"][method] = all(differences[str(a)]["matched_training_seed_wins"] == 2
                                                   and differences[str(a)]["mean"] > 0 for a in (1, 50))
    for name in ("fitted_model", "known_local_linear", "known_inverse"):
        summary["direct_feedback"][name] = {
            "gains": [runs[CANDIDATE][s]["feedback_controls"][name]["physical_gain" if name == "known_inverse" else "gain"] for s in seeds],
            "returns": {str(a): seed_statistics([runs[CANDIDATE][s]["feedback_controls"][name]["amplitude_results"][str(a)]["mean"]
                                                 for s in seeds]) for a in severities},
        }
        summary["candidate_minus_feedback"][name] = {
            str(a): contrast(values[CANDIDATE]["returns"][str(a)]["seed_values"],
                             summary["direct_feedback"][name]["returns"][str(a)]["seed_values"]) for a in severities
        }
        paired = summary["paired_feedback_return_gaps"][name] = {
            str(a): [[x - y for x, y in zip(runs[CANDIDATE][s]["amplitude_results"][str(a)]["returns"],
                                           runs[CANDIDATE][s]["feedback_controls"][name]["amplitude_results"][str(a)]["returns"])]
                     for s in seeds] for a in severities
        }
        summary["paired_feedback_diagnostics"][name] = {
            str(a): [{"training_seed": s, "mean": mean(gaps), "median": median(gaps),
                      "evaluation_path_wins": sum(x > 0 for x in gaps),
                      "worst_five_paths": [{"evaluation_seed": evaluation_seeds[i], "return_gap": gaps[i]}
                                           for i in sorted(range(len(gaps)), key=lambda i: gaps[i])[:5]]}
                     for s, gaps in zip(seeds, paired[str(a)])] for a in severities
        }
    assert manifest["learner_added_value_criterion"] == {
        "comparator": "own_fitted_model_feedback", "severities": [1, 50], "required_matched_seed_wins": 2,
    }
    fitted_gaps = summary["candidate_minus_feedback"]["fitted_model"]
    summary["learner_added_value_gate_passed"] = all(fitted_gaps[str(a)]["matched_training_seed_wins"] == 2 for a in (1, 50))
    summary["decision"] = {
        "learner_comparison_passed": all(summary["replication_gates"].values()),
        "retain_symbolic_candidate": True,
        "learner_added_value_over_fitted_feedback": summary["learner_added_value_gate_passed"],
        "next_step": "Diagnose paired high-load failures on workers before expanding seeds or changing the frozen environment",
    }
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    with (destination / "seed_mean_returns.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("task_id", "method", "training_seed", "load_severity", "mean_return"))
        writer.writerows(rows)
    print("8 complete runs; 48 seed/severity means; frozen configs, budgets, runtime and paired controls verified")
    for method in methods:
        print(method, "ID cost", -values[method]["returns"]["1"]["mean"], "50x cost", -values[method]["returns"]["50"]["mean"])
    print("learner comparison gates", summary["replication_gates"])
    print("added-RL-value gate", summary["learner_added_value_gate_passed"])
    print("actor minus own fitted 50x", fitted_gaps["50"])


if __name__ == "__main__":
    main()
