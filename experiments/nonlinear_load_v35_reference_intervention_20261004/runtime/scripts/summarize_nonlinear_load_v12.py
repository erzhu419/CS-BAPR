"""Verify frozen-policy evaluations and judge the new disturbance paths alone."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from summarize_lqr_critic_shift_v4 import contrast, seed_statistics
from summarize_nonlinear_load_v9 import validate_results
from summarize_nonlinear_load_v10 import gate_passes, paired_diagnostics
from summarize_nonlinear_load_v11 import CANDIDATE

ROOT = Path(__file__).resolve().parents[1]
REVISION = "nonlinear_load_v12_paths"


def validate_reproduction(reproduction, source_returns, path):
    assert reproduction["passed"] and reproduction["evaluation_seed"] == 50000, path
    assert reproduction["severities"] == [1, 50], path
    assert (reproduction["rtol"], reproduction["atol"]) == (1e-4, 1e-3), path
    assert set(reproduction["checks"]) == set(source_returns), path
    for name, checks in reproduction["checks"].items():
        assert set(checks) == {"1", "50"}, path
        for severity, result in checks.items():
            expected = source_returns[name][severity]
            assert result["source"] == expected, path
            assert math.isclose(result["actual"], expected, rel_tol=1e-4, abs_tol=1e-3), path
            assert math.isclose(result["return_gap"], result["actual"] - expected, abs_tol=1e-12), path


def main():
    study = ROOT / "experiments/nonlinear_load_v12_paths_20261002"
    destination = ROOT / "results" / REVISION
    manifest = json.loads((study / "dispatch_manifest.json").read_text())
    observations = json.loads((destination / "task_observations.json").read_text())
    seeds, severities, tasks = manifest["training_seeds"], manifest["severities"], manifest["tasks"]
    arms = {a["method"]: a for a in manifest["arms"]}
    evaluation_seeds = list(range(60000, 60100))
    assert seeds == manifest["confirmation_criterion_applies_to_training_seeds"] == [7, 8]
    assert severities == [1, 2, 5, 10, 20, 50] and len(arms) == 4 and CANDIDATE in arms
    assert (manifest["evaluation_seed_start"], manifest["evaluation_episodes_per_severity"]) == (60000, 100)
    assert len(tasks) == len({t["id"] for t in tasks}) == 8
    assert {(t["method"], t["seed"]) for t in tasks} == {(m, s) for m in arms for s in seeds}
    assert len(observations) == 8 and {o["id"] for o in observations} == {t["id"] for t in tasks}
    assert all(o["status"] == "done" and o["exit_code"] == 0 for o in observations)
    runs, paths, rows, known_controls = {m: {} for m in arms}, [], [], {}
    evaluation_count, reproduction_count = 0, 0
    for task in tasks:
        method, seed = task["method"], task["seed"]
        path = destination / method / f"seed{seed}" / f"{REVISION}_{method}_seed{seed}_evaluation.json"
        data = json.loads(path.read_text())
        payload = json.loads((study / "policy_inputs" / method / f"seed{seed}" / "policy_input.json").read_text())
        source = payload["source_protocol"]
        assert (data["revision"], data["source_revision"], data["source_task_id"], data["training_seed"],
                data["method"], data["checkpoint_selection"]) == (
                    REVISION, manifest["source_revision"], task["source_task_id"], seed, method, "last"), path
        assert data["source_task_id"] == payload["source_task_id"] and data["source_node"] == payload["source_node"], path
        assert (data["environment_steps"], data["training_updates"], data["training_episodes"]) == (100000, 99001, 500), path
        assert all(data[k] == 0 for k in ("optimizer_steps_taken", "new_training_interactions", "new_identification_interactions")), path
        assert data["config"] == source["config"] == arms[method]["config"], path
        assert data["environment_settings"] == source["environment_settings"] == manifest["environment_settings"], path
        assert data["environment"] == "nonlinear_load" and data["actor_type"] == arms[method]["config"]["actor_type"], path
        assert (data["evaluation_seed_start"], data["evaluation_episodes_per_severity"]) == (60000, 100), path
        assert data["runtime"]["device"] == "cuda", path
        for package, expected in manifest["runtime_requirement"].items():
            if package != "python_minor_versions":
                assert data["runtime"][package] == expected, path
        assert ".".join(data["runtime"]["python"].split()[0].split(".")[:2]) in manifest["runtime_requirement"]["python_minor_versions"], path
        assert data["sindy"] == payload["sindy_report"], path
        assert (data["sindy"] is not None) == arms[method]["symbolic"], path
        validate_reproduction(data["reproduction"], payload["source_path_returns"], path)
        validate_results(data["amplitude_results"], severities, evaluation_seeds, path)
        controls = data["feedback_controls"]
        assert set(controls) == ({"known_local_linear", "known_inverse", "fitted_model"} if arms[method]["symbolic"]
                                 else {"known_local_linear", "known_inverse"}), path
        for name, control in controls.items():
            gain_key = "physical_gain" if name == "known_inverse" else "gain"
            gain = control[gain_key]
            assert len(gain) == 2 and all(len(row) == (4 if name == "known_inverse" else 6) for row in gain), path
            assert all(math.isfinite(x) for row in gain for x in row), path
            validate_results(control["amplitude_results"], severities, evaluation_seeds, path)
            if name == "fitted_model":
                assert gain == payload["symbolic_reference_gain"], path
            else:
                previous = known_controls.setdefault(name, control)
                assert gain == previous[gain_key], path
                for a in severities:
                    assert all(math.isclose(x, y, rel_tol=1e-6, abs_tol=1e-7) for x, y in zip(
                        control["amplitude_results"][str(a)]["returns"],
                        previous["amplitude_results"][str(a)]["returns"])), path
        evaluation_count += (1 + len(controls)) * len(severities) * len(evaluation_seeds)
        reproduction_count += (1 + len(controls)) * 2
        runs[method][seed] = data
        paths.append(path)
        rows.extend((task["id"], method, seed, a, data["amplitude_results"][str(a)]["mean"]) for a in severities)

    assert evaluation_count == manifest["formal_evaluation_episodes_including_controls"] == 15600
    assert reproduction_count == manifest["reproduction_episodes_including_controls"] == 52
    summary = {
        "revision": REVISION, "source_revision": manifest["source_revision"], "completed_runs": 8,
        "training_seeds": seeds, "evaluation_seeds": evaluation_seeds, "severities": severities,
        "environment_settings": manifest["environment_settings"], "checkpoint_selection": "last",
        "methods": {m: {"returns": {str(a): seed_statistics([
            runs[m][s]["amplitude_results"][str(a)]["mean"] for s in seeds]) for a in severities}} for m in arms},
        "candidate_minus_comparator": {}, "replication_gates": {}, "direct_feedback": {},
        "candidate_minus_feedback": {}, "paired_feedback_diagnostics": {},
        "reference_is_optimal": False, "pooled_with_old_paths": False, "pooled_with_development_seeds": False,
        "statistics_unit": "two training-seed means on 100 new paired disturbance paths; no new training seeds",
        "new_optimizer_updates": 0, "new_training_interactions": 0, "new_identification_interactions": 0,
        "formal_evaluation_episodes_including_controls": evaluation_count,
        "reproduction_episodes_including_controls": reproduction_count,
        "final_evaluation_json_bytes": sum(p.stat().st_size for p in paths),
    }
    comparators = [m for m in arms if m != CANDIDATE]
    assert manifest["replication_criterion"] == {
        "comparators": comparators, "severities": [1, 50], "lower_mean_cost": True, "required_matched_seed_wins": 2,
    }
    candidate = summary["methods"][CANDIDATE]["returns"]
    for method in comparators:
        differences = summary["candidate_minus_comparator"][method] = {
            str(a): contrast(candidate[str(a)]["seed_values"], summary["methods"][method]["returns"][str(a)]["seed_values"])
            for a in severities
        }
        summary["replication_gates"][method] = gate_passes(differences)
    for name in ("fitted_model", "known_local_linear", "known_inverse"):
        summary["direct_feedback"][name] = {"returns": {
            str(a): seed_statistics([runs[CANDIDATE][s]["feedback_controls"][name]["amplitude_results"][str(a)]["mean"]
                                    for s in seeds]) for a in severities}}
        summary["candidate_minus_feedback"][name] = {
            str(a): contrast(candidate[str(a)]["seed_values"], summary["direct_feedback"][name]["returns"][str(a)]["seed_values"])
            for a in severities}
        summary["paired_feedback_diagnostics"][name] = {
            str(a): [{"training_seed": s, **paired_diagnostics(
                runs[CANDIDATE][s]["amplitude_results"][str(a)]["returns"],
                runs[CANDIDATE][s]["feedback_controls"][name]["amplitude_results"][str(a)]["returns"], evaluation_seeds)}
                     for s in seeds] for a in severities}
    assert manifest["learner_added_value_criterion"] == {
        "comparator": "own_fitted_model_feedback", "severities": [1, 50], "required_matched_seed_wins": 2,
    }
    summary["learner_added_value_gate_passed"] = gate_passes(summary["candidate_minus_feedback"]["fitted_model"])
    summary["decision"] = {
        "learner_comparison_passed": all(summary["replication_gates"].values()),
        "learner_added_value_over_fitted_feedback": summary["learner_added_value_gate_passed"],
        "next_step": "Keep policies and references frozen; test larger physical-state reset radii before claiming wider applicability",
        "new_tasks_submitted": False,
    }
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    with (destination / "seed_mean_returns.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("task_id", "method", "training_seed", "load_severity", "mean_return"))
        writer.writerows(rows)
    print("8 complete frozen-policy evaluations; 15,600 new-path episodes; no training or fitting")
    for method, values in summary["methods"].items():
        print(method, "ID cost", -values["returns"]["1"]["mean"], "50x cost", -values["returns"]["50"]["mean"])
    print("new-path comparison gates", summary["replication_gates"])
    print("new-path added-RL-value gate", summary["learner_added_value_gate_passed"])
    print("actor minus fitted 50x", summary["candidate_minus_feedback"]["fitted_model"]["50"])


if __name__ == "__main__":
    main()
