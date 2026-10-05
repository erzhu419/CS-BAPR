"""Verify the frozen seed7/8 confirmation cohort without pooling development runs."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from summarize_lqr_critic_shift_v4 import contrast, seed_statistics
from summarize_nonlinear_load_v9 import validate_results
from summarize_nonlinear_load_v10 import gate_passes, paired_diagnostics
from summarize_random_load_equilibrium_v7 import summarize_runs

ROOT = Path(__file__).resolve().parents[1]
REVISION = "nonlinear_load_v11_replication"
CANDIDATE = "csbapr-centered-mlp-jacobian-only-no-target-shift"


def main():
    destination = ROOT / "results" / REVISION
    study = ROOT / "experiments/nonlinear_load_v11_replication_20261002"
    manifest = json.loads((study / "dispatch_manifest.json").read_text())
    observations = json.loads((destination / "task_observations.json").read_text())
    startup = json.loads((destination / "startup_observations.json").read_text())
    seeds, severities, tasks = manifest["training_seeds"], manifest["severities"], manifest["tasks"]
    arms = {a["method"]: a for a in manifest["arms"]}
    evaluation_seeds = list(range(50000, 50100))
    assert seeds == manifest["confirmation_criterion_applies_to_training_seeds"] == [7, 8]
    assert severities == [1, 2, 5, 10, 20, 50] and len(arms) == 4 and CANDIDATE in arms
    assert (manifest["evaluation_seed_start"], manifest["evaluation_episodes_per_severity"]) == (50000, 100)
    assert len(tasks) == len({t["id"] for t in tasks}) == 8
    assert {(t["method"], t["seed"]) for t in tasks} == {(m, s) for m in arms for s in seeds}
    assert len(observations) == 8 and {o["id"] for o in observations} == {t["id"] for t in tasks}
    assert all(o["status"] == "done" and o["exit_code"] == 0 for o in observations)
    startup_by_id = {t["id"]: t for t in startup["tasks"]}
    runs, paths, rows, known_controls = {m: {} for m in arms}, [], [], {}
    evaluation_count = 0
    for task in tasks:
        method, seed = task["method"], task["seed"]
        path = destination / method / f"seed{seed}" / f"{REVISION}_{method}_seed{seed}_evaluation.json"
        data = json.loads(path.read_text())
        assert (data["revision"], data["method"], data["training_seed"], data["training_episodes"],
                data["environment"], data["checkpoint_selection"]) == (REVISION, method, seed, 500, "nonlinear_load", "last"), path
        assert (data["environment_steps"], data["training_updates"]) == (100000, 99001), path
        assert len(data["training_returns"]) == 500 and all(map(math.isfinite, data["training_returns"])), path
        assert data["config"] == arms[method]["config"], path
        assert data["environment_settings"] == manifest["environment_settings"], path
        assert data["actor_type"] == arms[method]["config"]["actor_type"], path
        assert (data["evaluation_seed_start"], data["evaluation_episodes_per_severity"]) == (50000, 100), path
        assert data["runtime"]["device"] == "cuda", path
        for package, expected in manifest["runtime_requirement"].items():
            if package != "python_minor_versions":
                assert data["runtime"][package] == expected, path
        minor = ".".join(data["runtime"]["python"].split()[0].split(".")[:2])
        assert minor in manifest["runtime_requirement"]["python_minor_versions"], path
        symbolic = arms[method]["symbolic"]
        assert (data["sindy"] is not None) == symbolic, path
        alignment = data["policy_alignment"]
        assert alignment["reference"] == ("fitted_feedback" if symbolic else "known_model_diagnostic"), path
        assert set(alignment["initial_states"]) == {str(a) for a in severities}, path
        assert all(v["n_states"] == 100 for v in alignment["initial_states"].values()), path
        if data["actor_type"] == "centered-tanh-mlp":
            assert alignment["origin_action"] == [0.0, 0.0], path
        validate_results(data["amplitude_results"], severities, evaluation_seeds, path)
        controls = data["feedback_controls"]
        assert set(controls) == ({"known_local_linear", "known_inverse", "fitted_model"} if symbolic
                                 else {"known_local_linear", "known_inverse"}), path
        for name, control in controls.items():
            gain = control["physical_gain" if name == "known_inverse" else "gain"]
            assert len(gain) == 2 and all(len(row) == (4 if name == "known_inverse" else 6) for row in gain), path
            assert all(math.isfinite(v) for row in gain for v in row), path
            validate_results(control["amplitude_results"], severities, evaluation_seeds, path)
            if name == "fitted_model":
                assert gain == startup_by_id[task["id"]]["reference_gain"], path
            else:
                previous = known_controls.setdefault(name, control)
                assert gain == previous["physical_gain" if name == "known_inverse" else "gain"], path
                for a in severities:
                    assert all(math.isclose(x, y, rel_tol=1e-6, abs_tol=1e-7) for x, y in zip(
                        control["amplitude_results"][str(a)]["returns"],
                        previous["amplitude_results"][str(a)]["returns"])), path
        evaluation_count += (1 + len(controls)) * len(severities) * len(evaluation_seeds)
        runs[method][seed] = data
        paths.append(path)
        rows.extend((task["id"], method, seed, a, data["amplitude_results"][str(a)]["mean"]) for a in severities)

    summary = {
        "revision": REVISION, "completed_runs": 8, "checkpoint_selection": "last",
        "training_seeds": seeds, "evaluation_seeds": evaluation_seeds, "severities": severities,
        "environment_settings": manifest["environment_settings"], "data_accounting": manifest["data_accounting"],
        "methods": {m: summarize_runs([runs[m][s] for s in seeds], severities) for m in arms},
        "candidate_minus_comparator": {}, "replication_gates": {}, "direct_feedback": {},
        "candidate_minus_feedback": {}, "paired_feedback_diagnostics": {},
        "reference_is_optimal": False, "pooled_with_development_seeds": False,
        "statistics_unit": "two independent training-seed means, each averaging 100 paired paths",
        "final_evaluation_json_bytes": sum(p.stat().st_size for p in paths),
        "formal_evaluation_episodes_including_controls": evaluation_count,
    }
    comparators = [m for m in arms if m != CANDIDATE]
    assert manifest["replication_criterion"] == {
        "comparators": comparators, "severities": [1, 50], "lower_mean_cost": True, "required_matched_seed_wins": 2,
    }
    candidate = summary["methods"][CANDIDATE]["returns"]
    for method in comparators:
        differences = summary["candidate_minus_comparator"][method] = {
            str(a): contrast(candidate[str(a)]["seed_values"],
                             summary["methods"][method]["returns"][str(a)]["seed_values"]) for a in severities
        }
        summary["replication_gates"][method] = gate_passes(differences)
    for name in ("fitted_model", "known_local_linear", "known_inverse"):
        summary["direct_feedback"][name] = {
            "returns": {str(a): seed_statistics([runs[CANDIDATE][s]["feedback_controls"][name]
                                                 ["amplitude_results"][str(a)]["mean"] for s in seeds]) for a in severities},
        }
        differences = summary["candidate_minus_feedback"][name] = {
            str(a): contrast(candidate[str(a)]["seed_values"],
                             summary["direct_feedback"][name]["returns"][str(a)]["seed_values"]) for a in severities
        }
        summary["paired_feedback_diagnostics"][name] = {
            str(a): [{"training_seed": s, **paired_diagnostics(
                runs[CANDIDATE][s]["amplitude_results"][str(a)]["returns"],
                runs[CANDIDATE][s]["feedback_controls"][name]["amplitude_results"][str(a)]["returns"],
                evaluation_seeds)} for s in seeds] for a in severities
        }
    assert manifest["learner_added_value_criterion"] == {
        "comparator": "own_fitted_model_feedback", "severities": [1, 50], "required_matched_seed_wins": 2,
    }
    summary["learner_added_value_gate_passed"] = gate_passes(summary["candidate_minus_feedback"]["fitted_model"])
    summary["decision"] = {
        "learner_comparison_passed": all(summary["replication_gates"].values()),
        "learner_added_value_over_fitted_feedback": summary["learner_added_value_gate_passed"],
        "next_step": "Freeze the candidate and evaluate saved policies on unused paired paths before further tuning or environment changes",
    }
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    with (destination / "seed_mean_returns.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("task_id", "method", "training_seed", "load_severity", "mean_return"))
        writer.writerows(rows)
    print("8 complete runs; frozen configs, budgets, paired controls and saved fitted references verified")
    for method, values in summary["methods"].items():
        print(method, "ID cost", -values["returns"]["1"]["mean"], "50x cost", -values["returns"]["50"]["mean"])
    print("new-seed learner comparison", summary["replication_gates"])
    print("new-seed added RL value", summary["learner_added_value_gate_passed"])
    print("actor minus own fitted 50x", summary["candidate_minus_feedback"]["fitted_model"]["50"])


if __name__ == "__main__":
    main()
