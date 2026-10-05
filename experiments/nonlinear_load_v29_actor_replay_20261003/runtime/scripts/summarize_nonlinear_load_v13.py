"""Verify radius stress results and apply the frozen gates at each radius."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean

from summarize_lqr_critic_shift_v4 import contrast, seed_statistics
from summarize_nonlinear_load_v9 import validate_results
from summarize_nonlinear_load_v10 import gate_passes, paired_diagnostics
from summarize_nonlinear_load_v11 import CANDIDATE

ROOT = Path(__file__).resolve().parents[1]
REVISION = "nonlinear_load_v13_radius"
DIAGNOSTICS = ("maximum_physical_state_norm", "saturated_component_fraction",
               "outside_unit_observation_ball_fraction")


def validate_radius_reproduction(data, source, seeds, path):
    reproduction = data["reproduction"]
    assert reproduction["passed"] and reproduction["radius"] == 0.2, path
    assert reproduction["source_revision"] == "nonlinear_load_v12_paths", path
    assert reproduction["evaluation_seeds"] == seeds and reproduction["severities"] == [1, 50], path
    assert (reproduction["rtol"], reproduction["atol"]) == (1e-4, 1e-3), path
    actual = data["radius_results"]["0.2"]
    modes = {"actor": actual["amplitude_results"],
             **{k: v["amplitude_results"] for k, v in actual["feedback_controls"].items()}}
    previous = {"actor": source["amplitude_results"],
                **{k: v["amplitude_results"] for k, v in source["feedback_controls"].items()}}
    gaps = reproduction["maximum_absolute_return_gaps"]
    assert set(modes) == set(previous) == set(gaps), path
    for name, values in modes.items():
        assert set(gaps[name]) == {"1", "50"}, path
        for severity in ("1", "50"):
            current, old = values[severity], previous[name][severity]
            assert current["evaluation_seeds"] == old["evaluation_seeds"] == seeds, path
            assert len(current["returns"]) == len(old["returns"]) == len(seeds) == 100, path
            differences = [abs(x - y) for x, y in zip(current["returns"], old["returns"])]
            assert all(d <= 1e-3 + 1e-4 * abs(y) for d, y in zip(differences, old["returns"])), path
            assert math.isclose(max(differences), gaps[name][severity], rel_tol=1e-12, abs_tol=1e-12), path


def validate_diagnostics(results, path):
    for result in results.values():
        for name in DIAGNOSTICS:
            values = result[name]
            assert len(values) == 100 and all(math.isfinite(x) and x >= 0 for x in values), path
            if name != "maximum_physical_state_norm":
                assert all(x <= 1 for x in values), path


def expansion_decision(radius_summaries):
    expanded = [radius_summaries[r] for r in ("0.5", "1.0")]
    comparison = all(all(r["replication_gates"].values()) for r in expanded)
    added_value = all(r["learner_added_value_gate_passed"] for r in expanded)
    return {"learner_comparison_passed_at_both_expanded_radii": comparison,
            "learner_added_value_passed_at_both_expanded_radii": added_value,
            "full_expansion_confirmed": comparison and added_value,
            "new_tasks_submitted": False}


def summarize_radius(runs, methods, seeds, severities, evaluation_seeds, radius):
    blocks = {m: {s: runs[m][s]["radius_results"][radius] for s in seeds} for m in methods}
    summary = {"radius": float(radius), "methods": {}, "candidate_minus_comparator": {},
               "replication_gates": {}, "direct_feedback": {}, "candidate_minus_feedback": {},
               "paired_feedback_diagnostics": {}}
    for method in methods:
        summary["methods"][method] = {
            "returns": {str(a): seed_statistics([blocks[method][s]["amplitude_results"][str(a)]["mean"]
                                                 for s in seeds]) for a in severities},
            "diagnostics": {str(a): [{"training_seed": s,
                **{name: mean(blocks[method][s]["amplitude_results"][str(a)][name]) for name in DIAGNOSTICS},
                "largest_physical_state_norm": max(blocks[method][s]["amplitude_results"][str(a)][DIAGNOSTICS[0]])}
                for s in seeds] for a in severities},
        }
    candidate = summary["methods"][CANDIDATE]["returns"]
    for method in methods[1:]:
        differences = summary["candidate_minus_comparator"][method] = {
            str(a): contrast(candidate[str(a)]["seed_values"],
                             summary["methods"][method]["returns"][str(a)]["seed_values"]) for a in severities}
        summary["replication_gates"][method] = gate_passes(differences)
    for name in ("fitted_model", "known_local_linear", "known_inverse"):
        controls = {s: blocks[CANDIDATE][s]["feedback_controls"][name] for s in seeds}
        summary["direct_feedback"][name] = {"returns": {
            str(a): seed_statistics([controls[s]["amplitude_results"][str(a)]["mean"] for s in seeds])
            for a in severities}}
        summary["candidate_minus_feedback"][name] = {
            str(a): contrast(candidate[str(a)]["seed_values"],
                             summary["direct_feedback"][name]["returns"][str(a)]["seed_values"])
            for a in severities}
        summary["paired_feedback_diagnostics"][name] = {
            str(a): [{"training_seed": s, **paired_diagnostics(
                blocks[CANDIDATE][s]["amplitude_results"][str(a)]["returns"],
                controls[s]["amplitude_results"][str(a)]["returns"], evaluation_seeds)} for s in seeds]
            for a in severities}
    summary["learner_added_value_gate_passed"] = gate_passes(summary["candidate_minus_feedback"]["fitted_model"])
    return summary


def main():
    study = ROOT / "experiments/nonlinear_load_v13_radius_20261002"
    destination = ROOT / "results" / REVISION
    manifest = json.loads((study / "dispatch_manifest.json").read_text())
    observations = json.loads((destination / "task_observations.json").read_text())
    seeds, severities, radii, tasks = (manifest[k] for k in ("training_seeds", "severities", "radii", "tasks"))
    arms = {a["method"]: a for a in manifest["arms"]}
    methods, evaluation_seeds = list(arms), list(range(60000, 60100))
    assert seeds == manifest["confirmation_criterion_applies_to_training_seeds"] == [7, 8]
    assert severities == [1, 50] and radii == [0.2, 0.5, 1.0]
    assert manifest["expansion_radii_required_to_pass_separately"] == [0.5, 1.0]
    assert len(methods) == 4 and methods[0] == CANDIDATE
    assert len(tasks) == len({t["id"] for t in tasks}) == 8
    assert {(t["method"], t["seed"]) for t in tasks} == {(m, s) for m in methods for s in seeds}
    assert len(observations) == 8 and {o["id"] for o in observations} == {t["id"] for t in tasks}
    assert all(o["status"] == "done" and o["exit_code"] == 0 for o in observations)
    assert manifest["replication_criterion"] == {
        "comparators": methods[1:], "severities": severities, "lower_mean_cost": True, "required_matched_seed_wins": 2}
    assert manifest["learner_added_value_criterion"] == {
        "comparator": "own_fitted_model_feedback", "severities": severities, "required_matched_seed_wins": 2}
    runs, rows, paths, known_controls = {m: {} for m in methods}, [], [], {}
    episode_count = 0
    for task in tasks:
        method, seed = task["method"], task["seed"]
        relative = Path(method) / f"seed{seed}"
        path = destination / relative / f"{REVISION}_{method}_seed{seed}_evaluation.json"
        data = json.loads(path.read_text())
        payload = json.loads((study / "policy_inputs" / relative / "policy_input.json").read_text())
        source = json.loads((study / "source_evaluations" / relative / "evaluation.json").read_text())
        assert (data["revision"], data["source_revision"], data["reproduction_revision"], data["method"],
                data["training_seed"], data["source_task_id"], data["checkpoint_selection"]) == (
                    REVISION, manifest["source_revision"], manifest["reproduction_revision"], method,
                    seed, task["source_task_id"], "last"), path
        assert data["source_task_id"] == payload["source_task_id"] == source["source_task_id"], path
        assert payload["source_node"] == source["source_node"], path
        assert (data["environment_steps"], data["training_updates"], data["training_episodes"]) == (100000, 99001, 500), path
        assert all(data[k] == 0 for k in ("optimizer_steps_taken", "new_training_interactions", "new_identification_interactions")), path
        assert data["config"] == payload["source_protocol"]["config"] == source["config"] == arms[method]["config"], path
        assert data["environment_settings"] == source["environment_settings"] == manifest["environment_settings"], path
        assert data["environment"] == "nonlinear_load" and data["actor_type"] == arms[method]["config"]["actor_type"], path
        assert (data["evaluation_seed_start"], data["evaluation_episodes_per_severity"], data["preflight_only"]) == (60000, 100, False), path
        assert data["radii"] == radii and data["severities"] == severities and data["runtime"]["device"] == "cuda", path
        for package, version in manifest["runtime_requirement"].items():
            if package != "python_minor_versions":
                assert data["runtime"][package] == version, path
        assert ".".join(data["runtime"]["python"].split()[0].split(".")[:2]) in manifest["runtime_requirement"]["python_minor_versions"], path
        assert set(data["radius_results"]) == {str(r) for r in radii}, path
        for radius in radii:
            block = data["radius_results"][str(radius)]
            assert block["environment_settings"] == {**manifest["environment_settings"], "initial_amplitude": radius}, path
            validate_results(block["amplitude_results"], severities, evaluation_seeds, path)
            validate_diagnostics(block["amplitude_results"], path)
            controls = block["feedback_controls"]
            assert set(controls) == ({"known_local_linear", "known_inverse", "fitted_model"} if arms[method]["symbolic"]
                                     else {"known_local_linear", "known_inverse"}), path
            for name, control in controls.items():
                validate_results(control["amplitude_results"], severities, evaluation_seeds, path)
                validate_diagnostics(control["amplitude_results"], path)
                gain_key = "physical_gain" if name == "known_inverse" else "gain"
                assert control[gain_key] == source["feedback_controls"][name][gain_key], path
                if name == "fitted_model":
                    assert control["gain"] == payload["symbolic_reference_gain"], path
                else:
                    previous = known_controls.setdefault((radius, name), control)
                    assert control[gain_key] == previous[gain_key], path
                    for severity in severities:
                        assert all(math.isclose(x, y, rel_tol=1e-6, abs_tol=1e-7) for x, y in zip(
                            control["amplitude_results"][str(severity)]["returns"],
                            previous["amplitude_results"][str(severity)]["returns"])), path
            episode_count += (1 + len(controls)) * len(severities) * len(evaluation_seeds)
            rows.extend((task["id"], method, seed, radius, a, block["amplitude_results"][str(a)]["mean"])
                        for a in severities)
        validate_radius_reproduction(data, source, evaluation_seeds, path)
        runs[method][seed] = data
        paths.append(path)
    assert episode_count == manifest["formal_evaluation_episodes_including_controls"] == 15600
    summary = {
        "revision": REVISION, "completed_runs": 8, "training_seeds": seeds, "evaluation_seeds": evaluation_seeds,
        "radii": radii, "severities": severities, "environment_settings": manifest["environment_settings"],
        "data_accounting": manifest["data_accounting"], "checkpoint_selection": "last",
        "radius_summaries": {str(r): summarize_radius(runs, methods, seeds, severities, evaluation_seeds, str(r)) for r in radii},
        "statistics_unit": "same two training policies; 100 already-used paired paths per radius and severity",
        "reference_is_optimal": False, "pooled_across_radii": False, "new_training_seeds": False,
        "new_evaluation_paths": False, "new_optimizer_updates": 0, "new_training_interactions": 0,
        "new_identification_interactions": 0, "formal_evaluation_episodes_including_controls": episode_count,
        "final_evaluation_json_bytes": sum(p.stat().st_size for p in paths),
        "remote_evaluation_json_bytes_pulled": sum(o["remote_bytes_pulled"] for o in observations),
    }
    summary["decision"] = expansion_decision(summary["radius_summaries"])
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    with (destination / "seed_mean_returns.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("task_id", "method", "training_seed", "reset_radius", "load_severity", "mean_return"))
        writer.writerows(rows)
    for radius, result in summary["radius_summaries"].items():
        print("radius", radius)
        for method, values in result["methods"].items():
            print(method, "ID/50x cost", [-values["returns"][str(a)]["mean"] for a in severities])
        print("comparison gates", result["replication_gates"], "added-value", result["learner_added_value_gate_passed"])
        print("actor minus fitted", result["candidate_minus_feedback"]["fitted_model"])
    print("expansion decision", summary["decision"])


if __name__ == "__main__":
    main()
