"""Verify frozen radius diagnostics and summarize paired cost mechanisms."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean

from summarize_lqr_critic_shift_v4 import seed_statistics

ROOT = Path(__file__).resolve().parents[1]
REVISION = "nonlinear_load_v14_readonly_diagnostic"
COSTS = ("state_costs", "command_costs", "early_state_costs", "early_command_costs",
         "late_state_costs", "late_command_costs")
SOURCE_MODES = {"fitted": "fitted_model", "known_local": "known_local_linear", "known_inverse": "known_inverse"}


def validate_costs(result, saved, seeds, path):
    assert result["evaluation_seeds"] == saved["evaluation_seeds"] == seeds, path
    assert len(result["returns"]) == len(seeds), path
    for actual, expected in zip(result["returns"], saved["returns"]):
        assert math.isfinite(actual) and abs(actual - expected) <= 1e-3 + 1e-4 * abs(expected), path
    assert math.isclose(result["maximum_absolute_reproduction_error"],
                        max(abs(x - y) for x, y in zip(result["returns"], saved["returns"])),
                        rel_tol=1e-12, abs_tol=1e-12), path
    assert math.isclose(result["mean_cost"], -mean(result["returns"]), rel_tol=1e-12), path
    for name in COSTS:
        values = result[name]
        assert len(values) == len(seeds) and all(math.isfinite(x) and x >= 0 for x in values), path
        assert math.isclose(mean(values), result[f"mean_{name[:-1]}"], rel_tol=1e-12, abs_tol=1e-12), path
    for i, value in enumerate(result["returns"]):
        assert math.isclose(-value, result["state_costs"][i] + result["command_costs"][i], rel_tol=1e-12, abs_tol=1e-12), path
        for kind in ("state", "command"):
            assert math.isclose(result[f"{kind}_costs"][i],
                                result[f"early_{kind}_costs"][i] + result[f"late_{kind}_costs"][i],
                                rel_tol=1e-12, abs_tol=1e-12), path


def main():
    study = ROOT / "experiments/nonlinear_load_v14_diagnostic_20261002"
    destination = ROOT / "results/nonlinear_load_v14_diagnostic"
    manifest = json.loads((study / "dispatch_manifest.json").read_text())
    observations = json.loads((destination / "task_observations.json").read_text())
    tasks, seeds, method = manifest["tasks"], manifest["training_seeds"], manifest["method"]
    evaluation_seeds = list(range(60000, 60100))
    assert len(tasks) == 2 and {t["seed"] for t in tasks} == set(seeds) == {7, 8}
    assert {o["id"] for o in observations} == {t["id"] for t in tasks} and len(observations) == 2
    assert all(o["status"] == "done" and o["exit_code"] == 0 for o in observations)
    scopes = {"initial_states": 100, "actor_trajectory": 2000, "largest_physical_state_quartile": 500,
              "largest_load_quartile": 500, "path60002": 20}
    runs, rows, paths = {}, [], []
    for task in tasks:
        seed = task["seed"]
        relative = Path(method) / f"seed{seed}"
        path = destination / relative / f"{REVISION}_{method}_seed{seed}_diagnostic.json"
        data = json.loads(path.read_text())
        source = json.loads((study / "source_evaluations" / relative / "evaluation.json").read_text())
        assert (data["revision"], data["source_revision"], data["policy_source_revision"], data["source_task_id"],
                data["method"], data["training_seed"], data["checkpoint_selection"]) == (
                    REVISION, manifest["source_revision"], manifest["policy_source_revision"], task["source_task_id"], method, seed, "last"), path
        assert data["source_task_id"] == source["source_task_id"] and data["config"] == source["config"] == manifest["config"], path
        assert data["environment_settings"] == source["environment_settings"] == manifest["environment_settings"], path
        assert (data["environment_steps"], data["training_updates"], data["training_episodes"]) == (100000, 99001, 500), path
        assert all(data[k] == 0 for k in ("optimizer_steps_taken", "new_training_interactions", "new_identification_interactions")), path
        assert not data["preflight_only"] and data["evaluation_seeds"] == evaluation_seeds and data["runtime"]["device"] == "cuda", path
        assert data["radii"] == manifest["radii"] == [0.2, 0.5, 1.0] and data["severities"] == manifest["severities"] == [1, 50], path
        assert (data["early_steps"], data["sample_interval"], data["focus_path"]) == (50, 10, 60002), path
        for package, version in manifest["runtime_requirement"].items():
            if package != "python_minor_versions":
                assert data["runtime"][package] == version, path
        assert ".".join(data["runtime"]["python"].split()[0].split(".")[:2]) in manifest["runtime_requirement"]["python_minor_versions"], path
        assert set(data["radius_results"]) == {"0.2", "0.5", "1.0"}, path
        for radius, values in data["radius_results"].items():
            assert set(values) == {"1", "50"}, path
            for severity, block in values.items():
                original = source["radius_results"][radius]
                assert block["reproduction_passed"] and block["environment_settings"] == original["environment_settings"], path
                assert set(block["controllers"]) == {"actor", *SOURCE_MODES}, path
                for mode, controller in block["controllers"].items():
                    saved = (original["amplitude_results"] if mode == "actor" else
                             original["feedback_controls"][SOURCE_MODES[mode]]["amplitude_results"])[severity]
                    validate_costs(controller, saved, evaluation_seeds, path)
                assert set(block["alignment"]) == set(scopes), path
                for scope, count in scopes.items():
                    assert block["alignment"][scope]["sampled_states"] == count, path
                    assert set(block["alignment"][scope]["controls"]) == set(block["controllers"]), path
                assert set(block["paired_excess_costs"]) == set(SOURCE_MODES), path
                for mode, comparison in block["paired_excess_costs"].items():
                    assert [p["evaluation_seed"] for p in comparison["per_path"]] == evaluation_seeds, path
                    actor, reference = block["controllers"]["actor"], block["controllers"][mode]
                    for i, p in enumerate(comparison["per_path"]):
                        assert math.isclose(p["total_cost"], reference["returns"][i] - actor["returns"][i], rel_tol=1e-9, abs_tol=1e-10), path
                        for name in COSTS:
                            assert math.isclose(p[name[:-1]], actor[name][i] - reference[name][i], rel_tol=1e-10, abs_tol=1e-10), path
                    for name in ("total_cost", *(k[:-1] for k in COSTS)):
                        assert math.isclose(comparison[f"mean_{name}"], mean(p[name] for p in comparison["per_path"]), rel_tol=1e-12, abs_tol=1e-12), path
                    rows.append((seed, radius, severity, mode, comparison["mean_total_cost"],
                                 comparison["mean_state_cost"], comparison["mean_command_cost"],
                                 comparison["mean_early_state_cost"] + comparison["mean_early_command_cost"],
                                 comparison["mean_late_state_cost"] + comparison["mean_late_command_cost"]))
        runs[seed] = data
        paths.append(path)
    summary = {"revision": REVISION, "completed_runs": 2, "training_seeds": seeds, "evaluation_seeds": evaluation_seeds,
               "radii": manifest["radii"], "severities": manifest["severities"], "radius_summaries": {},
               "diagnostic_only": True, "statistics_unit": "same two frozen training policies; retrospective paired mechanism analysis",
               "new_optimizer_updates": 0, "new_training_interactions": 0, "new_identification_interactions": 0,
               "formal_evaluation_episodes_including_controls": 4800,
               "historical_v13_decision_unchanged": manifest["historical_v13_decision_unchanged"],
               "diagnostic_json_bytes": sum(p.stat().st_size for p in paths),
               "remote_diagnostic_json_bytes_pulled": sum(o["remote_bytes_pulled"] for o in observations),
               "new_tasks_submitted": False}
    for radius in ("0.2", "0.5", "1.0"):
        summary["radius_summaries"][radius] = {}
        for severity in ("1", "50"):
            blocks = [runs[s]["radius_results"][radius][severity] for s in seeds]
            result = summary["radius_summaries"][radius][severity] = {"paired_excess_costs": {}, "fitted_alignment": {}, "path60002": []}
            for mode in SOURCE_MODES:
                pairs = [b["paired_excess_costs"][mode] for b in blocks]
                result["paired_excess_costs"][mode] = {name: seed_statistics([p[f"mean_{name}"] for p in pairs])
                    for name in ("total_cost", *(k[:-1] for k in COSTS))}
                result["paired_excess_costs"][mode].update(
                    early_total_cost=seed_statistics([p["mean_early_state_cost"] + p["mean_early_command_cost"] for p in pairs]),
                    late_total_cost=seed_statistics([p["mean_late_state_cost"] + p["mean_late_command_cost"] for p in pairs]))
            for scope in scopes:
                metrics = [b["alignment"][scope]["controls"]["fitted"] for b in blocks]
                result["fitted_alignment"][scope] = {key: seed_statistics([v[key] for v in metrics])
                                                     for key in metrics[0] if key != "mean_output_force_load_jacobian"}
            result["path60002"] = [{"training_seed": s, **next(p for p in b["paired_excess_costs"]["fitted"]["per_path"]
                                                               if p["evaluation_seed"] == 60002)} for s, b in zip(seeds, blocks)]
            print("radius/severity", radius, severity,
                  {name: result["paired_excess_costs"]["fitted"][name]["mean"]
                   for name in ("total_cost", "state_cost", "command_cost", "early_total_cost", "late_total_cost")})
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    with (destination / "seed_cost_excesses.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("training_seed", "reset_radius", "load_severity", "reference", "total_cost_excess",
                         "state_cost_excess", "command_cost_excess", "early_cost_excess", "late_cost_excess"))
        writer.writerows(rows)


if __name__ == "__main__":
    main()
