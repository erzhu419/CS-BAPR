"""Aggregate the completed worker-side v9 diagnostic, without training gates."""
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
MODES = ("actor", "fitted", "known_local", "known_inverse")


def main():
    destination = ROOT / "results/nonlinear_load_v9_diagnostic"
    reports = [json.loads((destination / f"seed{seed}.json").read_text()) for seed in (5, 6)]
    observations = json.loads((destination / "task_observations.json").read_text())
    assert [r["training_seed"] for r in reports] == [5, 6]
    assert [r["seed"] for r in observations] == [5, 6]
    assert all(r["exit_code"] == 0 for r in observations)
    summary = {
        "revision": "nonlinear_load_v9_readonly_diagnostic", "training_seeds": [5, 6],
        "evaluation_paths_per_severity": 100, "severities": [1, 20, 50],
        "evaluation_episodes": 2400, "optimizer_steps_taken": 0, "checkpoint_transfers": 0,
        "downloaded_result_json_bytes": sum((destination / f"seed{s}.json").stat().st_size for s in (5, 6)),
        "maximum_absolute_reproduction_error": max(
            r["results"][str(a)][m]["maximum_absolute_reproduction_error"]
            for r in reports for a in (1, 20, 50) for m in MODES),
        "mean_cost_decomposition": {}, "mean_actor_excess_cost_decomposition": {},
        "seed50_actor_minus_fitted": {}, "seed50_trajectory_alignment": {},
        "origin_feedback": {str(r["training_seed"]): r["origin"] for r in reports},
        "interpretation": "State-cost gap and inherited fitted-load Jacobian bias; constraint attribution needs ablation",
        "next_training_arms": {"critic_only": {"weight_sym": 0.01, "jac_weight": 0.0},
                               "jacobian_only": {"weight_sym": 0.0, "jac_weight": 0.01}},
        "new_training_submitted": False, "v9_added_rl_value_gate_changed": False,
    }
    for severity in (1, 20, 50):
        key = str(severity)
        summary["mean_cost_decomposition"][key] = {
            mode: {metric: mean(r["results"][key][mode][metric] for r in reports)
                   for metric in ("mean_cost", "mean_state_cost", "mean_command_cost")}
            for mode in MODES
        }
        summary["mean_actor_excess_cost_decomposition"][key] = {
            mode: {metric: mean(r["results"][key]["paired_excess_costs"][mode][metric] for r in reports)
                   for metric in ("mean_total_cost", "mean_state_cost", "mean_command_cost")}
            for mode in MODES[1:]
        }
    for r in reports:
        seed = str(r["training_seed"])
        summary["seed50_actor_minus_fitted"][seed] = r["results"]["50"]["paired_excess_costs"]["fitted"]
        summary["seed50_trajectory_alignment"][seed] = {
            name: r["results"]["50"][name] for name in ("alignment", "largest_load_quartile_alignment")
        }
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    completion = {key: summary[key] for key in (
        "revision", "training_seeds", "evaluation_episodes", "optimizer_steps_taken", "checkpoint_transfers",
        "downloaded_result_json_bytes", "maximum_absolute_reproduction_error", "new_training_submitted")}
    completion["status"] = "completed"
    (destination / "completion.json").write_text(json.dumps(completion, indent=2) + "\n")
    print(json.dumps(completion, indent=2))
    for mode, metrics in summary["mean_cost_decomposition"]["50"].items():
        print(mode, metrics)


if __name__ == "__main__":
    main()
