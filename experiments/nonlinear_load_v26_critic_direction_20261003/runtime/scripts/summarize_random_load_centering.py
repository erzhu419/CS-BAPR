"""Aggregate the nine frozen, read-only actor-centering diagnoses."""
import json
import math
from pathlib import Path

from summarize_lqr_critic_shift_v4 import contrast, seed_statistics

ROOT = Path(__file__).resolve().parents[1]


def main():
    destination = ROOT / "results/random_load_v6_centering"
    manifest = json.loads((ROOT / "experiments/random_load_v6_centering_20261001/manifest.json").read_text())
    methods = list(dict.fromkeys(r["method"] for r in manifest["runs"]))
    summary = {"revision": manifest["revision"], "completed_runs": 9, "optimizer_steps_taken": 0,
               "checkpoint_transfers": 0, "severities": [1, 20, 50], "methods": {}}
    assert len(manifest["runs"]) == 9 and len(methods) == 3
    for method in methods:
        runs = [json.loads((destination / f"{method}_seed{s}.json").read_text()) for s in (0, 1, 2)]
        for seed, run in enumerate(runs):
            assert (run["method"], run["training_seed"], run["optimizer_steps_taken"]) == (method, seed, 0)
            for severity in summary["severities"]:
                for mode in ("original", "mean_centered", "reference"):
                    result = run[mode][str(severity)]
                    assert result["evaluation_seeds"] == manifest["evaluation_seeds"]
                    assert len(result["returns"]) == 20 and all(math.isfinite(x) for x in result["returns"])
        values = summary["methods"][method] = {
            "original_cost": {}, "centered_cost": {}, "reference_cost": {},
            "centered_minus_original_return": {}, "centered_trajectory_metrics": {},
            "origin_actions": [r["origin_action"] for r in runs],
            "reference_gain_max_abs_errors": [r["reference_gain_max_abs_error"] for r in runs],
            "maximum_absolute_reproduction_error": max(r["original"][str(a)]["maximum_absolute_reproduction_error"] for r in runs for a in (1, 20, 50)),
        }
        for severity in summary["severities"]:
            key = str(severity)
            for mode, target in (("original", "original_cost"), ("mean_centered", "centered_cost"), ("reference", "reference_cost")):
                values[target][key] = seed_statistics([r[mode][key]["mean_cost"] for r in runs])
            values["centered_minus_original_return"][key] = contrast(
                [-x for x in values["centered_cost"][key]["seed_values"]],
                [-x for x in values["original_cost"][key]["seed_values"]],
            )
            metrics = [r["mean_centered"][key] for r in runs]
            values["centered_trajectory_metrics"][key] = {
                name: seed_statistics([m[name] for m in metrics])
                for name in ("mean_saturated_component_fraction", "physical_state_rms", "last_50_physical_state_rms",
                             "mean_final_position_norm", "mean_final_velocity_norm")
            }
            values["centered_trajectory_metrics"][key]["alignment"] = {
                name: seed_statistics([m["trajectory_alignment"][name] for m in metrics])
                for name in metrics[0]["trajectory_alignment"] if name != "n_states"
            }
        values["centering_improves_id_and_50_all_seeds"] = all(
            values["centered_minus_original_return"][str(a)]["matched_training_seed_wins"] == 3 for a in (1, 50)
        )
        print(method, {a: round(values["centered_cost"][a]["mean"], 5) for a in ("1", "20", "50")})
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
