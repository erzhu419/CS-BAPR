"""Compact per-seed terminal gradient diagnosis; no performance gate changes."""
import argparse
import json
from pathlib import Path

REVISION = "nonlinear_load_v16_gradient_diagnostic"


def summarize(root, manifest):
    seeds = {}
    for seed in (9, 10):
        result = json.loads((root / f"seed{seed}" / f"{REVISION}_seed{seed}.json").read_text())
        assert result["revision"] == REVISION and result["training_seed"] == seed
        assert result["optimizer_steps_taken"] == result["new_training_interactions"] == 0
        assert result["parameters_unchanged"] and not result["preflight_only"]
        assert result["noise_seeds"] == list(range(92000, 92008))
        assert result["evaluation_seeds"] == list(range(60000, 60100))
        assert result["source_tasks"] == {c: manifest["source_tasks"][c][str(seed)]["id"] for c in ("narrow", "balanced")}
        assert len(result["reproduction_errors"]) == 12 and len(result["bank_support"]) == 20
        arms = {}
        for condition in ("narrow", "balanced"):
            gradients = result["gradients"][condition]
            assert set(gradients) == set(result["bank_support"])
            assert all(len(g["draws"]) == 8 for g in gradients.values())
            replay = {}
            for source in ("narrow", "balanced"):
                for phase in ("all", "early", "late", "recent"):
                    name = f"replay_{source}_{phase}"
                    g = gradients[name]
                    replay[name] = {
                        "mean_gradient_norms": g["mean_gradient_norms"],
                        "sac_vs_jac": g["mean_sac_vs_jac"], "total_vs_jac": g["mean_total_vs_jac"],
                        "conflicting_draws": sum(d["sac_vs_jac"]["dot"] < 0 for d in g["draws"]),
                        "initial_r1_weighted_physical_jac_derivatives": {
                            load: {term: value["right_loss_derivative_under_left_descent"]
                                   for term, value in result["replay_descent_vs_initial_physical_jacobian"]
                                   [condition][name][f"initial_r1.0_load{load}"].items()}
                            for load in ("1", "50")}}
            arms[condition] = {"terminal_objective": result["terminal_objective"][condition],
                               "replay_support": result["replay_support"][condition],
                               "initial_r1": {load: gradients[f"initial_r1.0_load{load}"] for load in ("1", "50")},
                               "replay": replay}
        seeds[str(seed)] = {"source_tasks": result["source_tasks"], "arms": arms,
                            "maximum_reproduction_error": max(result["reproduction_errors"].values())}
    return {"revision": REVISION, "training_seeds": [9, 10], "optimizer_steps_taken": 0,
            "historical_v13_v15_gate": "HOLD", "seeds": seeds,
            "both_seeds_own_replay_total_descent_increases_initial_r1_jac": {
                condition: {phase: {load: all(seeds[str(s)]["arms"][condition]["replay"]
                    [f"replay_{condition}_{phase}"]["initial_r1_weighted_physical_jac_derivatives"][load]["total"] > 0
                    for s in (9, 10)) for load in ("1", "50")} for phase in ("all", "early", "late", "recent")}
                for condition in ("narrow", "balanced")}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    summary = summarize(args.result_root, manifest)
    path = args.result_root / "summary.json"
    path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(path)
    print(json.dumps(summary["both_seeds_own_replay_total_descent_increases_initial_r1_jac"], indent=2))


if __name__ == "__main__":
    main()
