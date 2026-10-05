"""Verify the frozen symbolic-loss ablation and compare the reused v9 arms."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean, median

from summarize_lqr_critic_shift_v4 import contrast
from summarize_nonlinear_load_v9 import validate_results
from summarize_random_load_equilibrium_v7 import summarize_runs

ROOT = Path(__file__).resolve().parents[1]
REVISION = "nonlinear_load_v10_ablation"
FULL = "csbapr-centered-mlp-no-target-shift"
CRITIC = "csbapr-centered-mlp-critic-only-no-target-shift"
JACOBIAN = "csbapr-centered-mlp-jacobian-only-no-target-shift"


def gate_passes(differences):
    return all(differences[str(a)]["matched_training_seed_wins"] == 2 and
               differences[str(a)]["mean"] > 0 for a in (1, 50))


def paired_diagnostics(learner, reference, seeds):
    gaps = [x - y for x, y in zip(learner, reference)]
    assert len(gaps) == len(seeds) == 100
    return {
        "mean": mean(gaps), "median": median(gaps), "path_wins": sum(x > 0 for x in gaps),
        "return_gaps": gaps,
        "worst_five_paths": [{"evaluation_seed": seeds[i], "return_gap": gaps[i]}
                             for i in sorted(range(100), key=lambda i: gaps[i])[:5]],
    }


def main():
    destination = ROOT / "results" / REVISION
    manifest = json.loads((ROOT / "experiments/nonlinear_load_v10_ablation_20261002/dispatch_manifest.json").read_text())
    observations = json.loads((destination / "task_observations.json").read_text())
    prior = json.loads((ROOT / "results/nonlinear_load_v9_training/summary.json").read_text())
    seeds, severities, tasks = manifest["training_seeds"], manifest["severities"], manifest["tasks"]
    assert seeds == [5, 6] and severities == [1, 2, 5, 10, 20, 50]
    evaluation_seeds = list(range(50000, 50100))
    assert manifest["evaluation_seed_start"] == 50000 and manifest["evaluation_episodes_per_severity"] == 100
    configs = {a["method"]: a["config"] for a in manifest["arms"]}
    assert set(configs) == {CRITIC, JACOBIAN}
    assert len(tasks) == len({t["id"] for t in tasks}) == 4
    assert {(t["method"], t["seed"]) for t in tasks} == {(m, s) for m in configs for s in seeds}
    assert len(observations) == 4 and {t["id"] for t in tasks} == {o["id"] for o in observations}
    assert all(o["status"] == "done" and o["exit_code"] == 0 for o in observations)
    assert manifest["environment_settings"] == prior["environment_settings"]
    assert prior["training_seeds"] == seeds and prior["evaluation_seeds"] == evaluation_seeds
    assert prior["severities"] == severities
    base_runs = {s: json.loads((ROOT / "results/nonlinear_load_v9_training" / FULL / f"seed{s}" /
                               f"nonlinear_load_v9_training_{FULL}_seed{s}_evaluation.json").read_text()) for s in seeds}
    runs, rows, paths = {method: {} for method in configs}, [], []
    for task in tasks:
        method, seed = task["method"], task["seed"]
        path = destination / method / f"seed{seed}" / f"{REVISION}_{method}_seed{seed}_evaluation.json"
        data = json.loads(path.read_text())
        assert (data["revision"], data["method"], data["training_seed"], data["training_episodes"],
                data["environment"], data["checkpoint_selection"]) == (REVISION, method, seed, 500, "nonlinear_load", "last"), path
        assert (data["environment_steps"], data["training_updates"]) == (100000, 99001), path
        assert len(data["training_returns"]) == 500 and all(map(math.isfinite, data["training_returns"])), path
        assert data["config"] == configs[method] and data["environment_settings"] == manifest["environment_settings"], path
        removed = "jac_weight" if method == CRITIC else "weight_sym"
        assert data["config"] == {**base_runs[seed]["config"], removed: 0.0}, path
        assert data["actor_type"] == "centered-tanh-mlp" and data["sindy"] is not None, path
        assert data["runtime"]["device"] == "cuda", path
        for package, expected in manifest["runtime_requirement"].items():
            if package != "python_minor_versions":
                assert data["runtime"][package] == expected, path
        minor = ".".join(data["runtime"]["python"].split()[0].split(".")[:2])
        assert minor in manifest["runtime_requirement"]["python_minor_versions"], path
        alignment = data["policy_alignment"]
        assert alignment["reference"] == "fitted_feedback" and alignment["origin_action"] == [0.0, 0.0], path
        assert set(alignment["initial_states"]) == {str(a) for a in severities}, path
        assert all(v["n_states"] == 100 for v in alignment["initial_states"].values()), path
        validate_results(data["amplitude_results"], severities, evaluation_seeds, path)
        assert set(data["feedback_controls"]) == {"fitted_model", "known_local_linear", "known_inverse"}, path
        for name, control in data["feedback_controls"].items():
            previous = base_runs[seed]["feedback_controls"][name]
            gain = "physical_gain" if name == "known_inverse" else "gain"
            assert control[gain] == previous[gain], path
            validate_results(control["amplitude_results"], severities, evaluation_seeds, path)
            for key, result in control["amplitude_results"].items():
                assert all(math.isclose(x, y, rel_tol=1e-6, abs_tol=1e-7)
                           for x, y in zip(result["returns"], previous["amplitude_results"][key]["returns"])), path
        runs[method][seed] = data
        paths.append(path)
        rows.extend((task["id"], method, seed, a, data["amplitude_results"][str(a)]["mean"]) for a in severities)

    summary = {
        "revision": REVISION, "completed_new_runs": 4, "checkpoint_selection": "last",
        "training_seeds": seeds, "evaluation_seeds": evaluation_seeds, "severities": severities,
        "methods": {m: summarize_runs([runs[m][s] for s in seeds], severities) for m in configs},
        "reused_v9_methods": prior["methods"], "contrasts": {}, "replication_gates": {},
        "added_rl_value_gates": {}, "paired_own_fitted_diagnostics": {},
        "new_final_evaluation_json_bytes": sum(p.stat().st_size for p in paths),
        "formal_new_evaluation_episodes_including_controls": 9600,
        "statistics_unit": "two independent training-seed means, each averaging 100 paired paths",
        "fitted_gains_and_paired_controls_match_v9": True,
        "historical_v9_added_value_gate_changed": False,
    }
    assert manifest["replication_criterion"] == {
        "comparators": ["csbapr-centered-mlp-no-sindy-no-target-shift", "bapr-no-target-shift", "resac-no-target-shift"],
        "severities": [1, 50], "lower_mean_cost": True, "required_matched_seed_wins": 2,
    }
    assert manifest["learner_added_value_criterion"] == {
        "comparator": "own_fitted_model_feedback", "severities": [1, 50], "required_matched_seed_wins": 2,
    }
    for method in configs:
        values = summary["methods"][method]["returns"]
        contrasts = summary["contrasts"][method] = {}
        for other in [*prior["methods"], "own_fitted_feedback"]:
            contrasts[other] = {
                str(a): contrast(values[str(a)]["seed_values"],
                                 [runs[method][s]["feedback_controls"]["fitted_model"]["amplitude_results"][str(a)]["mean"] for s in seeds]
                                 if other == "own_fitted_feedback" else prior["methods"][other]["returns"][str(a)]["seed_values"])
                for a in severities
            }
        summary["replication_gates"][method] = {
            other: gate_passes(contrasts[other]) for other in manifest["replication_criterion"]["comparators"]
        }
        summary["added_rl_value_gates"][method] = gate_passes(contrasts["own_fitted_feedback"])
        summary["paired_own_fitted_diagnostics"][method] = {
            str(a): [{"training_seed": s, **paired_diagnostics(
                runs[method][s]["amplitude_results"][str(a)]["returns"],
                runs[method][s]["feedback_controls"]["fitted_model"]["amplitude_results"][str(a)]["returns"],
                evaluation_seeds)} for s in seeds] for a in severities
        }
    summary["decision"] = {
        "selected_candidate": JACOBIAN,
        "critic_symbolic_penalty_required_for_current_performance": False,
        "jacobian_only_passes_learner_comparison": all(summary["replication_gates"][JACOBIAN].values()),
        "jacobian_only_passes_added_rl_value": summary["added_rl_value_gates"][JACOBIAN],
        "next_step": "Freeze Jacobian-only and confirm on new training seeds against no-symbolic/BAPR/RE-SAC before changing the environment or reference",
        "new_training_submitted": False,
    }
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    with (destination / "seed_mean_returns.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("task_id", "method", "training_seed", "load_severity", "mean_return"))
        writer.writerows(rows)
    print("4 complete runs; 24 seed/severity means; frozen configs, budgets and paired controls verified")
    for method, values in summary["methods"].items():
        print(method, "ID cost", -values["returns"]["1"]["mean"], "50x cost", -values["returns"]["50"]["mean"])
        print("learner-comparison gates", summary["replication_gates"][method], "added-RL-value", summary["added_rl_value_gates"][method])
        print("actor minus fitted 50x", summary["contrasts"][method]["own_fitted_feedback"]["50"])


if __name__ == "__main__":
    main()
