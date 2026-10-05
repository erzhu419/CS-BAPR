"""Summarize the frozen identification diagnostic without pooling seeds."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scripts.diagnose_nonlinear_load_v18 import RADII, REVISION, SEVERITIES, TRAINING_SEEDS, decide

ROOT = Path(__file__).resolve().parents[1]


def main():
    study = ROOT / "experiments/nonlinear_load_v18_identification_20261002"
    results = ROOT / "results" / REVISION
    manifest = json.loads((study / "dispatch_manifest.json").read_text())
    tasks = manifest["tasks"]
    assert len(tasks) == len({t["id"] for t in tasks}) == 2
    assert {t["seed"] for t in tasks} == set(TRAINING_SEEDS)
    observations = json.loads((results / "task_observations.json").read_text())
    assert {t["id"] for t in observations} == {t["id"] for t in tasks}
    assert all(t["status"] == "done" and t["exit_code"] == 0 for t in observations)
    runs, paths = [], []
    for task in tasks:
        path = results / f"seed{task['seed']}" / f"{REVISION}_seed{task['seed']}_diagnostic.json"
        run = json.loads(path.read_text())
        assert run["revision"] == REVISION and run["training_seed"] == task["seed"]
        assert run["source_revision"] == manifest["source_revision"]
        assert run["config"] == manifest["config"] and run["environment_settings"] == manifest["environment_settings"]
        assert run["original_fit_reproduced"] and run["existing_controls_reproduced"] and run["device"] == "cuda"
        assert (run["identification_interactions"], run["original_validation_interactions"],
                run["independent_validation_interactions"], run["evaluation_episodes_including_controls"],
                run["optimizer_updates"], run["new_actor_training_interactions"]) == (10000, 1000, 2000, 2400, 0, 0)
        for radius in RADII:
            for severity in SEVERITIES:
                block = run["radius_results"][str(radius)][str(severity)]
                for controller in block["controllers"].values():
                    assert controller["evaluation_seeds"] == list(range(60000, 60100))
                    returns = np.asarray(controller["returns"])
                    assert returns.shape == (100,) and np.isfinite(returns).all()
                    np.testing.assert_allclose(-returns, np.asarray(controller["state_costs"]) + controller["command_costs"], rtol=1e-12, atol=1e-12)
                expected = np.asarray(block["controllers"]["linear"]["returns"]) - block["controllers"]["cubic"]["returns"]
                np.testing.assert_allclose([p["total_cost"] for p in block["cubic_minus_linear"]["per_path"]], expected)
        assert run["decision"] == decide([run])
        runs.append(run)
        paths.append(path)
    output = {"revision": REVISION, "completed_task_ids": [t["id"] for t in tasks],
              "completed_runs": 2, "optimizer_updates": 0, "new_actor_training_interactions": 0,
              "identification_interactions": 20000, "original_validation_interactions": 2000,
              "independent_validation_interactions": 4000, "evaluation_episodes_including_controls": 4800,
              "result_json_bytes": sum(p.stat().st_size for p in paths), "decision": decide(runs),
              "per_seed": [{"training_seed": run["training_seed"], "validation": run["validation"],
                            "fitted_load_gain_errors": {k: v["parameter_errors"]["load_gain_max_abs_error"] for k, v in run["fits"].items()},
                            "direct_feedback_changes": {str(r): {str(s): run["radius_results"][str(r)][str(s)]["cubic_minus_linear"]["mean_total_cost"]
                                                                   for s in SEVERITIES} for r in RADII}} for run in runs]}
    (results / "summary.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(output["decision"], indent=2))


if __name__ == "__main__":
    main()
