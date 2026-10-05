"""Validate and collect per-seed first-action diagnostics without training admission."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scripts.diagnose_nonlinear_load_v21 import MODES, RADII, REVISION, SEVERITIES, summarize_block

ROOT = Path(__file__).resolve().parents[1]


def validate(data, manifest, task, preflight=False):
    assert (data["revision"], data["source_revision"], data["source_task_id"], data["training_seed"],
            data["preflight_only"], data["checkpoint_selection"]) == (
        REVISION, manifest["source_revision"], manifest["source_tasks"][str(task["seed"])]["id"], task["seed"], preflight, "last")
    assert data["config"] == manifest["config"] and data["environment_settings"] == manifest["environment_settings"]
    assert data["parameters_unchanged"] and data["optimizer_updates"] == data["fitting_interactions"] == 0
    assert data["gamma"] == .99 and data["runtime"]["device"] == "cuda"
    for package, version in manifest["runtime_requirement"].items():
        if package == "python_minor_versions":
            assert '.'.join(data["runtime"]["python"].split()[0].split('.')[:2]) in version
        else:
            assert data["runtime"][package] == version
    count, radii, loads = (2, (1.,), (50,)) if preflight else (100, RADII, SEVERITIES)
    assert data["controller_episodes"] == len(radii) * len(loads) * count * 2
    assert set(data["radius_results"]) == {str(r) for r in radii}
    for radius in radii:
        for load in loads:
            b = data["radius_results"][str(radius)][str(load)]
            assert b["evaluation_seeds"] == list(range(60000, 60000 + count)) and b["disturbance_paths_matched"]
            totals, scores = {}, {}
            for mode in MODES:
                totals[mode] = {k: np.asarray(v) for k, v in b["continuations"][mode].items()}
                scores[mode] = {k: np.asarray(v) for k, v in b["initial_scores"][mode].items()}
                assert all(v.shape == (count,) and np.isfinite(v).all() for v in totals[mode].values())
                assert all(v.shape == ((count, 2) if k == "action" else (count,)) and np.isfinite(v).all()
                           for k, v in scores[mode].items())
                np.testing.assert_allclose(totals[mode]["total_cost"], totals[mode]["state_cost"] + totals[mode]["command_cost"])
                np.testing.assert_allclose(totals[mode]["state_cost"], totals[mode]["early_state_cost"] + totals[mode]["late_state_cost"])
            assert b["summary"] == summarize_block(totals, scores)


def main():
    study = ROOT / "experiments/nonlinear_load_v21_objective_20261003"
    result = ROOT / "results" / REVISION
    manifest = json.loads((study / "dispatch_manifest.json").read_text())
    observations = json.loads((result / "task_observations.json").read_text())
    assert {o["id"] for o in observations} == {t["id"] for t in manifest["tasks"]}
    assert all(o["status"] == "done" and o["exit_code"] == 0 for o in observations)
    assert {t["seed"] for t in manifest["tasks"]} == {15, 16} and len(manifest["tasks"]) == 2
    collected, paths = {}, []
    for task in manifest["tasks"]:
        path = result / f"seed{task['seed']}" / f"{REVISION}_seed{task['seed']}.json"
        data = json.loads(path.read_text())
        validate(data, manifest, task)
        paths.append(path)
        collected[str(task["seed"])] = {r: {s: b["summary"] for s, b in values.items()}
                                       for r, values in data["radius_results"].items()}
    output = {"revision": REVISION, "completed_task_ids": [t["id"] for t in manifest["tasks"]],
              "controller_episodes": 2400, "optimizer_updates": 0, "fitting_interactions": 0,
              "result_json_bytes": sum(p.stat().st_size for p in paths), "per_seed": collected,
              "historical_hold_unchanged": True, "diagnostic_only": True}
    (result / "summary.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print("Validated two frozen first-action diagnostics; no training-admission verdict inferred.")


if __name__ == "__main__":
    main()
