"""Frozen entropy-floor gates, keeping each fresh training seed separate."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scripts.summarize_nonlinear_load_v19 import validate
from scripts.train_nonlinear_load_v22 import INPUT_COLUMNS, MODES, RADII, REDUCTION, REVISION, SEVERITIES, TRAINING_SEEDS

ROOT = Path(__file__).resolve().parents[1]


def summarize(runs, seeds):
    differences, fitted_gates = {}, {}
    for radius in RADII:
        differences[str(radius)], fitted_gates[str(radius)] = {}, True
        for severity in SEVERITIES:
            values = []
            for seed in seeds:
                modified = runs[("no-floor", seed)]["radius_results"][str(radius)][str(severity)]
                baseline = runs[("floor", seed)]["radius_results"][str(radius)][str(severity)]
                actor, control = modified["controllers"]["actor"], baseline["controllers"]["actor"]
                excess = modified["paired_excess_costs"]["fitted"]["mean_total_cost"]
                fitted_gates[str(radius)] &= excess < 0
                values.append({"training_seed": seed, "modified_cost": actor["mean_cost"], "control_cost": control["mean_cost"],
                               "total_cost_change": actor["mean_cost"] - control["mean_cost"],
                               "early_state_cost_change": actor["mean_early_state_cost"] - control["mean_early_state_cost"],
                               "own_fitted_excess_cost": excess})
            differences[str(radius)][str(severity)] = values
    repair = all(v["total_cost_change"] < 0 and v["early_state_cost_change"] < 0
                 for values in differences["1.0"].values() for v in values)
    retention = all(v["total_cost_change"] <= 0 for values in differences["0.2"].values() for v in values)
    added_value = all(fitted_gates.values())
    return {"modified_minus_control": differences, "own_fitted_added_value_per_radius": fitted_gates,
            "decision": {"radius1_repair_passed": repair, "small_radius_retention_passed": retention,
                         "all_radius_own_fitted_added_value_passed": added_value,
                         "ready_for_learner_replication": repair and retention and added_value,
                         "historical_hold_unchanged": True}}


def validate_temperature(data):
    trace = data["temperature_trace"]
    updates = data["training_updates"]
    assert len(trace) == data["training_episodes"]
    assert [t["episode"] for t in trace] == list(range(1, len(trace) + 1))
    assert [t["training_updates"] for t in trace] == [max(0, (e + 1) * 200 - 999) for e in range(len(trace))]
    enabled = data["entropy_mode"] == "floor"
    assert data["config"]["enable_entropy_floor"] == enabled and data["config"]["entropy_floor"] == .01
    for t in trace:
        assert all(np.isfinite(t[k]) and t[k] > 0 for k in ("alpha", "raw_alpha", "reward_std"))
        assert 0 <= t["floor_binding_updates"] <= t["below_nominal_floor_updates"] <= t["training_updates"]
        if t["training_updates"]:
            expected = max(t["raw_alpha"], .01) if enabled else t["raw_alpha"]
            assert t["alpha"] == expected
        assert t["floor_binding_updates"] == (t["below_nominal_floor_updates"] if enabled else 0)
    assert 0 <= data["below_nominal_floor_updates"] <= updates
    assert data["floor_binding_updates"] == trace[-1]["floor_binding_updates"]
    assert data["below_nominal_floor_updates"] == trace[-1]["below_nominal_floor_updates"]
    assert all(a["below_nominal_floor_updates"] <= b["below_nominal_floor_updates"] and
               a["floor_binding_updates"] <= b["floor_binding_updates"] for a, b in zip(trace, trace[1:]))


def compare_pre_floor_prefix(runs, seeds):
    comparisons = {}
    for seed in seeds:
        a, b = [runs[(mode, seed)] for mode in MODES]
        traces = [d["temperature_trace"] for d in (a, b)]
        count = min(next(t["episode"] for t in trace if t["below_nominal_floor_updates"] > 0)
                    for trace in traces) - 1
        fields = {"training_return": [d["training_returns"][:count] for d in (a, b)]}
        fields.update({name: [[t[name] for t in trace[:count]] for trace in traces]
                       for name in ("alpha", "raw_alpha", "reward_std")})
        differences = {}
        for name, (left, right) in fields.items():
            delta = np.asarray(right) - np.asarray(left)
            different = np.flatnonzero(delta)
            differences[name] = {"different_episodes": len(different),
                                 "first_different_episode": int(different[0] + 1) if len(different) else None,
                                 "max_abs_difference": float(np.max(np.abs(delta)))}
        comparisons[str(seed)] = {"episodes_before_either_floor_crossing": count,
                                  "floor_gpu": a["runtime"]["gpu"], "no_floor_gpu": b["runtime"]["gpu"],
                                  "recorded_episode_traces_match": all(v["different_episodes"] == 0
                                                                       for v in differences.values()),
                                  "differences": differences}
    return comparisons


def main():
    study = ROOT / "experiments/nonlinear_load_v22_entropy_floor_20261003"
    destination = ROOT / "results" / REVISION
    manifest = json.loads((study / "dispatch_manifest.json").read_text())
    tasks = manifest["tasks"]
    assert len(tasks) == len({t["id"] for t in tasks}) == 4
    assert {(t["mode"], t["seed"]) for t in tasks} == {(a, s) for a in MODES for s in TRAINING_SEEDS}
    observations = json.loads((destination / "task_observations.json").read_text())
    assert {o["id"] for o in observations} == {t["id"] for t in tasks}
    assert all(o["status"] == "done" and o["exit_code"] == 0 for o in observations)
    runs, paths, known = {}, [], {}
    for task in tasks:
        mode, seed = task["mode"], task["seed"]
        path = destination / mode / f"seed{seed}" / f"{REVISION}_{mode}_seed{seed}_evaluation.json"
        data = json.loads(path.read_text())
        validate(data, {**manifest, "config": manifest["configs"][mode]}, {**task, "reference": "cubic-local"}, revision=REVISION)
        validate_temperature(data)
        assert (data["jacobian_scope"], data["jacobian_input_columns"], data["jacobian_reduction"]) == (
            "full", INPUT_COLUMNS, REDUCTION)
        assert data["entropy_mode"] == mode
        runs[(mode, seed)] = data
        paths.append(path)
        for radius in RADII:
            for severity in SEVERITIES:
                for control_mode in ("known_local", "known_inverse"):
                    returns = data["radius_results"][str(radius)][str(severity)]["controllers"][control_mode]["returns"]
                    np.testing.assert_allclose(returns, known.setdefault((radius, severity, control_mode), returns), rtol=1e-6, atol=1e-7)
    for seed in TRAINING_SEEDS:
        a, b = [runs[(mode, seed)] for mode in MODES]
        left, right = a["reference_identification"], b["reference_identification"]
        for name in ("linear", "cubic-local"):
            np.testing.assert_allclose(left["gains"][name], right["gains"][name], rtol=1e-6, atol=1e-7)
        for model in ("linear", "cubic"):
            np.testing.assert_allclose(left["fits"][model]["coefficients"], right["fits"][model]["coefficients"], rtol=1e-6, atol=1e-8)
        for radius in RADII:
            for severity in SEVERITIES:
                left, right = [r["radius_results"][str(radius)][str(severity)]["controllers"]["fitted"]["returns"] for r in (a, b)]
                np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-7)
    output = {"revision": REVISION, "training_seeds": list(TRAINING_SEEDS), "completed_task_ids": [t["id"] for t in tasks],
              "completed_runs": len(runs), "formal_learner_interactions": 400000, "formal_training_updates": 396004,
              "formal_identification_interactions": 40000, "formal_validation_interactions": 4000,
              "formal_evaluation_episodes_including_controls": 9600, "evaluation_json_bytes": sum(p.stat().st_size for p in paths),
              "paired_known_controls_verified": True, "matched_identification_verified": True,
              "paired_fitted_reference_verified": True, "reference_is_optimal": False,
              "pre_floor_prefix_comparison": compare_pre_floor_prefix(runs, TRAINING_SEEDS),
              "temperature": {mode: {str(seed): {"terminal": runs[(mode, seed)]["temperature_trace"][-1],
                                                   "below_nominal_floor_updates": runs[(mode, seed)]["below_nominal_floor_updates"],
                                                   "floor_binding_updates": runs[(mode, seed)]["floor_binding_updates"]}
                                      for seed in TRAINING_SEEDS} for mode in MODES},
              **summarize(runs, TRAINING_SEEDS)}
    (destination / "summary.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(output["decision"], indent=2))


if __name__ == "__main__":
    main()
