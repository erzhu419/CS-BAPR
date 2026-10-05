"""Matched-data actuator identification, without learner updates or checkpoints."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pysindy
import scipy
from scipy.linalg import solve_discrete_are
import sklearn
import torch

from csbapr.envs.lqr_env import A, B, ACTION_SCALE, EP_LEN, Q, R
from csbapr.envs.nonlinear_load_env import NonlinearLoadEnv
from csbapr.sindy.data_collector import collect_trajectories
from csbapr.sindy.load_feedback import fitted_load_reference
from csbapr.sindy.world_model import SymbolicWorldModel
from scripts.diagnose_nonlinear_load_v14 import paired_excess_costs, rollout

REVISION = "nonlinear_load_v18_identification"
RADII, SEVERITIES, TRAINING_SEEDS = (.2, .5, 1.), (1, 50), (11, 12)
GAIN_TOLERANCE = .001


def augmented_actions(actions):
    return np.hstack((actions, actions ** 3))


def fit_models(states, actions):
    models = {}
    for name, n_control in (("linear", 2), ("cubic", 4)):
        model = SymbolicWorldModel(6, n_control, poly_degree=1, threshold=.001, discrete_time=True)
        controls = actions if name == "linear" else [augmented_actions(u) for u in actions]
        model.fit(states, U=controls, multiple_trajectories=True, t=1.)
        models[name] = model
    return models


def physical_matrices(model):
    powers = model.library.powers_
    physical = model.coeffs[:4]
    assert np.max(np.abs(physical[:, powers.sum(axis=1) == 0])) <= 1e-6
    linear = physical[:, powers.sum(axis=1) == 1]
    cubic = linear[:, 8:10] if model.n_control == 4 else np.zeros((4, 2))
    return np.eye(4) + linear[:, :4], linear[:, 4:6], linear[:, 6:8], cubic


def local_gain(model):
    dynamics, load, control, _ = physical_matrices(model)
    action_cost = R * ACTION_SCALE ** 2
    value = solve_discrete_are(dynamics, control, Q, action_cost)
    physical = np.linalg.solve(action_cost + control.T @ value @ control,
                               control.T @ value @ dynamics)
    return np.hstack((physical, np.linalg.lstsq(control, load, rcond=None)[0]))


def physical_metrics(model, states, actions, targets):
    controls = augmented_actions(actions) if model.n_control == 4 else actions
    truth = targets[:, :4]
    residual = model.predict(states, controls)[:, :4] - truth
    return {"physical_rmse": float(np.sqrt(np.mean(residual ** 2))),
            "physical_rmse_by_output": np.sqrt(np.mean(residual ** 2, axis=0)).tolist(),
            "physical_r_squared": float(1 - np.sum(residual ** 2) /
                                         max(np.sum((truth - truth.mean(axis=0)) ** 2), 1e-10))}


def fit_report(model, gain):
    dynamics, load, control, cubic = physical_matrices(model)
    expected_load_gain = np.eye(2) / (ACTION_SCALE * .5)
    errors = {"dynamics_max_abs_error": float(np.max(np.abs(dynamics - A))),
              "load_matrix_max_abs_error": float(np.max(np.abs(load - B))),
              "linear_action_matrix_max_abs_error": float(np.max(np.abs(control - 2.5 * B))),
              "cubic_action_matrix_max_abs_error": float(np.max(np.abs(cubic - 2.5 * B))),
              "load_gain_max_abs_error": float(np.max(np.abs(gain[:, 4:] - expected_load_gain)))}
    return {"gain": gain.tolist(), "dynamics": dynamics.tolist(), "load_matrix": load.tolist(),
            "linear_action_matrix": control.tolist(), "cubic_action_matrix": cubic.tolist(),
            "parameter_errors": errors,
            "feature_names": model.get_feature_names(), "coefficients": model.coeffs.tolist()}


def decide(runs):
    validation = all(run["validation"][str(load)]["cubic"]["physical_rmse"] <
                     run["validation"][str(load)]["linear"]["physical_rmse"]
                     for run in runs for load in SEVERITIES)
    gains = all(run["fits"]["cubic"]["parameter_errors"]["load_gain_max_abs_error"] <= GAIN_TOLERANCE
                for run in runs)
    feedback = all(run["radius_results"][str(radius)][str(load)]["cubic_minus_linear"]["mean_total_cost"] < 0
                   for run in runs for radius in RADII for load in SEVERITIES)
    return {"physical_validation_improved_each_seed_and_load": validation,
            "local_load_gain_recovered_each_seed": gains,
            "direct_feedback_improved_all_blocks": feedback,
            "ready_for_matched_actor_experiment": validation and gains and feedback,
            "historical_v13_v15_v17_hold_unchanged": True}


def diagnose(source, device):
    seed = source["training_seed"]
    env = NonlinearLoadEnv(severity=1, seed=seed)
    env.action_space.seed(seed)
    assert env.specification() == source["environment_settings"]
    states, actions, _ = collect_trajectories(env, n_episodes=50, max_steps=EP_LEN)
    models = fit_models(states, actions)
    gains = {"linear": fitted_load_reference(models["linear"]).gain.numpy(),
             "cubic": local_gain(models["cubic"])}
    np.testing.assert_allclose(local_gain(models["linear"]), gains["linear"], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(gains["linear"], source["feedback_gains"]["fitted"], rtol=0, atol=1e-6)
    validation_states, validation_actions, _ = collect_trajectories(env, n_episodes=5, max_steps=EP_LEN)
    x, u, y = models["linear"].training_arrays(validation_states, validation_actions, multiple_trajectories=True)
    prediction = models["linear"].predict(x, u)
    original_r_squared = float(1 - np.sum((y - prediction) ** 2) /
                               max(np.sum((y - y.mean(axis=0)) ** 2), 1e-10))
    assert np.isclose(original_r_squared, source["sindy"]["r_squared"], rtol=0, atol=1e-10)
    validation = {}
    for severity in SEVERITIES:
        validation_seed = (71000 if severity == 1 else 72000) + seed
        independent = NonlinearLoadEnv(severity=severity, seed=validation_seed)
        independent.action_space.seed(validation_seed)
        xv, uv, _ = collect_trajectories(independent, n_episodes=5, max_steps=EP_LEN)
        x, u, y = models["linear"].training_arrays(xv, uv, multiple_trajectories=True)
        validation[str(severity)] = {name: physical_metrics(model, x, u, y) for name, model in models.items()}
        print(f"Validation load={severity}: " + str({k: v["physical_rmse"] for k, v in validation[str(severity)].items()}), flush=True)

    # The existing rollout helper uses a module only to select the tensor device.
    device_anchor = torch.nn.Linear(6, 2, bias=False).to(device).requires_grad_(False)
    seeds = list(range(60000, 60100))
    results = {}
    for radius in RADII:
        results[str(radius)] = {}
        for severity in SEVERITIES:
            controllers = {}
            for name in ("linear", "cubic", "known_local", "known_inverse"):
                gain = (gains[name] if name in gains else source["feedback_gains"]["known_local"])
                tensors = {"fitted": torch.as_tensor(gain, dtype=torch.float32, device=device)}
                mode = "known_inverse" if name == "known_inverse" else "fitted"
                controllers[name], _ = rollout(device_anchor, tensors, mode, radius, severity, seeds)
                if name != "cubic":
                    original = "fitted" if name == "linear" else name
                    expected = source["radius_results"][str(radius)][str(severity)]["controllers"][original]
                    np.testing.assert_allclose(controllers[name]["returns"], expected["returns"], rtol=1e-6, atol=1e-7)
            difference = paired_excess_costs(controllers["cubic"], controllers["linear"], seeds)
            results[str(radius)][str(severity)] = {"controllers": controllers, "cubic_minus_linear": difference}
            print(f"Feedback radius={radius} load={severity}: delta={difference['mean_total_cost']:+.6f}", flush=True)
    return {"training_seed": seed, "original_fit_reproduced": True,
            "original_full_state_r_squared": original_r_squared,
            "fits": {name: fit_report(model, gains[name]) for name, model in models.items()},
            "validation": validation, "radius_results": results,
            "identification_interactions": 10000, "original_validation_interactions": 1000,
            "independent_validation_interactions": 2000, "evaluation_episodes_including_controls": 2400,
            "optimizer_updates": 0, "new_actor_training_interactions": 0,
            "existing_controls_reproduced": True, "reference_is_optimal": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=TRAINING_SEEDS, required=True)
    parser.add_argument("--source-evaluation", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    manifest = json.loads((Path(__file__).resolve().parents[2] / "dispatch_manifest.json").read_text())
    source = json.loads(args.source_evaluation.read_text())
    assert manifest["revision"] == REVISION and source["revision"] == manifest["source_revision"]
    assert source["training_seed"] == args.seed and source["sampling"] == "replay"
    assert source["method"] == manifest["method"] and source["config"] == manifest["config"]
    assert source["environment_settings"] == manifest["environment_settings"]
    assert (manifest["training_seeds"], manifest["radii"], manifest["severities"],
            manifest["load_gain_tolerance"], manifest["evaluation_seed_start"],
            manifest["evaluation_episodes_per_severity"]) == (list(TRAINING_SEEDS), list(RADII), list(SEVERITIES), GAIN_TOLERANCE, 60000, 100)
    versions = {"torch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
                "pysindy": pysindy.__version__, "sklearn": sklearn.__version__}
    for package, version in manifest["runtime_requirement"].items():
        if package != "python_minor_versions":
            assert versions[package] == version
    assert f"{sys.version_info.major}.{sys.version_info.minor}" in manifest["runtime_requirement"]["python_minor_versions"]
    output = {"revision": REVISION, "source_revision": source["revision"], "method": source["method"],
              "config": source["config"], "environment_settings": source["environment_settings"],
              "independent_validation_seeds": {"1": 71000 + args.seed, "50": 72000 + args.seed},
              "radii": list(RADII), "severities": list(SEVERITIES), "evaluation_seeds": list(range(60000, 60100)),
              "device": args.device, "runtime": {**versions, "python": sys.version},
              **diagnose(source, args.device)}
    output["decision"] = decide([output])
    args.result_dir.mkdir(parents=True, exist_ok=True)
    path = args.result_dir / f"{REVISION}_seed{args.seed}_diagnostic.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)
    print(f"Eval complete: {path}; optimizer updates=0", flush=True)


if __name__ == "__main__":
    main()
