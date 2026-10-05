"""Frozen non-learning nonlinear-load pilot; no RL updates or tuning."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.optimize import brentq
import torch
from tqdm import tqdm

from csbapr.envs.lqr_env import A, B, Q, R, ACTION_SCALE, EP_LEN, get_optimal_gain
from csbapr.envs.nonlinear_load_env import NonlinearLoadEnv, LINEAR_SHARE, actuator_force
from csbapr.envs.random_load_env import SEVERITIES
from csbapr.sindy.data_collector import collect_trajectories
from csbapr.sindy.load_feedback import fitted_load_reference
from csbapr.sindy.world_model import SymbolicWorldModel

FIT_SEEDS = (91000, 92000)


def inverse_force(force):
    return np.array([brentq(lambda u: actuator_force(u) - f, -1.0, 1.0) for f in force])


def local_linear_gain():
    control = B * ACTION_SCALE * LINEAR_SHARE
    action_cost = R * ACTION_SCALE ** 2
    p = solve_discrete_are(A, control, Q, action_cost)
    gain = np.linalg.solve(action_cost + control.T @ p @ control, control.T @ p @ A)
    return np.hstack((gain, np.eye(2) / (ACTION_SCALE * LINEAR_SHARE)))


def fit_reference(seed):
    env = NonlinearLoadEnv(1, seed)
    env.action_space.seed(seed)
    x, u, _ = collect_trajectories(env, n_episodes=50, max_steps=EP_LEN)
    model = SymbolicWorldModel(6, 2, poly_degree=1, threshold=0.001, discrete_time=True)
    model.fit(x, U=u, multiple_trajectories=True)
    x, u, _ = collect_trajectories(env, n_episodes=5, max_steps=EP_LEN)
    states, controls, targets = model.training_arrays(x, u, multiple_trajectories=True)
    error = model.predict(states, controls) - targets
    physical_error, physical_target = error[:, :4], targets[:, :4]
    report = {
        "identification_seed": seed, "identification_interactions": 10000,
        "validation_interactions": 1000, "coefficients": model.coeffs.tolist(),
        "library_powers": model.library.powers_.tolist(),
        "validation_physical_increment_rmse": float(np.sqrt(np.mean(physical_error ** 2))),
        "validation_physical_increment_r_squared": float(1 - np.sum(physical_error ** 2) /
                                                          np.sum((physical_target - physical_target.mean(0)) ** 2)),
        "physical_intercept": model.coeffs[:4, model.library.powers_.sum(1) == 0].reshape(-1).tolist(),
        "reference_gain": None, "reference_builder_error": None,
    }
    try:
        report["reference_gain"] = fitted_load_reference(model).gain.numpy().tolist()
    except (ValueError, np.linalg.LinAlgError) as error:
        report["reference_builder_error"] = str(error)
    return report


def action_for(policy, observation, local_gain, fitted_gain=None):
    if policy == "zero":
        return np.zeros(2)
    if policy == "inverse_force_cancel_only":
        return inverse_force(-observation[4:])
    if policy == "local_linear_feedback":
        return np.tanh(-local_gain @ observation)
    if policy.startswith("linear_fitted_"):
        return np.tanh(-fitted_gain @ observation)
    feedback = -get_optimal_gain() @ observation[:4]
    if policy == "inverse_feedback":
        feedback -= observation[4:] / ACTION_SCALE
    return inverse_force(ACTION_SCALE * np.tanh(feedback))


def rollout(policy, severity, seed, local_gain, fitted_gain=None):
    env = NonlinearLoadEnv(severity, seed)
    observation, _ = env.reset()
    total, saturated, maximum, physical_square, force_square, storms = 0.0, 0, 0.0, 0.0, 0.0, 0
    for step in range(EP_LEN):
        action = action_for(policy, observation, local_gain, fitted_gain)
        assert np.isfinite(action).all() and np.max(np.abs(action)) <= 1.0
        norm = float(np.linalg.norm(env.state))
        maximum = max(maximum, norm)
        physical_square += norm ** 2
        saturated += int(np.sum(np.abs(action) > 0.95))
        observation, reward, terminated, truncated, info = env.step(action)
        assert not truncated and terminated == (step == EP_LEN - 1)
        assert np.isfinite(observation).all() and np.isfinite(reward)
        total += reward
        force_square += float(np.mean(info["applied_disturbance"] ** 2))
        storms += info["storm"]
    return {
        "return": total, "maximum_physical_state_norm": max(maximum, float(np.linalg.norm(env.state))),
        "saturated_component_fraction": saturated / (2 * EP_LEN),
        "physical_state_rms": np.sqrt(physical_square / EP_LEN),
        "force_rms": np.sqrt(force_square / EP_LEN), "storm_fraction": storms / EP_LEN,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-eval", type=int, default=100)
    parser.add_argument("--evaluation-seed-start", type=int, default=50000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    seeds = list(range(args.evaluation_seed_start, args.evaluation_seed_start + args.n_eval))
    fits = [fit_reference(seed) for seed in FIT_SEEDS]
    gain = local_linear_gain()
    policies = {name: None for name in ("zero", "inverse_force_cancel_only", "local_linear_feedback",
                                       "inverse_feedback_no_force", "inverse_feedback")}
    for fit in fits:
        if fit["reference_gain"] is not None:
            policies[f"linear_fitted_{fit['identification_seed']}"] = np.asarray(fit["reference_gain"])
    results = {}
    progress = tqdm(total=len(SEVERITIES) * len(policies) * len(seeds), desc="non-learning control episodes")
    for severity in SEVERITIES:
        values = results[str(severity)] = {}
        for policy, fitted_gain in policies.items():
            episodes = []
            for seed in seeds:
                episodes.append(rollout(policy, severity, seed, gain, fitted_gain))
                progress.update()
            values[policy] = {
                "mean_cost": -mean(r["return"] for r in episodes),
                "returns": [r["return"] for r in episodes],
                "maximum_physical_state_norm": max(r["maximum_physical_state_norm"] for r in episodes),
                **{f"mean_{key}": mean(r[key] for r in episodes) for key in (
                    "saturated_component_fraction", "physical_state_rms", "force_rms", "storm_fraction")},
            }
        print("severity", severity, {k: round(v["mean_cost"], 6) for k, v in values.items()}, flush=True)
    progress.close()
    checks = {
        "inverse_reference_beats_zero_all_severities": all(v["inverse_feedback"]["mean_cost"] < v["zero"]["mean_cost"] for v in results.values()),
        "inverse_reference_50x_cost_below_fifth_of_zero": results["50"]["inverse_feedback"]["mean_cost"] < 0.2 * results["50"]["zero"]["mean_cost"],
        "inverse_reference_stays_in_unit_physical_ball": all(v["inverse_feedback"]["maximum_physical_state_norm"] <= 1.0 for v in results.values()),
        "inverse_reference_saturation_below_ten_percent": all(v["inverse_feedback"]["mean_saturated_component_fraction"] < 0.1 for v in results.values()),
        "both_existing_linear_reference_builds_succeed": all(f["reference_gain"] is not None for f in fits),
    }
    report = {
        "revision": "nonlinear_load_v9_pilot", "environment_settings": NonlinearLoadEnv.specification(),
        "evaluation_seeds": seeds, "identification_fits": fits, "local_linear_gain": gain.tolist(),
        "inverse_reference_gain": get_optimal_gain().tolist(), "results": results, "checks": checks,
        "controllability_passed": all(v for k, v in checks.items() if k != "both_existing_linear_reference_builds_succeed"),
        "existing_symbolic_interface_passed": checks["both_existing_linear_reference_builds_succeed"],
        "reference_is_optimal": False, "rl_training_runs": 0, "optimizer_updates": 0,
        "identification_interactions": 20000, "validation_interactions": 2000,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("checks", checks, flush=True)
    print("Eval complete: nonlinear-load pilot; RL training runs=0, optimizer updates=0", flush=True)


if __name__ == "__main__":
    main()
