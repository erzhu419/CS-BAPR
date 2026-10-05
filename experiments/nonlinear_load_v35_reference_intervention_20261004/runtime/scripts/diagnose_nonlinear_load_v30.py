"""Qualify sustained load without RL training, fitting or checkpoints."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import scipy
from tqdm import tqdm

from csbapr.envs.lqr_env import ACTION_SCALE, EP_LEN, Q, R, get_optimal_gain
from csbapr.envs.nonlinear_load_env import NonlinearLoadEnv, inverse_force, local_linear_gain
from csbapr.envs.sustained_load_env import SustainedLoadEnv

REVISION = "nonlinear_load_v30_environment_qualification"
RADII = (0.2, 0.5, 1.0)
SEVERITIES = (1, 50)
PROFILES = {"original": NonlinearLoadEnv, "sustained": SustainedLoadEnv}
CONTROLLERS = ("zero", "known_local", "known_inverse", "no_load_feedback", "force_cancel_feedback")
WINDOW = 50
SEED_START = 61000


def control_actions(mode, observations):
    if mode == "zero":
        return np.zeros((len(observations), 2))
    if mode == "known_local":
        return np.tanh(-observations @ local_linear_gain().T)
    feedback = -observations[:, :4] @ get_optimal_gain().T
    if mode == "force_cancel_feedback":
        desired_force = np.clip(ACTION_SCALE * feedback - observations[:, 4:], -ACTION_SCALE, ACTION_SCALE)
    else:
        if mode == "known_inverse":
            feedback -= observations[:, 4:] / ACTION_SCALE
        else:
            assert mode == "no_load_feedback"
        desired_force = ACTION_SCALE * np.tanh(feedback)
    return np.stack([inverse_force(force) for force in desired_force])


def rollout(profile, mode, radius, severity, seeds, *, action_fn=None):
    envs = [PROFILES[profile](severity, seed) for seed in seeds]
    for env in envs:
        env.amplitude = radius
    observations = np.stack([env.reset()[0] for env in envs])
    count = len(seeds)
    state_cost, command_cost = np.zeros((2, count, EP_LEN // WINDOW))
    returns, maximum, saturated, force_square, force_max, changes = np.zeros((6, count))
    for step in range(EP_LEN):
        physical = np.stack([env.state for env in envs])
        applied_load = np.stack([env.disturbance for env in envs])
        actions = control_actions(mode, observations) if action_fn is None else action_fn(observations)
        assert np.isfinite(actions).all() and np.max(np.abs(actions)) <= 1
        window = step // WINDOW
        command = ACTION_SCALE * actions.astype(float)
        state_cost[:, window] += np.einsum("bi,ij,bj->b", physical, Q, physical)
        command_cost[:, window] += np.einsum("bi,ij,bj->b", command, R, command)
        maximum = np.maximum(maximum, np.linalg.norm(physical, axis=1))
        saturated += np.sum(np.abs(actions) > .95, axis=1)
        force_square += np.sum(applied_load ** 2, axis=1)
        force_max = np.maximum(force_max, np.max(np.abs(applied_load), axis=1))
        transitions = [env.step(action) for env, action in zip(envs, actions)]
        assert all(t[2] == (step == EP_LEN - 1) and not t[3] for t in transitions)
        returns += [t[1] for t in transitions]
        observations = np.stack([t[0] for t in transitions])
        changes += np.any(observations[:, 4:] != applied_load.astype(np.float32), axis=1)
    maximum = np.maximum(maximum, np.linalg.norm(np.stack([env.state for env in envs]), axis=1))
    np.testing.assert_allclose(-returns, np.sum(state_cost + command_cost, axis=1), rtol=1e-12, atol=1e-12)
    return {
        "evaluation_seeds": seeds, "returns": returns.tolist(),
        "state_costs_by_window": state_cost.tolist(), "command_costs_by_window": command_cost.tolist(),
        "maximum_physical_state_norms": maximum.tolist(),
        "saturated_component_fractions": (saturated / (2 * EP_LEN)).tolist(),
        "force_rms": np.sqrt(force_square / (2 * EP_LEN)).tolist(),
        "maximum_load_components": force_max.tolist(), "force_change_counts": changes.tolist(),
        "mean_cost": float(-returns.mean()),
        "mean_state_costs_by_window": state_cost.mean(0).tolist(),
        "mean_command_costs_by_window": command_cost.mean(0).tolist(),
        "late_cost_share": float(np.sum(state_cost[:, 1:] + command_cost[:, 1:]) / -returns.sum()),
    }


def qualify_radius(results, radius):
    controls = results["sustained"]
    references = [controls[str(load)]["force_cancel_feedback"] for load in SEVERITIES]
    ood = controls["50"]
    ref_late_state = sum(ood["force_cancel_feedback"]["mean_state_costs_by_window"][1:])
    no_load_late_state = sum(ood["no_load_feedback"]["mean_state_costs_by_window"][1:])
    checks = {
        "reference_bounded_and_low_saturation": all(
            max(c["maximum_physical_state_norms"]) <= max(1.0, 2 * radius)
            and np.mean(c["saturated_component_fractions"]) < .1 for c in references),
        "reference_cost_below_fifth_of_zero": all(
            c["mean_cost"] < .2 * controls[str(load)]["zero"]["mean_cost"]
            for load, c in zip(SEVERITIES, references)),
        "ood_majority_cost_after_step50": ood["force_cancel_feedback"]["late_cost_share"] >= .5,
        "current_load_halves_ood_late_state_cost": ref_late_state < .5 * no_load_late_state,
    }
    return {"checks": checks, "qualified": all(checks.values()),
            "ood_reference_late_state_cost": ref_late_state,
            "ood_no_load_late_state_cost": no_load_late_state}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--radius", type=float, choices=RADII, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    count = 2 if args.preflight_only else 100
    seeds = list(range(SEED_START, SEED_START + count))
    started = time.monotonic()
    results = {}
    progress = tqdm(total=len(PROFILES) * len(SEVERITIES) * len(CONTROLLERS) * count, desc="qualification episodes")
    for profile in PROFILES:
        results[profile] = {}
        for severity in SEVERITIES:
            controllers = results[profile][str(severity)] = {}
            for mode in CONTROLLERS:
                controllers[mode] = rollout(profile, mode, args.radius, severity, seeds)
                progress.update(count)
            print("profile", profile, "radius", args.radius, "load", severity,
                  {m:round(c["mean_cost"], 6) for m,c in controllers.items()}, flush=True)
    progress.close()
    report = {
        "revision": REVISION, "radius": args.radius, "preflight_only": args.preflight_only,
        "environment_settings": {p:cls.specification() for p,cls in PROFILES.items()},
        "severities": list(SEVERITIES), "controllers": list(CONTROLLERS), "window_steps": WINDOW,
        "evaluation_seeds": seeds, "controller_episodes": progress.total,
        "evaluation_interactions": progress.total * EP_LEN,
        "rl_training_runs": 0, "optimizer_updates": 0, "fitting_interactions": 0,
        "reference_is_optimal": False, "historical_hold_unchanged": True,
        "runtime": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__, "device": "cpu"},
        "results": results, "qualification": qualify_radius(results, args.radius),
        "elapsed_seconds": time.monotonic() - started,
    }
    args.result_dir.mkdir(parents=True, exist_ok=True)
    path = args.result_dir / f"{REVISION}_radius{args.radius}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print("Eval complete: environment qualification; RL training runs=0; optimizer updates=0", flush=True)


if __name__ == "__main__":
    main()
