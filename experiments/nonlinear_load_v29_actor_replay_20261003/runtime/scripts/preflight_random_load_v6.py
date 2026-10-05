"""Non-learning random-load environment pilot; no CS-BAPR ranking or tuning."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from csbapr.envs.lqr_env import EP_LEN, ACTION_SCALE, get_optimal_gain
from csbapr.envs.random_load_env import RandomLoadEnv, SEVERITIES


def control(action_name, observation):
    feedback = -get_optimal_gain() @ observation[:4]
    cancellation = observation[4:] / ACTION_SCALE
    if action_name == "zero":
        return np.zeros(2)
    if action_name == "force_cancel_only":
        return -cancellation
    if action_name == "feedback_no_force":
        return np.tanh(feedback)
    return np.tanh(feedback - cancellation)


def main():
    policies = ("zero", "force_cancel_only", "feedback_no_force", "feedback_with_force")
    seeds = list(range(40000, 40100))
    results = {}
    for severity in SEVERITIES:
        values = results[str(severity)] = {}
        for policy in policies:
            returns, saturated, state_maxima, storm_fractions, force_rms = [], [], [], [], []
            for seed in seeds:
                env = RandomLoadEnv(severity, seed)
                obs, _ = env.reset()
                total, saturation, maximum, storms, force_square = 0.0, 0, 0.0, 0, 0.0
                for step in range(EP_LEN):
                    action = control(policy, obs)
                    saturation += int(np.sum(np.abs(action) > 0.95))
                    maximum = max(maximum, float(np.linalg.norm(obs[:4])))
                    obs, reward, terminated, truncated, info = env.step(action)
                    total += reward
                    storms += info["storm"]
                    force_square += float(np.mean(info["applied_disturbance"] ** 2))
                    assert not truncated and terminated == (step == EP_LEN - 1)
                    assert np.max(np.abs(info["applied_disturbance"])) <= env.force_cap
                returns.append(total)
                saturated.append(saturation / (2 * EP_LEN))
                state_maxima.append(maximum)
                storm_fractions.append(storms / EP_LEN)
                force_rms.append(np.sqrt(force_square / EP_LEN))
            values[policy] = {
                "mean_cost": -mean(returns), "returns": returns,
                "mean_saturated_component_fraction": mean(saturated),
                "maximum_state_norm": max(state_maxima),
                "mean_storm_fraction": mean(storm_fractions),
                "mean_force_rms": mean(force_rms),
            }
        print(severity, {p: round(v["mean_cost"], 4) for p, v in values.items()}, flush=True)
    checks = {
        "reference_beats_zero_all_severities": all(v["feedback_with_force"]["mean_cost"] < v["zero"]["mean_cost"] for v in results.values()),
        "reference_50x_cost_below_fifth_of_zero": results["50"]["feedback_with_force"]["mean_cost"] < 0.2 * results["50"]["zero"]["mean_cost"],
        "reference_saturation_below_ten_percent": all(v["feedback_with_force"]["mean_saturated_component_fraction"] < 0.1 for v in results.values()),
        "observed_force_is_useful_at_50x": results["50"]["feedback_with_force"]["mean_cost"] < results["50"]["feedback_no_force"]["mean_cost"],
        "storm_fraction_grows_at_least_fivefold": results["50"]["zero"]["mean_storm_fraction"] > 5 * results["1"]["zero"]["mean_storm_fraction"],
    }
    output = ROOT / "results/random_load_v6/environment_pilot.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "revision": "random_load_v6_environment_pilot", "evaluation_seeds": seeds,
        "environment_settings": RandomLoadEnv.specification(),
        "results": results, "checks": checks, "passed": all(checks.values()),
        "reference_is_optimal": False, "training_runs": 0,
    }, indent=2, allow_nan=False) + "\n")
    print(checks, flush=True)
    if not all(checks.values()):
        raise SystemExit("Environment pilot failed; do not dispatch formal training")


if __name__ == "__main__":
    main()
