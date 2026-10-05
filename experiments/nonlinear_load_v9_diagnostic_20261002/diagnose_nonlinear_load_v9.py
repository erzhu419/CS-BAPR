"""Read-only cost and feedback diagnostics; final checkpoints stay on workers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

MODES = ("actor", "fitted", "known_local", "known_inverse")
SEVERITIES = (1, 20, 50)


def control_action(actor, gains, mode, observations):
    from csbapr.losses.jacobian import deterministic_action
    from scripts.preflight_nonlinear_load_v9 import action_for

    if mode == "known_inverse":
        return np.stack([action_for("inverse_feedback", obs, None) for obs in observations])
    states = torch.as_tensor(observations, dtype=torch.float32)
    with torch.no_grad():
        actions = (deterministic_action(actor, states) if mode == "actor" else
                   torch.tanh(-states @ gains[mode].T))
    return actions.numpy()


def control_derivatives(actor, gains, mode, observations):
    from csbapr.envs.lqr_env import ACTION_SCALE, get_optimal_gain
    from csbapr.envs.nonlinear_load_env import LINEAR_SHARE
    from csbapr.losses.jacobian import action_jacobian, deterministic_action

    if mode == "actor":
        states = torch.as_tensor(observations, dtype=torch.float32).requires_grad_(True)
        actions = deterministic_action(actor, states)
        jacobian = action_jacobian(actions, states, False).detach().numpy()
        actions = actions.detach().numpy()
    elif mode == "known_inverse":
        actions = control_action(actor, gains, mode, observations)
        gain = np.hstack((get_optimal_gain(), np.eye(2) / ACTION_SCALE))
        linear_actions = np.tanh(-observations @ gain.T)
        slope = ACTION_SCALE * (LINEAR_SHARE + 3 * (1 - LINEAR_SHARE) * actions ** 2)
        # Chain rule through the monotone inverse actuator.
        jacobian = -ACTION_SCALE * (1 - linear_actions ** 2)[:, :, None] * gain / slope[:, :, None]
    else:
        actions = control_action(actor, gains, mode, observations)
        jacobian = -(1 - actions ** 2)[:, :, None] * gains[mode].numpy()
    slope = ACTION_SCALE * (LINEAR_SHARE + 3 * (1 - LINEAR_SHARE) * actions ** 2)
    return actions, jacobian, slope[:, :, None] * jacobian


def alignment(actor, gains, observations):
    from csbapr.envs.nonlinear_load_env import actuator_force

    controls = {mode: control_derivatives(actor, gains, mode, observations) for mode in MODES}
    metrics = {}
    for mode, (actions, jacobian, force_jacobian) in controls.items():
        actual, actual_jacobian, actual_force_jacobian = controls["actor"]
        difference = actual_jacobian - jacobian
        force_difference = actual_force_jacobian - force_jacobian
        metrics[mode] = {
            "action_component_rmse_from_actor": float(np.sqrt(np.mean((actions - actual) ** 2))),
            "output_force_component_rmse_from_actor": float(np.sqrt(np.mean(
                (actuator_force(actions) - actuator_force(actual)) ** 2))),
            "physical_input_action_jacobian_error_sq": float(np.mean(np.sum(difference[:, :, :4] ** 2, (1, 2)))),
            "load_input_action_jacobian_error_sq": float(np.mean(np.sum(difference[:, :, 4:] ** 2, (1, 2)))),
            "load_input_output_force_jacobian_error_sq": float(np.mean(np.sum(force_difference[:, :, 4:] ** 2, (1, 2)))),
            "load_cancellation_jacobian_residual_sq": float(np.mean(np.sum(
                (np.eye(2) + force_jacobian[:, :, 4:]) ** 2, (1, 2)))),
            "mean_output_force_load_jacobian": force_jacobian[:, :, 4:].mean(0).tolist(),
        }
    return {"sampled_states": len(observations), "controls": metrics}


def rollout(actor, gains, mode, severity, seeds):
    from csbapr.envs.lqr_env import Q, R, ACTION_SCALE, EP_LEN
    from csbapr.envs.nonlinear_load_env import NonlinearLoadEnv

    envs = [NonlinearLoadEnv(severity, seed) for seed in seeds]
    observations = np.stack([env.reset()[0] for env in envs])
    returns, state_cost, command_cost = (np.zeros(len(seeds)) for _ in range(3))
    saturated, maximum_norm = np.zeros(len(seeds)), np.zeros(len(seeds))
    samples = []
    for step in range(EP_LEN):
        if mode == "actor" and step % 10 == 0:
            samples.append(observations.copy())
        actions = control_action(actor, gains, mode, observations)
        assert np.isfinite(actions).all() and np.max(np.abs(actions)) <= 1
        physical = np.stack([env.state for env in envs])
        command = ACTION_SCALE * actions.astype(float)
        state_cost += np.einsum("bi,ij,bj->b", physical, Q, physical)
        command_cost += np.einsum("bi,ij,bj->b", command, R, command)
        maximum_norm = np.maximum(maximum_norm, np.linalg.norm(physical, axis=1))
        saturated += np.sum(np.abs(actions) > 0.95, axis=1)
        transitions = [env.step(action) for env, action in zip(envs, actions)]
        assert all(t[2] == (step == EP_LEN - 1) and not t[3] for t in transitions)
        returns += [t[1] for t in transitions]
        observations = np.stack([t[0] for t in transitions])
    maximum_norm = np.maximum(maximum_norm, np.linalg.norm(np.stack([env.state for env in envs]), axis=1))
    np.testing.assert_allclose(-returns, state_cost + command_cost, rtol=1e-12, atol=1e-12)
    result = {
        "evaluation_seeds": seeds, "returns": returns.tolist(),
        "state_costs": state_cost.tolist(), "command_costs": command_cost.tolist(),
        "mean_cost": float(-returns.mean()), "mean_state_cost": float(state_cost.mean()),
        "mean_command_cost": float(command_cost.mean()),
        "maximum_physical_state_norm": float(maximum_norm.max()),
        "mean_saturated_component_fraction": float(saturated.mean() / (2 * EP_LEN)),
    }
    return result, np.concatenate(samples) if samples else None


def diagnose(checkpoint_path, evaluation_path, output):
    from csbapr.envs.nonlinear_load_env import NonlinearLoadEnv
    from csbapr.networks.policy import EquilibriumGaussianPolicy

    evaluation = json.loads(evaluation_path.read_text())
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    extra = checkpoint["extra_state"]
    protocol = extra["protocol"]
    assert protocol["revision"] == evaluation["revision"] == "nonlinear_load_v9_training"
    assert protocol["environment"] == evaluation["environment"] == "nonlinear_load"
    assert protocol["method"] == evaluation["method"] == "csbapr-centered-mlp-no-target-shift"
    assert protocol["training_seed"] == evaluation["training_seed"] in (5, 6)
    assert protocol["config"] == evaluation["config"]
    assert protocol["environment_settings"] == evaluation["environment_settings"] == NonlinearLoadEnv.specification()
    assert (extra["next_ep"], extra["env_steps"], checkpoint["training_steps"]) == (500, 100000, 99001)
    config = protocol["config"]
    assert (config["state_dim"], config["action_dim"], config["actor_type"]) == (6, 2, "centered-tanh-mlp")
    assert not config["use_context"] and not config["use_nau_actor"]
    actor = EquilibriumGaussianPolicy(6, 2, config["hidden_dim"], activation="tanh").eval()
    actor.load_state_dict(checkpoint["actor"])
    actor.requires_grad_(False)
    gains = {"fitted": torch.as_tensor(checkpoint["symbolic_reference_gain"], dtype=torch.float32),
             "known_local": torch.tensor(evaluation["feedback_controls"]["known_local_linear"]["gain"], dtype=torch.float32)}
    torch.testing.assert_close(gains["fitted"], torch.tensor(evaluation["feedback_controls"]["fitted_model"]["gain"]), rtol=0, atol=0)
    del checkpoint
    origin = np.zeros((1, 6), dtype=np.float32)
    report = {
        "revision": "nonlinear_load_v9_readonly_diagnostic", "training_seed": protocol["training_seed"],
        "method": protocol["method"], "checkpoint_selection": "last", "optimizer_steps_taken": 0,
        "analysis_device": "cpu", "torch_version": torch.__version__, "severities": list(SEVERITIES),
        "reproduction_rtol": 1e-4, "reproduction_atol": 1e-3,
        "alignment_sampling": "actor trajectory, every 10 steps; same states for all four controls",
        "origin": {mode: {"action": a[0].tolist(), "action_jacobian": j[0].tolist(),
                          "output_force_load_jacobian": f[0, :, 4:].tolist()}
                   for mode in MODES for a, j, f in [control_derivatives(actor, gains, mode, origin)]},
        "results": {},
    }
    for severity in SEVERITIES:
        key = str(severity)
        saved = evaluation["amplitude_results"][key]
        seeds = saved["evaluation_seeds"]
        assert seeds == list(range(50000, 50100))
        results = report["results"][key] = {}
        actor_states = None
        for mode in MODES:
            result, states = rollout(actor, gains, mode, severity, seeds)
            original = saved if mode == "actor" else evaluation["feedback_controls"][
                {"fitted": "fitted_model", "known_local": "known_local_linear", "known_inverse": "known_inverse"}[mode]
            ]["amplitude_results"][key]
            assert original["evaluation_seeds"] == seeds
            np.testing.assert_allclose(result["returns"], original["returns"], rtol=1e-4, atol=1e-3)
            result["maximum_absolute_reproduction_error"] = float(np.max(np.abs(
                np.array(result["returns"]) - original["returns"])))
            results[mode] = result
            if mode == "actor":
                actor_states = states
        results["alignment"] = alignment(actor, gains, actor_states)
        load_norm = np.linalg.norm(actor_states[:, 4:], axis=1)
        high_load = np.argsort(load_norm)[-len(actor_states) // 4:]
        results["largest_load_quartile_alignment"] = alignment(actor, gains, actor_states[high_load])
        results["paired_excess_costs"] = {}
        for mode in MODES[1:]:
            delta_state = np.array(results["actor"]["state_costs"]) - results[mode]["state_costs"]
            delta_command = np.array(results["actor"]["command_costs"]) - results[mode]["command_costs"]
            delta = delta_state + delta_command
            results["paired_excess_costs"][mode] = {
                "mean_state_cost": float(delta_state.mean()), "mean_command_cost": float(delta_command.mean()),
                "mean_total_cost": float(delta.mean()), "actor_better_paths": int(np.sum(delta < 0)),
                "five_largest_actor_excesses": [{"evaluation_seed": seeds[i], "state_cost": float(delta_state[i]),
                                                "command_cost": float(delta_command[i]), "total_cost": float(delta[i])}
                                               for i in np.argsort(delta)[-5:][::-1]],
            }
        print("severity", severity, {mode: round(results[mode]["mean_cost"], 6) for mode in MODES}, flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("Eval complete: nonlinear-load read-only diagnostic; optimizer updates=0", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root))
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    diagnose(args.checkpoint, args.evaluation, args.output)


if __name__ == "__main__":
    main()
