"""Read-only random-load actor recentering; checkpoints remain on workers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


def policy_action(actor, states, origin_mean=None):
    mean = actor(states)[0]
    if origin_mean is not None:
        mean = mean - origin_mean
    return float(getattr(actor, "action_range", 1.0)) * torch.tanh(mean)


def alignment(actor, reference, observations, origin_mean):
    from csbapr.losses.jacobian import action_jacobian

    states = torch.as_tensor(observations, dtype=torch.float32).requires_grad_(True)
    actions = policy_action(actor, states, origin_mean)
    targets = reference(states)
    difference = action_jacobian(actions, states, False) - action_jacobian(targets, states, False)
    return {
        "n_states": len(states),
        "action_component_rmse": (actions - targets).square().mean().sqrt().item(),
        "physical_input_jacobian_squared_frobenius_mean": difference[:, :, :4].square().sum((1, 2)).mean().item(),
        "force_input_jacobian_squared_frobenius_mean": difference[:, :, 4:].square().sum((1, 2)).mean().item(),
    }


def rollout(actor, reference, severity, seeds, mode, origin_mean):
    from csbapr.envs.random_load_env import RandomLoadEnv, EP_LEN

    envs = [RandomLoadEnv(severity, seed) for seed in seeds]
    states = np.stack([env.reset()[0] for env in envs])
    returns, physical_squared = np.zeros(len(seeds)), np.zeros(len(seeds))
    last_50_squared, saturated = np.zeros(len(seeds)), np.zeros(len(seeds))
    sampled = []
    for step in range(EP_LEN):
        if step % 10 == 0:
            sampled.append(states.copy())
        with torch.no_grad():
            inputs = torch.from_numpy(states)
            actions = (reference(inputs) if mode == "reference" else
                       policy_action(actor, inputs, origin_mean if mode == "mean_centered" else None)).numpy()
        squared_norm = np.sum(states[:, :4].astype(float) ** 2, axis=1)
        physical_squared += squared_norm
        if step >= EP_LEN - 50:
            last_50_squared += squared_norm
        saturated += np.sum(np.abs(actions) > 0.95, axis=1)
        transitions = [env.step(action) for env, action in zip(envs, actions)]
        assert all(t[2] == (step == EP_LEN - 1) and not t[3] for t in transitions)
        returns += [t[1] for t in transitions]
        states = np.stack([t[0] for t in transitions])
    result = {
        "evaluation_seeds": seeds, "returns": returns.tolist(), "mean_cost": float(-returns.mean()),
        "mean_saturated_component_fraction": float(saturated.mean() / (2 * EP_LEN)),
        "physical_state_rms": float(np.sqrt(physical_squared.mean() / EP_LEN)),
        "last_50_physical_state_rms": float(np.sqrt(last_50_squared.mean() / 50)),
        "mean_final_position_norm": float(np.linalg.norm(states[:, :2], axis=1).mean()),
        "mean_final_velocity_norm": float(np.linalg.norm(states[:, 2:4], axis=1).mean()),
    }
    if mode != "reference":
        result["trajectory_alignment"] = alignment(
            actor, reference, np.concatenate(sampled), origin_mean if mode == "mean_centered" else None,
        )
    return result


def diagnose(checkpoint_path, evaluation_path, output):
    from csbapr.agent import CSBAPRAgent
    from csbapr.config import CSBAPRConfig
    from csbapr.sindy.feedback import LQRFeedbackReference
    from csbapr.sindy.load_feedback import known_load_reference

    evaluation = json.loads(evaluation_path.read_text())
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    extra = checkpoint["extra_state"]
    protocol = extra["protocol"]
    assert protocol["revision"] == evaluation["revision"] == "random_load_v6"
    assert protocol["environment"] == evaluation["environment"] == "random_load"
    assert (protocol["method"], protocol["training_seed"]) == (evaluation["method"], evaluation["training_seed"])
    assert protocol["config"] == evaluation["config"]
    assert protocol["environment_settings"] == evaluation["environment_settings"]
    assert (extra["next_ep"], extra["env_steps"], checkpoint["training_steps"]) == (500, 100000, 99001)
    config = CSBAPRConfig(**protocol["config"])
    assert config.state_dim == 6 and not config.use_context
    config.device = "cpu"
    agent = CSBAPRAgent(config.state_dim, config.action_dim, config)
    actor = agent.actor.eval()
    actor.load_state_dict(checkpoint["actor"])
    gain = checkpoint["symbolic_reference_gain"]
    reference = LQRFeedbackReference(gain) if gain is not None else known_load_reference()
    reference.requires_grad_(False)
    del checkpoint, agent
    with torch.no_grad():
        origin_mean = actor(torch.zeros(1, 6))[0].detach()
        origin_action = policy_action(actor, torch.zeros(1, 6))[0].tolist()
    report = {
        "revision": "random_load_v6_readonly_centering", "method": protocol["method"],
        "training_seed": protocol["training_seed"], "checkpoint_selection": "last",
        "optimizer_steps_taken": 0, "analysis_device": "cpu", "torch_version": torch.__version__,
        "severities": [1, 20, 50], "origin_mean": origin_mean[0].tolist(), "origin_action": origin_action,
        "reference_kind": "saved_fitted_feedback" if gain is not None else "known_model_diagnostic",
        "reference_gain_max_abs_error": float((reference.gain - known_load_reference().gain).abs().max()),
        "reproduction_rtol": 1e-4, "reproduction_atol": 1e-3,
        "original": {}, "mean_centered": {}, "reference": {},
    }
    for severity in report["severities"]:
        key = str(severity)
        saved = evaluation["amplitude_results"][key]
        seeds = saved["evaluation_seeds"]
        assert seeds == list(range(10000, 10020))
        for mode in ("original", "mean_centered", "reference"):
            report[mode][key] = rollout(actor, reference, severity, seeds, mode, origin_mean)
        actual = report["original"][key]["returns"]
        np.testing.assert_allclose(actual, saved["returns"], rtol=1e-4, atol=1e-3)
        report["original"][key]["maximum_absolute_reproduction_error"] = float(np.max(np.abs(np.array(actual) - saved["returns"])))
        print(protocol["method"], protocol["training_seed"], severity,
              {mode: round(report[mode][key]["mean_cost"], 5) for mode in ("original", "mean_centered", "reference")}, flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


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
