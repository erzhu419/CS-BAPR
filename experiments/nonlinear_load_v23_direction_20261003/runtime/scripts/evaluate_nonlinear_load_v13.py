"""Stress frozen policies with larger reset radii and unchanged disturbance paths."""
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
import sklearn
import torch

from csbapr.agent import CSBAPRAgent
from csbapr.config import CSBAPRConfig
from csbapr.envs.lqr_env import EP_LEN, get_optimal_gain
from csbapr.envs.nonlinear_load_env import NonlinearLoadEnv
from csbapr.sindy.feedback import LQRFeedbackReference
from scripts.evaluate_nonlinear_load_v12 import load_actor_parameters
from scripts.preflight_nonlinear_load_v9 import action_for
from scripts.test_lqr_convergence import known_reference

REVISION = "nonlinear_load_v13_radius"
RADII = (0.2, 0.5, 1.0)
SEVERITIES = (1, 50)


def make_radius_env(severity, seed, radius):
    env = NonlinearLoadEnv(severity, seed)
    env.amplitude = float(radius)
    return env


def rollout(agent, references, mode, radius, severity, seed):
    env = make_radius_env(severity, seed, radius)
    observation, _ = env.reset()
    total, maximum, saturated, outside = 0.0, 0.0, 0, 0
    for step in range(EP_LEN):
        maximum = max(maximum, float(np.linalg.norm(env.state)))
        outside += int(np.linalg.norm(observation) > 1.0)
        if mode == "actor":
            action = agent.select_action(observation, deterministic=True)
        elif mode == "known_inverse":
            action = action_for("inverse_feedback", observation, None)
        else:
            with torch.no_grad():
                inputs = torch.as_tensor(observation, dtype=torch.float32, device=agent.device)
                action = references[mode](inputs).cpu().numpy()
        assert np.isfinite(action).all() and np.max(np.abs(action)) <= 1.0
        saturated += int(np.sum(np.abs(action) > 0.95))
        observation, reward, terminated, truncated, _ = env.step(action)
        assert not truncated and terminated == (step == EP_LEN - 1)
        assert np.isfinite(observation).all() and np.isfinite(reward)
        total += reward
    return {"return": total,
            "maximum_physical_state_norm": max(maximum, float(np.linalg.norm(env.state))),
            "saturated_component_fraction": saturated / (2 * EP_LEN),
            "outside_unit_observation_ball_fraction": outside / EP_LEN}


def evaluate_radius(agent, references, radius, seeds):
    modes = ["actor", *references, "known_inverse"]
    results = {}
    for mode in modes:
        values = results[mode] = {}
        for severity in SEVERITIES:
            episodes = [rollout(agent, references, mode, radius, severity, seed) for seed in seeds]
            returns = [e["return"] for e in episodes]
            values[str(severity)] = {
                "mean": float(np.mean(returns)), "std": float(np.std(returns)),
                "returns": returns, "evaluation_seeds": seeds,
                **{name: [e[name] for e in episodes] for name in (
                    "maximum_physical_state_norm", "saturated_component_fraction",
                    "outside_unit_observation_ball_fraction")},
            }
        print(f"radius={radius} mode={mode}: ID/50x evaluation complete", flush=True)
    controls = {name: {"gain": reference.gain.cpu().tolist(), "amplitude_results": results[name]}
                for name, reference in references.items()}
    controls["known_inverse"] = {"physical_gain": get_optimal_gain().tolist(),
                                  "reference_is_optimal": False,
                                  "amplitude_results": results["known_inverse"]}
    return {"environment_settings": {**NonlinearLoadEnv.specification(), "initial_amplitude": radius},
            "amplitude_results": results["actor"], "feedback_controls": controls}


def check_reproduction(actual, source, seeds):
    modes = {"actor": actual["amplitude_results"],
             **{k: v["amplitude_results"] for k, v in actual["feedback_controls"].items()}}
    previous = {"actor": source["amplitude_results"],
                **{k: v["amplitude_results"] for k, v in source["feedback_controls"].items()}}
    assert set(modes) == set(previous)
    errors = {}
    for name, values in modes.items():
        errors[name] = {}
        for severity in SEVERITIES:
            old = previous[name][str(severity)]
            assert old["evaluation_seeds"][:len(seeds)] == seeds
            expected = old["returns"][:len(seeds)]
            current = values[str(severity)]["returns"]
            np.testing.assert_allclose(current, expected, rtol=1e-4, atol=1e-3)
            errors[name][str(severity)] = float(np.max(np.abs(np.asarray(current) - expected)))
    return {"passed": True, "radius": 0.2, "source_revision": "nonlinear_load_v12_paths",
            "evaluation_seeds": seeds, "severities": list(SEVERITIES),
            "rtol": 1e-4, "atol": 1e-3, "maximum_absolute_return_gaps": errors}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-input", type=Path, required=True)
    parser.add_argument("--source-evaluation", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    payload = json.loads(args.policy_input.read_text())
    source = json.loads(args.source_evaluation.read_text())
    protocol = payload["source_protocol"]
    manifest = json.loads((Path(__file__).resolve().parents[2] / "dispatch_manifest.json").read_text())
    arm = next(a for a in manifest["arms"] if a["method"] == protocol["method"])
    assert protocol["revision"] == "nonlinear_load_v11_replication" and source["revision"] == "nonlinear_load_v12_paths"
    assert protocol["training_seed"] == source["training_seed"] in (7, 8)
    assert protocol["method"] == source["method"] and payload["source_task_id"] == source["source_task_id"]
    assert protocol["config"] == source["config"] == arm["config"]
    assert protocol["environment_settings"] == source["environment_settings"] == manifest["environment_settings"]
    assert (payload["next_episode"], payload["environment_steps"], payload["training_updates"]) == (500, 100000, 99001)
    assert (manifest["radii"], manifest["severities"], manifest["evaluation_seed_start"],
            manifest["evaluation_episodes_per_severity"]) == (list(RADII), list(SEVERITIES), 60000, 100)
    versions = {"torch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
                "pysindy": pysindy.__version__, "sklearn": sklearn.__version__}
    for package, version in manifest["runtime_requirement"].items():
        if package != "python_minor_versions":
            assert versions[package] == version
    assert f"{sys.version_info.major}.{sys.version_info.minor}" in manifest["runtime_requirement"]["python_minor_versions"]
    config = CSBAPRConfig(**{**protocol["config"], "device": args.device})
    assert not config.use_context and not config.use_nau_actor
    agent = CSBAPRAgent(config.state_dim, config.action_dim, config)
    assert str(agent.device) == args.device
    load_actor_parameters(agent.actor, payload["actor_parameters"])
    references = {"known_local_linear": known_reference("nonlinear_load").to(agent.device)}
    if arm["symbolic"]:
        assert payload["symbolic_reference_gain"] == source["feedback_controls"]["fitted_model"]["gain"]
        references["fitted_model"] = LQRFeedbackReference(payload["symbolic_reference_gain"]).to(agent.device)
    else:
        assert payload["symbolic_reference_gain"] is None
    seeds = list(range(60000, 60001 if args.preflight_only else 60100))
    baseline = evaluate_radius(agent, references, 0.2, seeds)
    reproduction = check_reproduction(baseline, source, seeds)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{REVISION}_{protocol['method']}_seed{protocol['training_seed']}"
    (args.result_dir / f"{tag}_reproduction.json").write_text(json.dumps(reproduction, indent=2) + "\n")
    print("Radius-0.2 reproduction passed; optimizer updates=0", flush=True)
    radius_results = {"0.2": baseline}
    for radius in RADII[1:]:
        radius_results[str(radius)] = evaluate_radius(agent, references, radius, seeds)
    assert agent.training_steps == 0 and agent.replay_buffer.size == 0
    output = {**protocol, "revision": REVISION, "source_revision": protocol["revision"],
              "reproduction_revision": source["revision"], "source_task_id": payload["source_task_id"],
              "checkpoint_selection": "last", "training_updates": payload["training_updates"],
              "environment_steps": payload["environment_steps"], "actor_type": config.actor_type,
              "optimizer_steps_taken": 0, "new_training_interactions": 0, "new_identification_interactions": 0,
              "radii": list(RADII), "severities": list(SEVERITIES), "evaluation_seed_start": 60000,
              "evaluation_episodes_per_severity": len(seeds), "preflight_only": args.preflight_only,
              "reproduction": reproduction, "radius_results": radius_results,
              "runtime": {**versions, "python": sys.version, "device": str(agent.device),
                          "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None}}
    path = args.result_dir / f"{tag}_evaluation.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)
    print(f"Radius evaluation complete: {path}; optimizer updates=0", flush=True)


if __name__ == "__main__":
    main()
