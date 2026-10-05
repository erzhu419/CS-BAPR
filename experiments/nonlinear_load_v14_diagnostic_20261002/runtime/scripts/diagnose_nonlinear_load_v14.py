"""Read-only radius diagnostics from portable frozen actor parameters."""
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

from csbapr.envs.lqr_env import ACTION_SCALE, EP_LEN, Q, R
from csbapr.networks.policy import EquilibriumGaussianPolicy
from scripts.diagnose_nonlinear_load_v9 import MODES, alignment, control_action
from scripts.evaluate_nonlinear_load_v12 import load_actor_parameters
from scripts.evaluate_nonlinear_load_v13 import RADII, SEVERITIES, make_radius_env

REVISION = "nonlinear_load_v14_readonly_diagnostic"
METHOD = "csbapr-centered-mlp-jacobian-only-no-target-shift"
EARLY_STEPS = 50
SAMPLE_INTERVAL = 10
SOURCE_MODES = {"fitted": "fitted_model", "known_local": "known_local_linear", "known_inverse": "known_inverse"}


def rollout(actor, gains, mode, radius, severity, seeds):
    envs = [make_radius_env(severity, seed, radius) for seed in seeds]
    observations = np.stack([env.reset()[0] for env in envs])
    returns, state_cost, command_cost, early_state, early_command = (np.zeros(len(seeds)) for _ in range(5))
    samples = []
    for step in range(EP_LEN):
        if mode == "actor" and step % SAMPLE_INTERVAL == 0:
            samples.append(observations.copy())
        actions = control_action(actor, gains, mode, observations)
        assert np.isfinite(actions).all() and np.max(np.abs(actions)) <= 1
        physical = np.stack([env.state for env in envs])
        command = ACTION_SCALE * actions.astype(float)
        current_state = np.einsum("bi,ij,bj->b", physical, Q, physical)
        current_command = np.einsum("bi,ij,bj->b", command, R, command)
        state_cost += current_state
        command_cost += current_command
        if step < EARLY_STEPS:
            early_state += current_state
            early_command += current_command
        transitions = [env.step(action) for env, action in zip(envs, actions)]
        assert all(t[2] == (step == EP_LEN - 1) and not t[3] for t in transitions)
        returns += [t[1] for t in transitions]
        observations = np.stack([t[0] for t in transitions])
    np.testing.assert_allclose(-returns, state_cost + command_cost, rtol=1e-12, atol=1e-12)
    costs = {"state_costs": state_cost, "command_costs": command_cost,
             "early_state_costs": early_state, "early_command_costs": early_command,
             "late_state_costs": state_cost - early_state, "late_command_costs": command_cost - early_command}
    result = {"evaluation_seeds": seeds, "returns": returns.tolist(), "mean_cost": float(-returns.mean()),
              **{name: values.tolist() for name, values in costs.items()},
              **{f"mean_{name[:-1]}": float(values.mean()) for name, values in costs.items()}}
    return result, np.stack(samples) if samples else None


def paired_excess_costs(actor, reference, seeds):
    differences = {name: np.asarray(actor[name]) - reference[name] for name in (
        "state_costs", "command_costs", "early_state_costs", "early_command_costs",
        "late_state_costs", "late_command_costs")}
    total = differences["state_costs"] + differences["command_costs"]
    np.testing.assert_allclose(total, np.asarray(reference["returns"]) - actor["returns"], rtol=1e-10, atol=1e-10)
    return {"mean_total_cost": float(total.mean()), "actor_better_paths": int(np.sum(total < 0)),
            **{f"mean_{name[:-1]}": float(values.mean()) for name, values in differences.items()},
            "per_path": [{"evaluation_seed": seed, "total_cost": float(total[i]),
                          **{name[:-1]: float(values[i]) for name, values in differences.items()}}
                         for i, seed in enumerate(seeds)],
            "five_largest_actor_excesses": [{"evaluation_seed": seeds[i], "total_cost": float(total[i])}
                                           for i in np.argsort(total)[-5:][::-1]]}


def diagnose_block(actor, gains, source, radius, severity, seeds):
    original = source["radius_results"][str(radius)]
    results, samples = {}, None
    indices = [original["amplitude_results"][str(severity)]["evaluation_seeds"].index(seed) for seed in seeds]
    for mode in MODES:
        result, observations = rollout(actor, gains, mode, radius, severity, seeds)
        saved = (original["amplitude_results"] if mode == "actor" else
                 original["feedback_controls"][SOURCE_MODES[mode]]["amplitude_results"])[str(severity)]
        expected = [saved["returns"][i] for i in indices]
        np.testing.assert_allclose(result["returns"], expected, rtol=1e-4, atol=1e-3)
        result["maximum_absolute_reproduction_error"] = float(np.max(np.abs(np.asarray(result["returns"]) - expected)))
        results[mode] = result
        if mode == "actor":
            samples = observations
    trajectory = samples.reshape(-1, 6)
    physical_norm, load_norm = np.linalg.norm(trajectory[:, :4], axis=1), np.linalg.norm(trajectory[:, 4:], axis=1)
    quartile_count = len(trajectory) // 4
    focus = seeds.index(60002)
    return {"environment_settings": original["environment_settings"], "controllers": results,
            "reproduction_passed": True,
            "alignment": {"initial_states": alignment(actor, gains, samples[0]),
                          "actor_trajectory": alignment(actor, gains, trajectory),
                          "largest_physical_state_quartile": alignment(actor, gains, trajectory[np.argsort(physical_norm)[-quartile_count:]]),
                          "largest_load_quartile": alignment(actor, gains, trajectory[np.argsort(load_norm)[-quartile_count:]]),
                          "path60002": alignment(actor, gains, samples[:, focus])},
            "path60002_initial_observation": samples[0, focus].tolist(),
            "paired_excess_costs": {mode: paired_excess_costs(results["actor"], results[mode], seeds) for mode in MODES[1:]}}


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
    protocol = payload["source_protocol"]
    source = json.loads(args.source_evaluation.read_text())
    manifest = json.loads((Path(__file__).resolve().parents[2] / "dispatch_manifest.json").read_text())
    assert manifest["revision"] == REVISION and manifest["source_revision"] == source["revision"] == "nonlinear_load_v13_radius"
    assert protocol["revision"] == source["source_revision"] == "nonlinear_load_v11_replication"
    assert protocol["method"] == source["method"] == manifest["method"] == METHOD
    assert protocol["training_seed"] == source["training_seed"] in manifest["training_seeds"] == [7, 8]
    assert (payload["next_episode"], payload["environment_steps"], payload["training_updates"]) == (500, 100000, 99001)
    assert payload["source_task_id"] == source["source_task_id"]
    assert protocol["config"] == source["config"] == manifest["config"]
    assert protocol["environment_settings"] == source["environment_settings"] == manifest["environment_settings"]
    assert (source["radii"], source["severities"], manifest["radii"], manifest["severities"]) == (list(RADII), list(SEVERITIES)) * 2
    assert (manifest["early_steps"], manifest["sample_interval"], manifest["focus_path"],
            manifest["evaluation_seed_start"], manifest["evaluation_episodes_per_severity"]) == (50, 10, 60002, 60000, 100)
    versions = {"torch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
                "pysindy": pysindy.__version__, "sklearn": sklearn.__version__}
    for package, version in manifest["runtime_requirement"].items():
        if package != "python_minor_versions":
            assert versions[package] == version
    assert f"{sys.version_info.major}.{sys.version_info.minor}" in manifest["runtime_requirement"]["python_minor_versions"]
    config = protocol["config"]
    assert config["actor_type"] == "centered-tanh-mlp" and not config["use_nau_actor"] and not config["use_context"]
    actor = EquilibriumGaussianPolicy(6, 2, config["hidden_dim"], activation="tanh").to(args.device)
    load_actor_parameters(actor, payload["actor_parameters"])
    actor.requires_grad_(False)
    controls = source["radius_results"]["0.2"]["feedback_controls"]
    assert controls["fitted_model"]["gain"] == payload["symbolic_reference_gain"]
    gains = {"fitted": torch.tensor(payload["symbolic_reference_gain"], dtype=torch.float32, device=args.device),
             "known_local": torch.tensor(controls["known_local_linear"]["gain"], dtype=torch.float32, device=args.device)}
    for block in source["radius_results"].values():
        for name, control in controls.items():
            key = "physical_gain" if name == "known_inverse" else "gain"
            assert block["feedback_controls"][name][key] == control[key]
    seeds = [60002] if args.preflight_only else list(range(60000, 60100))
    results = {}
    for radius in RADII:
        results[str(radius)] = {}
        for severity in SEVERITIES:
            results[str(radius)][str(severity)] = diagnose_block(actor, gains, source, radius, severity, seeds)
            print(f"radius={radius} severity={severity}: reproduction and cost/Jacobian diagnostics passed; updates=0", flush=True)
    assert all(not p.requires_grad and p.grad is None for p in actor.parameters())
    output = {**protocol, "revision": REVISION, "source_revision": source["revision"],
              "policy_source_revision": protocol["revision"], "source_task_id": payload["source_task_id"],
              "checkpoint_selection": "last", "environment_steps": payload["environment_steps"],
              "training_updates": payload["training_updates"], "optimizer_steps_taken": 0,
              "new_training_interactions": 0, "new_identification_interactions": 0,
              "radii": list(RADII), "severities": list(SEVERITIES), "evaluation_seeds": seeds,
              "early_steps": EARLY_STEPS, "sample_interval": SAMPLE_INTERVAL, "focus_path": 60002,
              "alignment_sampling": "actor pre-step states every 10 steps; all controllers share each sampled state",
              "preflight_only": args.preflight_only, "reference_is_optimal": False,
              "runtime": {**versions, "python": sys.version, "device": args.device,
                          "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None},
              "radius_results": results}
    args.result_dir.mkdir(parents=True, exist_ok=True)
    path = args.result_dir / f"{REVISION}_{METHOD}_seed{protocol['training_seed']}_diagnostic.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)
    print(f"Eval complete: read-only radius diagnostics: {path}; optimizer updates=0", flush=True)


if __name__ == "__main__":
    main()
