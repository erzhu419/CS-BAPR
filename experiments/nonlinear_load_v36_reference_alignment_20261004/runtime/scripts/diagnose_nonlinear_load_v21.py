"""Read-only first-action interventions with matched frozen-actor continuations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pysindy
import scipy
import sklearn
import torch
from torch.distributions import Normal

from csbapr.envs.lqr_env import ACTION_SCALE, EP_LEN, Q, R
from csbapr.losses.jacobian import deterministic_action
from csbapr.networks.critic import EnsembleQNet
from csbapr.networks.policy import EquilibriumGaussianPolicy
from scripts.diagnose_nonlinear_load_v9 import control_action
from scripts.evaluate_nonlinear_load_v12 import load_actor_parameters
from scripts.evaluate_nonlinear_load_v13 import RADII, SEVERITIES, make_radius_env

REVISION = "nonlinear_load_v21_objective_diagnostic"
SOURCE_REVISION = "nonlinear_load_v20_jacobian_scope"
TRAINING_SEEDS = (15, 16)
GAMMA = .99
MODES = ("actor", "fitted-first")


def export_checkpoint(path, evaluation_path, source_task_id, output):
    source = json.loads(evaluation_path.read_text())
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    extra = checkpoint["extra_state"]
    protocol = extra["protocol"]
    assert protocol["revision"] == source["revision"] == SOURCE_REVISION
    assert protocol["jacobian_scope"] == source["jacobian_scope"] == "full"
    assert protocol["reference"] == source["reference"] == "cubic-local"
    assert protocol["training_seed"] == source["training_seed"] in TRAINING_SEEDS
    assert protocol["config"] == source["config"] and not protocol["preflight_only"]
    assert (extra["next_ep"], extra["env_steps"], checkpoint["training_steps"]) == (500, 100000, 99001)
    gain = checkpoint["symbolic_reference_gain"].tolist()
    assert gain == source["feedback_gains"]["fitted"]
    assert extra["reference_identification"] == source["reference_identification"]
    state = checkpoint["training_state"]
    belief = state["belief"]
    weights = torch.tensor(belief["belief"], dtype=torch.float32)
    weighted_lambda = float((weights * torch.exp(-.1 * torch.arange(belief["max_H"]))).sum())
    payload = {"revision": REVISION, "source_task_id": source_task_id, "source_protocol": protocol,
               "source_budget": {"next_episode": extra["next_ep"], "environment_steps": extra["env_steps"],
                                 "training_updates": checkpoint["training_steps"]},
               "actor_parameters": {k: v.tolist() for k, v in checkpoint["actor"].items()},
               "critic_parameters": {k: v.tolist() for k, v in checkpoint["critic"].items()},
               "fitted_gain": gain, "alpha": float(checkpoint["alpha"]), "weighted_lambda": weighted_lambda,
               "reward_std": max(float(state["reward_ema_var"]) ** .5, 1e-6),
               "source_actor_results": {str(r): {str(s): source["radius_results"][str(r)][str(s)]["controllers"]["actor"]
                                                 for s in SEVERITIES} for r in RADII}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")
    print(f"Export complete: {output}; bytes={output.stat().st_size}; replay/optimizers/checkpoint excluded", flush=True)


def initial_scores(actor, critic, gain, observations, alpha, beta):
    states = torch.as_tensor(observations, dtype=torch.float32, device=next(actor.parameters()).device)
    with torch.no_grad():
        mean, log_std = actor(states)
        actions = {"actor": deterministic_action(actor, states), "fitted-first": torch.tanh(-states @ gain.T)}
        results = {}
        for mode, action in actions.items():
            q = critic(states, action)
            q_mean, q_std = q.mean(0), q.std(0)
            # Same tanh correction and epsilon as GaussianPolicy.evaluate.
            latent = torch.atanh((action / actor.action_range).clamp(-1 + 1e-6, 1 - 1e-6))
            log_prob = (Normal(mean, log_std.exp()).log_prob(latent)
                        - torch.log(1 - (action / actor.action_range).square() + 1e-6)
                        - np.log(actor.action_range)).sum(-1)
            risk_score = q_mean + beta * q_std
            results[mode] = {"action": action.cpu().numpy(), "q_mean": q_mean.cpu().numpy(),
                             "q_std": q_std.cpu().numpy(), "risk_score": risk_score.cpu().numpy(),
                             "log_probability": log_prob.cpu().numpy(),
                             "soft_surrogate_score": (risk_score - alpha * log_prob).cpu().numpy()}
    return results


def paired_rollout(actor, gain, radius, severity, seeds, gamma=GAMMA):
    environments = {mode: [make_radius_env(severity, seed, radius) for seed in seeds] for mode in MODES}
    observations = {mode: np.stack([env.reset()[0] for env in envs]) for mode, envs in environments.items()}
    np.testing.assert_array_equal(observations["actor"], observations["fitted-first"])
    initial = observations["actor"].copy()
    names = ("total_cost", "discounted_cost", "state_cost", "command_cost", "early_state_cost", "late_state_cost")
    totals = {mode: {name: np.zeros(len(seeds)) for name in names} for mode in MODES}
    for step in range(EP_LEN):
        applied = {}
        for mode, envs in environments.items():
            controller = "fitted" if mode == "fitted-first" and step == 0 else "actor"
            actions = control_action(actor, {"fitted": gain}, controller, observations[mode])
            assert np.isfinite(actions).all() and np.max(np.abs(actions)) <= 1
            physical = np.stack([env.state for env in envs])
            state_cost = np.einsum("bi,ij,bj->b", physical, Q, physical)
            command = ACTION_SCALE * actions.astype(float)
            command_cost = np.einsum("bi,ij,bj->b", command, R, command)
            transitions = [env.step(action) for env, action in zip(envs, actions)]
            assert all(t[2] == (step == EP_LEN - 1) and not t[3] for t in transitions)
            cost = -np.asarray([t[1] for t in transitions])
            np.testing.assert_allclose(cost, state_cost + command_cost, rtol=1e-12, atol=1e-12)
            totals[mode]["total_cost"] += cost
            totals[mode]["discounted_cost"] += gamma ** step * cost
            totals[mode]["state_cost"] += state_cost
            totals[mode]["command_cost"] += command_cost
            totals[mode]["early_state_cost" if step < 50 else "late_state_cost"] += state_cost
            observations[mode] = np.stack([t[0] for t in transitions])
            applied[mode] = np.stack([t[4]["applied_disturbance"] for t in transitions])
        np.testing.assert_array_equal(applied["actor"], applied["fitted-first"])
    return initial, totals


def ordering(cost_delta, score_delta):
    return {"fitted_first_lower_cost_paths": int(np.sum(cost_delta < 0)),
            "fitted_first_higher_cost_paths": int(np.sum(cost_delta > 0)),
            "score_prefers_fitted_first_paths": int(np.sum(score_delta > 0)),
            "score_prefers_actor_paths": int(np.sum(score_delta < 0)),
            "actor_preferred_despite_lower_fitted_first_cost_paths": int(np.sum((cost_delta < 0) & (score_delta < 0))),
            "fitted_first_preferred_despite_higher_cost_paths": int(np.sum((cost_delta > 0) & (score_delta > 0)))}


def summarize_block(totals, scores):
    delta = {name: totals["fitted-first"][name] - totals["actor"][name] for name in totals["actor"]}
    score_deltas = {name: scores["fitted-first"][name] - scores["actor"][name]
                    for name in ("q_mean", "risk_score", "soft_surrogate_score")}
    return {"mean_fitted_first_minus_actor_cost": {k: float(v.mean()) for k, v in delta.items()},
            "mean_fitted_first_minus_actor_score": {k: float(v.mean()) for k, v in score_deltas.items()},
            "discount_changes_path_ordering_count": int(np.sum(delta["total_cost"] * delta["discounted_cost"] < 0)),
            "discount_changes_mean_ordering": bool(delta["total_cost"].mean() * delta["discounted_cost"].mean() < 0),
            "ordering": {cost: {score: ordering(delta[cost], difference) for score, difference in score_deltas.items()}
                         for cost in ("total_cost", "discounted_cost")}}


def diagnose(payload, device, count, radii, severities):
    config = payload["source_protocol"]["config"]
    assert config["gamma"] == GAMMA and config["actor_type"] == "centered-tanh-mlp"
    assert config["policy_q_std_cap"] == 0 and not config["use_context"]
    actor = EquilibriumGaussianPolicy(6, 2, config["hidden_dim"], activation="tanh").to(device).eval()
    critic = EnsembleQNet(6, 2, config["hidden_dim"], config["num_critics"]).to(device).eval()
    load_actor_parameters(actor, payload["actor_parameters"])
    load_actor_parameters(critic, payload["critic_parameters"])
    actor.requires_grad_(False)
    critic.requires_grad_(False)
    before = [p.detach().clone() for net in (actor, critic) for p in net.parameters()]
    gain = torch.as_tensor(payload["fitted_gain"], dtype=torch.float32, device=device)
    seeds = list(range(60000, 60000 + count))
    beta = -2 - config["penalty_scale"] * payload["weighted_lambda"]
    blocks = {}
    for radius in radii:
        blocks[str(radius)] = {}
        for severity in severities:
            initial, totals = paired_rollout(actor, gain, radius, severity, seeds)
            saved = payload["source_actor_results"][str(radius)][str(severity)]
            assert saved["evaluation_seeds"][:count] == seeds
            errors = {}
            for name, key in (("total_cost", "returns"), ("state_cost", "state_costs"), ("command_cost", "command_costs")):
                expected = np.asarray(saved[key][:count]) * (-1 if name == "total_cost" else 1)
                np.testing.assert_allclose(totals["actor"][name], expected, rtol=1e-4, atol=1e-3)
                errors[name] = float(np.max(np.abs(totals["actor"][name] - expected)))
            scores = initial_scores(actor, critic, gain, initial, payload["alpha"], beta)
            blocks[str(radius)][str(severity)] = {
                "evaluation_seeds": seeds, "initial_observations": initial.tolist(),
                "reproduction_max_abs_errors": errors, "disturbance_paths_matched": True,
                "continuations": {mode: {name: array.tolist() for name, array in values.items()} for mode, values in totals.items()},
                "initial_scores": {mode: {name: array.tolist() for name, array in values.items()} for mode, values in scores.items()},
                "summary": summarize_block(totals, scores)}
            print(f"radius={radius} load={severity}: matched first-action diagnostic complete; updates=0", flush=True)
    assert all(torch.equal(old, p) and p.grad is None for old, p in zip(before, [p for net in (actor, critic) for p in net.parameters()]))
    return {"gamma": GAMMA, "alpha": payload["alpha"], "weighted_lambda": payload["weighted_lambda"],
            "risk_beta": beta, "reward_std": payload["reward_std"], "radius_results": blocks,
            "parameters_unchanged": True, "optimizer_updates": 0, "fitting_interactions": 0,
            "controller_episodes": len(radii) * len(severities) * count * 2}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-checkpoint", type=Path)
    parser.add_argument("--source-evaluation", type=Path)
    parser.add_argument("--source-task-id")
    parser.add_argument("--export-output", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--seed", type=int, choices=TRAINING_SEEDS)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.export_checkpoint:
        export_checkpoint(args.export_checkpoint, args.source_evaluation, args.source_task_id, args.export_output)
        return
    manifest = json.loads((Path(__file__).resolve().parents[2] / "dispatch_manifest.json").read_text())
    payload = json.loads(args.input.read_text())
    protocol = payload["source_protocol"]
    assert manifest["revision"] == payload["revision"] == REVISION
    assert protocol["revision"] == manifest["source_revision"] == SOURCE_REVISION
    assert protocol["training_seed"] == args.seed and protocol["jacobian_scope"] == "full"
    assert payload["source_task_id"] == manifest["source_tasks"][str(args.seed)]["id"]
    assert protocol["config"] == manifest["config"] and protocol["environment_settings"] == manifest["environment_settings"]
    assert payload["source_budget"] == {"next_episode": 500, "environment_steps": 100000, "training_updates": 99001}
    assert (manifest["gamma"], manifest["radii"], manifest["severities"], manifest["evaluation_seed_start"],
            manifest["evaluation_episodes_per_block"], manifest["modes"]) == (GAMMA, list(RADII), list(SEVERITIES), 60000, 100, list(MODES))
    versions = {"torch": torch.__version__, "numpy": np.__version__, "pysindy": pysindy.__version__,
                "scipy": scipy.__version__, "sklearn": sklearn.__version__}
    for package, version in manifest["runtime_requirement"].items():
        if package == "python_minor_versions":
            assert f"{sys.version_info.major}.{sys.version_info.minor}" in version
        else:
            assert versions[package] == version
    count, radii, severities = (2, (1.,), (50,)) if args.preflight_only else (100, RADII, SEVERITIES)
    output = {"revision": REVISION, "source_revision": SOURCE_REVISION, "source_task_id": payload["source_task_id"],
              "training_seed": args.seed, "preflight_only": args.preflight_only, "checkpoint_selection": "last",
              "runtime": {**versions, "python": sys.version, "device": args.device},
              "config": protocol["config"], "environment_settings": protocol["environment_settings"],
              **diagnose(payload, args.device, count, radii, severities)}
    args.result_dir.mkdir(parents=True, exist_ok=True)
    path = args.result_dir / f"{REVISION}_seed{args.seed}.json"
    path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(f"Diagnostic complete: {path}; optimizer updates=0", flush=True)


if __name__ == "__main__":
    main()
