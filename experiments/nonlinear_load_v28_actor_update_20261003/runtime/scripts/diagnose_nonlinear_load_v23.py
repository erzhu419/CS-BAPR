"""Frozen latent-mean directions and paired deterministic first-action derivatives."""
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
from csbapr.networks.critic import EnsembleQNet
from csbapr.networks.policy import EquilibriumGaussianPolicy
from scripts.evaluate_nonlinear_load_v12 import load_actor_parameters
from scripts.evaluate_nonlinear_load_v13 import make_radius_env

REVISION = "nonlinear_load_v23_direction_diagnostic"
SOURCE_REVISION = "nonlinear_load_v22_entropy_floor"
SOURCES = ("floor", "no-floor")
RADII, SEVERITIES, STEPS = (.2, 1.), (1, 50), (.01, .005)
NOISE_SEEDS = tuple(range(92000, 92008))
GAMMA = .99
BRANCHES = {"actor": 0., **{f"{sign}_{h}": value * h for h in STEPS for sign, value in (("plus", 1), ("minus", -1))}}


def export_checkpoint(path, evaluation_path, source_id, output):
    source = json.loads(evaluation_path.read_text())
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    extra, state = ckpt["extra_state"], ckpt["training_state"]
    protocol = extra["protocol"]
    assert protocol["revision"] == source["revision"] == SOURCE_REVISION
    assert protocol["training_seed"] == source["training_seed"] == 18
    assert protocol["entropy_mode"] == source["entropy_mode"] in SOURCES
    assert protocol["config"] == source["config"] and not protocol["preflight_only"]
    assert protocol["jacobian_scope"] == source["jacobian_scope"] == "full"
    assert protocol["reference"] == source["reference"] == "cubic-local"
    assert (extra["next_ep"], extra["env_steps"], ckpt["training_steps"]) == (500, 100000, 99001)
    gain = ckpt["symbolic_reference_gain"].tolist()
    assert gain == source["feedback_gains"]["fitted"]
    assert extra["reference_identification"] == source["reference_identification"]
    belief = state["belief"]
    weights = torch.tensor(belief["belief"], dtype=torch.float32)
    weighted_lambda = float((weights * torch.exp(-.1 * torch.arange(belief["max_H"]))).sum())
    payload = {"revision": REVISION, "source_id": source_id, "source_mode": protocol["entropy_mode"],
               "source_protocol": protocol, "source_budget": [500, 100000, 99001],
               "actor_parameters": {k: v.tolist() for k, v in ckpt["actor"].items()},
               "critic_parameters": {k: v.tolist() for k, v in ckpt["critic"].items()},
               "fitted_gain": gain, "alpha": float(ckpt["alpha"]), "weighted_lambda": weighted_lambda,
               "reward_std": max(float(state["reward_ema_var"]) ** .5, 1e-6),
               "source_actor_results": {str(r): {str(s): source["radius_results"][str(r)][str(s)]["controllers"]["actor"]
                                                 for s in SEVERITIES} for r in RADII}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")
    print(f"Exported {payload['source_mode']}: {output.stat().st_size} bytes; no replay/optimizer/checkpoint", flush=True)


def mean_direction(actor, gain, states):
    with torch.no_grad():
        mean, log_std = actor(states)
        difference = -states @ gain.T - mean
        distance = difference.norm(dim=-1, keepdim=True)
        direction = difference / distance.clamp_min(1e-12)
    return mean, log_std, direction, distance.squeeze(-1)


def directional_scores(actor, critic, gain, observations, alpha, beta, noise_seeds=NOISE_SEEDS):
    device = next(actor.parameters()).device
    states = torch.as_tensor(observations, dtype=torch.float32, device=device)
    mean, log_std, direction, distance = mean_direction(actor, gain, states)
    result = {}
    for name, seeds in (("deterministic", (None,)), ("expected", noise_seeds)):
        draws = {k: [] for k in ("q_mean", "risk_penalty", "entropy_bonus", "risk_score", "soft_score")}
        for seed in seeds:
            z = (torch.zeros_like(mean) if seed is None else
                 torch.randn(mean.shape, generator=torch.Generator().manual_seed(seed)).to(device))
            shifted_mean = mean.detach().clone().requires_grad_(True)
            latent = shifted_mean + log_std.exp() * z
            action0 = latent.tanh()
            action = actor.action_range * action0
            q = critic(states, action)
            components = {"q_mean": q.mean(0), "risk_penalty": beta * q.std(0)}
            log_prob = (Normal(shifted_mean, log_std.exp()).log_prob(latent)
                        - torch.log(1 - action0.square() + 1e-6) - np.log(actor.action_range)).sum(-1)
            components["entropy_bonus"] = -alpha * log_prob
            for key, score in components.items():
                gradient = torch.autograd.grad(score.sum(), shifted_mean, retain_graph=True)[0]
                draws[key].append((gradient * direction).sum(-1).detach().cpu().numpy().astype(float))
            draws["risk_score"].append(draws["q_mean"][-1] + draws["risk_penalty"][-1])
            draws["soft_score"].append(draws["risk_score"][-1] + draws["entropy_bonus"][-1])
        result[name] = {k: np.stack(v).tolist() for k, v in draws.items()}
    return {"latent_direction": direction.cpu().tolist(), "latent_distance_to_fitted": distance.cpu().tolist(),
            "noise_seeds": list(noise_seeds), **result}


def paired_rollout(actors, gains, radius, severity, seeds, steps=STEPS):
    branches = {"actor": 0., **{f"{sign}_{h}": value * h for h in steps for sign, value in (("plus", 1), ("minus", -1))}}
    environments = {(source, branch): [make_radius_env(severity, seed, radius) for seed in seeds]
                    for source in actors for branch in branches}
    observations = {key: np.stack([env.reset()[0] for env in envs]) for key, envs in environments.items()}
    initial = next(iter(observations.values())).copy()
    for state in observations.values():
        np.testing.assert_array_equal(state, initial)
    names = ("total_cost", "discounted_cost", "state_cost", "command_cost", "early_state_cost", "late_state_cost")
    totals = {key: {name: np.zeros(len(seeds)) for name in names} for key in environments}
    directions = {}
    for source, actor in actors.items():
        states = torch.as_tensor(initial, dtype=torch.float32, device=next(actor.parameters()).device)
        directions[source] = mean_direction(actor, gains[source], states)[2]
    for step in range(EP_LEN):
        applied = []
        for (source, branch), envs in environments.items():
            actor = actors[source]
            states = torch.as_tensor(observations[source, branch], dtype=torch.float32, device=next(actor.parameters()).device)
            with torch.no_grad():
                mean, _ = actor(states)
                if step == 0:
                    mean = mean + branches[branch] * directions[source]
                actions = (actor.action_range * mean.tanh()).cpu().numpy()
            physical = np.stack([env.state for env in envs])
            state_cost = np.einsum("bi,ij,bj->b", physical, Q, physical)
            command = ACTION_SCALE * actions.astype(float)
            command_cost = np.einsum("bi,ij,bj->b", command, R, command)
            transitions = [env.step(action) for env, action in zip(envs, actions)]
            assert all(t[2] == (step == EP_LEN - 1) and not t[3] for t in transitions)
            cost = -np.asarray([t[1] for t in transitions])
            np.testing.assert_allclose(cost, state_cost + command_cost, rtol=1e-12, atol=1e-12)
            values = totals[source, branch]
            for name, value in (("total_cost", cost), ("discounted_cost", GAMMA ** step * cost),
                                ("state_cost", state_cost), ("command_cost", command_cost),
                                ("early_state_cost" if step < 50 else "late_state_cost", state_cost)):
                values[name] += value
            observations[source, branch] = np.stack([t[0] for t in transitions])
            applied.append(np.stack([t[4]["applied_disturbance"] for t in transitions]))
        for forces in applied[1:]:
            np.testing.assert_array_equal(applied[0], forces)
    return initial, {source: {branch: {k: v.tolist() for k, v in totals[source, branch].items()}
                             for branch in branches} for source in actors}


def summarize_block(continuations, scores, steps=STEPS):
    derivatives, comparisons = {}, {}
    score_means = {mode: {k: np.asarray(v).mean(0) for k, v in scores[mode].items()}
                   for mode in ("deterministic", "expected")}
    for h in steps:
        derivatives[str(h)] = {key: (np.asarray(continuations[f"plus_{h}"][key])
                                     - continuations[f"minus_{h}"][key]) / (2 * h)
                               for key in continuations["actor"]}
        comparisons[str(h)] = {cost: {mode: {score: {
            "toward_fitted_lower_cost_paths": int(np.sum(delta < 0)),
            "score_opposes_lower_cost_direction_paths": int(np.sum((delta < 0) & (gradient < 0))),
            "score_favors_higher_cost_direction_paths": int(np.sum((delta > 0) & (gradient > 0)))}
            for score, gradient in components.items()} for mode, components in score_means.items()}
            for cost, delta in derivatives[str(h)].items() if cost in ("total_cost", "discounted_cost")}
    return {"mean_cost_directional_derivatives": {h: {k: float(v.mean()) for k, v in values.items()}
                                                 for h, values in derivatives.items()},
            "mean_score_directional_derivatives": {mode: {k: float(v.mean()) for k, v in values.items()}
                                                  for mode, values in score_means.items()},
            "finite_difference_step_sign_disagreements": {
                k: int(np.sum(derivatives[str(steps[0])][k] * derivatives[str(steps[1])][k] < 0))
                for k in ("total_cost", "discounted_cost")}, "ordering": comparisons}


def diagnose(payloads, device, count, radii, severities, noise_seeds=NOISE_SEEDS):
    actors, critics, gains, before = {}, {}, {}, {}
    for mode, payload in payloads.items():
        config = payload["source_protocol"]["config"]
        actors[mode] = EquilibriumGaussianPolicy(6, 2, config["hidden_dim"], activation="tanh").to(device).eval()
        critics[mode] = EnsembleQNet(6, 2, config["hidden_dim"], config["num_critics"]).to(device).eval()
        for net, key in ((actors[mode], "actor_parameters"), (critics[mode], "critic_parameters")):
            load_actor_parameters(net, payload[key])
            net.requires_grad_(False)
        gains[mode] = torch.as_tensor(payload["fitted_gain"], dtype=torch.float32, device=device)
        before[mode] = [p.detach().clone() for net in (actors[mode], critics[mode]) for p in net.parameters()]
    torch.testing.assert_close(gains["floor"], gains["no-floor"], rtol=0, atol=0)
    seeds, blocks = list(range(60000, 60000 + count)), {}
    for radius in radii:
        blocks[str(radius)] = {}
        for load in severities:
            initial, continuations = paired_rollout(actors, gains, radius, load, seeds)
            models = {}
            for mode, payload in payloads.items():
                saved = payload["source_actor_results"][str(radius)][str(load)]
                assert saved["evaluation_seeds"][:count] == seeds
                errors = {}
                for name, key in (("total_cost", "returns"), ("state_cost", "state_costs"), ("command_cost", "command_costs")):
                    expected = np.asarray(saved[key][:count]) * (-1 if name == "total_cost" else 1)
                    actual = np.asarray(continuations[mode]["actor"][name])
                    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-3)
                    errors[name] = float(np.max(np.abs(actual - expected)))
                beta = -2 - payload["source_protocol"]["config"]["penalty_scale"] * payload["weighted_lambda"]
                scores = directional_scores(actors[mode], critics[mode], gains[mode], initial, payload["alpha"], beta, noise_seeds)
                models[mode] = {"continuations": continuations[mode], "scores": scores, "reproduction_max_abs_errors": errors,
                                "summary": summarize_block(continuations[mode], scores)}
            blocks[str(radius)][str(load)] = {"evaluation_seeds": seeds, "initial_observations": initial.tolist(),
                                              "disturbance_paths_matched": True, "models": models}
            print(f"radius={radius} load={load}: two frozen models, five branches, common paths; updates=0", flush=True)
    for mode in payloads:
        params = [p for net in (actors[mode], critics[mode]) for p in net.parameters()]
        assert all(torch.equal(old, p) and p.grad is None for old, p in zip(before[mode], params))
    return {"radius_results": blocks, "parameters_unchanged": True, "optimizer_updates": 0, "fitting_interactions": 0,
            "controller_episodes": len(radii) * len(severities) * count * len(BRANCHES) * len(payloads)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-checkpoint", type=Path)
    parser.add_argument("--source-evaluation", type=Path)
    parser.add_argument("--source-id")
    parser.add_argument("--export-output", type=Path)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.export_checkpoint:
        export_checkpoint(args.export_checkpoint, args.source_evaluation, args.source_id, args.export_output)
        return
    m = json.loads((Path(__file__).resolve().parents[2] / "dispatch_manifest.json").read_text())
    assert (m["revision"], m["source_revision"], m["radii"], m["severities"], m["finite_difference_steps"],
            m["noise_seeds"], m["evaluation_seed_start"], m["evaluation_episodes_per_block"], m["gamma"]) == (
        REVISION, SOURCE_REVISION, list(RADII), list(SEVERITIES), list(STEPS), list(NOISE_SEEDS), 60000, 100, GAMMA)
    payloads = {mode: json.loads((args.input_dir / f"{mode}_seed18.json").read_text()) for mode in SOURCES}
    for mode, p in payloads.items():
        assert (p["revision"], p["source_mode"], p["source_id"], p["source_budget"]) == (
            REVISION, mode, m["source_tasks"][mode]["id"], [500, 100000, 99001])
        assert p["source_protocol"]["revision"] == SOURCE_REVISION and p["source_protocol"]["training_seed"] == 18
        assert p["source_protocol"]["config"] == m["configs"][mode]
        assert p["source_protocol"]["environment_settings"] == m["environment_settings"]
    versions = {"torch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
                "pysindy": pysindy.__version__, "sklearn": sklearn.__version__}
    for package, version in m["runtime_requirement"].items():
        if package == "python_minor_versions":
            assert f"{sys.version_info.major}.{sys.version_info.minor}" in version
        else:
            assert versions[package] == version
    count, radii, loads, noise = (2, (1.,), (50,), NOISE_SEEDS[:2]) if args.preflight_only else (100, RADII, SEVERITIES, NOISE_SEEDS)
    out = {"revision": REVISION, "source_revision": SOURCE_REVISION, "training_seed": 18,
           "source_ids": {mode: p["source_id"] for mode, p in payloads.items()}, "configs": m["configs"],
           "environment_settings": m["environment_settings"], "finite_difference_steps": list(STEPS), "gamma": GAMMA,
           "terminal": {mode: {k: p[k] for k in ("alpha", "weighted_lambda", "reward_std")} for mode, p in payloads.items()},
           "checkpoint_selection": "last", "preflight_only": args.preflight_only,
           "runtime": {**versions, "python": sys.version, "device": args.device},
           **diagnose(payloads, args.device, count, radii, loads, noise)}
    args.result_dir.mkdir(parents=True, exist_ok=True)
    path = args.result_dir / f"{REVISION}_seed18.json"
    path.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(f"Diagnostic complete: {path}; optimizer updates=0", flush=True)


if __name__ == "__main__":
    main()
