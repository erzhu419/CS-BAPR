"""Frozen-checkpoint, state-matched actor gradients and replay support."""
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

from csbapr.envs.lqr_env import EP_LEN, Q
from csbapr.losses.jacobian import action_jacobian, compute_jacobian_loss, deterministic_action
from csbapr.networks.critic import EnsembleQNet
from csbapr.networks.policy import EquilibriumGaussianPolicy
from csbapr.sindy.feedback import LQRFeedbackReference
from scripts.diagnose_nonlinear_load_v14 import rollout
from scripts.evaluate_nonlinear_load_v12 import load_actor_parameters
from scripts.evaluate_nonlinear_load_v13 import RADII, SEVERITIES

REVISION = "nonlinear_load_v16_gradient_diagnostic"
SOURCE_REVISION = "nonlinear_load_v15_coverage"
METHOD = "csbapr-centered-mlp-jacobian-only-no-target-shift"
BATCH_SIZE = 256
SAMPLE_SEED = 91000
NOISE_SEEDS = tuple(range(92000, 92008))
PHASES = ("all", "early", "late", "recent")


def support(states):
    physical = np.linalg.norm(states[:, :4], axis=1)
    load = np.linalg.norm(states[:, 4:], axis=1)
    return {"count": len(states),
            "physical_norm_quantiles": np.quantile(physical, [0, .25, .5, .75, .95, 1]).tolist(),
            "load_norm_quantiles": np.quantile(load, [0, .25, .5, .75, .95, 1]).tolist(),
            "physical_norm_bin_edges": [0, .2, .5, 1, None],
            "physical_norm_bin_counts": np.histogram(physical, [0, .2, .5, 1, np.inf])[0].tolist(),
            "mean_state_cost": float(np.einsum("bi,ij,bj->b", states[:, :4], Q, states[:, :4]).mean())}


def replay_payload(replay, condition):
    rows = replay["buffer"]
    assert len(rows) == replay["position"] == 100000 < replay["capacity"]
    assert all(len(row) == 5 and bool(row[4]) == (i % EP_LEN == EP_LEN - 1)
               for i, row in enumerate(rows))
    states = np.stack([row[0] for row in rows])
    assert states.shape == (100000, 6) and np.isfinite(states).all()
    indices = np.arange(len(states))
    phase_indices = {"all": indices, "early": indices[indices % EP_LEN < 50],
                     "late": indices[indices % EP_LEN >= 50], "recent": indices[-20000:]}
    cycle = np.asarray([.2, .5, 1] if condition == "balanced" else [.2])
    reset_radii = cycle[(indices // EP_LEN) % len(cycle)]
    assert np.all(np.linalg.norm(states[::EP_LEN, :4], axis=1) <= reset_radii[::EP_LEN] + 1e-6)
    samples = {}
    for offset, (name, ids) in enumerate(phase_indices.items()):
        selected = np.random.default_rng(SAMPLE_SEED + offset).choice(ids, BATCH_SIZE, replace=True)
        samples[name] = {"indices": selected.tolist(), "states": states[selected].tolist()}
    return {"support": {name: support(states[ids]) for name, ids in phase_indices.items()},
            "reset_support": {str(r): support(states[(indices % EP_LEN == 0) & (reset_radii == r)])
                              for r in cycle},
            "reset_fraction": 500 / len(states), "samples": samples}


def export_checkpoint(path, source_task_id, result_path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    extra = checkpoint["extra_state"]
    protocol = extra["protocol"]
    assert protocol["revision"] == SOURCE_REVISION and protocol["method"] == METHOD
    assert protocol["training_seed"] in (9, 10) and not protocol["preflight_only"]
    assert (extra["next_ep"], extra["env_steps"], checkpoint["training_steps"]) == (500, 100000, 99001)
    state = checkpoint["training_state"]
    belief = state["belief"]
    # This is the terminal coefficient, without another belief/alpha/critic update.
    weights = torch.tensor(belief["belief"], dtype=torch.float32)
    weighted_lambda = float((weights * torch.exp(-.1 * torch.arange(belief["max_H"]))).sum())
    payload = {"revision": REVISION, "source_protocol": protocol, "source_task_id": source_task_id,
               "next_episode": extra["next_ep"], "environment_steps": extra["env_steps"],
               "training_updates": checkpoint["training_steps"], "alpha": float(checkpoint["alpha"]),
               "weighted_lambda": weighted_lambda,
               "actor_parameters": {k: v.tolist() for k, v in checkpoint["actor"].items()},
               "critic_parameters": {k: v.tolist() for k, v in checkpoint["critic"].items()},
               "symbolic_reference_gain": checkpoint["symbolic_reference_gain"].tolist(),
               "replay": replay_payload(state["replay"], protocol["condition"])}
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")
    print(f"Export complete: {result_path}; bytes={result_path.stat().st_size}; optimizer/replay checkpoint excluded", flush=True)


def flat_gradient(loss, parameters):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                      for p, g in zip(parameters, gradients)]).detach()


def relation(left, right):
    dot = float(torch.dot(left, right))
    product = float(left.norm() * right.norm())
    return {"dot": dot, "cosine": dot / product if product else None,
            "right_loss_derivative_under_left_descent": -dot}


def gradient_terms(actor, critic, reference, states, alpha, beta, jac_weight, noise_seed):
    torch.manual_seed(noise_seed)
    action, log_prob, _, _, _ = actor.evaluate(states)
    q = critic(states, action)
    losses = {"q_mean": -q.mean(0).mean(), "risk": -beta * q.std(0).mean(),
              "entropy": alpha * log_prob.mean()}
    differentiable_states = states.detach().requires_grad_(True)
    jac_pi = action_jacobian(deterministic_action(actor, differentiable_states), differentiable_states, True)
    jac_ref = action_jacobian(reference(differentiable_states), differentiable_states, False).detach()
    error = (jac_pi - jac_ref).square().mean(0)
    losses["jac_physical"] = jac_weight * error[:, :4].sum()
    losses["jac_load"] = jac_weight * error[:, 4:].sum()
    parameters = tuple(actor.parameters())
    gradients = {name: flat_gradient(loss, parameters) for name, loss in losses.items()}
    gradients["sac"] = sum(gradients[k] for k in ("q_mean", "risk", "entropy"))
    gradients["jac"] = gradients["jac_physical"] + gradients["jac_load"]
    gradients["total"] = gradients["sac"] + gradients["jac"]
    assert all(torch.isfinite(g).all() for g in gradients.values())
    return {name: float(loss.detach()) for name, loss in losses.items()}, gradients


def diagnose_bank(actor, critic, reference, states, payload, noise_seeds):
    config = payload["source_protocol"]["config"]
    beta = -2 - config["penalty_scale"] * payload["weighted_lambda"]
    draws, sums = [], {}
    for seed in noise_seeds:
        losses, gradients = gradient_terms(actor, critic, reference, states, payload["alpha"], beta,
                                          config["jac_weight"], seed)
        draws.append({"noise_seed": seed, "losses": losses,
                      "gradient_norms": {k: float(g.norm()) for k, g in gradients.items()},
                      "sac_vs_jac": relation(gradients["sac"], gradients["jac"]),
                      "total_vs_jac": relation(gradients["total"], gradients["jac"]),
                      "clip_scale": min(1., config["grad_clip"] / float(gradients["total"].norm()))})
        for name, gradient in gradients.items():
            sums[name] = sums.get(name, 0) + gradient
    means = {k: g / len(noise_seeds) for k, g in sums.items()}
    summary = {"draws": draws, "mean_gradient_norms": {k: float(g.norm()) for k, g in means.items()},
               "mean_sac_vs_jac": relation(means["sac"], means["jac"]),
               "mean_total_vs_jac": relation(means["total"], means["jac"]),
               "mean_loss": {k: float(np.mean([d["losses"][k] for d in draws])) for k in draws[0]["losses"]}}
    return summary, means


def load_networks(payload, device):
    config = payload["source_protocol"]["config"]
    assert config["actor_type"] == "centered-tanh-mlp" and not config["use_context"]
    assert config["jac_weight"] == .01 and config["beta_bc"] == config["policy_q_std_cap"] == 0
    actor = EquilibriumGaussianPolicy(6, 2, config["hidden_dim"], activation="tanh").to(device)
    critic = EnsembleQNet(6, 2, config["hidden_dim"], config["num_critics"]).to(device)
    load_actor_parameters(actor, payload["actor_parameters"])
    load_actor_parameters(critic, payload["critic_parameters"])
    critic.requires_grad_(False)
    reference = LQRFeedbackReference(np.asarray(payload["symbolic_reference_gain"])).to(device)
    return actor, critic, reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-checkpoint", type=Path)
    parser.add_argument("--source-task-id")
    parser.add_argument("--export-output", type=Path)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--seed", type=int, choices=[9, 10])
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.export_checkpoint:
        export_checkpoint(args.export_checkpoint, args.source_task_id, args.export_output)
        return
    manifest = json.loads((Path(__file__).resolve().parents[2] / "dispatch_manifest.json").read_text())
    assert manifest["revision"] == REVISION and manifest["source_revision"] == SOURCE_REVISION
    assert (manifest["batch_size"], manifest["sample_seed"], manifest["noise_seeds"]) == (BATCH_SIZE, SAMPLE_SEED, list(NOISE_SEEDS))
    versions = {"torch": torch.__version__, "numpy": np.__version__, "pysindy": pysindy.__version__,
                "scipy": scipy.__version__, "sklearn": sklearn.__version__}
    for package, version in manifest["runtime_requirement"].items():
        if package != "python_minor_versions":
            assert versions[package] == version
    assert f"{sys.version_info.major}.{sys.version_info.minor}" in manifest["runtime_requirement"]["python_minor_versions"]
    payloads = {c: json.loads((args.input_dir / f"{c}_seed{args.seed}.json").read_text())
                for c in ("narrow", "balanced")}
    sources = {c: json.loads((args.input_dir / f"{c}_seed{args.seed}_evaluation.json").read_text())
               for c in payloads}
    networks, before = {}, {}
    for condition, payload in payloads.items():
        protocol = payload["source_protocol"]
        source = sources[condition]
        assert payload["revision"] == REVISION and source["revision"] == protocol["revision"] == SOURCE_REVISION
        assert (protocol["condition"], protocol["training_seed"], protocol["method"]) == (condition, args.seed, METHOD)
        assert (payload["next_episode"], payload["environment_steps"], payload["training_updates"]) == (500, 100000, 99001)
        assert source["config"] == protocol["config"] == manifest["config"]
        assert source["feedback_gains"]["fitted"] == payload["symbolic_reference_gain"]
        assert payload["source_task_id"] == manifest["source_tasks"][condition][str(args.seed)]["id"]
        networks[condition] = load_networks(payload, args.device)
        before[condition] = [p.detach().clone() for net in networks[condition][:2] for p in net.parameters()]
    assert payloads["narrow"]["symbolic_reference_gain"] == payloads["balanced"]["symbolic_reference_gain"]
    banks, reproduction = {}, {}
    radii = (1.,) if args.preflight_only else RADII
    severities = (50,) if args.preflight_only else SEVERITIES
    evaluation_seeds = [60002] if args.preflight_only else list(range(60000, 60100))
    for radius in radii:
        for severity in severities:
            trajectories = []
            label = f"r{radius}_load{severity}"
            for condition, (actor, _, _) in networks.items():
                result, samples = rollout(actor, {}, "actor", radius, severity, evaluation_seeds)
                saved = sources[condition]["radius_results"][str(radius)][str(severity)]["controllers"]["actor"]
                expected = [saved["returns"][saved["evaluation_seeds"].index(s)] for s in evaluation_seeds]
                np.testing.assert_allclose(result["returns"], expected, rtol=1e-4, atol=1e-3)
                reproduction[f"{condition}_{label}"] = float(np.max(np.abs(np.asarray(result["returns"]) - expected)))
                if not trajectories:
                    initial = samples[0]
                else:
                    np.testing.assert_array_equal(initial, samples[0])
                trajectories.append(samples[5:].reshape(-1, 6))
            selected = np.random.default_rng(SAMPLE_SEED).choice(len(initial), BATCH_SIZE, replace=True)
            banks[f"initial_{label}"] = initial[selected]
            late = np.concatenate(trajectories)
            selected = np.random.default_rng(SAMPLE_SEED).choice(len(late), BATCH_SIZE, replace=True)
            banks[f"late_trajectory_{label}"] = late[selected]
    for condition, payload in payloads.items():
        phases = ("all",) if args.preflight_only else PHASES
        for phase in phases:
            banks[f"replay_{condition}_{phase}"] = np.asarray(payload["replay"]["samples"][phase]["states"])
    results, cross = {}, {}
    noise_seeds = NOISE_SEEDS[:2] if args.preflight_only else NOISE_SEEDS
    for condition, (actor, critic, reference) in networks.items():
        results[condition], mean_gradients = {}, {}
        payload = payloads[condition]
        for name, values in banks.items():
            states = torch.tensor(values, dtype=torch.float32, device=args.device)
            result, gradients = diagnose_bank(actor, critic, reference, states, payload, noise_seeds)
            native = float(compute_jacobian_loss(actor, reference, states).detach())
            split = (result["mean_loss"]["jac_physical"] + result["mean_loss"]["jac_load"]) / manifest["config"]["jac_weight"]
            np.testing.assert_allclose(split, native, rtol=1e-5, atol=1e-6)
            results[condition][name] = result
            mean_gradients[name] = gradients
            print(f"{condition} {name}: native Jacobian matched; gradients finite; updates=0", flush=True)
        cross[condition] = {
            replay: {initial: {term: relation(gradients[term], initial_grad["jac_physical"])
                              for term in ("q_mean", "risk", "entropy", "sac", "jac", "total")}
                     for initial, initial_grad in mean_gradients.items() if initial.startswith("initial_")}
            for replay, gradients in mean_gradients.items() if replay.startswith("replay_")}
        after = [p for net in networks[condition][:2] for p in net.parameters()]
        assert all(torch.equal(old, new) and new.grad is None for old, new in zip(before[condition], after))
    output = {"revision": REVISION, "source_revision": SOURCE_REVISION, "training_seed": args.seed,
              "source_tasks": {c: p["source_task_id"] for c, p in payloads.items()},
              "optimizer_steps_taken": 0, "new_training_interactions": 0, "parameters_unchanged": True,
              "preflight_only": args.preflight_only, "noise_seeds": list(noise_seeds),
              "evaluation_seeds": evaluation_seeds, "reproduction_errors": reproduction,
              "terminal_objective": {c: {"alpha": p["alpha"], "weighted_lambda": p["weighted_lambda"],
                                        "beta": -2 - manifest["config"]["penalty_scale"] * p["weighted_lambda"]}
                                     for c, p in payloads.items()},
              "replay_support": {c: {k: v for k, v in p["replay"].items() if k != "samples"} for c, p in payloads.items()},
              "bank_support": {name: support(values) for name, values in banks.items()},
              "gradients": results, "replay_descent_vs_initial_physical_jacobian": cross,
              "runtime": {**versions, "python": sys.version,
                          "device": args.device, "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None}}
    args.result_dir.mkdir(parents=True, exist_ok=True)
    path = args.result_dir / f"{REVISION}_seed{args.seed}.json"
    path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(f"Eval complete: {path}; parameters unchanged; optimizer updates=0", flush=True)


if __name__ == "__main__":
    main()
