"""Paired actor coverage, native Adam directions and full-policy continuations."""
from copy import deepcopy
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

from csbapr.losses.jacobian import compute_jacobian_loss, deterministic_action
from csbapr.sindy.feedback import LQRFeedbackReference
from scripts.diagnose_nonlinear_load_v16 import flat_gradient, relation, support
from scripts.diagnose_nonlinear_load_v24 import (
    COSTS, RETURNS, SCORES, EP_LEN, policy_sample, stochastic_rollout,
)
from scripts.diagnose_nonlinear_load_v26 import (
    NOISE_SEEDS, RADII, SEVERITIES, TRAINING_SEEDS,
    EnsembleQNet, EquilibriumGaussianPolicy, load_actor_parameters,
)

REVISION = "nonlinear_load_v28_actor_update_diagnostic"
SOURCE_REVISION = "nonlinear_load_v25_critic_replay"
ARMS = ("uniform", "early")
BATCHES = 8
TRAIN_NOISE = tuple(range(94000, 94008))
PARAMETER_SCALES = (.5, 1.)
COMPONENTS = ("q_mean", "risk", "entropy", "jacobian")


def sampled_batches(states, numpy_state, jacobian_state, seed, count=BATCHES):
    assert states.shape == (100000, 6)
    native = np.random.RandomState()
    native.set_state(numpy_state)
    jacobian = np.random.default_rng()
    jacobian.bit_generator.state = deepcopy(jacobian_state)
    replacement = np.random.default_rng(94000 + seed)
    batches = []
    for _ in range(count):
        uniform = native.randint(0, len(states), size=256)
        resets = replacement.integers(0, 500, size=64) * EP_LEN
        ranks = replacement.integers(0, 500 * 49, size=64)
        early = np.concatenate((uniform[:128], resets, ranks // 49 * EP_LEN + ranks % 49 + 1))
        jac = np.concatenate((uniform[:128], jacobian.integers(0, 500, size=128) * EP_LEN))
        batches.append({name: {"indices": ids.tolist(), "states": states[ids].tolist(),
                              "support": {**support(states[ids]), "episode_phase_counts": {
                                  "reset": int(np.sum(ids % EP_LEN == 0)),
                                  "steps1_49": int(np.sum((ids % EP_LEN > 0) & (ids % EP_LEN < 50))),
                                  "steps50_199": int(np.sum(ids % EP_LEN >= 50))}}}
                        for name, ids in (("uniform", uniform), ("early", early), ("jacobian", jac))})
    return batches


def json_optimizer(optimizer):
    return {"param_groups": optimizer["param_groups"],
            "state": {str(k): {name: value.tolist() if isinstance(value, torch.Tensor) else value
                               for name, value in state.items()}
                      for k, state in optimizer["state"].items()}}


def tensor_optimizer(payload):
    return {"param_groups": deepcopy(payload["param_groups"]),
            "state": {int(k): {name: torch.tensor(value, dtype=torch.float32)
                               for name, value in state.items()}
                      for k, state in payload["state"].items()}}


def export_checkpoint(checkpoint, portable, source_id, output):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    source = json.loads(portable.read_text())
    extra, state = ckpt["extra_state"], ckpt["training_state"]
    assert extra["protocol"] == source["source_protocol"]
    assert source["source_id"] == source_id and source["source_mode"] == "early"
    assert source["source_protocol"]["revision"] == SOURCE_REVISION
    assert (extra["next_ep"], extra["env_steps"], ckpt["training_steps"]) == (500, 100000, 99001)
    for net in ("actor", "critic"):
        for name, value in ckpt[net].items():
            torch.testing.assert_close(value, torch.tensor(source[f"{net}_parameters"][name]), rtol=0, atol=0)
    assert ckpt["alpha"] == source["alpha"] and state["reward_ema_var"] ** .5 == source["reward_std"]
    replay = state["replay"]
    assert len(replay["buffer"]) == replay["position"] == 100000 < replay["capacity"]
    states = np.stack([row[0] for row in replay["buffer"]])
    assert all(bool(row[4]) == (i % EP_LEN == EP_LEN - 1) for i, row in enumerate(replay["buffer"]))
    batches = sampled_batches(states, state["numpy_rng"], extra["jacobian_sampler"]["rng"], source["source_seed"])
    payload = {"revision": REVISION, "source_id": source_id, "training_seed": source["source_seed"],
               "source_budget": source["source_budget"], "source_protocol": extra["protocol"],
               "actor_optimizer": json_optimizer(ckpt["actor_optimizer"]), "batches": batches,
               "replay_support": {"all": support(states), "reset": support(states[::EP_LEN]),
                                  "early": support(states[np.arange(len(states)) % EP_LEN < 50])},
               "source_jacobian_batches": extra["jacobian_sampler"]["batches"],
               "source_critic_batches": extra["critic_sampler"]["batches"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")
    print(f"Exported Adam and sampled states: {output.stat().st_size} bytes; full replay/checkpoint stays remote", flush=True)


def loss_terms(actor, critic, reference, states, jac_states, alpha, beta, jac_weight, noise):
    action, log_prob = policy_sample(actor, states, noise)
    q = critic(states, action)
    return {"q_mean": -q.mean(0).mean(), "risk": -beta * q.std(0).mean(),
            "entropy": alpha * log_prob.mean(),
            "jacobian": jac_weight * compute_jacobian_loss(actor, reference, jac_states)}


def adam_delta(actor, optimizer_state, gradient, clip):
    shadow = deepcopy(actor)
    parameters = tuple(shadow.parameters())
    optimizer = torch.optim.Adam(parameters)
    optimizer.load_state_dict(deepcopy(optimizer_state))
    offset = 0
    for parameter in parameters:
        parameter.grad = gradient[offset:offset + parameter.numel()].reshape_as(parameter).clone()
        offset += parameter.numel()
    assert offset == gradient.numel()
    norm = torch.nn.utils.clip_grad_norm_(parameters, clip)
    optimizer.step()
    delta = torch.cat([(p.detach() - old.detach()).flatten() for p, old in zip(parameters, actor.parameters())])
    return delta, float(norm), min(1., clip / (float(norm) + 1e-6))


def displaced_actor(actor, delta, scale):
    shadow, offset = deepcopy(actor), 0
    with torch.no_grad():
        for parameter in shadow.parameters():
            parameter.add_(scale * delta[offset:offset + parameter.numel()].reshape_as(parameter))
            offset += parameter.numel()
    assert offset == delta.numel()
    return shadow.eval().requires_grad_(False)


def vector_relation(left, right):
    return {key: value for key, value in relation(left, right).items() if key in ("dot", "cosine")}


def update_directions(actor, critic, reference, source, exported, preflight=False):
    config, device = source["source_protocol"]["config"], next(actor.parameters()).device
    optimizer = tensor_optimizer(exported["actor_optimizer"])
    beta = -2 - config["penalty_scale"] * source["weighted_lambda"]
    parameters = tuple(actor.parameters())
    zero = torch.zeros(sum(p.numel() for p in parameters), device=device)
    momentum, _, _ = adam_delta(actor, optimizer, zero, config["grad_clip"])
    output, directions = {}, {}
    batches = exported["batches"][:1] if preflight else exported["batches"]
    noises = TRAIN_NOISE[:2] if preflight else TRAIN_NOISE
    for arm in ARMS:
        draws, deltas = [], []
        for index, bank in enumerate(batches):
            states, jac = [torch.tensor(bank[name]["states"], device=device) for name in (arm, "jacobian")]
            for seed in noises:
                noise = torch.randn((len(states), 2), generator=torch.Generator().manual_seed(seed)).to(device)
                losses = loss_terms(actor, critic, reference, states, jac, source["alpha"], beta,
                                    config["jac_weight"], noise)
                grads = {k: flat_gradient(v, parameters) for k, v in losses.items()}
                total = sum(grads.values())
                direct = flat_gradient(sum(losses.values()), parameters)
                torch.testing.assert_close(total, direct, rtol=1e-4, atol=1e-7)
                delta, norm, clipping = adam_delta(actor, optimizer, total, config["grad_clip"])
                assert torch.isfinite(delta).all()
                draws.append({"batch": index, "noise_seed": seed, "losses": {k: float(v.detach()) for k, v in losses.items()},
                              "gradient_norms": {k: float(v.norm()) for k, v in grads.items()},
                              "total_gradient_norm": norm, "clip_scale": clipping,
                              "adam_delta_l2": float(delta.norm()),
                              "component_loss_derivatives_under_adam": {k: float(torch.dot(g, delta)) for k, g in grads.items()},
                              "total_descent_vs_jacobian": relation(total, grads["jacobian"]),
                              "adam_vs_zero_data_step": vector_relation(delta, momentum)})
                deltas.append(delta)
        directions[arm] = torch.stack(deltas).mean(0)
        output[arm] = {"draws": draws, "mean_adam_delta_l2": float(directions[arm].norm()),
                       "mean_adam_vs_zero_data_step": vector_relation(directions[arm], momentum)}
    output["uniform_vs_early"] = vector_relation(directions["uniform"], directions["early"])
    output["zero_data_adam_delta_l2"] = float(momentum.norm())
    return directions, output


def compact_continuations(raw, baseline, scales):
    output = {}
    for arm in ARMS:
        output[arm] = {}
        for h in scales:
            plus, minus = [raw[f"{arm}_{sign}_{h}"] for sign in ("plus", "minus")]
            groups = {}
            for group in ("continuations", "critic_values"):
                p, m = plus[group]["actor"], minus[group]["actor"]
                derivatives = {k: (p[k] - m[k]) / (2 * h) for k in p}
                groups[group] = {"mean_derivatives": {k: float(v.mean()) for k, v in derivatives.items()},
                                 "per_draw_mean_derivatives": {k: v.mean(1).tolist() for k, v in derivatives.items()},
                                 "per_path_mean_derivatives": {k: v.mean(0).tolist() for k, v in derivatives.items()}}
            actual, predicted = groups["continuations"], groups["critic_values"]
            plus_cost = plus["continuations"]["actor"]["discounted_cost"] - baseline["continuations"]["actor"]["discounted_cost"]
            groups["plus_step_discounted_cost_change"] = {"mean": float(plus_cost.mean()),
                "per_draw_means": plus_cost.mean(1).tolist(), "per_path_means": plus_cost.mean(0).tolist()}
            groups["soft_score_vs_actual_soft_actor_path_sign_disagreements"] = int(np.sum(
                np.asarray(actual["per_path_mean_derivatives"]["soft_actor_return"])
                * np.asarray(predicted["per_path_mean_derivatives"]["soft_score"]) < 0))
            output[arm][str(h)] = groups
    return output


def run(source, previous, exported, device, raw_dir, ckpt_dir, preflight=False):
    config = source["source_protocol"]["config"]
    actor = EquilibriumGaussianPolicy(6, 2, config["hidden_dim"], activation="tanh").to(device).eval()
    critic = EnsembleQNet(6, 2, config["hidden_dim"], config["num_critics"]).to(device).eval().requires_grad_(False)
    load_actor_parameters(actor, source["actor_parameters"])
    load_actor_parameters(critic, source["critic_parameters"])
    reference = LQRFeedbackReference(source["fitted_gain"]).to(device).eval().requires_grad_(False)
    before = [p.detach().clone() for net in (actor, critic) for p in net.parameters()]
    terminal = {k: source[k] for k in ("alpha", "weighted_lambda", "reward_std")}
    terminal["beta"] = -2 - config["penalty_scale"] * source["weighted_lambda"]
    assert terminal == previous["terminal"]["early"]
    protocol = {"revision": REVISION, "seed": source["source_seed"], "source_id": source["source_id"],
                "preflight_only": preflight, "scales": list(PARAMETER_SCALES)}
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    progress = ckpt_dir / "directions_last.pt"
    if progress.exists():
        saved = torch.load(progress, map_location=device, weights_only=False)
        assert saved["protocol"] == protocol
        directions, gradients = saved["directions"], saved["gradients"]
    else:
        directions, gradients = update_directions(actor, critic, reference, source, exported, preflight)
        torch.save({"protocol": protocol, "directions": directions, "gradients": gradients}, progress)
    actors = {"baseline": deepcopy(actor).requires_grad_(False)}
    for arm in ARMS:
        for h in PARAMETER_SCALES:
            for sign, direction in (("plus", 1), ("minus", -1)):
                actors[f"{arm}_{sign}_{h}"] = displaced_actor(actor, directions[arm], direction * h)
    critics = {k: critic for k in actors}
    gains = {k: reference.gain for k in actors}
    terminals = {k: terminal for k in actors}
    radii, loads = ((1.,), (1,)) if preflight else (RADII, SEVERITIES)
    seeds = list(range(60000, 60002 if preflight else 60100))
    noises = NOISE_SEEDS[:2] if preflight else NOISE_SEEDS
    raw_dir.mkdir(parents=True, exist_ok=True)
    blocks, raw_files = {}, []
    for radius in radii:
        blocks[str(radius)] = {}
        for load in loads:
            raw_path = raw_dir / f"radius{radius}_load{load}.pt"
            if raw_path.exists():
                saved = torch.load(raw_path, weights_only=False, map_location="cpu")
                assert saved["protocol"] == protocol and saved["seeds"] == seeds and saved["noise_seeds"] == list(noises)
                initial, raw = saved["initial"], saved["raw"]
            else:
                initial, raw = stochastic_rollout(actors, critics, gains, terminals, radius, load, seeds, noises, branches={"actor": 0.})
                torch.save({"protocol": protocol, "seeds": seeds, "noise_seeds": list(noises), "initial": initial, "raw": raw}, raw_path)
            baseline = raw["baseline"]
            old = previous["radius_results"][str(radius)][str(load)]
            np.testing.assert_array_equal(initial, np.asarray(old["initial_observations"])[:len(seeds)].astype(np.float32))
            if not preflight:
                for group in ("continuations", "critic_values"):
                    for key, values in baseline[group]["actor"].items():
                        np.testing.assert_allclose(values.mean(0), old["models"]["early"][group]["actor"][key], rtol=1e-4, atol=1e-5)
            states = torch.tensor(initial, device=device)
            projection = {arm: {k: [] for k in COMPONENTS + ("action_reference_mse",)} for arm in ARMS}
            for noise_seed in noises:
                noise = torch.randn((len(states), 2), generator=torch.Generator().manual_seed(noise_seed)).to(device)
                terms = loss_terms(actor, critic, reference, states, states, terminal["alpha"], terminal["beta"], config["jac_weight"], noise)
                terms["action_reference_mse"] = (deterministic_action(actor, states) - reference(states)).square().mean()
                for key, value in terms.items():
                    gradient = flat_gradient(value, tuple(actor.parameters()))
                    for arm in ARMS:
                        projection[arm][key].append(float(torch.dot(gradient, directions[arm])))
            blocks[str(radius)][str(load)] = {"evaluation_seeds": seeds, "noise_seeds": list(noises),
                "disturbance_paths_matched": True, "action_noise_matched": True,
                "baseline_v26_reproduced": not preflight, "initial_component_loss_derivatives": projection,
                "arms": compact_continuations(raw, baseline, PARAMETER_SCALES)}
            raw_files.append({"path": str(raw_path.resolve()), "bytes": raw_path.stat().st_size})
    params = [p for net in (actor, critic) for p in net.parameters()]
    assert all(torch.equal(p, old) and p.grad is None for p, old in zip(params, before))
    return {"gradients": gradients, "radius_results": blocks, "terminal": terminal,
            "raw_files": raw_files, "original_parameters_unchanged": True, "retained_optimizer_updates": 0,
            "shadow_adam_steps": 1 + len(ARMS) * (2 if preflight else BATCHES * len(TRAIN_NOISE)),
            "controller_episodes": len(radii) * len(loads) * len(seeds) * len(noises) * len(actors),
            "training_environment_interactions": 0}


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    for name in ("checkpoint", "portable", "output"):
        export.add_argument(f"--{name}", type=Path, required=True)
    export.add_argument("--source-id", required=True)
    diagnose = commands.add_parser("diagnose")
    diagnose.add_argument("--seed", type=int, choices=TRAINING_SEEDS, required=True)
    for name in ("input-dir", "result-dir", "raw-dir", "ckpt-dir"):
        diagnose.add_argument(f"--{name}", type=Path, required=True)
    diagnose.add_argument("--resume", type=Path)
    diagnose.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    diagnose.add_argument("--threads", type=int, default=2)
    diagnose.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(getattr(args, "threads", 2))
    torch.set_num_interop_threads(1)
    if args.command == "export":
        export_checkpoint(args.checkpoint, args.portable, args.source_id, args.output)
        return
    manifest = json.loads((Path(__file__).resolve().parents[2] / "dispatch_manifest.json").read_text())
    assert manifest["revision"] == REVISION
    source, previous, exported = [json.loads((args.input_dir / f"{name}_seed{args.seed}.json").read_text())
                                   for name in ("early", "v26", "actor_state")]
    assert exported["revision"] == REVISION and exported["source_protocol"] == source["source_protocol"]
    assert exported["source_id"] == source["source_id"] == manifest["source_tasks"][str(args.seed)]["training_id"]
    assert exported["training_seed"] == source["source_seed"] == args.seed
    assert source["source_protocol"]["config"] == manifest["config"] and source["source_budget"] == [500, 100000, 99001]
    assert len(exported["batches"]) == BATCHES and exported["source_jacobian_batches"] == exported["source_critic_batches"] == 99001
    group, = exported["actor_optimizer"]["param_groups"]
    assert group["lr"] == manifest["config"]["lr_actor"] and group["weight_decay"] == manifest["config"]["actor_weight_decay"]
    if args.resume:
        assert args.resume.exists() and args.resume.parent.resolve() == args.ckpt_dir.resolve()
    versions = {"torch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
                "pysindy": pysindy.__version__, "sklearn": sklearn.__version__}
    for key, value in versions.items():
        assert value == manifest["runtime_requirement"][key]
    result = run(source, previous, exported, args.device, args.raw_dir, args.ckpt_dir, args.preflight_only)
    result.update(revision=REVISION, source_id=source["source_id"], training_seed=args.seed,
                  config=manifest["config"], environment_settings=manifest["environment_settings"],
                  preflight_only=args.preflight_only, parameter_scales=list(PARAMETER_SCALES),
                  runtime={**versions, "python": sys.version, "device": args.device},
                  batch_support=[{name: values["support"] for name, values in bank.items()} for bank in exported["batches"]],
                  replay_support=exported["replay_support"])
    result["evaluation_environment_interactions"] = result["controller_episodes"] * EP_LEN
    args.result_dir.mkdir(parents=True, exist_ok=True)
    path = args.result_dir / f"{REVISION}_seed{args.seed}.json"
    path.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n")
    print(f"Completed full-parameter diagnostic: {path.stat().st_size} bytes; original parameters unchanged", flush=True)


if __name__ == "__main__":
    main()
