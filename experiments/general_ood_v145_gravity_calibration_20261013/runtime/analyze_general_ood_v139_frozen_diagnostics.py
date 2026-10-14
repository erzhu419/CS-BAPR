"""Zero-update replay forensics on final, full-state v137/v138 checkpoints."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.distributions import Normal, TanhTransform

import run_general_ood_v132_source_stability as v132
from csbapr.config import CSBAPRConfig

REVISION = "general_ood_v139_frozen_diagnostics"
CASES = ("v138_seed3_friction-only", "v138_seed4_gravity-only",
         "v138_seed4_friction-only", "v138_seed3_gravity-only", "v137_seed3_id", "v137_seed4_id")
AGE_BOUNDS = (0, 25000, 50000, 100000, 200000, 300000)


def replay_indices(size, per_bin, seed=139000):
    assert size == AGE_BOUNDS[-1]
    rng = np.random.default_rng(seed)
    return [(lo, hi, rng.choice(np.arange(lo, hi), per_bin, replace=False))
            for lo, hi in zip(AGE_BOUNDS[:-1], AGE_BOUNDS[1:])]


def describe(tensor):
    values = tensor.detach().double().reshape(-1)
    finite = values[torch.isfinite(values)]
    result = dict(count=values.numel(), nonfinite=int((~torch.isfinite(values)).sum()))
    if finite.numel():
        result.update(mean=float(finite.mean()), abs_mean=float(finite.abs().mean()),
                      min=float(finite.min()), max=float(finite.max()),
                      p50=float(finite.quantile(.5)), p95=float(finite.quantile(.95)))
    return result


def density_terms(mean, log_std, noise, action_range=1.):
    std = log_std.exp()
    u = mean + std*noise
    action = u.tanh()
    normal_native = Normal(mean, std).log_prob(u)
    normal_reparam = Normal(torch.zeros_like(noise), torch.ones_like(noise)).log_prob(noise)-log_std
    native_jac = torch.log(1.-action.square()+1e-6)
    stable_jac = TanhTransform().log_abs_det_jacobian(u, action)
    scale = np.log(action_range)
    return dict(action=action*action_range, u=u,
                native=(normal_native-native_jac-scale).sum(1),
                stable=(normal_native-stable_jac-scale).sum(1),
                reparam_stable=(normal_reparam-stable_jac-scale).sum(1),
                normal_gap=normal_native-normal_reparam)


def actor_probe(actor, state, noise):
    with torch.no_grad():
        mean, log_std = actor(state)
    mean, log_std = mean.detach().requires_grad_(), log_std.detach().requires_grad_()
    terms = density_terms(mean, log_std, noise, actor.action_range)
    gradients = {}
    for name in ("native", "stable", "reparam_stable"):
        gm, gs = torch.autograd.grad(terms[name].sum(), (mean, log_std), retain_graph=True)
        gradients[name] = (gm.detach(), gs.detach())
    report = {name: describe(value) for name, value in terms.items() if name != "action"}
    report.update(mean=describe(mean), log_std=describe(log_std), std=describe(log_std.exp()),
        saturation_fraction=float((terms["u"].tanh().abs() == 1.).float().mean()),
        log_std_min_fraction=float((log_std == actor.log_std_min).float().mean()),
        log_std_max_fraction=float((log_std == actor.log_std_max).float().mean()),
        stable_minus_native=describe(terms["stable"]-terms["native"]))
    for name, (gm, gs) in gradients.items():
        report[name+"_mean_gradient"] = describe(gm)
        report[name+"_log_std_gradient"] = describe(gs)
    report["mean_gradient_gap"] = describe(gradients["stable"][0]-gradients["native"][0])
    return {key: value.detach() for key, value in terms.items()}, report


def target_terms(reward, done, q_min, log_prob, alpha, gamma, reward_std):
    mask = (1.-done)*gamma
    return dict(reward=reward/reward_std, bootstrap=mask*q_min,
                entropy=-mask*alpha*log_prob)


def probe_agent(agent, per_bin):
    assert torch.device(agent.device).type == "cpu" and agent.config.weight_reg == 0.
    replay = agent.replay_buffer
    assert replay.size < replay.capacity and replay.position == replay.size
    noise_rng = torch.Generator(device="cpu").manual_seed(139001)
    reward_std = max(agent._reward_ema_var**.5, 1e-6)
    bins = []
    for lo, hi, indices in replay_indices(replay.size, per_bin):
        rows = [replay.buffer[int(i)] for i in indices]
        states, actions, rewards, next_states, dones, contexts, next_contexts = zip(*rows)
        tensors = [torch.as_tensor(np.stack(x), dtype=torch.float32) for x in
                   (states, actions, rewards, next_states, dones, contexts, next_contexts)]
        state, action, reward, next_state, done, context, next_context = tensors
        state = agent._augment_state_tensor(state, context)
        next_state = agent._augment_state_tensor(next_state, next_context)
        noise = torch.randn((2, per_bin, agent.config.action_dim), generator=noise_rng)
        current, current_report = actor_probe(agent.actor, state, noise[0])
        future, future_report = actor_probe(agent.actor, next_state, noise[1])
        with torch.no_grad():
            behavior_q = agent.critic(state, action)
            policy_q = agent.critic(state, current["action"])
            next_current_q = agent.critic(next_state, future["action"])
            next_target_q = agent.target_critic(next_state, future["action"])
            terms = target_terms(reward, done, next_target_q.min(0)[0], future["native"],
                                 agent.alpha, agent.config.gamma, reward_std)
            target = sum(terms.values())
            stable_terms = target_terms(reward, done, next_target_q.min(0)[0], future["stable"],
                                        agent.alpha, agent.config.gamma, reward_std)
            stable_target = sum(stable_terms.values())
            residual = behavior_q-target
            values = dict(behavior_q=behavior_q, policy_q=policy_q,
                policy_minus_behavior_q=policy_q-behavior_q,
                policy_q_mean_minus_2std=policy_q.mean(0)-2.*policy_q.std(0),
                behavior_action_delta=(current["action"]-action).norm(dim=1),
                next_current_q=next_current_q, next_target_q=next_target_q,
                next_current_minus_target_q=next_current_q-next_target_q,
                target=target, target_alpha_zero=terms["reward"]+terms["bootstrap"],
                stable_target_minus_native=stable_target-target,
                td_residual=residual, td_squared=residual.square(),
                **{key+"_term": value for key, value in terms.items()})
            top = target.abs().topk(min(3, per_bin)).indices.tolist()
            examples = [dict(replay_index=int(indices[i]), terminated=float(done[i]),
                target=float(target[i]), stable_target=float(stable_target[i]),
                reward_term=float(terms["reward"][i]), bootstrap_term=float(terms["bootstrap"][i]),
                entropy_term=float(terms["entropy"][i]), behavior_q_mean=float(behavior_q[:, i].mean()),
                policy_q_mean=float(policy_q[:, i].mean()), next_log_prob=float(future["native"][i]),
                next_pretanh=future["u"][i].tolist()) for i in top]
        bins.append(dict(collection_steps=[lo+1, hi], indices=indices.tolist(),
            terminated_count=int(done.sum()), quantities={k: describe(v) for k, v in values.items()},
            actor=current_report, next_actor=future_report, largest_targets=examples))
    return dict(alpha=float(agent.alpha), reward_normalization_std=reward_std,
        reward_normalization="saved final EMA std; no hypothetical EMA update",
        training_interactions=0, optimizer_updates=0, environment_steps=0,
        replay_size=replay.size, source_optimizer_updates=agent.training_steps,
        samples_per_bin=per_bin, sampling_seed=139000, action_noise_seed=139001,
        gradient_coordinates="detached actor mean/log_std heads, no model parameter gradients",
        bins=bins)


def run(args):
    started = time.monotonic()
    torch.set_num_threads(2)
    source = json.loads(args.source_json.read_text())
    revision, seed, profile = args.case.split("_", 2)
    assert source["revision"].startswith("general_ood_"+revision+"_")
    assert source["training_seed"] == int(seed.removeprefix("seed"))
    assert source["training_profile"] == profile and not source["smoke"]
    assert source["training"]["interactions"] == 300000
    config = CSBAPRConfig(**dict(source["config"], device="cpu"))
    agent = v132.SourceStabilityAgent(config.state_dim, config.action_dim, config, source["arm"])
    extra = agent.load(source["resumable_checkpoint"])
    assert extra["revision"] == v132.REVISION and extra["spec"] == source["spec"]
    assert extra["seed"] == source["training_seed"] and extra["loop"]["step"] == 300000
    assert agent.training_steps == source["training"]["optimizer_updates"] == 299745
    assert agent.context_tracker._count == source["final_context_count"] == 300000
    assert asdict(agent.config) == dict(source["config"], device="cpu")
    row = dict(revision=REVISION, case=args.case, smoke=args.smoke, device="cpu",
        source=dict(revision=source["revision"], seed=source["training_seed"], profile=profile,
                    checkpoint=source["resumable_checkpoint"], training_steps=300000,
                    config=source["config"], spec=source["spec"]),
        diagnostic=probe_agent(agent, 8 if args.smoke else 128),
        limitation="Final frozen weights on collection-age-stratified replay, not historical weights or causal attribution",
        runtime=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__),
        seconds=time.monotonic()-started)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    destination = args.result_dir/"diagnostics.json"
    destination.write_text(json.dumps(row, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] {args.case}: {row['diagnostic']['samples_per_bin']*5} replay samples; zero updates", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
