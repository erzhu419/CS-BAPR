"""Frozen replay/gradient diagnosis of the predeclared v130 source states."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import numpy as np
import torch

import run_general_ood_v122_target_l1 as v122
from csbapr.utils import compute_reg_norm

REVISION = "general_ood_v131_source_diagnostics"
SOURCE_CELLS = ((3, 100000), (4, 100000), (1, 300000), (2, 300000),
                (3, 300000), (4, 300000))
NOISE_SEEDS = (13100, 13101, 13102)


def describe(values):
    values = torch.as_tensor(values, dtype=torch.float64).detach().cpu().flatten()
    if not values.numel():
        return {"count": 0}
    assert torch.isfinite(values).all()
    quantiles = torch.quantile(values, torch.tensor([.5, .9, .99], dtype=torch.float64))
    return dict(count=values.numel(), mean=values.mean().item(), min=values.min().item(),
                p50=quantiles[0].item(), p90=quantiles[1].item(),
                p99=quantiles[2].item(), max=values.max().item())


def residual_summary(td):
    td = td.detach().double().cpu()
    if td.shape[1] == 0:
        return {"transitions": 0}
    per_transition = td.square().mean(0)
    total = per_transition.sum().item()
    descending = per_transition.sort(descending=True).values
    share = lambda fraction: (descending[:math.ceil(len(descending)*fraction)].sum().item()/total
                              if total else 0.)
    return dict(transitions=td.shape[1], mse=per_transition.mean().item(),
                abs_td=describe(td.abs()), top1pct_loss_share=share(.01),
                top10pct_loss_share=share(.1))


def gradient_norm(loss, parameters):
    gradients = torch.autograd.grad(loss, tuple(parameters), retain_graph=True, allow_unused=True)
    return math.sqrt(sum(g.detach().double().square().sum().item()
                         for g in gradients if g is not None))


def diagnose_batch(agent, indices):
    sample = [agent.replay_buffer.buffer[int(i)] for i in indices]
    arrays = [np.stack([row[k] for row in sample]) for k in range(7)]
    raw_states, actions, rewards, raw_next, dones, contexts, next_contexts = [
        torch.as_tensor(x, dtype=torch.float32, device=agent.device) for x in arrays]
    states = agent._augment_state_tensor(raw_states, contexts)
    next_states = agent._augment_state_tensor(raw_next, next_contexts)
    policy_action, policy_log_prob, _, mean, log_std = agent.actor.evaluate(states)
    with torch.no_grad():
        next_action, next_log_prob, *_ = agent.actor.evaluate(next_states)
        variance = .99*agent._reward_ema_var+.01*rewards.var().item()
        reward_std = max(variance**.5, 1e-6)
        normalized = rewards/reward_std
        bootstrap = agent.config.gamma*(1.-dones)
        next_q = agent.target_critic(next_states, next_action).min(0).values
        discounted_q = bootstrap*next_q
        discounted_entropy = bootstrap*(-agent.alpha*next_log_prob)
        norms = compute_reg_norm(agent.target_critic)
        uplift = bootstrap[None, :]*agent.config.weight_reg*norms[:, None]
        target = normalized[None, :]+discounted_q[None, :]+discounted_entropy[None, :]+uplift
        native_loss, *_ = agent._compute_q_loss_bapr_fallback(
            states, actions, normalized[:, None], next_states, dones[:, None],
            next_action, next_log_prob, norms, len(indices))
    predicted = agent.critic(states, actions)
    td = predicted-target
    critic_mse = td.square().mean()
    critic_disagreement = agent.config.beta_ood*predicted.std(0).mean()
    critic_loss = critic_mse+critic_disagreement
    torch.testing.assert_close(critic_loss.detach(), native_loss)
    policy_q = agent.critic(states, policy_action)
    beta = agent._effective_policy_beta(0.)
    assert beta == -2.
    components = dict(q_mean=-policy_q.mean(),
                      uncertainty=-beta*policy_q.std(0).mean(),
                      entropy=agent.alpha*policy_log_prob.mean(),
                      behavior_cloning=agent.config.beta_bc*torch.nn.functional.mse_loss(policy_action, actions))
    actor_loss = sum(components.values())
    gradients = dict(critic_total=gradient_norm(critic_loss, agent.critic.parameters()),
                     critic_mse=gradient_norm(critic_mse, agent.critic.parameters()),
                     critic_disagreement=gradient_norm(critic_disagreement, agent.critic.parameters()),
                     actor_total=gradient_norm(actor_loss, agent.actor.parameters()))
    gradients.update({"actor_"+key: gradient_norm(loss, agent.actor.parameters())
                      for key, loss in components.items()})
    values = dict(predicted_q=predicted, target=target, td_error=td, done=dones,
        raw_reward=rewards, normalized_reward=normalized, discounted_bootstrap_q=discounted_q,
        discounted_entropy=discounted_entropy, discounted_l1_uplift=uplift,
        state_abs_max=raw_states.abs().amax(1), next_state_abs_max=raw_next.abs().amax(1),
        augmented_state_abs_max=states.abs().amax(1),
        context_abs_max=contexts.abs().amax(1), next_context_abs_max=next_contexts.abs().amax(1),
        policy_action_abs_max=policy_action.abs().amax(1), policy_mean_abs_max=mean.abs().amax(1),
        policy_std_mean=log_std.exp().mean(1), policy_log_prob=policy_log_prob,
        next_policy_log_prob=next_log_prob)
    return ({key: value.detach().cpu() for key, value in values.items()},
            dict(critic_loss=critic_loss.item(), actor_loss=actor_loss.item(),
                 reward_std=reward_std, **gradients))


def diagnose_pool(agent, indices, noise_seed, batch_size=256):
    values, batches = {}, []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(noise_seed)
        for offset in range(0, len(indices), batch_size):
            batch_values, metrics = diagnose_batch(agent, indices[offset:offset+batch_size])
            batches.append(metrics)
            for key, value in batch_values.items():
                values.setdefault(key, []).append(value)
    values = {key: torch.cat(parts, dim=-1) for key, parts in values.items()}
    td, dones = values["td_error"], values["done"].bool()
    per_transition = td.double().square().mean(0)
    worst = per_transition.topk(min(5, len(indices))).indices.tolist()
    return dict(samples=len(indices), batches=len(batches), noise_seed=noise_seed,
        residuals=residual_summary(td), replay_done=residual_summary(td[:, dones]),
        replay_not_done=residual_summary(td[:, ~dones]),
        per_critic=[dict(member=i, **residual_summary(td[i:i+1])) for i in range(td.shape[0])],
        scales={key: describe(value) for key, value in values.items() if key != "td_error"},
        batch_metrics={key: describe([b[key] for b in batches]) for key in batches[0]},
        native_critic_loss_reconstructed=True,
        worst_transitions=[dict(replay_index=int(indices[i]), replay_done=bool(dones[i]),
            raw_reward=values["raw_reward"][i].item(),
            state_abs_max=values["state_abs_max"][i].item(),
            next_state_abs_max=values["next_state_abs_max"][i].item(),
            predicted_q=values["predicted_q"][:, i].tolist(), target=values["target"][:, i].tolist(),
            td_error=td[:, i].tolist(), discounted_bootstrap_q=values["discounted_bootstrap_q"][i].item(),
            discounted_entropy=values["discounted_entropy"][i].item()) for i in worst])


def run(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    source = json.loads(args.source_json.read_text())
    expected_revision = v122.REVISION if args.source_steps == 100000 else "general_ood_v123_source_extension"
    assert (args.seed, args.source_steps) in SOURCE_CELLS
    assert source["revision"] == expected_revision and not source["smoke"]
    assert source["training_seed"] == args.seed and source["training"]["interactions"] == args.source_steps
    checkpoint = Path(source["checkpoint"]).with_name("latest.pt")
    env = gym.make("Hopper-v4")
    agent = v122.make_agent(env, "cpu")
    env.close()
    assert asdict(agent.config) == dict(source["config"], device="cpu")
    extra = agent.load(str(checkpoint))
    assert extra["seed"] == args.seed and extra["loop"]["step"] == args.source_steps
    assert agent.training_steps == args.source_steps-255 and agent.replay_buffer.size == args.source_steps
    assert agent.context_tracker._count == args.source_steps
    assert agent.config.weight_reg == agent.config.penalty_scale == 0.
    assert agent.config.use_context and agent.config.enable_min_q_target and agent.config.enable_reward_ema
    assert not agent.config.use_nau_actor and agent.config.weight_sym == agent.config.jac_weight == 0.
    assert agent.pi_sym_torch is None and agent._actor_auxiliary_loss(None) is None
    before = (agent.training_steps, agent.replay_buffer.size, agent._reward_ema_var,
              agent.context_tracker._count, agent.alpha)
    samples = 256 if args.smoke else 4096
    noise_seeds = NOISE_SEEDS[:1] if args.smoke else NOISE_SEEDS
    pools = {}
    for name, start in (("uniform_training", 0), ("last10k", args.source_steps-10000)):
        indices = np.linspace(start, args.source_steps-1, samples).astype(int)
        pools[name] = [diagnose_pool(agent, indices, noise) for noise in noise_seeds]
        print(f"[PROBE] seed{args.seed} age{args.source_steps} {name}: samples={samples}", flush=True)
    assert before == (agent.training_steps, agent.replay_buffer.size, agent._reward_ema_var,
                      agent.context_tracker._count, agent.alpha)
    assert all(p.grad is None for p in (*agent.actor.parameters(), *agent.critic.parameters()))
    assert not torch.cuda.is_initialized()
    result = dict(revision=REVISION, training_seed=args.seed, source_steps=args.source_steps,
        source_revision=source["revision"], checkpoint=str(checkpoint), smoke=args.smoke,
        checkpoint_selection="predeclared-final-full-state", source_readout_refitted=False,
        new_training_interactions=0, environment_step_calls=0, optimizer_updates=0,
        gradient_evaluation_only=True, critic_gradient_clip=agent.config.grad_clip, alpha=agent.alpha,
        frozen_reward_ema_var=agent._reward_ema_var, noise_seeds=list(noise_seeds), pools=pools,
        loss_scope="Frozen endpoint objectives; no alpha/critic step before actor gradients. Reward std is recomputed per native-size batch without advancing its EMA.",
        done_semantics="Replay stores terminated-or-truncated; these groups do not separate physical termination from time limits.",
        runtime=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
                     gymnasium=gym.__version__, mujoco=__import__("mujoco").__version__,
                     device="cpu", cuda_initialized=False),
        seconds=time.monotonic()-started)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    (args.result_dir/"source_diagnostics.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] seed{args.seed} age{args.source_steps}: optimizer_updates=0 environment_steps=0", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--source-steps", type=int, choices=(100000, 300000), required=True)
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
