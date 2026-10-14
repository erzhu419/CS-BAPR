"""Read-only reward, source-policy and frozen-gradient diagnosis of v119."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_general_ood_v119_actor_feedback as v119
from scripts import probe_general_ood_v70_physical_audit as physical

REVISION = "general_ood_v120_mechanism"
NOISE_SEEDS = (12000, 12001, 12002)


def reward_settings(env):
    raw = env.unwrapped
    assert raw._terminate_when_unhealthy and raw._healthy_reward == 1.
    return dict(healthy_reward=float(raw._healthy_reward), ctrl_cost_weight=float(raw._ctrl_cost_weight),
                forward_reward_weight=float(raw._forward_reward_weight))


def decompose(rewards, actions, settings):
    cost = settings["ctrl_cost_weight"]*np.square(actions).sum(axis=-1)
    forward = np.asarray(rewards)-settings["healthy_reward"]+cost
    velocity = forward/settings["forward_reward_weight"]
    return cost, forward, velocity


def reward_statistics(trace, settings, spec):
    cost, forward, velocity = decompose(trace["rewards"], trace["executed_actions"], settings)
    output = {}
    for name, (a, b) in v119.windows(spec).items():
        output[name] = dict(mean_reward=float(trace["rewards"][a:b].mean()),
            mean_healthy_reward=settings["healthy_reward"], mean_control_cost=float(cost[a:b].mean()),
            mean_forward_reward=float(forward[a:b].mean()), mean_forward_velocity=float(velocity[a:b].mean()),
            fraction_abs_velocity_below_point1=float(np.mean(np.abs(velocity[a:b]) < .1)),
            mean_risk=float(trace["risk"][a:b].mean()), physical_terminations=int(trace["terminated"][a:b].sum()))
    return output


def probe_pools(saved, trace, smoke=False):
    buffer = saved["training_state"]["replay"]["buffer"]
    assert len(buffer) == 12144 and saved["training_steps"] == 105745
    pools = {}
    for name, a, b in (("prechange", 3500, 3800), ("post_compound", 5700, 6000)):
        indices = np.linspace(a, b-1, 16 if smoke else 256).astype(int)
        samples = [buffer[6144+int(i)] for i in indices]
        states, actions, contexts = (np.stack([r[k] for r in samples]) for k in (0, 1, 5))
        np.testing.assert_array_equal(states, trace["states"][indices])
        np.testing.assert_array_equal(actions, trace["actions"][indices])
        pools[name] = (states, actions, contexts)
    return pools


def gradient_probe(agent, pool, noise_seed):
    states, recorded_actions, contexts = [torch.as_tensor(x, dtype=torch.float32, device=agent.device) for x in pool]
    torch.manual_seed(noise_seed)
    actions, log_prob, _, mean, log_std = agent.actor.evaluate(agent._augment_state_tensor(states, contexts))
    q = agent.critic(agent._augment_state_tensor(states, contexts), actions)
    q_mean, q_std = q.mean(dim=0), q.std(dim=0)
    cap = float(agent.config.policy_q_std_cap or 0.)
    sigma = q_std.clamp(max=cap) if cap > 0 else q_std
    losses = dict(q_mean=-q_mean.mean(), q_std_unit=sigma.mean(), entropy=agent.alpha*log_prob.mean(),
                  bc=agent.config.beta_bc*torch.nn.functional.mse_loss(actions, recorded_actions))
    params = list(agent.actor.parameters())
    vectors = {}
    for name, loss in losses.items():
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        vectors[name] = torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p, g in zip(params, grads)])
    base = vectors["q_mean"]+2.*vectors["q_std_unit"]+vectors["entropy"]+vectors["bc"]
    stronger = base+vectors["q_std_unit"]
    norms = {k: float(v.norm().item()) for k, v in vectors.items()}
    def cosine(x, y):
        denom = float(x.norm()*y.norm())
        return float(torch.dot(x, y).item()/denom) if denom else None
    base_norm = float(base.norm().item())
    result = dict(noise_seed=noise_seed, samples=len(states), mean_q=float(q_mean.mean().item()),
        mean_q_std=float(q_std.mean().item()), mean_policy_q_std=float(sigma.mean().item()), alpha=float(agent.alpha),
        reward_normalization_std=max(float(getattr(agent, "_reward_ema_var", 1.))**.5, 1e-6),
        mean_policy_std=float(log_std.exp().mean().item()), mean_abs_sampled_action=float(actions.abs().mean().item()),
        mean_abs_deterministic_action=float(mean.tanh().abs().mean().item()),
        loss_components={k: float(v.detach().item()) for k, v in losses.items()}, gradient_norms=norms,
        beta_minus2_gradient_norm=base_norm, beta_minus3_gradient_norm=float(stronger.norm().item()),
        unit_beta_increment_over_base_gradient=norms["q_std_unit"]/base_norm if base_norm else None,
        beta_minus2_vs_minus3_gradient_cosine=cosine(base, stronger),
        q_mean_vs_std_gradient_cosine=cosine(vectors["q_mean"], vectors["q_std_unit"]),
        grad_clip=float(agent.config.grad_clip), optimizer_steps=0,
        gradient_scope="raw actor-parameter loss gradients before clipping, Adam and weight decay; not historical updates")
    assert all(np.isfinite(vectors[k].detach().cpu().numpy()).all() for k in vectors)
    return result


def evaluate_source(agent, mode, seed, episodes, max_steps):
    import gymnasium as gym

    snapshot, updates = deepcopy(agent.snapshot_context()), agent.training_steps
    env = gym.make("Hopper-v4")
    physical.apply_mode(env, mode)
    settings, rows = reward_settings(env), []
    for episode in range(episodes):
        agent.restore_context(deepcopy(snapshot))
        agent.reset_context()
        state, _ = env.reset(seed=seed+episode)
        rewards, velocities, costs = [], [], []
        for _ in range(max_steps):
            action = agent.select_action(state, deterministic=True)
            executed = np.clip(action*env.action_space.high, env.action_space.low, env.action_space.high)
            next_state, reward, term, trunc, info = env.step(executed)
            cost, forward, velocity = decompose(np.asarray([reward]), np.asarray([executed]), settings)
            np.testing.assert_allclose(velocity[0], info["x_velocity"], rtol=1e-9, atol=1e-9)
            rewards.append(float(reward))
            velocities.append(float(info["x_velocity"]))
            costs.append(float(cost[0]))
            agent.observe_context(state, action, reward, next_state, term or trunc)
            state = next_state
            if term or trunc:
                break
        rows.append(dict(seed=seed+episode, episode_return=float(np.sum(rewards)), length=len(rewards),
            mean_forward_velocity=float(np.mean(velocities)), mean_control_cost=float(np.mean(costs)),
            physical_termination=bool(term), time_limit_truncation=bool(trunc)))
    env.close()
    agent.restore_context(snapshot)
    assert agent.training_steps == updates
    return dict(mode=mode, episodes=rows, mean_return=float(np.mean([r["episode_return"] for r in rows])),
        mean_episode_velocity=float(np.mean([r["mean_forward_velocity"] for r in rows])),
        environment_step_calls=sum(r["length"] for r in rows), reward_formula_verified_against_info=True)


def run(args):
    import gymnasium as gym

    torch.set_num_threads(2)
    assert args.seed in v119.SEEDS
    modes = ("normal", "compound") if args.smoke else physical.MODES
    noises = NOISE_SEEDS[:1] if args.smoke else NOISE_SEEDS
    spec = dict(seed=args.seed, smoke=args.smoke, modes=list(modes), noise_seeds=list(noises),
                evaluation_episodes=1 if args.smoke else 3, evaluation_max_steps=64 if args.smoke else 1000,
                gradient_samples=16 if args.smoke else 256)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint_dir/"progress.pt"
    progress = torch.load(args.resume_from, map_location="cpu", weights_only=False) if args.resume_from else dict(
        revision=REVISION, spec=spec, trace_records=[], source_evaluation=[], gradients={})
    assert progress["revision"] == REVISION and progress["spec"] == spec
    def save():
        temporary = checkpoint.with_suffix(".tmp")
        torch.save(progress, temporary)
        temporary.replace(checkpoint)
    env = gym.make("Hopper-v4")
    settings = reward_settings(env)
    agent = v119.ContinuationAgent(env, "fixed-minus2", "cpu")
    env.close()
    population = [(arm, mode) for arm in v119.ARMS for mode in modes]
    for arm, mode in population[len(progress["trace_records"]):]:
        directory = args.v119_dir/f"run_checkpoints/seed{args.seed}_{arm}"
        with np.load(directory/f"{arm}_{mode}.npz") as trace:
            assert len(trace["states"]) == 6000
            stats = reward_statistics(trace, settings, v119.SPEC)
        data = json.loads((args.v119_dir/f"run_results/seed{args.seed}_{arm}/actor_feedback.json").read_text())
        previous = next(r for r in data["records"] if r["mode"] == mode)
        for window, row in stats.items():
            for key in ("mean_reward", "mean_risk", "physical_terminations"):
                np.testing.assert_allclose(row[key], previous["windows"][window][key], rtol=0, atol=1e-12)
        progress["trace_records"].append(dict(arm=arm, mode=mode, windows=stats))
        save()
    shared = torch.load(args.shared_dir/"shared.pt", map_location="cpu", weights_only=False)
    agent.start_shared(args.source_dir/v119.v53.SOURCE_NAME, shared)
    assert agent.training_steps == 99745 and agent.replay_buffer.size == 6144
    for mode in modes[len(progress["source_evaluation"]):]:
        seed = 140000+1000*v119.SEEDS.index(args.seed)+100*physical.MODES.index(mode)
        row = evaluate_source(agent, mode, seed, spec["evaluation_episodes"], spec["evaluation_max_steps"])
        progress["source_evaluation"].append(row)
        save()
        print(f"[SOURCE] seed{args.seed} {mode} return={row['mean_return']:.3f} velocity={row['mean_episode_velocity']:.5f}", flush=True)
    directory = args.v119_dir/f"run_checkpoints/seed{args.seed}_fixed-minus2"
    saved = torch.load(directory/"latest.pt", map_location="cpu", weights_only=False)
    assert saved["extra_state"]["progress"]["next_cell"] == 5
    assert saved["extra_state"]["progress"]["records"][-1]["mode"] == "compound"
    with np.load(directory/"fixed-minus2_compound.npz") as trace:
        pools = probe_pools(saved, trace, args.smoke)
    del saved
    for label in ("source", *v119.ARMS):
        if label in progress["gradients"]:
            continue
        if label == "source":
            agent.start_shared(args.source_dir/v119.v53.SOURCE_NAME, shared)
        else:
            extra = agent.load(str(args.v119_dir/f"run_checkpoints/seed{args.seed}_{label}/latest.pt"))
            assert agent.training_steps == 105745 and extra["progress"]["next_cell"] == 5
        progress["gradients"][label] = {p: [gradient_probe(agent, pool, seed) for seed in noises] for p, pool in pools.items()}
        save()
        print(f"[GRADIENT] seed{args.seed} {label}: two common pools", flush=True)
    result = dict(**progress, passed=True, source_revision=v119.REVISION, reward_settings=settings,
        optimizer_updates=0, existing_trace_paths=len(population), readout_refitted=False, readout_adopted=False,
        evaluation_interactions=sum(r["environment_step_calls"] for r in progress["source_evaluation"]),
        probe_pool="same fixed-minus2 compound replay, prechange3500:3800 and post5700:6000; saved online contexts",
        final_checkpoints="compound endpoints only; other mode actors were overwritten by v119 latest.pt",
        limitations=["Read-only development diagnosis conditional on the same three environment seeds and one trained source.",
            "Raw endpoint gradients are not recorded training updates or predicted return improvements.",
            "Source evaluation starts from the same shared context snapshot as v119; noise repeats are not independent seeds."])
    args.result_dir.mkdir(parents=True, exist_ok=True)
    path = args.result_dir/("native_smoke.json" if args.smoke else "mechanism.json")
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] {REVISION}: {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("v119-dir", "source-dir", "shared-dir", "checkpoint-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=v119.SEEDS, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
