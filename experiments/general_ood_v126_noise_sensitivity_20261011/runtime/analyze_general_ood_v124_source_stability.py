"""Frozen source-policy stability and actor-gradient diagnosis; no updates."""
import argparse
from collections import deque
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import numpy as np
import torch

import run_general_ood_v122_target_l1 as v122

REVISION = "general_ood_v124_source_stability"
NOISE_SEEDS = (12400, 12401, 12402)


def stage_spec(smoke=False):
    return dict(smoke=smoke, ages=[100000, 300000], reset_seeds=list(range(150000, 150001 if smoke else 150005)),
                noise_seeds=list(NOISE_SEEDS[:1] if smoke else NOISE_SEEDS),
                max_steps=64 if smoke else 1000, gradient_samples=16 if smoke else 256)


@contextmanager
def frozen_context_rng(agent):
    context = deepcopy(agent.snapshot_context())
    numpy_rng, python_rng = np.random.get_state(), random.getstate()
    updates, size = agent.training_steps, agent.replay_buffer.size
    try:
        with torch.random.fork_rng(devices=[]):
            yield
    finally:
        agent.restore_context(context)
        np.random.set_state(numpy_rng)
        random.setstate(python_rng)
    assert (agent.training_steps, agent.replay_buffer.size) == (updates, size)


def physical_state(raw):
    z, angle = map(float, raw.data.qpos[1:3])
    state = raw.state_vector()[2:]
    checks = dict(height=raw._healthy_z_range[0] < z < raw._healthy_z_range[1],
                  angle=raw._healthy_angle_range[0] < angle < raw._healthy_angle_range[1],
                  state=bool(np.all((raw._healthy_state_range[0] < state) &
                                    (state < raw._healthy_state_range[1]))))
    assert all(checks.values()) == bool(raw.is_healthy)
    return dict(height=z, angle=angle, max_abs_qvel=float(np.max(np.abs(raw.data.qvel))),
                unhealthy_conditions=[k for k, healthy in checks.items() if not healthy])


def evaluate_episode(agent, reset_seed, noise_seed, max_steps):
    with frozen_context_rng(agent):
        if noise_seed is not None:
            torch.random.default_generator.manual_seed(noise_seed)
        agent.reset_context()
        env = gym.make("Hopper-v4")
        raw, tail = env.unwrapped, deque(maxlen=5)
        rewards, velocities, differences, stds = [], [], [], []
        try:
            state, _ = env.reset(seed=reset_seed)
            for step in range(max_steps):
                tensor = torch.as_tensor(agent._augment_state_np(state), device=agent.device).unsqueeze(0)
                with torch.no_grad():
                    mean, log_std = agent.actor(tensor)
                deterministic_action = (agent.actor.action_range*mean.tanh()).squeeze(0).numpy()
                action = agent.select_action(state, deterministic=noise_seed is None)
                executed = np.clip(action*env.action_space.high, env.action_space.low, env.action_space.high)
                pre = physical_state(raw)
                next_state, reward, term, trunc, info = env.step(executed)
                tail.append(dict(step=step+1, before=pre, after=physical_state(raw),
                                 action=action.tolist(), velocity=float(info["x_velocity"])))
                rewards.append(float(reward))
                velocities.append(float(info["x_velocity"]))
                differences.append(float(np.linalg.norm(action-deterministic_action)))
                stds.append(float(log_std.exp().mean()))
                agent.observe_context(state, action, reward, next_state, term or trunc)
                state = next_state
                if term or trunc:
                    break
            return dict(reset_seed=reset_seed, noise_seed=noise_seed,
                        action_mode="deterministic" if noise_seed is None else "sampled",
                        length=len(rewards), episode_return=float(np.sum(rewards)),
                        mean_velocity=float(np.mean(velocities)), physical_termination=bool(term),
                        time_limit_truncation=bool(trunc), diagnostic_cutoff=not(term or trunc),
                        mean_action_noise_l2=float(np.mean(differences)), mean_policy_std=float(np.mean(stds)),
                        final_physical_state=physical_state(raw), tail=list(tail))
        finally:
            env.close()


def replay_pools(agent, samples):
    buffer, size = agent.replay_buffer.buffer, agent.replay_buffer.size
    pools = {}
    for name, start in (("full", 0), ("recent10k", max(0, size-10000))):
        indices = np.linspace(start, size-1, min(samples, size-start)).astype(int)
        pools[name] = tuple(np.stack([buffer[int(i)][k] for i in indices]) for k in (0, 1, 5))
    return pools


def gradient_probe(agent, pool, noise_seed):
    with frozen_context_rng(agent):
        states, recorded, contexts = [torch.as_tensor(x, dtype=torch.float32, device=agent.device) for x in pool]
        torch.random.default_generator.manual_seed(noise_seed)
        state = agent._augment_state_tensor(states, contexts)
        actions, log_prob, _, mean, log_std = agent.actor.evaluate(state)
        q = agent.critic(state, actions)
        q_mean, q_std = q.mean(dim=0), q.std(dim=0)
        cap = float(agent.config.policy_q_std_cap or 0.)
        sigma = q_std.clamp(max=cap) if cap > 0 else q_std
        losses = dict(q=-q_mean.mean(), risk=2.*sigma.mean(), entropy=agent.alpha*log_prob.mean(),
                      bc=agent.config.beta_bc*torch.nn.functional.mse_loss(actions, recorded))
        params = list(agent.actor.parameters())
        def vector(loss):
            grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p, g in zip(params, grads)])
        vectors = {k: vector(v) for k, v in losses.items()}
        total = sum(vectors.values())
        torch.testing.assert_close(total, vector(sum(losses.values())), rtol=1e-4, atol=1e-5)
        without_risk = total-vectors["risk"]
        def cosine(x, y):
            denominator = float(x.norm()*y.norm())
            return float(torch.dot(x, y)/denominator) if denominator else None
        norms = {k: float(v.norm()) for k, v in vectors.items()}
        assert all(torch.isfinite(v).all() for v in vectors.values())
        return dict(noise_seed=noise_seed, samples=len(states), mean_q=float(q_mean.detach().mean()),
                    mean_q_std=float(q_std.detach().mean()), mean_policy_std=float(log_std.detach().exp().mean()),
                    mean_abs_deterministic_action=float(mean.detach().tanh().abs().mean()), alpha=float(agent.alpha),
                    losses={k: float(v.detach()) for k, v in losses.items()}, gradient_norms=norms,
                    total_gradient_norm=float(total.norm()), without_risk_gradient_norm=float(without_risk.norm()),
                    risk_over_without_risk=norms["risk"]/float(without_risk.norm()) if without_risk.norm() else None,
                    q_vs_risk_cosine=cosine(vectors["q"], vectors["risk"]),
                    total_vs_without_risk_cosine=cosine(total, without_risk), optimizer_updates=0,
                    gradient_scope="frozen raw actor gradients, before clipping, Adam and weight decay; not historical updates")


def run(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    spec, started = stage_spec(args.smoke), time.monotonic()
    spec["training_seed"] = args.seed
    progress = json.loads(args.resume_from.read_text()) if args.resume_from else dict(
        revision=REVISION, spec=spec, models=[])
    assert progress["revision"] == REVISION and progress["spec"] == json.loads(json.dumps(spec))
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    def save():
        path = args.checkpoint_dir/"progress.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(progress, indent=2, allow_nan=False)+"\n")
        temporary.replace(path)
    for age, runtime, revision in ((100000, args.v122_runtime, v122.REVISION),
                                   (300000, args.v123_runtime, "general_ood_v123_source_extension")):
        source = json.loads((runtime/f"run_results/seed{args.seed}/source.json").read_text())
        assert source["revision"] == revision and source["training_seed"] == args.seed and not source["smoke"]
        env = gym.make("Hopper-v4")
        agent = v122.make_agent(env, "cpu")
        env.close()
        assert asdict(agent.config) == dict(source["config"], device="cpu")
        versions = dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
                        gymnasium=gym.__version__, mujoco=__import__("mujoco").__version__)
        assert all(versions[k] == source["runtime"][k] for k in versions)
        checkpoint = runtime/f"run_checkpoints/seed{args.seed}/latest.pt"
        agent.load(str(checkpoint))
        assert agent.training_steps == age-agent.config.batch_size+1 and agent.replay_buffer.size == age
        row = next((r for r in progress["models"] if r["age"] == age), None)
        if row is None:
            row = dict(age=age, checkpoint=str(checkpoint), source_revision=revision,
                       source_final_gpu_evaluation=source["final_evaluation"], evaluations=[], gradients={})
            progress["models"].append(row)
        cells = [(reset, noise) for reset in spec["reset_seeds"] for noise in (None, *spec["noise_seeds"])]
        for reset, noise in cells[len(row["evaluations"]):]:
            result = evaluate_episode(agent, reset, noise, spec["max_steps"])
            row["evaluations"].append(result)
            save()
            print(f"[EVAL] seed{args.seed} age={age} cell={len(row['evaluations'])}/{len(cells)} "
                  f"mode={result['action_mode']} length={result['length']}", flush=True)
        for name, pool in replay_pools(agent, spec["gradient_samples"]).items():
            if name not in row["gradients"]:
                row["gradients"][name] = [gradient_probe(agent, pool, noise) for noise in spec["noise_seeds"]]
                save()
        assert agent.training_steps == age-agent.config.batch_size+1 and agent.replay_buffer.size == age
        del agent
    result = dict(**progress, training_seed=args.seed, optimizer_updates=0, source_readout_refitted=False,
                  evaluation_device="cpu", runtime=versions, seconds=time.monotonic()-started,
                  evaluation_interactions=sum(e["length"] for r in progress["models"] for e in r["evaluations"]))
    args.result_dir.mkdir(parents=True, exist_ok=True)
    (args.result_dir/"stability.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] seed{args.seed} frozen_models=2 optimizer_updates=0 "
          f"evaluation_steps={result['evaluation_interactions']}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--v122-runtime", type=Path, required=True)
    parser.add_argument("--v123-runtime", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
