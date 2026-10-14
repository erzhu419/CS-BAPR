"""Fresh normal-only Hopper sources; checkpoints stay on the training node."""
import argparse
from collections import deque
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import numpy as np
import torch

from scripts import run_general_ood_v39 as v39
from scripts.run_general_ood_v52 import environment_state, restore_environment
from scripts.train_csbapr import _recent_rollout_payload

REVISION = "general_ood_v121_locomotion_source"
SEEDS = (0, 1, 2)
METRICS = ("q_loss", "policy_loss", "q_mean", "q_std", "alpha", "effective_beta",
           "reward_normalization_std", "policy_entropy")


def stage_spec(smoke=False):
    return dict(steps=512 if smoke else 100000, checkpoint_interval=256 if smoke else 10000,
        diagnostic_interval=256 if smoke else 5000, evaluation_interval=256 if smoke else 25000,
        evaluation_episodes=1 if smoke else 3, final_evaluation_episodes=1 if smoke else 5,
        evaluation_max_steps=64 if smoke else 1000, evaluation_seed=149000,
        final_evaluation_seed=150000, random_action_warmup_steps=0)


def evaluate(agent, seed, episodes, max_steps):
    snapshot, updates = deepcopy(agent.snapshot_context()), agent.training_steps
    numpy_rng, python_rng = np.random.get_state(), random.getstate()
    devices = [torch.cuda.current_device()] if torch.device(agent.device).type == "cuda" else []
    env, records = gym.make("Hopper-v4"), []
    try:
        with torch.random.fork_rng(devices=devices):
            raw = env.unwrapped
            assert raw._terminate_when_unhealthy and raw._healthy_reward == 1.
            for episode in range(episodes):
                agent.restore_context(deepcopy(snapshot))
                agent.reset_context()
                state, _ = env.reset(seed=seed+episode)
                rewards, velocities, costs = [], [], []
                for _ in range(max_steps):
                    action = agent.select_action(state, deterministic=True)
                    executed = np.clip(action*env.action_space.high, env.action_space.low, env.action_space.high)
                    next_state, reward, terminated, truncated, info = env.step(executed)
                    cost = raw._ctrl_cost_weight*float(np.square(executed).sum())
                    velocity = float(info["x_velocity"])
                    np.testing.assert_allclose(reward, 1.+raw._forward_reward_weight*velocity-cost,
                                               rtol=1e-9, atol=1e-9)
                    rewards.append(float(reward))
                    velocities.append(velocity)
                    costs.append(cost)
                    agent.observe_context(state, action, reward, next_state, terminated or truncated)
                    state = next_state
                    if terminated or truncated:
                        break
                records.append(dict(seed=seed+episode, episode_return=float(np.sum(rewards)), length=len(rewards),
                    mean_forward_velocity=float(np.mean(velocities)), mean_control_cost=float(np.mean(costs)),
                    fraction_abs_velocity_below_point1=float(np.mean(np.abs(velocities) < .1)),
                    physical_termination=bool(terminated), time_limit_truncation=bool(truncated)))
    finally:
        env.close()
        agent.restore_context(snapshot)
        np.random.set_state(numpy_rng)
        random.setstate(python_rng)
    assert agent.training_steps == updates
    return dict(episodes=records, mean_return=float(np.mean([r["episode_return"] for r in records])),
        mean_episode_velocity=float(np.mean([r["mean_forward_velocity"] for r in records])),
        mean_episode_length=float(np.mean([r["length"] for r in records])),
        physical_terminations=sum(r["physical_termination"] for r in records),
        environment_step_calls=sum(r["length"] for r in records))


def save_progress(agent, env, loop, spec, seed, path):
    temporary = path.with_suffix(".tmp")
    agent.save(str(temporary), extra_state=dict(revision=REVISION, seed=seed, spec=spec,
        loop=loop, environment=environment_state(env)))
    temporary.replace(path)


def train(agent, env, seed, spec, checkpoint, resume_from=None):
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if resume_from is not None:
        saved = agent.load(str(resume_from))
        assert (saved["revision"], saved["seed"], saved["spec"]) == (REVISION, seed, spec)
        loop = saved["loop"]
        env.reset(seed=0)
        restore_environment(env, saved["environment"])
    else:
        state, _ = env.reset(seed=seed*10000)
        agent.reset_context()
        loop = dict(step=0, episodes_completed=0, episode_age=0, episode_return=0., state=state,
            recent={k: deque(maxlen=agent.config.surprise_window) for k in ("state", "action", "reward", "context")},
            recent_returns=deque(maxlen=20), blocks=[], rows=[], evaluations=[])
        loop["evaluations"].append(dict(step=0, **evaluate(agent, spec["evaluation_seed"],
            spec["evaluation_episodes"], spec["evaluation_max_steps"])))
    for step in range(loop["step"], spec["steps"]):
        state = loop["state"]
        action = agent.select_action(state)
        context = agent._current_context_np().copy()
        executed = np.clip(action*env.action_space.high, env.action_space.low, env.action_space.high)
        next_state, reward, terminated, truncated, info = env.step(executed)
        done = terminated or truncated
        for key, value in (("state", state.copy()), ("action", action.copy()),
                           ("reward", float(reward)), ("context", context)):
            loop["recent"][key].append(value)
        agent.store_transition(state, action, reward, next_state, float(done))
        metrics = agent.update(recent_rollout=_recent_rollout_payload(loop["recent"]))
        if metrics is not None:
            assert metrics["effective_beta"] == -2.
        loop["rows"].append(dict(reward=float(reward), velocity=float(info["x_velocity"]),
            terminated=bool(terminated), truncated=bool(truncated),
            metrics=None if metrics is None else {k: float(metrics[k]) for k in METRICS}))
        loop["step"], loop["state"] = step+1, next_state
        loop["episode_age"] += 1
        loop["episode_return"] += float(reward)
        if done:
            loop["recent_returns"].append(loop["episode_return"])
            loop["episodes_completed"] += 1
            loop["episode_age"], loop["episode_return"] = 0, 0.
            if step+1 < spec["steps"]:
                loop["state"], _ = env.reset(seed=seed*10000+loop["episodes_completed"])
                agent.reset_context()
        if (step+1) % spec["diagnostic_interval"] == 0 or step+1 == spec["steps"]:
            rows = loop["rows"]
            updates = [r["metrics"] for r in rows if r["metrics"] is not None]
            loop["blocks"].append(dict(step=step+1, interactions=len(rows), updates=len(updates),
                mean_reward=float(np.mean([r["reward"] for r in rows])),
                mean_forward_velocity=float(np.mean([r["velocity"] for r in rows])),
                fraction_abs_velocity_below_point1=float(np.mean([abs(r["velocity"]) < .1 for r in rows])),
                physical_terminations=sum(r["terminated"] for r in rows),
                time_limit_truncations=sum(r["truncated"] for r in rows),
                metrics={k: float(np.mean([r[k] for r in updates])) for k in METRICS} if updates else {}))
            loop["rows"] = []
            print(f"[TRAIN] seed{seed} {step+1}/{spec['steps']} updates={agent.training_steps} "
                  f"velocity={loop['blocks'][-1]['mean_forward_velocity']:.4f}", flush=True)
        if (step+1) % spec["evaluation_interval"] == 0 and step+1 < spec["steps"]:
            result = evaluate(agent, spec["evaluation_seed"], spec["evaluation_episodes"], spec["evaluation_max_steps"])
            loop["evaluations"].append(dict(step=step+1, **result))
            print(f"[EVAL] seed{seed} step={step+1} velocity={result['mean_episode_velocity']:.4f} "
                  f"length={result['mean_episode_length']:.1f}", flush=True)
        if (step+1) % spec["checkpoint_interval"] == 0 or step+1 == spec["steps"]:
            save_progress(agent, env, loop, spec, seed, checkpoint)
    assert agent.training_steps == max(0, spec["steps"]-agent.config.batch_size+1)
    assert agent.context_tracker._count == spec["steps"]
    return dict(interactions=loop["step"], optimizer_updates=agent.training_steps,
        episodes_completed=loop["episodes_completed"], partial_episode_steps=loop["episode_age"],
        partial_episode_return=loop["episode_return"], recent_completed_episode_returns=list(loop["recent_returns"]),
        diagnostic_blocks=loop["blocks"], periodic_evaluations=loop["evaluations"])


def run(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env = gym.make("Hopper-v4")
    agent = v39.make_agent("Hopper-v4", "fixed-minus2", args.device, env)
    raw = env.unwrapped
    reward_settings = dict(healthy_reward=float(raw._healthy_reward), ctrl_cost_weight=float(raw._ctrl_cost_weight),
                          forward_reward_weight=float(raw._forward_reward_weight))
    spec, started = stage_spec(args.smoke), time.monotonic()
    training = train(agent, env, args.seed, spec, args.checkpoint_dir/"latest.pt", args.resume_from)
    final = args.checkpoint_dir/"final.pt"
    agent.save(str(final))
    restored = v39.make_agent("Hopper-v4", "fixed-minus2", args.device, env)
    restored.load(str(final))
    env.close()
    result = dict(revision=REVISION, training_seed=args.seed, smoke=args.smoke, spec=spec, config=asdict(agent.config),
        environment="Hopper-v4", training_mode="normal-only", effective_beta=-2., trained_from_scratch=True,
        reward_settings=reward_settings, checkpoint_selection="final-not-best", checkpoint=str(final),
        periodic_evaluation_selects_checkpoint=False, optimizer_updates_during_evaluation=0,
        source_readout_refitted=False, training=training,
        final_evaluation=evaluate(restored, spec["final_evaluation_seed"], spec["final_evaluation_episodes"],
                                  spec["evaluation_max_steps"]),
        runtime=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
            gymnasium=gym.__version__, mujoco=__import__("mujoco").__version__,
            gpu=torch.cuda.get_device_name() if args.device == "cuda" else None), seconds=time.monotonic()-started)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    (args.result_dir/"source.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] seed{args.seed} steps={spec['steps']} "
          f"final_velocity={result['final_evaluation']['mean_episode_velocity']:.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
