"""Fixed-interaction belief diagnostic; checkpoints stay on the worker."""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import numpy as np
import torch

from csbapr.agent import CSBAPRAgent
from csbapr.envs.extreme_ood import ExtremeOODConfig, ExtremeOODWrapper
from scripts.train_csbapr import (
    MUJOCO_MODE_PROFILES, _evaluate_agent, _recent_rollout_payload,
    apply_mode_to_env, make_config as preset_config,
)
from scripts.v38_legacy_belief import BeliefTracker as LegacyBeliefTracker

REVISION = "general_ood_v38_belief"
VARIANTS = ("legacy-belief", "corrected-belief", "fixed-penalty")
BUDGETS = {"Pendulum-v1": 100000, "Hopper-v4": 300000}
EVALUATION_SEED_START = 80000
TRAIN_SWITCH_STEPS = 5000
METRICS = (
    "surprise", "belief_reset_probability", "belief_effective_window",
    "weighted_lambda_effective", "effective_beta", "q_std", "alpha", "policy_loss", "q_loss",
)


def make_config(environment, variant, device):
    config = preset_config(environment, "bapr-stable-context")
    config.device = device
    if variant == "fixed-penalty":
        config.penalty_scale = 0.0
    return config


def make_agent(environment, variant, device, env):
    config = make_config(environment, variant, device)
    config.state_dim = env.observation_space.shape[0]
    config.action_dim = env.action_space.shape[0]
    agent = CSBAPRAgent(env.observation_space.shape[0], env.action_space.shape[0], config)
    if variant == "legacy-belief":
        agent.belief_tracker = LegacyBeliefTracker(
            config.max_run_length, config.hazard_rate, config.base_variance,
        )
    return agent


def train(agent, env, environment, seed, budget, label=""):
    modes = MUJOCO_MODE_PROFILES[environment]
    rng = np.random.RandomState(seed)
    recent = {key: deque(maxlen=agent.config.surprise_window)
              for key in ("state", "action", "reward", "context")}
    steps, episodes, updates = 0, 0, 0
    events, diagnostic_blocks, metric_rows = [], [], []
    completed_rewards = deque(maxlen=20)
    started = time.monotonic()
    env.action_space.seed(seed)
    apply_mode_to_env(env, environment, modes["normal"])
    while steps < budget:
        state, _ = env.reset(seed=seed * 10000 + episodes)
        agent.reset_context()
        done, reward_sum = False, 0.0
        while not done and steps < budget:
            if steps % TRAIN_SWITCH_STEPS == 0:
                mode = "normal" if steps == 0 else str(rng.choice(list(modes)))
                apply_mode_to_env(env, environment, modes[mode])
                events.append({"step": steps, "mode": mode})
            action = agent.select_action(state)
            context = agent._current_context_np().copy()
            executed = np.clip(action * env.action_space.high[0],
                               env.action_space.low[0], env.action_space.high[0])
            next_state, reward, terminated, truncated, _ = env.step(executed)
            done = terminated or truncated
            for key, value in (("state", state.copy()), ("action", action.copy()),
                               ("reward", float(reward)), ("context", context)):
                recent[key].append(value)
            # Budget exhaustion is a collection cutoff, not an MDP terminal.
            agent.store_transition(state, action, reward, next_state, float(done))
            metrics = agent.update(recent_rollout=_recent_rollout_payload(recent))
            if metrics is not None:
                metric_rows.append({key: float(metrics[key]) for key in METRICS})
                updates += 1
            state = next_state
            reward_sum += reward
            steps += 1
            if steps % 10000 == 0 or steps == budget:
                if metric_rows:
                    diagnostic_blocks.append({
                        "step": steps, "updates": len(metric_rows),
                        **{key: float(np.mean([r[key] for r in metric_rows])) for key in METRICS},
                    })
                    metric_rows.clear()
                print(f"[TRAIN] {label} steps={steps}/{budget} episodes={episodes} "
                      f"updates={updates} time={time.monotonic()-started:.0f}s", flush=True)
        if done:
            completed_rewards.append(float(reward_sum))
        episodes += 1
    return {
        "interactions": steps, "optimizer_update_steps": updates,
        "episode_segments": episodes, "last_segment_terminated": bool(done),
        "recent_completed_episode_returns": list(completed_rewards),
        "mode_events": events, "diagnostic_blocks": diagnostic_blocks,
        "seconds": time.monotonic() - started,
    }


def evaluate(agent, environment, primary_count, fixed_count):
    def rollout(env, count, seed):
        context = agent.snapshot_context()
        result = _evaluate_agent(agent, env, agent.config.max_steps_per_episode, count, seed)
        agent.restore_context(context)
        env.close()
        return result

    def ood(schedule, profile=None):
        return ExtremeOODWrapper(gym.make(environment), config=ExtremeOODConfig(
            intensity=50.0, profile_family="structured", schedule=schedule,
            seed=EVALUATION_SEED_START, static_profile=profile,
            switch_every_steps=50 if environment == "Pendulum-v1" else 200,
            switch_probability=1.0,
        ))

    agent.actor.eval()
    agent.actor.requires_grad_(False)
    updates = agent.training_steps
    modes = {
        "id": rollout(gym.make(environment), primary_count, EVALUATION_SEED_START),
        "markov_50x": rollout(ood("markov"), primary_count, EVALUATION_SEED_START),
    }
    probe = ood("static")
    profiles = [p["name"] for p in probe.profiles if p["name"] != "id"]
    probe.close()
    fixed = {profile: rollout(ood("static", profile), fixed_count,
                              EVALUATION_SEED_START + index * 100)
             for index, profile in enumerate(profiles)}
    assert agent.training_steps == updates
    return modes, fixed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, choices=BUDGETS)
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--seed", required=True, type=int, choices=(0, 1))
    parser.add_argument("--stage", choices=("preflight", "formal"), default="formal")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env = gym.make(args.env)
    agent = make_agent(args.env, args.variant, "cuda", env)
    budget = 512 if args.stage == "preflight" else BUDGETS[args.env]
    primary_count, fixed_count = (2, 1) if args.stage == "preflight" else (20, 5)
    training = train(agent, env, args.env, args.seed, budget, args.variant)
    env.close()
    assert training["interactions"] == budget
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint_dir / f"{REVISION}_{args.env}_{args.variant}_seed{args.seed}_final.pt"
    agent.save(str(checkpoint))
    # Evaluate the saved final checkpoint, not an in-memory or best policy.
    fresh = gym.make(args.env)
    restored = make_agent(args.env, args.variant, "cuda", fresh)
    fresh.close()
    restored.load(str(checkpoint))
    loaded_context_count = int(restored.context_tracker._count)
    assert loaded_context_count == agent.context_tracker._count
    modes, fixed = evaluate(restored, args.env, primary_count, fixed_count)
    output = {
        "revision": REVISION, "stage": args.stage, "environment": args.env,
        "variant": args.variant, "training_seed": args.seed, "config": asdict(agent.config),
        "checkpoint": checkpoint.name, "checkpoint_selection": "final",
        "loaded_training_context_count": loaded_context_count,
        "restore_training_context_before_each_evaluation_block": True,
        "training_budget": budget, "training": training,
        "training_switch_steps": TRAIN_SWITCH_STEPS,
        "evaluation_seed_start": EVALUATION_SEED_START,
        "evaluation_episodes": primary_count, "fixed_profile_episodes": fixed_count,
        "ood_intensity": 50.0, "ood_profile_family": "structured",
        "ood_schedule": "markov", "ood_switch_probability": 1.0,
        "ood_switch_steps": 50 if args.env == "Pendulum-v1" else 200,
        "evaluation_only": True, "optimizer_updates_during_evaluation": 0,
        "results": modes, "fixed_profile_results": fixed,
        "runtime": {"python": sys.version, "torch": torch.__version__,
                    "numpy": np.__version__, "gymnasium": gym.__version__,
                    "mujoco": __import__("mujoco").__version__,
                    "gpu": torch.cuda.get_device_name()},
    }
    args.result_dir.mkdir(parents=True, exist_ok=True)
    path = args.result_dir / f"{REVISION}_{args.env}_{args.variant}_seed{args.seed}.json"
    path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(f"[DONE] {args.env} {args.variant} seed{args.seed}: "
          f"steps={budget}, result={path}", flush=True)


if __name__ == "__main__":
    main()
