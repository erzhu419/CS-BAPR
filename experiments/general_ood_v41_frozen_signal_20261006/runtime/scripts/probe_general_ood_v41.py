"""Paired frozen-policy mode probes; auxiliary detectors never affect actions."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import numpy as np
import torch

from csbapr.belief.surprise import RelativeSurpriseComputer
from csbapr.belief.tracker import BeliefTracker
from csbapr.utils import compute_reg_norm
from scripts import run_general_ood_v40 as v40
from scripts.train_csbapr import MUJOCO_MODE_PROFILES, apply_mode_to_env

REVISION = "general_ood_v41_frozen_signal"
POLICIES = ("fixed-minus2", "relative-adaptive")
OBSERVERS = ("step-overlap", "chunk256", "previous-chunk-reference")
WINDOW, SWITCH_STEP, PROBE_STEPS = 256, 1024, 1536
METRICS = ("surprise", "surprise_raw", "surprise_reward_zscore", "surprise_q_std_spike",
           "surprise_reg_selected", "belief_reset_probability", "belief_effective_window",
           "belief_entropy", "weighted_lambda", "hypothetical_beta")


class PreviousChunkReference:
    """Predict a new chunk from the preceding chunk, before incorporating it."""

    def __init__(self, ema_alpha):
        self.ema_alpha, self.ema_surprise = ema_alpha, 0.0
        self.previous_rewards, self.previous_q = None, None
        self.last_components = {}

    def compute(self, rewards, q_mean, reg_signal):
        previous = self.previous_rewards
        previous_q = self.previous_q
        self.previous_rewards, self.previous_q = rewards.clone(), q_mean
        if previous is None:
            return None
        reward_z = abs(rewards.mean().item() - previous.mean().item()) / max(previous.std(unbiased=False).item(), 1e-6)
        q_spike = max(q_mean / previous_q - 1.0, 0.0) if previous_q > 1e-6 else 0.0
        raw = max(reward_z, q_spike, reg_signal)
        self.last_components = {"surprise_raw": raw, "surprise_reward_zscore": reward_z,
                                "surprise_q_std_spike": q_spike, "surprise_reg_selected": reg_signal}
        self.ema_surprise = self.ema_alpha * raw + (1 - self.ema_alpha) * self.ema_surprise
        return self.ema_surprise


def observe_trace(rewards, q_spread, current_norm, target_norm, config):
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    q_spread = torch.as_tensor(q_spread, dtype=torch.float32)
    current_norm, target_norm = current_norm.cpu(), target_norm.cpu()
    reg_signal = .5 * ((current_norm - target_norm).abs() / target_norm.abs().clamp_min(1e-6)).mean().item()
    detectors = {name: RelativeSurpriseComputer(config.surprise_ema) for name in OBSERVERS[:2]}
    detectors[OBSERVERS[2]] = PreviousChunkReference(config.surprise_ema)
    beliefs = {name: BeliefTracker(config.max_run_length, config.hazard_rate, config.base_variance)
               for name in OBSERVERS}
    bins = {name: [[] for _ in range(len(rewards) // WINDOW)] for name in OBSERVERS}
    schedule = np.exp(-.1 * np.arange(config.max_run_length))
    for end in range(WINDOW, len(rewards) + 1):
        window = rewards[end - WINDOW:end]
        q_mean = q_spread[end - WINDOW:end].mean().item()
        for name in OBSERVERS:
            if name != "step-overlap" and end % WINDOW:
                continue
            detector, belief = detectors[name], beliefs[name]
            surprise = (detector.compute(window, q_mean, reg_signal) if name == OBSERVERS[2]
                        else detector.compute(window, q_mean, current_norm, target_norm))
            if surprise is None:
                continue
            belief.update(surprise)
            weight = float(np.sum(belief.belief * schedule))
            components = detector.last_components
            bins[name][(end - 1) // WINDOW].append({
                "surprise": surprise, **{k: components[k] for k in METRICS[1:5]},
                "belief_reset_probability": float(belief.belief[0]),
                "belief_effective_window": float(belief.effective_window),
                "belief_entropy": float(belief.entropy), "weighted_lambda": weight,
                "hypothetical_beta": -2.0 - weight,
            })
    return {name: [{"start": i * WINDOW, "end": (i + 1) * WINDOW, "observations": len(rows),
                    "means": {k: float(np.mean([r[k] for r in rows])) for k in METRICS} if rows else None,
                    "maxima": {k: max(r[k] for r in rows) for k in ("surprise", "belief_reset_probability")} if rows else None}
                   for i, rows in enumerate(groups)] for name, groups in bins.items()}


@torch.no_grad()
def collect_trajectory(agent, env, seed, target_mode, context_snapshot,
                       budget=PROBE_STEPS, switch_step=SWITCH_STEP):
    agent.restore_context(deepcopy(context_snapshot))
    agent.reset_context()
    profiles = MUJOCO_MODE_PROFILES["Hopper-v4"]
    apply_mode_to_env(env, "Hopper-v4", profiles["normal"])
    state, _ = env.reset(seed=seed)
    episodes, rows = 0, {k: [] for k in ("states", "actions", "contexts", "rewards", "dones")}
    for step in range(budget):
        if step == switch_step:
            apply_mode_to_env(env, "Hopper-v4", profiles[target_mode])
        action = agent.select_action(state, deterministic=True)
        context = agent._current_context_np()
        executed = np.clip(action * env.action_space.high, env.action_space.low, env.action_space.high)
        next_state, reward, terminated, truncated, _ = env.step(executed)
        done = terminated or truncated
        for key, value in (("states", state.copy()), ("actions", action.copy()), ("contexts", context.copy()),
                           ("rewards", float(reward)), ("dones", bool(done))):
            rows[key].append(value)
        agent.observe_context(state, action, reward, next_state, done=done)
        state = next_state
        if done and step + 1 < budget:
            episodes += 1
            state, _ = env.reset(seed=seed + episodes * 10000)
            agent.reset_context()
    trace = {k: np.asarray(v) for k, v in rows.items()}
    q_spread = []
    for start in range(0, budget, WINDOW):
        states = torch.as_tensor(trace["states"][start:start + WINDOW], dtype=torch.float32, device=agent.device)
        contexts = torch.as_tensor(trace["contexts"][start:start + WINDOW], dtype=torch.float32, device=agent.device)
        actions = torch.as_tensor(trace["actions"][start:start + WINDOW], dtype=torch.float32, device=agent.device)
        values = agent.critic(agent._augment_state_tensor(states, contexts), actions)
        q_spread.extend(values.std(dim=0).cpu().numpy().tolist())
    trace["q_spread"] = np.asarray(q_spread, dtype=np.float32)
    return trace


def prefix_equal(control, changed, switch_step=SWITCH_STEP):
    return all(np.array_equal(control[k][:switch_step], changed[k][:switch_step]) for k in control)


def paired_contrasts(control, changed):
    return {name: {phase: {k: changed[name][i]["means"][k] - control[name][i]["means"][k] for k in METRICS}
                   for phase, i in (("pre256", 3), ("first_post256", 4), ("second_post256", 5))}
            for name in OBSERVERS}


def run(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(1)
    np.random.seed(1)
    seeds = list(range(99000, 99002)) if args.stage == "preflight" else list(range(90000, 90020))
    targets = ("normal", "high_grav") if args.stage == "preflight" else tuple(MUJOCO_MODE_PROFILES["Hopper-v4"])
    args.result_dir.mkdir(parents=True, exist_ok=True)
    for policy in POLICIES:
        started = time.monotonic()
        fresh = gym.make("Hopper-v4")
        agent = v40.make_agent("Hopper-v4", policy, "cuda", fresh)
        fresh.close()
        checkpoint = args.checkpoint_dir / f"{v40.REVISION}_Hopper-v4_{policy}_seed1_final.pt"
        agent.load(str(checkpoint))
        assert agent.training_steps == 99745 and agent.context_tracker._count == 100000
        for model in (agent.actor, agent.critic, agent.target_critic):
            model.eval()
            model.requires_grad_(False)
        context_snapshot = agent.snapshot_context()
        current_norm, target_norm = compute_reg_norm(agent.critic).cpu(), compute_reg_norm(agent.target_critic).cpu()
        records = []
        for seed in seeds:
            control, control_observers = None, None
            for mode in targets:
                env = gym.make("Hopper-v4", max_episode_steps=PROBE_STEPS)
                trace = collect_trajectory(agent, env, seed, mode, context_snapshot)
                env.close()
                observers = observe_trace(trace["rewards"], trace["q_spread"], current_norm, target_norm, agent.config)
                if mode == "normal":
                    control, control_observers = trace, observers
                assert prefix_equal(control, trace), (policy, seed, mode, "unpaired-prefix")
                records.append({
                    "seed": seed, "target_mode": mode, "actual_change": mode != "normal",
                    "interactions": len(trace["rewards"]), "paired_prefix_equal": True,
                    "episode_endings_by_256": [int(trace["dones"][i:i + WINDOW].sum()) for i in range(0, PROBE_STEPS, WINDOW)],
                    "reward_means_by_256": [float(trace["rewards"][i:i + WINDOW].mean()) for i in range(0, PROBE_STEPS, WINDOW)],
                    "observers": observers, "change_minus_nochange": paired_contrasts(control_observers, observers),
                })
            print(f"[PROBE] {policy} seed={seed}: {len(targets)} modes, no optimizer updates", flush=True)
        agent.restore_context(deepcopy(context_snapshot))
        assert agent.context_tracker._count == 100000 and agent.training_steps == 99745
        assert agent.replay_buffer.size == 0
        output = {
            "revision": REVISION, "stage": args.stage, "environment": "Hopper-v4", "source_policy": policy,
            "source_checkpoint": checkpoint.name, "source_training_seed": 1,
            "source_training_interactions": 100000, "loaded_training_update_steps": 99745,
            "loaded_training_context_count": 100000, "config": asdict(agent.config),
            "evaluation_seeds": seeds, "target_modes": list(targets), "observer_names": list(OBSERVERS),
            "probe_interactions_per_rollout": PROBE_STEPS, "switch_step": SWITCH_STEP,
            "window": WINDOW, "probe_episode_time_limit": PROBE_STEPS,
            "reward_normalization_std": 1.0, "optimizer_updates_during_probe": 0, "replay_insertions": 0,
            "actor_critic_target_frozen": True, "restore_training_context_before_each_rollout": True,
            "auxiliary_belief_controls_actions": False, "hypothetical_beta_penalty_scale": 1.0,
            "auxiliary_belief_initialization": "uniform; fresh each rollout; no optimizer warmup",
            "auxiliary_hazard_per_observation": agent.config.hazard_rate,
            "current_critic_norms": current_norm.tolist(), "target_critic_norms": target_norm.tolist(),
            "total_probe_interactions": len(records) * PROBE_STEPS, "records": records,
            "seconds": time.monotonic() - started,
            "runtime": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                        "gymnasium": gym.__version__, "mujoco": __import__("mujoco").__version__,
                        "gpu": torch.cuda.get_device_name()},
        }
        path = args.result_dir / f"{REVISION}_{policy}.json"
        path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
        print(f"[DONE] frozen signal {policy}: {len(records)} rollouts, result={path}", flush=True)
        del agent
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "formal"), default="formal")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
