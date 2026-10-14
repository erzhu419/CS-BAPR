"""Four matched closed-loop arms; full continuation state stays server-side."""
from collections import Counter, deque
from copy import deepcopy
from dataclasses import asdict
import argparse
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm

from csbapr.envs.extreme_ood import ExtremeOODConfig, ExtremeOODWrapper
from scripts import probe_general_ood_v53_behavior_coverage as v53
from scripts import run_general_ood_v52 as v52
from scripts.train_csbapr import _evaluate_agent, _recent_rollout_payload
from scripts.v58_agent import ClosedLoopAgent, VARIANTS

REVISION = "general_ood_v58_closed_loop"
BUDGET = 6000
EVALUATION_STARTS = {116000: 120000, 116001: 121000}
EVALUATION_EPISODES = 2
METRICS = ("q_loss", "policy_loss", "q_std", "alpha", "effective_beta", "weighted_lambda_effective",
           "signal", "readout_coefficient", "reward")


def mode_events(seed):
    order = np.random.RandomState(seed).permutation([m for m in v53.MODES if m != "normal"])
    events = [dict(step=0, mode="normal")]
    events.extend(dict(step=800 + 1000 * i, mode=str(mode)) for i, mode in enumerate(order))
    events.append(dict(step=4800, mode="normal"))
    return [e for e in events if e["step"] < BUDGET]


def train(agent, env, seed, checkpoint, resume_from=None):
    schedule = {e["step"]: e["mode"] for e in mode_events(seed)}
    modes = v53.MUJOCO_MODE_PROFILES["Hopper-v4"]
    if resume_from:
        payload = agent.load(str(resume_from))
        loop = payload["loop"]
        assert loop["seed"] == seed and loop["variant"] == agent.variant
        agent.readouts, agent.readout_coefficient = payload["readouts"], loop["readout_coefficient"]
        v53.apply_mode_to_env(env, "Hopper-v4", modes[loop["mode"]])
        env.reset(seed=seed)
        v52.restore_environment(env, payload["environment"])
    else:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        v53.apply_mode_to_env(env, "Hopper-v4", modes["normal"])
        state, _ = env.reset(seed=seed)
        agent.reset_context()
        loop = dict(seed=seed, variant=agent.variant, step=0, episode=0, episode_age=0,
            reward_sum=0., state=state, mode="normal", readout_coefficient=0.,
            recent={k: deque(maxlen=agent.config.surprise_window) for k in ("state", "action", "reward", "context")},
            rows=[], blocks=[], mode_events=[], returns=[], mode_statistics={})
    initial = loop["step"]
    for step in tqdm(range(initial, BUDGET), initial=initial, total=BUDGET, desc=f"{seed}/{agent.variant}", mininterval=10):
        if step in schedule:
            loop["mode"] = schedule[step]
            v53.apply_mode_to_env(env, "Hopper-v4", modes[loop["mode"]])
            loop["mode_events"].append(dict(step=step, mode=loop["mode"]))
        state = loop["state"]
        action = agent.select_action(state, deterministic=False)
        context = agent._current_context_np().copy()
        executed = np.clip(action * env.action_space.high, env.action_space.low, env.action_space.high)
        next_state, reward, terminated, truncated, _ = env.step(executed)
        done = terminated or truncated
        if step == 0:
            loop["first_transition"] = dict(state=state.tolist(), action=action.tolist(), executed_action=executed.tolist(),
                                           reward=float(reward), next_state=next_state.tolist(), done=bool(done))
        agent.observe_transition(state, executed, next_state, loop["episode_age"])
        for k, v in (("state", state.copy()), ("action", action.copy()), ("reward", float(reward)), ("context", context)):
            loop["recent"][k].append(v)
        agent.store_transition(state, action, reward, next_state, float(done))
        metrics = agent.update(recent_rollout=_recent_rollout_payload(loop["recent"]))
        assert metrics is not None and abs(metrics["effective_beta"] + 2. + agent.penalty_coefficient) < 1e-12
        metrics.update(signal=agent.readouts.signal, readout_coefficient=agent.readout_coefficient, reward=float(reward))
        row = {k: float(metrics[k]) for k in METRICS}
        assert all(np.isfinite(v) for v in row.values()), "nonfinite-closed-loop-metric"
        loop["rows"].append(dict(mode=loop["mode"], **row))
        totals = loop["mode_statistics"].setdefault(loop["mode"], dict(count=0, sums={k: 0. for k in METRICS}))
        totals["count"] += 1
        for k, v in row.items():
            totals["sums"][k] += v
        loop["readout_coefficient"] = agent.readout_coefficient
        loop["state"], loop["step"] = next_state, step + 1
        loop["reward_sum"] += reward
        loop["episode_age"] += 1
        if done:
            loop["returns"].append(float(loop["reward_sum"]))
            loop["episode"] += 1
            loop["episode_age"], loop["reward_sum"] = 0, 0.
            loop["state"], _ = env.reset(seed=seed + 10000 * loop["episode"])
            agent.reset_context()
        if (step + 1) % 1000 == 0 or step + 1 == BUDGET:
            loop["blocks"].append(dict(step=step + 1, updates=len(loop["rows"]),
                mode_counts=dict(Counter(r["mode"] for r in loop["rows"])),
                **{k: float(np.mean([r[k] for r in loop["rows"]])) for k in METRICS}))
            loop["rows"].clear()
        if (step + 1) % 500 == 0 or step + 1 == BUDGET:
            v52.save_training(agent, env, loop, checkpoint)
            print(f"[TRAIN] {seed}/{agent.variant} {step+1}/{BUDGET} beta={metrics['effective_beta']:.5f}", flush=True)
    return dict(interactions=loop["step"], optimizer_updates=agent.training_steps - 99745,
        first_transition=loop["first_transition"], mode_events=loop["mode_events"], diagnostic_blocks=loop["blocks"],
        mode_statistics={m: dict(count=v["count"], **{k: total / v["count"] for k, total in v["sums"].items()})
                         for m, v in loop["mode_statistics"].items()},
        completed_episode_returns=loop["returns"], context_count=agent.context_tracker._count,
        replay_size=agent.replay_buffer.size, resumed_at=initial)


def evaluate(agent, seed):
    start, snapshot, updates = EVALUATION_STARTS[seed], agent.snapshot_context(), agent.training_steps
    agent.actor.eval()
    def rollout(env, seed_start):
        rewards, lengths, profiles = [], [], Counter()
        for episode in range(EVALUATION_EPISODES):
            agent.restore_context(deepcopy(snapshot))
            result = _evaluate_agent(agent, env, 1000, 1, seed_start + episode)
            rewards.extend(result["rewards"])
            lengths.extend(result["lengths"])
            profiles.update(result["profile_counts"])
        env.close()
        return dict(mean=float(np.mean(rewards)), std=float(np.std(rewards)), rewards=rewards,
                    lengths=lengths, profile_counts=dict(profiles), seed_start=seed_start)
    results = dict(id=rollout(gym.make("Hopper-v4"), start))
    for index, mode in enumerate(m for m in v53.MODES if m != "normal"):
        env = gym.make("Hopper-v4")
        v53.apply_mode_to_env(env, "Hopper-v4", v53.MUJOCO_MODE_PROFILES["Hopper-v4"][mode])
        results[mode] = rollout(env, start + 100 * (index + 1))
    extreme = ExtremeOODWrapper(gym.make("Hopper-v4"), config=ExtremeOODConfig(
        intensity=50., profile_family="structured", schedule="markov", seed=start + 500,
        switch_every_steps=200, switch_probability=1.))
    results["markov_50x"] = rollout(extreme, start + 500)
    agent.restore_context(snapshot)
    assert agent.training_steps == updates
    return results


def run(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    shared = torch.load(args.shared_dir / "shared.pt", map_location="cpu", weights_only=False)
    dynamics = torch.load(args.dynamics_checkpoint, map_location="cpu", weights_only=False)["extra_state"]["progress"]
    frozen = torch.load(args.reference_checkpoint, map_location="cpu", weights_only=False)["extra_state"]["progress"]
    assert dynamics["stage"] == "preflight" and dynamics["next_cell"] == 9
    assert frozen["stage"] == "formal" and frozen["next_cell"] == 22
    model, refs = dynamics["models"]["broad"][0], frozen["references"]
    pooled, phase = refs["matched_pooled"], refs["phase_calibrated"]
    reference_before = deepcopy(dict(pooled=pooled.metadata(), phase=phase.metadata(), model=model.metadata()))
    del dynamics, frozen
    env = gym.make("Hopper-v4")
    agent = ClosedLoopAgent(env, args.variant)
    agent.start_shared(args.source_dir / v53.SOURCE_NAME, shared, model, pooled, phase)
    assert agent.training_steps == 99745 and agent.context_tracker._count == 100000 and agent.replay_buffer.size == 6144
    initial_reward_ema_var = float(getattr(agent, "_reward_ema_var", 1.))
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint_dir / f"{args.variant}_latest.pt"
    training = train(agent, env, args.seed, checkpoint, args.resume_from)
    env.close()
    fresh = gym.make("Hopper-v4")
    restored = ClosedLoopAgent(fresh, args.variant)
    fresh.close()
    restored.load(str(checkpoint))
    assert training["interactions"] == training["optimizer_updates"] == BUDGET
    assert restored.training_steps == 99745 + BUDGET
    results = evaluate(restored, args.seed)
    assert reference_before == dict(pooled=pooled.metadata(), phase=phase.metadata(), model=model.metadata())
    output = dict(revision=REVISION, variant=args.variant, continuation_seed=args.seed, source_training_seed=1,
        source_checkpoint=v53.SOURCE_NAME, initial_replay_size=6144, initial_reward_ema_var=initial_reward_ema_var,
        initial_source_updates=99745, initial_context_count=100000, shared_replay_split="fit-only",
        checkpoint_selection="final", config=asdict(agent.config), reference_frozen=True,
        penalty_enters_actor_loss_only=True, beta_mapping="-2-coefficient, no gain tuning",
        readout_metadata=reference_before, training=training, results=results,
        evaluation_interactions=sum(sum(v["lengths"]) for v in results.values()),
        optimizer_updates_during_evaluation=0, restore_training_context_before_each_evaluation_episode=True,
        evaluation_seed_start=EVALUATION_STARTS[args.seed], evaluation_episodes=EVALUATION_EPISODES,
        limitations=["Two continuation seeds from one trained source, short budget and two evaluation episodes per condition; pilot only.",
                     "Normal references cover initial and early later episodes, not all subsequent learning distributions."])
    args.result_dir.mkdir(parents=True, exist_ok=True)
    destination = args.result_dir / f"{REVISION}_{args.variant}_seed{args.seed}.json"
    destination.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(f"[DONE] {args.variant}/seed{args.seed}: {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", type=int, choices=EVALUATION_STARTS, required=True)
    for name in ("source-dir", "shared-dir", "dynamics-checkpoint", "reference-checkpoint", "checkpoint-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
