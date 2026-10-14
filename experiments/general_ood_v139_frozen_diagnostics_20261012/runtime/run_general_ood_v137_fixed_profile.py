"""Matched-budget, fixed-profile learnability probes; not an adaptation study."""
import argparse
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

import run_general_ood_v132_source_stability as v132
import analyze_general_ood_v135_ood_context as v135

REVISION = "general_ood_v137_fixed_profile"
PROFILES = ("id", "easy_surface", "mega_event")


def profile_settings(profile):
    assert profile in PROFILES
    return next(p for p in v135.make_ood_profiles("Hopper-v4", intensity=50., family="structured") if p["name"] == profile)


def stage_spec(profile, smoke=False):
    assert profile in PROFILES
    return dict(v132.stage_spec("mse-control", smoke), steps=512 if smoke else 300000,
        study_revision=REVISION, training_profile=profile, profile_family="structured", intensity=50.,
        training_mode="fixed-profile-from-scratch", effective_beta=-2.,
        reward_changes=[], policy_receives_profile_labels=False)


def make_env(profile):
    assert profile in PROFILES
    condition = "id" if profile == "id" else "static/"+profile
    wrapper = v135.make_env(condition)
    wrapper.reset(seed=0)
    metadata = dict(profile=profile, actual_multipliers=v135.actual_multipliers(wrapper),
        body_mass=wrapper.unwrapped.model.body_mass.tolist(), gravity=wrapper.unwrapped.model.opt.gravity.tolist(),
        geom_friction=wrapper.unwrapped.model.geom_friction.tolist())
    # The profile is constant: native TimeLimit retains the existing full-state resume path.
    return wrapper.env, metadata


def assert_physics(env, metadata):
    raw = env.unwrapped
    for name, array in (("body_mass", raw.model.body_mass), ("gravity", raw.model.opt.gravity),
                        ("geom_friction", raw.model.geom_friction)):
        np.testing.assert_array_equal(array, metadata[name])


def evaluate(agent, profile, seed, episodes, max_steps, env_factory=None):
    snapshot, updates = deepcopy(agent.snapshot_context()), agent.training_steps
    numpy_rng, python_rng = np.random.get_state(), random.getstate()
    devices = [torch.cuda.current_device()] if torch.device(agent.device).type == "cuda" else []
    env, physics = (make_env if env_factory is None else env_factory)(profile)
    records = []
    try:
        with torch.random.fork_rng(devices=devices):
            raw = env.unwrapped
            assert raw._terminate_when_unhealthy and raw._healthy_reward == 1.
            for episode in range(episodes):
                agent.restore_context(deepcopy(snapshot))
                agent.reset_context()
                state, _ = env.reset(seed=seed+episode)
                assert_physics(env, physics)
                rewards, velocities, costs = [], [], []
                for _ in range(max_steps):
                    action = agent.select_action(state, deterministic=True)
                    executed = np.clip(action*env.action_space.high, env.action_space.low, env.action_space.high)
                    next_state, reward, terminated, truncated, info = env.step(executed)
                    cost = raw._ctrl_cost_weight*float(np.square(executed).sum())
                    velocity = float(info["x_velocity"])
                    np.testing.assert_allclose(reward, 1.+raw._forward_reward_weight*velocity-cost, rtol=1e-9, atol=1e-9)
                    rewards.append(float(reward)); velocities.append(velocity); costs.append(cost)
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
    return dict(profile=profile, actual_multipliers=physics["actual_multipliers"],
        episodes=records, mean_return=float(np.mean([r["episode_return"] for r in records])),
        mean_episode_velocity=float(np.mean([r["mean_forward_velocity"] for r in records])),
        mean_episode_length=float(np.mean([r["length"] for r in records])),
        physical_terminations=sum(r["physical_termination"] for r in records),
        environment_step_calls=sum(r["length"] for r in records))


def evaluation_spec(smoke=False):
    return dict(reset_seeds=list(range(154000, 154001 if smoke else 154005)),
        action_noise_seeds=[137000] if smoke else [137000, 137001, 137002],
        max_steps=64 if smoke else 1000, device="cpu", context_mode="online",
        sampled_noise_scale=1., profile_schedule="matching-static",
        noise_schedule="v126 CPU torch.Generator; one float32 (1,3) draw per step")


def final_evaluation(agent, profile, smoke=False, env_factory=None):
    assert torch.device(agent.device).type == "cpu" and agent.replay_buffer.size == 0
    spec, before = evaluation_spec(smoke), agent.training_steps
    condition = "id" if profile == "id" else "static/"+profile
    episodes = []
    for reset in spec["reset_seeds"]:
        for noise in (None, *spec["action_noise_seeds"]):
            cell = dict(condition=condition, mode="deterministic" if noise is None else "sampled",
                        reset_seed=reset, noise_seed=noise)
            episodes.append(v135.evaluate_episode(agent, cell, "online", spec["max_steps"], env_factory=env_factory))
    assert agent.training_steps == before
    return dict(protocol=spec, device="cpu", profile=profile, episodes=episodes, optimizer_updates=0,
        replay_loaded=False, evaluation_interactions=sum(e["length"] for e in episodes))


def run(args, *, revision=REVISION, env_factory=make_env, spec_factory=stage_spec, final_evaluator=final_evaluation):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    env, physics = env_factory(args.profile)
    agent = v132.make_agent(env, args.device, "mse-control")
    raw = env.unwrapped
    reward_settings = dict(healthy_reward=float(raw._healthy_reward), ctrl_cost_weight=float(raw._ctrl_cost_weight),
                          forward_reward_weight=float(raw._forward_reward_weight))
    spec, started = spec_factory(args.profile, args.smoke), time.monotonic()
    training = v132.train(agent, env, args.seed, spec, args.checkpoint_dir/"latest.pt", args.resume_from,
        evaluation_fn=lambda a, seed, episodes, limit: evaluate(a, args.profile, seed, episodes, limit, env_factory=env_factory))
    assert_physics(env, physics)
    final = args.checkpoint_dir/"final.pt"
    agent.save(str(final))
    restored = v132.make_agent(env, "cpu", "mse-control")
    restored.load(str(final))
    assert restored.training_steps == training["optimizer_updates"] and restored.context_tracker._count == spec["steps"]
    for model in (restored.actor, restored.critic, restored.target_critic):
        model.eval().requires_grad_(False)
    env.close()
    result = dict(revision=revision, training_driver_revision=v132.REVISION, training_seed=args.seed,
        training_profile=args.profile, arm="mse-control", smoke=args.smoke, spec=spec, config=asdict(agent.config),
        environment="Hopper-v4", training_mode="fixed-profile-from-scratch", effective_beta=-2., trained_from_scratch=True,
        training_device=args.device, physics=physics, reward_settings=reward_settings, reward_changes=[],
        policy_receives_profile_labels=False, checkpoint_selection="final-not-best", checkpoint=str(final),
        resumable_checkpoint=str(args.checkpoint_dir/"latest.pt"), periodic_evaluation_selects_checkpoint=False,
        optimizer_updates_during_evaluation=0, source_readout_refitted=False, training=training,
        final_evaluation=final_evaluator(restored, args.profile, args.smoke),
        final_context_count=restored.context_tracker._count,
        runtime=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__, gymnasium=gym.__version__,
            mujoco=__import__("mujoco").__version__, gpu=torch.cuda.get_device_name() if args.device == "cuda" else None),
        seconds=time.monotonic()-started,
        scope="Known-regime learnability at matched budgets, not unknown-regime adaptation or baseline superiority")
    args.result_dir.mkdir(parents=True, exist_ok=True)
    (args.result_dir/"source.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    lengths = [e["length"] for e in result["final_evaluation"]["episodes"]]
    print(f"[DONE] seed{args.seed} profile={args.profile} steps={spec['steps']} "
          f"updates={training['optimizer_updates']} mean_final_length={np.mean(lengths):.2f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(3, 4), required=True)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
