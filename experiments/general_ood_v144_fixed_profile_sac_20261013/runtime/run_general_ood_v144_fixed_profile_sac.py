"""Standard SB3 SAC learnability controls on the frozen v138 Hopper physics."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import io
import json
from pathlib import Path
import random
import sys
import time

HERE = Path(__file__).resolve().parent
RUNTIME = (HERE.parents[0]/"experiments/general_ood_v144_fixed_profile_sac_20261013/runtime"
           if HERE.name == "scripts" else HERE)
sys.path.insert(0, str(RUNTIME))
sys.path.insert(0, str(RUNTIME/"vendor"))

import gymnasium as gym
import numpy as np
import stable_baselines3 as sb3
from stable_baselines3 import SAC
import torch

import run_general_ood_v142_budget_extension as physics_worker

REVISION = "general_ood_v144_fixed_profile_sac"
PROFILES = ("friction-only", "gravity-only", "id")
v137, v138 = physics_worker.v137, physics_worker.source_worker.v138
evaluation = v137.v135.v134.evaluation
environment_state, restore_environment = v137.v132.environment_state, v137.v132.restore_environment
SETTINGS = dict(learning_rate=.0003, buffer_size=1000000, learning_starts=255,
    batch_size=256, gamma=.99, tau=.005, train_freq=1, gradient_steps=1,
    ent_coef="auto_0.2", target_entropy="auto", use_sde=False,
    policy_kwargs=dict(net_arch=[256, 256], n_critics=2))


def stage_spec(profile, smoke=False):
    assert profile in PROFILES
    return dict(revision=REVISION, profile=profile, smoke=smoke, steps=512 if smoke else 600000,
        block_steps=256 if smoke else 5000, checkpoint_interval=256 if smoke else 25000,
        evaluation_interval=256 if smoke else 25000, milestones=[256, 512] if smoke else [300000, 600000],
        bootstrap_mask="terminated-only", episode_boundary="terminated-or-truncated",
        training_reset_schedule="seed*10000 + completed episodes", settings=SETTINGS)


class TrainingResets(gym.Wrapper):
    def __init__(self, env, seed):
        super().__init__(env)
        self.seed, self.resets = seed, 0

    def reset(self, **kwargs):
        kwargs["seed"] = self.seed*10000+self.resets
        self.resets += 1
        return self.env.reset(**kwargs)


def make_model(env, seed, device):
    assert sb3.__version__ == "2.7.0"
    return SAC("MlpPolicy", TrainingResets(env, seed), seed=seed, device=device,
               verbose=0, **deepcopy(SETTINGS))


@contextmanager
def frozen_rng(device):
    numpy_rng, python_rng = np.random.get_state(), random.getstate()
    devices = [torch.cuda.current_device()] if torch.device(device).type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        np.random.set_state(numpy_rng)
        random.setstate(python_rng)


@torch.no_grad()
def evaluate_episode(model, profile, cell, max_steps):
    env = v137.v135.make_env("id") if profile == "id" else v138.make_wrapped_env("static/"+profile)
    noise = evaluation.noise_schedule(cell["noise_seed"], max_steps, 3)
    scale = 0. if cell["mode"] == "deterministic" else 1.
    rewards, velocities, stds, differences = [], [], [], []
    try:
        state, _ = env.reset(seed=cell["reset_seed"])
        raw = env.unwrapped
        assert raw._terminate_when_unhealthy and raw._healthy_reward == 1.
        multipliers = v137.v135.actual_multipliers(env)
        for step in range(max_steps):
            tensor = torch.as_tensor(state, dtype=torch.float32, device=model.device).unsqueeze(0)
            mean, log_std, _ = model.actor.get_action_dist_params(tensor)
            std = log_std.exp()
            deterministic = model.policy.unscale_action(mean.tanh().squeeze(0).cpu().numpy())
            scaled = (mean+scale*std*noise[step:step+1].to(model.device)).tanh().squeeze(0).cpu().numpy()
            action = model.policy.unscale_action(scaled)
            executed = np.clip(action*env.action_space.high, env.action_space.low, env.action_space.high)
            state, reward, term, trunc, info = env.step(executed)
            velocity = float(info["x_velocity"])
            expected = 1.+raw._forward_reward_weight*velocity-raw._ctrl_cost_weight*float(np.square(executed).sum())
            np.testing.assert_allclose(reward, expected, rtol=1e-9, atol=1e-9)
            assert info["ood_profile"] == profile
            rewards.append(float(reward)); velocities.append(velocity)
            stds.append(float(std.mean()))
            differences.append(float(np.linalg.norm(action-deterministic)))
            if term or trunc:
                break
        return dict(**cell, context_mode="not-applicable", context_updates=0, noise_scale=scale,
            length=len(rewards), episode_return=float(np.sum(rewards)), mean_velocity=float(np.mean(velocities)),
            physical_termination=bool(term), time_limit_truncation=bool(trunc), diagnostic_cutoff=not(term or trunc),
            mean_policy_std=float(np.mean(stds)), mean_action_noise_l2=float(np.mean(differences)),
            profile_counts={profile: len(rewards)}, non_id_steps=0 if profile == "id" else len(rewards),
            switches=0, profile_events=[dict(step=1, profile=profile, actual_multipliers=multipliers)],
            final_physical_state=evaluation.v124.physical_state(raw), native_reward_verified=True)
    finally:
        env.close()


def final_evaluation(checkpoint, profile, smoke, training_device):
    protocol = dict(v137.evaluation_spec(smoke), context_mode="not-applicable")
    if smoke:
        protocol["max_steps"] = 256
    with frozen_rng(training_device):
        restored = SAC.load(checkpoint, device="cpu")
        assert restored.replay_buffer.size() == 0
        restored.replay_buffer = None
        restored.policy.set_training_mode(False)
        before = restored._n_updates
        condition = "id" if profile == "id" else "static/"+profile
        episodes = [evaluate_episode(restored, profile, dict(condition=condition,
            mode="deterministic" if noise is None else "sampled", reset_seed=reset, noise_seed=noise), protocol["max_steps"])
            for reset in protocol["reset_seeds"] for noise in (None, *protocol["action_noise_seeds"])]
        assert restored._n_updates == before
    return dict(protocol=protocol, device="cpu", profile=profile, replay_loaded=False,
        optimizer_updates=0, episodes=episodes, evaluation_interactions=sum(e["length"] for e in episodes))


def periodic_evaluation(model, profile, smoke):
    with frozen_rng(model.device):
        model.policy.set_training_mode(False)
        condition = "id" if profile == "id" else "static/"+profile
        episodes = [evaluate_episode(model, profile, dict(condition=condition, mode="deterministic",
            reset_seed=seed, noise_seed=None), 64 if smoke else 1000)
            for seed in range(149000, 149001 if smoke else 149003)]
    return dict(device=str(model.device), episodes=episodes)


def save_progress(model, env, progress, spec, seed, path):
    serialized = io.BytesIO()
    model.save(serialized)
    monitor = model.env.envs[0]
    seeded = monitor.env
    assert isinstance(seeded, TrainingResets)
    payload = dict(revision=REVISION, spec=spec, seed=seed, model=serialized.getvalue(),
        replay=model.replay_buffer, environment=environment_state(env), resets=seeded.resets,
        monitor={k: deepcopy(getattr(monitor, k)) for k in ("rewards", "needs_reset", "total_steps", "episode_returns", "episode_lengths", "episode_times")},
        numpy_rng=np.random.get_state(), python_rng=random.getstate(), torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state() if model.device.type == "cuda" else None, progress=progress)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_progress(path, env, spec, seed, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert (payload["revision"], payload["spec"], payload["seed"]) == (REVISION, spec, seed)
    model = SAC.load(io.BytesIO(payload["model"]), env=TrainingResets(env, seed), device=device, force_reset=False)
    model.replay_buffer = payload["replay"]
    model.replay_buffer.device = torch.device(device)
    env.reset(seed=0)
    restore_environment(env, payload["environment"])
    monitor = model.env.envs[0]
    monitor.env.resets = payload["resets"]
    for k, value in payload["monitor"].items():
        setattr(monitor, k, value)
    np.random.set_state(payload["numpy_rng"])
    random.setstate(payload["python_rng"])
    torch.set_rng_state(payload["torch_rng"])
    if device == "cuda":
        torch.cuda.set_rng_state(payload["cuda_rng"])
    assert model._last_obs is not None and model.num_timesteps == payload["progress"]["step"]
    return model, payload["progress"]


def run(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    env, physics = physics_worker.make_env(args.profile)
    spec, started = stage_spec(args.profile, args.smoke), time.monotonic()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest = args.checkpoint_dir/"latest.pt"
    if args.resume_from:
        model, progress = load_progress(args.resume_from, env, spec, args.seed, args.device)
    else:
        model = make_model(env, args.seed, args.device)
        progress = dict(step=0, blocks=[], periodic_evaluations=[], snapshots=[])
        progress["periodic_evaluations"].append(dict(step=0, **periodic_evaluation(model, args.profile, args.smoke)))
    initial = model.num_timesteps
    while model.num_timesteps < spec["steps"]:
        before = model._n_updates
        model.learn(spec["block_steps"], reset_num_timesteps=False, log_interval=None)
        step = progress["step"] = model.num_timesteps
        assert step <= spec["steps"] and model._n_updates == max(0, step-255)
        values = model.logger.name_to_value
        metrics = {k: float(values[k]) for k in ("train/actor_loss", "train/critic_loss", "train/ent_coef") if k in values}
        assert all(np.isfinite(v) for v in metrics.values())
        progress["blocks"].append(dict(step=step, interactions=spec["block_steps"],
            updates=model._n_updates-before, metrics=metrics))
        print(f"[TRAIN] seed{args.seed} {args.profile} {step}/{spec['steps']} updates={model._n_updates}", flush=True)
        if step % spec["evaluation_interval"] == 0 and step < spec["steps"]:
            progress["periodic_evaluations"].append(dict(step=step, **periodic_evaluation(model, args.profile, args.smoke)))
        if step in spec["milestones"]:
            checkpoint = args.checkpoint_dir/f"step{step}.zip"
            model.save(checkpoint)
            progress["snapshots"].append(dict(step=step, optimizer_updates=model._n_updates, checkpoint=str(checkpoint),
                final_evaluation=final_evaluation(checkpoint, args.profile, args.smoke, model.device)))
        if step % spec["checkpoint_interval"] == 0 or step == spec["steps"]:
            save_progress(model, env, progress, spec, args.seed, latest)
    v137.assert_physics(env, physics)
    assert model.observation_space.shape == (11,) and model.replay_buffer.handle_timeout_termination
    raw = env.unwrapped
    result = dict(revision=REVISION, training_seed=args.seed, training_profile=args.profile, smoke=args.smoke,
        spec=spec, config=SETTINGS, environment="Hopper-v4", training_device=args.device,
        actual_new_training_interactions=model.num_timesteps-initial, cumulative_training_interactions=model.num_timesteps,
        optimizer_updates=model._n_updates, physics=physics,
        reward_settings=dict(healthy_reward=float(raw._healthy_reward), ctrl_cost_weight=float(raw._ctrl_cost_weight), forward_reward_weight=float(raw._forward_reward_weight)),
        reward_changes=[], healthy_termination_changes=[], policy_receives_profile_labels=False,
        observation_dim=11, use_context=False, use_symbolic_constraints=False, reward_normalization=False,
        snapshots=progress["snapshots"], training=progress, checkpoint_selection="predeclared300k600k-not-best",
        resumable_checkpoint=str(latest), bootstrap_mask="terminated-only", replay_handle_timeout_termination=True,
        optimizer_updates_during_evaluation=0, runtime=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
            gymnasium=gym.__version__, mujoco=__import__("mujoco").__version__, stable_baselines3=sb3.__version__,
            gpu=torch.cuda.get_device_name() if args.device == "cuda" else None),
        peak_cuda_allocated_mb=torch.cuda.max_memory_allocated()/1024**2 if args.device == "cuda" else 0.,
        seconds=time.monotonic()-started,
        scope="Native standard-SAC fixed-profile learnability positive control, not an algorithm-only CS-BAPR ablation")
    model.env.close()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    (args.result_dir/"source.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] seed{args.seed} {args.profile}: {model.num_timesteps} interactions / {model._n_updates} updates", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(3, 4), required=True)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
