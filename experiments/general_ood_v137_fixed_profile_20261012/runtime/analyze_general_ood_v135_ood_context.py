"""Frozen MSE policies under structured OOD and inference-context ablation."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import numpy as np
import torch

from csbapr.envs.extreme_ood import ExtremeOODConfig, ExtremeOODWrapper, make_ood_profiles
import analyze_general_ood_v134_frozen_robustness as v134

REVISION = "general_ood_v135_frozen_ood_context"
CONTEXT_MODES = ("online", "zero-context")
CONDITIONS = ("id", "static/easy_surface", "static/heavy_traffic", "static/mega_event", "static/slippery_rush", "markov")


def stage_spec(seed, context_mode, smoke=False):
    return dict(training_seed=seed, context_mode=context_mode, source_arm="mse-control", source_steps=300000,
        smoke=smoke, device="cpu", profile_family="structured", intensity=50.,
        conditions=["id", "static/mega_event", "markov"] if smoke else list(CONDITIONS),
        reset_seeds=list(range(152000, 152001 if smoke else 152005)),
        action_noise_seeds=[135000] if smoke else [135000, 135001, 135002],
        sampled_noise_scale=1., switch_every_steps=50, switch_probability=1.,
        max_steps=128 if smoke else 1000, sensor_action_reward_noise=False,
        noise_schedule="v126 CPU torch.Generator; one float32 (1,3) draw per step",
        context_protocol="restore source normalizer, reset EMA; online observes transitions, zero-context supplies zeros",
        policy_receives_profile_labels=False, reward_changes=[])


def cells(spec):
    return [dict(condition=condition, mode="deterministic" if noise is None else "sampled",
                 reset_seed=reset, noise_seed=noise)
            for condition in spec["conditions"] for reset in spec["reset_seeds"]
            for noise in (None, *spec["action_noise_seeds"])]


def make_env(condition):
    assert condition in CONDITIONS
    profile = "id" if condition == "id" else condition.removeprefix("static/")
    return ExtremeOODWrapper(gym.make("Hopper-v4"), config=ExtremeOODConfig(
        intensity=50., profile_family="structured", schedule="markov" if condition == "markov" else "static",
        static_profile=None if condition == "markov" else profile, switch_every_steps=50, switch_probability=1.,
        include_sensor_noise=False, include_action_noise=False, include_reward_noise=False))


def actual_multipliers(env):
    raw, base, profile = env.unwrapped, env._base, env.current_profile
    expected = {key: float(profile.get(key, 1.)) for key in ("body_mass", "gravity", "friction")}
    np.testing.assert_allclose(raw.model.body_mass, base["body_mass"]*expected["body_mass"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(raw.model.opt.gravity, base["gravity"]*expected["gravity"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(raw.model.geom_friction, base["geom_friction"]*expected["friction"], rtol=1e-12, atol=1e-12)
    return dict(body_mass=float(raw.model.body_mass.sum()/base["body_mass"].sum()),
                gravity=float(np.linalg.norm(raw.model.opt.gravity)/np.linalg.norm(base["gravity"])),
                friction=expected["friction"])


def evaluate_episode(agent, cell, context_mode, max_steps):
    assert context_mode in CONTEXT_MODES
    evaluation = v134.evaluation
    with evaluation.v124.frozen_context_rng(agent):
        agent.reset_context()
        initial_count = agent.context_tracker._count
        env = make_env(cell["condition"])
        rewards, velocities, stds, differences, contexts = [], [], [], [], []
        counts, events = Counter(), []
        noise = evaluation.noise_schedule(cell["noise_seed"], max_steps, agent.config.action_dim)
        scale = 0. if cell["mode"] == "deterministic" else 1.
        try:
            state, _ = env.reset(seed=cell["reset_seed"])
            raw = env.unwrapped
            assert raw._terminate_when_unhealthy and raw._healthy_reward == 1.
            for step in range(max_steps):
                context = agent._current_context_np() if context_mode == "online" else np.zeros(agent.context_dim, dtype=np.float32)
                tensor = torch.as_tensor(agent._augment_state_np(state, context), device=agent.device).unsqueeze(0)
                action_tensor, mean, std = evaluation.action_from_noise(agent.actor, tensor, noise[step:step+1], scale)
                action = action_tensor.squeeze(0).numpy()
                deterministic = (agent.actor.action_range*mean.tanh()).squeeze(0).numpy()
                executed = np.clip(action*env.action_space.high, env.action_space.low, env.action_space.high)
                next_state, reward, term, trunc, info = env.step(executed)
                velocity = float(info["x_velocity"])
                native_reward = 1.+raw._forward_reward_weight*velocity-raw._ctrl_cost_weight*float(np.square(executed).sum())
                np.testing.assert_allclose(reward, native_reward, rtol=1e-9, atol=1e-9)
                profile = info["ood_profile"]
                counts[profile] += 1
                if not events or events[-1]["profile"] != profile:
                    events.append(dict(step=step+1, profile=profile, actual_multipliers=actual_multipliers(env)))
                rewards.append(float(reward)); velocities.append(velocity)
                stds.append(float(std.mean())); differences.append(float(np.linalg.norm(action-deterministic)))
                contexts.append(float(np.linalg.norm(context)))
                if context_mode == "online":
                    agent.observe_context(state, action, reward, next_state, term or trunc)
                state = next_state
                if term or trunc:
                    break
            context_updates = agent.context_tracker._count-initial_count
            assert context_updates == (len(rewards) if context_mode == "online" else 0)
            return dict(**cell, context_mode=context_mode, noise_scale=scale, length=len(rewards),
                episode_return=float(np.sum(rewards)), mean_velocity=float(np.mean(velocities)),
                physical_termination=bool(term), time_limit_truncation=bool(trunc), diagnostic_cutoff=not(term or trunc),
                mean_policy_std=float(np.mean(stds)), mean_action_noise_l2=float(np.mean(differences)),
                mean_context_l2=float(np.mean(contexts)), context_updates=context_updates,
                profile_counts=dict(counts), non_id_steps=sum(n for name, n in counts.items() if name != "id"),
                profile_events=events, switches=len(events)-1,
                final_physical_state=evaluation.v124.physical_state(raw), native_reward_verified=True)
        finally:
            env.close()


def collect(agent, spec, progress, save):
    expected = cells(spec)
    assert [{k: e[k] for k in expected[0]} for e in progress["evaluations"]] == expected[:len(progress["evaluations"])]
    for cell in expected[len(progress["evaluations"]):]:
        episode = evaluate_episode(agent, cell, spec["context_mode"], spec["max_steps"])
        progress["evaluations"].append(episode)
        save()
        print(f"[EVAL] seed{spec['training_seed']} {spec['context_mode']} "
              f"cell={len(progress['evaluations'])}/{len(expected)} {cell['condition']} "
              f"mode={cell['mode']} length={episode['length']} non_id={episode['non_id_steps']}", flush=True)
    assert agent.training_steps == 299745 and agent.replay_buffer.size == 0


def run(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    versions = dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
                    gymnasium=gym.__version__, mujoco=__import__("mujoco").__version__)
    source = json.loads(args.source_json.read_text())
    agent = v134.load_source(source, args.seed, "mse-control", versions)
    spec, started = stage_spec(args.seed, args.context_mode, args.smoke), time.monotonic()
    metadata = dict(revision=source["revision"], checkpoint=source["checkpoint"], config=asdict(agent.config),
        context_count=agent.context_tracker._count, training_steps=agent.training_steps,
        initial_context_normalizer=dict(count=agent.context_tracker._count,
            mean=agent.context_tracker._mean.tolist(), m2=agent.context_tracker._m2.tolist()))
    progress = json.loads(args.resume_from.read_text()) if args.resume_from else dict(
        revision=REVISION, spec=spec, runtime=versions, source=metadata, evaluations=[])
    assert (progress["revision"], progress["spec"], progress["runtime"], progress["source"]) == (REVISION, spec, versions, metadata)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    def save():
        path = args.checkpoint_dir/"progress.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(progress, indent=2, allow_nan=False)+"\n")
        temporary.replace(path)
    collect(agent, spec, progress, save)
    assert not torch.cuda.is_initialized()
    result = dict(**progress, training_seed=args.seed, context_mode=args.context_mode, smoke=args.smoke,
        profiles=make_ood_profiles("Hopper-v4", intensity=50., family="structured"), evaluation_device="cpu",
        cuda_initialized=False, optimizer_updates=0, new_training_interactions=0, replay_loaded=False,
        source_readout_refitted=False, checkpoint_selection="predeclared-final-only", reward_changes=[],
        policy_receives_profile_labels=False, ablation_scope="inference context, not training representation",
        seconds=time.monotonic()-started, evaluation_interactions=sum(e["length"] for e in progress["evaluations"]))
    args.result_dir.mkdir(parents=True, exist_ok=True)
    (args.result_dir/"ood_context.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] seed{args.seed} {args.context_mode} episodes={len(cells(spec))} "
          f"optimizer_updates=0 evaluation_steps={result['evaluation_interactions']}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(3, 4), required=True)
    parser.add_argument("--context-mode", choices=CONTEXT_MODES, required=True)
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
