"""Paired abrupt-OOD probes after an identical normal-policy warmup."""
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

import analyze_general_ood_v135_ood_context as v135

REVISION = "general_ood_v136_warmup_context"
CONTEXT_MODES = ("online", "held-context")
CONDITIONS = ("id", "abrupt/easy_surface", "abrupt/heavy_traffic", "abrupt/mega_event", "abrupt/slippery_rush")


def stage_spec(seed, smoke=False):
    return dict(training_seed=seed, source_arm="mse-control", source_steps=300000, smoke=smoke, device="cpu",
        context_modes=list(CONTEXT_MODES), conditions=["id", "abrupt/mega_event"] if smoke else list(CONDITIONS),
        reset_seeds=list(range(153000, 153001 if smoke else 153005)),
        action_noise_seeds=[136000] if smoke else [136000, 136001, 136002], sampled_noise_scale=1.,
        warmup_steps=200, first_ood_step=201, max_steps=256 if smoke else 1000,
        profile_family="structured", intensity=50., reward_changes=[], policy_receives_profile_labels=False,
        noise_schedule="v126 CPU torch.Generator; one float32 (1,3) draw per step",
        context_protocol="restore source normalizer and reset EMA; both update200 ID transitions; held-context stops thereafter",
        prefix_check="exact state/context/action/transition equality through first post-warmup step, in memory only")


def cells(spec):
    return [dict(condition=condition, mode="deterministic" if noise is None else "sampled", reset_seed=reset, noise_seed=noise)
            for condition in spec["conditions"] for reset in spec["reset_seeds"]
            for noise in (None, *spec["action_noise_seeds"])]


def make_env(condition, warmup_steps=200):
    assert condition in CONDITIONS
    names = {"id", condition.removeprefix("abrupt/")}
    profiles = [p for p in v135.make_ood_profiles("Hopper-v4", intensity=50., family="structured") if p["name"] in names]
    return v135.ExtremeOODWrapper(gym.make("Hopper-v4"), profiles=profiles, config=v135.ExtremeOODConfig(
        intensity=50., profile_family="structured", schedule="abrupt", abrupt_step=warmup_steps,
        include_sensor_noise=False, include_action_noise=False, include_reward_noise=False))


def evaluate_episode(agent, cell, context_mode, max_steps, warmup_steps=200):
    assert context_mode in CONTEXT_MODES
    evaluation = v135.v134.evaluation
    with evaluation.v124.frozen_context_rng(agent):
        agent.reset_context()
        initial_count = agent.context_tracker._count
        env = make_env(cell["condition"], warmup_steps)
        rewards, velocities, contexts, changes, differences, prefix = [], [], [], [], [], []
        counts, events = Counter(), []
        boundary_context, boundary_state = None, None
        noise = evaluation.noise_schedule(cell["noise_seed"], max_steps, agent.config.action_dim)
        scale = 0. if cell["mode"] == "deterministic" else 1.
        try:
            state, _ = env.reset(seed=cell["reset_seed"])
            raw = env.unwrapped
            assert raw._terminate_when_unhealthy and raw._healthy_reward == 1.
            for step in range(max_steps):
                context = agent._current_context_np()
                if context_mode == "held-context" and step >= warmup_steps:
                    np.testing.assert_array_equal(context, boundary_context)
                tensor = torch.as_tensor(agent._augment_state_np(state, context), device=agent.device).unsqueeze(0)
                action_tensor, mean, _ = evaluation.action_from_noise(agent.actor, tensor, noise[step:step+1], scale)
                action = action_tensor.squeeze(0).numpy()
                deterministic = (agent.actor.action_range*mean.tanh()).squeeze(0).numpy()
                executed = np.clip(action*env.action_space.high, env.action_space.low, env.action_space.high)
                next_state, reward, term, trunc, info = env.step(executed)
                velocity = float(info["x_velocity"])
                native_reward = 1.+raw._forward_reward_weight*velocity-raw._ctrl_cost_weight*float(np.square(executed).sum())
                np.testing.assert_allclose(reward, native_reward, rtol=1e-9, atol=1e-9)
                if step <= warmup_steps:
                    prefix.append(np.concatenate((state, context, action, next_state, raw.data.qpos.copy(), raw.data.qvel.copy(), [reward, velocity])))
                profile = info["ood_profile"]
                counts[profile] += 1
                if not events or events[-1]["profile"] != profile:
                    events.append(dict(step=step+1, profile=profile, actual_multipliers=v135.actual_multipliers(env)))
                rewards.append(float(reward)); velocities.append(velocity)
                contexts.append(float(np.linalg.norm(context)))
                differences.append(float(np.linalg.norm(action-deterministic)))
                if step >= warmup_steps:
                    changes.append(float(np.linalg.norm(context-boundary_context)))
                if context_mode == "online" or step < warmup_steps:
                    agent.observe_context(state, action, reward, next_state, term or trunc)
                state = next_state
                if step+1 == warmup_steps:
                    boundary_context = agent._current_context_np().copy()
                    boundary_state = evaluation.v124.physical_state(raw)
                if term or trunc:
                    break
            length = len(rewards)
            updates = agent.context_tracker._count-initial_count
            assert updates == (length if context_mode == "online" else min(length, warmup_steps))
            episode = dict(**cell, context_mode=context_mode, noise_scale=scale, length=length,
                episode_return=float(np.sum(rewards)), mean_velocity=float(np.mean(velocities)),
                physical_termination=bool(term), time_limit_truncation=bool(trunc), diagnostic_cutoff=not(term or trunc),
                warmup_steps=min(length, warmup_steps), warmup_return=float(np.sum(rewards[:warmup_steps])),
                intervention_reached=length > warmup_steps, post_warmup_steps=max(length-warmup_steps, 0),
                post_warmup_return=float(np.sum(rewards[warmup_steps:])),
                post_warmup_mean_velocity=float(np.mean(velocities[warmup_steps:])) if changes else None,
                context_updates=updates, mean_context_l2=float(np.mean(contexts)),
                mean_post_context_change_l2=float(np.mean(changes)) if changes else None,
                mean_action_noise_l2=float(np.mean(differences)),
                context_at_intervention=None if boundary_context is None else boundary_context.tolist(),
                warmup_end_physical_state=boundary_state, profile_counts=dict(counts), profile_events=events,
                non_id_steps=sum(n for name, n in counts.items() if name != "id"), switches=len(events)-1,
                final_physical_state=evaluation.v124.physical_state(raw), native_reward_verified=True)
            return episode, np.stack(prefix)
        finally:
            env.close()


def evaluate_pair(agent, cell, max_steps, warmup_steps=200):
    online, prefix = evaluate_episode(agent, cell, "online", max_steps, warmup_steps)
    held, comparison = evaluate_episode(agent, cell, "held-context", max_steps, warmup_steps)
    np.testing.assert_array_equal(prefix, comparison)
    assert online["context_at_intervention"] == held["context_at_intervention"]
    assert online["warmup_end_physical_state"] == held["warmup_end_physical_state"]
    return dict(**cell, episodes=[online, held], prefix_equal=True, prefix_steps=len(prefix),
                prefix_max_abs_difference=float(np.max(np.abs(prefix-comparison))))


def collect(agent, spec, progress, save):
    expected = cells(spec)
    assert [{k: p[k] for k in expected[0]} for p in progress["pairs"]] == expected[:len(progress["pairs"])]
    for cell in expected[len(progress["pairs"]):]:
        pair = evaluate_pair(agent, cell, spec["max_steps"], spec["warmup_steps"])
        progress["pairs"].append(pair)
        save()
        print(f"[EVAL] seed{spec['training_seed']} pair={len(progress['pairs'])}/{len(expected)} "
              f"{cell['condition']} {cell['mode']} post_steps={[e['post_warmup_steps'] for e in pair['episodes']]} prefix_equal=True", flush=True)
    assert agent.training_steps == 299745 and agent.replay_buffer.size == 0


def run(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    versions = dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
                    gymnasium=gym.__version__, mujoco=__import__("mujoco").__version__)
    source = json.loads(args.source_json.read_text())
    agent = v135.v134.load_source(source, args.seed, "mse-control", versions)
    spec, started = stage_spec(args.seed, args.smoke), time.monotonic()
    metadata = dict(revision=source["revision"], checkpoint=source["checkpoint"], config=asdict(agent.config),
        context_count=agent.context_tracker._count, training_steps=agent.training_steps,
        initial_context_normalizer=dict(count=agent.context_tracker._count,
            mean=agent.context_tracker._mean.tolist(), m2=agent.context_tracker._m2.tolist()))
    progress = json.loads(args.resume_from.read_text()) if args.resume_from else dict(
        revision=REVISION, spec=spec, runtime=versions, source=metadata, pairs=[])
    assert (progress["revision"], progress["spec"], progress["runtime"], progress["source"]) == (REVISION, spec, versions, metadata)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    def save():
        path = args.checkpoint_dir/"progress.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(progress, indent=2, allow_nan=False)+"\n")
        temporary.replace(path)
    collect(agent, spec, progress, save)
    assert not torch.cuda.is_initialized()
    result = dict(**progress, training_seed=args.seed, smoke=args.smoke,
        profiles=v135.make_ood_profiles("Hopper-v4", intensity=50., family="structured"), evaluation_device="cpu",
        cuda_initialized=False, optimizer_updates=0, new_training_interactions=0, replay_loaded=False,
        source_readout_refitted=False, checkpoint_selection="predeclared-final-only", reward_changes=[],
        policy_receives_profile_labels=False, ablation_scope="post-warmup inference context, not training representation",
        seconds=time.monotonic()-started,
        evaluation_interactions=sum(e["length"] for p in progress["pairs"] for e in p["episodes"]))
    args.result_dir.mkdir(parents=True, exist_ok=True)
    (args.result_dir/"warmup_context.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] seed{args.seed} episodes={2*len(cells(spec))} optimizer_updates=0 evaluation_steps={result['evaluation_interactions']}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(3, 4), required=True)
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
