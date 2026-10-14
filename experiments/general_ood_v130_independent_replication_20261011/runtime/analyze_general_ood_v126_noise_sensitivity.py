"""Frozen noise-scale and common-noise CPU/CUDA source-policy diagnosis."""
import argparse
from collections import deque
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

import run_general_ood_v125_actor_beta as v125

v124 = v125.v124
REVISION = "general_ood_v126_noise_sensitivity"
NOISE_GROUPS = {"legacy": [12400, 12401, 12402], "fresh": [12600, 12601, 12602]}


def stage_spec(seed, device="cpu", smoke=False):
    assert device == "cpu" or seed == 2
    return dict(training_seed=seed, device=device, smoke=smoke, ages=[300000, 350000],
                reset_seeds=list(range(150000, 150001 if smoke else 150005)),
                noise_groups={k: v[:1] if smoke else v for k, v in NOISE_GROUPS.items()},
                noise_scales=[0., .25, .5, 1.], max_steps=64 if smoke else 1000,
                noise_schedule="CPU torch.Generator, one float32 (1,3) draw per environment step")


def cells(spec):
    return [(reset, 0., None, "deterministic") for reset in spec["reset_seeds"]] + [
        (reset, scale, noise, group) for scale in spec["noise_scales"][1:]
        for group, noises in spec["noise_groups"].items()
        for reset in spec["reset_seeds"] for noise in noises]


def noise_schedule(seed, steps, action_dim):
    if seed is None:
        return torch.zeros(steps, action_dim)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    # Step-sized draws preserve the legacy CPU get_action noise sequence.
    return torch.cat([torch.randn(1, action_dim, generator=generator) for _ in range(steps)])


def action_from_noise(actor, state, noise, scale):
    with torch.no_grad():
        mean, log_std = actor(state)
        action = actor.action_range * torch.tanh(mean + scale * log_std.exp() * noise)
    return action, mean, log_std.exp()


def evaluate_episode(agent, reset_seed, scale, noise_seed, max_steps, parity_agent=None):
    with v124.frozen_context_rng(agent):
        agent.reset_context()
        env = gym.make("Hopper-v4")
        raw, tail = env.unwrapped, deque(maxlen=3)
        noise = noise_schedule(noise_seed, max_steps, agent.config.action_dim)
        rewards, velocities, differences, stds, contexts = [], [], [], [], []
        parity = dict(samples=0, max_mean_abs_difference=0., max_std_abs_difference=0.,
                      max_action_abs_difference=0.)
        try:
            state, _ = env.reset(seed=reset_seed)
            for step in range(max_steps):
                augmented = agent._augment_state_np(state)
                tensor = torch.as_tensor(augmented, device=agent.device).unsqueeze(0)
                epsilon = noise[step:step+1].to(agent.device)
                sampled, mean, std = action_from_noise(agent.actor, tensor, epsilon, scale)
                action = sampled.squeeze(0).cpu().numpy()
                deterministic = (agent.actor.action_range * mean.tanh()).squeeze(0).cpu().numpy()
                context = agent._current_context_np()
                if parity_agent is not None and step % 25 == 0:
                    other = action_from_noise(parity_agent.actor, tensor.cpu(), epsilon.cpu(), scale)
                    parity["samples"] += 1
                    for key, x, y in zip(("max_action_abs_difference", "max_mean_abs_difference",
                                          "max_std_abs_difference"), (sampled, mean, std), other):
                        parity[key] = max(parity[key], float((x.cpu()-y).abs().max()))
                before = v124.physical_state(raw)
                executed = np.clip(action*env.action_space.high, env.action_space.low, env.action_space.high)
                next_state, reward, term, trunc, info = env.step(executed)
                context_norm = float(np.linalg.norm(context))
                tail.append(dict(step=step+1, before=before, after=v124.physical_state(raw),
                    action=action.tolist(), policy_mean=mean.squeeze(0).cpu().tolist(),
                    policy_std=std.squeeze(0).cpu().tolist(), context=context.tolist(),
                    velocity=float(info["x_velocity"])))
                rewards.append(float(reward)); velocities.append(float(info["x_velocity"]))
                differences.append(float(np.linalg.norm(action-deterministic)))
                stds.append(float(std.mean())); contexts.append(context_norm)
                agent.observe_context(state, action, reward, next_state, term or trunc)
                state = next_state
                if term or trunc:
                    break
            return dict(reset_seed=reset_seed, noise_seed=noise_seed, noise_scale=scale,
                length=len(rewards), episode_return=float(np.sum(rewards)),
                mean_velocity=float(np.mean(velocities)), physical_termination=bool(term),
                time_limit_truncation=bool(trunc), diagnostic_cutoff=not(term or trunc),
                mean_policy_std=float(np.mean(stds)), mean_action_noise_l2=float(np.mean(differences)),
                mean_context_l2=float(np.mean(contexts)), final_physical_state=v124.physical_state(raw),
                tail=list(tail), same_input_parity=parity if parity_agent is not None else None)
        finally:
            env.close()


def run(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    spec, started = stage_spec(args.seed, args.device, args.smoke), time.monotonic()
    versions = dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
                    gymnasium=gym.__version__, mujoco=__import__("mujoco").__version__)
    progress = json.loads(args.resume_from.read_text()) if args.resume_from else dict(
        revision=REVISION, spec=spec, runtime=versions, models=[])
    assert (progress["revision"], progress["spec"], progress["runtime"]) == (REVISION, spec, versions)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    def save():
        path = args.checkpoint_dir/"progress.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(progress, indent=2, allow_nan=False)+"\n")
        temporary.replace(path)
    for age, runtime, relative, revision in (
        (300000, args.v123_runtime, f"seed{args.seed}/source.json", v125.v123.REVISION),
        (350000, args.v125_runtime, f"seed{args.seed}_beta-minus2/actor_beta.json", v125.REVISION)):
        source = json.loads((runtime/"run_results"/relative).read_text())
        assert source["revision"] == revision and source["training_seed"] == args.seed and not source["smoke"]
        assert source["checkpoint_selection"] == "final-not-best" and source["effective_beta"] == -2.
        assert source["training"]["interactions"] == age and source["config"]["weight_reg"] == 0.
        assert all(versions[k] == source["runtime"][k] for k in versions)
        checkpoint = Path(source["checkpoint"])
        env = gym.make("Hopper-v4")
        agent = v125.make_agent(env, args.device, -2.)
        assert asdict(agent.config) == dict(source["config"], device=args.device)
        agent.load(str(checkpoint))
        assert agent.training_steps == age-agent.config.batch_size+1 and agent.context_tracker._count == age
        initial_context = dict(count=agent.context_tracker._count,
            mean=agent.context_tracker._mean.tolist(), m2=agent.context_tracker._m2.tolist())
        parity_agent = None
        if args.device == "cuda":
            parity_agent = v125.make_agent(env, "cpu", -2.)
            parity_agent.load(str(checkpoint))
        env.close()
        row = next((r for r in progress["models"] if r["age"] == age), None)
        if row is None:
            row = dict(age=age, checkpoint=str(checkpoint), source_revision=revision,
                config=dict(source["config"], device="cpu"), alpha=agent.alpha,
                initial_context_normalizer=initial_context, evaluations=[])
            progress["models"].append(row)
        assert row["checkpoint"] == str(checkpoint) and row["initial_context_normalizer"] == initial_context
        expected = cells(spec)
        assert [(e["reset_seed"], e["noise_scale"], e["noise_seed"], e["noise_group"])
                for e in row["evaluations"]] == expected[:len(row["evaluations"])]
        for reset, scale, noise, group in expected[len(row["evaluations"]):]:
            result = evaluate_episode(agent, reset, scale, noise, spec["max_steps"], parity_agent)
            row["evaluations"].append(dict(result, noise_group=group))
            save()
            print(f"[EVAL] seed{args.seed} device={args.device} age={age} "
                  f"cell={len(row['evaluations'])}/{len(expected)} scale={scale} "
                  f"group={group} length={result['length']}", flush=True)
        del agent, parity_agent
    result = dict(**progress, training_seed=args.seed, evaluation_device=args.device,
        gpu=torch.cuda.get_device_name() if args.device == "cuda" else None,
        optimizer_updates=0, new_training_interactions=0, source_readout_refitted=False,
        checkpoint_selection="final-not-best", reward_or_physics_changes=[],
        seconds=time.monotonic()-started,
        evaluation_interactions=sum(e["length"] for r in progress["models"] for e in r["evaluations"]))
    args.result_dir.mkdir(parents=True, exist_ok=True)
    (args.result_dir/"noise.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] seed{args.seed} device={args.device} optimizer_updates=0 "
          f"evaluation_steps={result['evaluation_interactions']}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--v123-runtime", type=Path, required=True)
    parser.add_argument("--v125-runtime", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
