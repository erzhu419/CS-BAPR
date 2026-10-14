"""Extend all stable v141 full states from300k to600k without changing rates."""
import argparse
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

import run_general_ood_v141_actor_rate as source_worker

v132, v137 = source_worker.v132, source_worker.v137
REVISION = "general_ood_v142_budget_extension"
PROFILES = source_worker.PROFILES
make_agent = source_worker.make_agent
make_env = source_worker.make_env
final_evaluation = source_worker.final_evaluation


def stage_spec(profile, smoke=False):
    source = source_worker.stage_spec(profile)
    return dict(source, steps=source["steps"]+(512 if smoke else 300000),
                source_steps=source["steps"], study_revision=REVISION, intervention="training-budget-only")


def continue_training(agent, env, seed, spec, source_spec, source_checkpoint, checkpoint,
                      resume_from=None, evaluation_fn=None):
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if resume_from is None:
        saved = agent.load(str(source_checkpoint))
        assert (saved["revision"], saved["seed"], saved["spec"]) == (v132.REVISION, seed, source_spec)
        loop = saved["loop"]
        assert loop["step"] == spec["source_steps"] == agent.context_tracker._count
        assert agent.training_steps == loop["step"]-agent.config.batch_size+1
        assert agent.replay_buffer.size == loop["step"]
        env.reset(seed=0)
        v132.restore_environment(env, saved["environment"])
        if loop["episode_age"] == 0:
            loop["state"], _ = env.reset(seed=seed*10000+loop["episodes_completed"])
            agent.reset_context()
        # The old endpoint skipped its periodic evaluation; include it as a new observation.
        if loop["step"] % spec["evaluation_interval"] == 0:
            result = evaluation_fn(agent, spec["evaluation_seed"], spec["evaluation_episodes"], spec["evaluation_max_steps"])
            loop["evaluations"].append(dict(step=loop["step"], **result))
        v132.save_progress(agent, env, loop, spec, seed, checkpoint)
        resume_from = checkpoint
        print(f"[CONTINUE] seed{seed} {spec['training_profile']} {loop['step']}/{spec['steps']} "
              f"updates={agent.training_steps} replay={agent.replay_buffer.size} episode_age={loop['episode_age']}", flush=True)
    return v132.train(agent, env, seed, spec, checkpoint, resume_from, evaluation_fn=evaluation_fn)


def run(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    source = json.loads(args.source_json.read_text())
    assert (source["revision"], source["training_seed"], source["training_profile"], source["smoke"]) == (
        source_worker.REVISION, args.seed, args.profile, False)
    assert source["spec"] == source_worker.stage_spec(args.profile)
    assert source["training"]["interactions"] == 300000 and source["training"]["optimizer_updates"] == 299745
    env, physics = make_env(args.profile)
    agent = make_agent(env, args.device, "mse-control")
    assert asdict(agent.config) == dict(source["config"], device=args.device)
    assert physics == source["physics"]
    raw = env.unwrapped
    reward_settings = dict(healthy_reward=float(raw._healthy_reward), ctrl_cost_weight=float(raw._ctrl_cost_weight),
                          forward_reward_weight=float(raw._forward_reward_weight))
    assert reward_settings == source["reward_settings"]
    runtime = dict(python=sys.version, torch=torch.__version__, numpy=np.__version__, gymnasium=gym.__version__,
        mujoco=__import__("mujoco").__version__, gpu=torch.cuda.get_device_name() if args.device == "cuda" else None)
    assert all(runtime[k] == source["runtime"][k] for k in runtime if k != "gpu")
    spec, started = stage_spec(args.profile, args.smoke), time.monotonic()
    callback = lambda a, seed, episodes, limit: v137.evaluate(a, args.profile, seed, episodes, limit, env_factory=make_env)
    source_checkpoint = Path(source["resumable_checkpoint"])
    training = continue_training(agent, env, args.seed, spec, source["spec"], source_checkpoint,
                                 args.checkpoint_dir/"latest.pt", args.resume_from, callback)
    new_steps = training["interactions"]-source["training"]["interactions"]
    new_updates = training["optimizer_updates"]-source["training"]["optimizer_updates"]
    assert new_steps == new_updates == (512 if args.smoke else 300000)
    v137.assert_physics(env, physics)
    final = args.checkpoint_dir/"final.pt"
    agent.save(str(final))
    restored = make_agent(env, "cpu", "mse-control")
    restored.load(str(final))
    assert restored.training_steps == training["optimizer_updates"] and restored.context_tracker._count == spec["steps"]
    for model in (restored.actor, restored.critic, restored.target_critic):
        model.eval().requires_grad_(False)
    env.close()
    row = dict(revision=REVISION, training_driver_revision=v132.REVISION, training_seed=args.seed,
        training_profile=args.profile, arm="mse-control", smoke=args.smoke, spec=spec, config=asdict(agent.config),
        source_revision=source_worker.REVISION, source_json=str(args.source_json), source_checkpoint=str(source_checkpoint),
        source_interactions=300000, source_optimizer_updates=299745, source_runtime=source["runtime"],
        actual_new_training_interactions=new_steps, actual_new_optimizer_updates=new_updates,
        trained_from_scratch=False, training_mode="fixed-profile-continuation", training_device=args.device,
        environment="Hopper-v4", effective_beta=-2., physics=physics, reward_settings=reward_settings,
        reward_changes=[], policy_receives_profile_labels=False, checkpoint_selection="final-not-best",
        checkpoint=str(final), resumable_checkpoint=str(args.checkpoint_dir/"latest.pt"),
        periodic_evaluation_selects_checkpoint=False, optimizer_updates_during_evaluation=0, source_readout_refitted=False,
        training=training, final_evaluation=final_evaluation(restored, args.profile, args.smoke),
        final_context_count=restored.context_tracker._count, runtime=runtime, seconds=time.monotonic()-started,
        scope="Budget-only continuation of v141, not a matched-budget comparison against original300k baselines")
    args.result_dir.mkdir(parents=True, exist_ok=True)
    (args.result_dir/"source.json").write_text(json.dumps(row, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] seed{args.seed} {args.profile} total_steps={spec['steps']} new_steps={new_steps}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(3, 4), required=True)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
