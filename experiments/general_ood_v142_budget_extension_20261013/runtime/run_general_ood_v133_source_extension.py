"""Continue every v132 endpoint; the only intervention is training budget."""
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

import run_general_ood_v132_source_stability as v132

REVISION = "general_ood_v133_source_extension"


def stage_spec(arm, smoke=False):
    source = v132.stage_spec(arm)
    return dict(v132.stage_spec(arm, smoke), steps=source["steps"]+(512 if smoke else 200000),
        source_steps=source["steps"], study_revision=REVISION, intervention="training-budget-only")


def continue_training(agent, env, seed, spec, source_checkpoint, checkpoint, resume_from=None):
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    assert agent.arm == spec["arm"]
    if resume_from is None:
        saved = agent.load(str(source_checkpoint))
        assert (saved["revision"], saved["seed"], saved["spec"]) == (
            v132.REVISION, seed, v132.stage_spec(spec["arm"]))
        loop = saved["loop"]
        assert loop["step"] == spec["source_steps"] == agent.context_tracker._count
        assert agent.training_steps == loop["step"]-agent.config.batch_size+1
        assert agent.replay_buffer.size == loop["step"] and agent.config.weight_reg == 0.
        env.reset(seed=0)
        v132.restore_environment(env, saved["environment"])
        # The prefix driver defers an episode reset when its final step ends a run.
        if loop["episode_age"] == 0:
            loop["state"], _ = env.reset(seed=seed*10000+loop["episodes_completed"])
            agent.reset_context()
        v132.save_progress(agent, env, loop, spec, seed, checkpoint)
        resume_from = checkpoint
        print(f"[CONTINUE] seed{seed} {spec['arm']} source_steps={loop['step']} "
              f"target_steps={spec['steps']} updates={agent.training_steps} "
              f"episode_age={loop['episode_age']} replay={agent.replay_buffer.size} "
              f"context_count={agent.context_tracker._count}", flush=True)
    return v132.train(agent, env, seed, spec, checkpoint, resume_from)


def run(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    env = gym.make("Hopper-v4")
    agent = v132.make_agent(env, args.device, args.arm)
    source = json.loads(args.source_json.read_text())
    assert (source["revision"], source["training_seed"], source["arm"], source["smoke"]) == (
        v132.REVISION, args.seed, args.arm, False)
    assert source["spec"] == v132.stage_spec(args.arm)
    assert source["training"]["interactions"] == 100000 and source["training"]["optimizer_updates"] == 99745
    assert source["checkpoint_selection"] == "final-not-best"
    assert asdict(agent.config) == dict(source["config"], device=args.device)
    runtime = dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
        gymnasium=gym.__version__, mujoco=__import__("mujoco").__version__,
        gpu=torch.cuda.get_device_name() if args.device == "cuda" else None)
    assert all(runtime[k] == source["runtime"][k] for k in runtime if k != "gpu")
    raw = env.unwrapped
    reward_settings = dict(healthy_reward=float(raw._healthy_reward), ctrl_cost_weight=float(raw._ctrl_cost_weight),
                          forward_reward_weight=float(raw._forward_reward_weight))
    assert reward_settings == source["reward_settings"] and env.spec.max_episode_steps == 1000
    spec, started = stage_spec(args.arm, args.smoke), time.monotonic()
    source_checkpoint = Path(source["resumable_checkpoint"])
    training = continue_training(agent, env, args.seed, spec, source_checkpoint,
                                 args.checkpoint_dir/"latest.pt", args.resume_from)
    new_steps = training["interactions"]-source["training"]["interactions"]
    new_updates = training["optimizer_updates"]-source["training"]["optimizer_updates"]
    assert new_steps == new_updates == (512 if args.smoke else 200000)
    final = args.checkpoint_dir/"final.pt"
    agent.save(str(final))
    restored = v132.make_agent(env, args.device, args.arm)
    restored.load(str(final))
    env.close()
    result = dict(revision=REVISION, training_driver_revision=v132.REVISION, training_seed=args.seed,
        arm=args.arm, smoke=args.smoke, spec=spec, config=asdict(agent.config), runtime=runtime,
        environment="Hopper-v4", training_mode="normal-only", effective_beta=-2., trained_from_scratch=False,
        source_revision=v132.REVISION, source_json=str(args.source_json), source_checkpoint=str(source_checkpoint),
        source_runtime=source["runtime"], source_interactions=spec["source_steps"],
        actual_new_training_interactions=new_steps, actual_new_optimizer_updates=new_updates,
        reward_settings=reward_settings, checkpoint_selection="final-not-best", checkpoint=str(final),
        resumable_checkpoint=str(args.checkpoint_dir/"latest.pt"),
        periodic_evaluation_selects_checkpoint=False, optimizer_updates_during_evaluation=0,
        source_readout_refitted=False, training=training,
        final_evaluation=v132.evaluate(restored, spec["final_evaluation_seed"], spec["final_evaluation_episodes"],
                                      spec["evaluation_max_steps"]), seconds=time.monotonic()-started)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    (args.result_dir/"source.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] seed{args.seed} {args.arm} total_steps={spec['steps']} new_steps={new_steps} "
          f"final_velocity={result['final_evaluation']['mean_episode_velocity']:.4f} "
          f"length={result['final_evaluation']['mean_episode_length']:.1f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=v132.SEEDS, required=True)
    parser.add_argument("--arm", choices=v132.ARMS, required=True)
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
