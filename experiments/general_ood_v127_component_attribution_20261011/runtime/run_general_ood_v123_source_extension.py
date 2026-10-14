"""Continue all three v122 source prefixes without further hyperparameter changes."""
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

import run_general_ood_v122_target_l1 as v122

v121 = v122.v121
REVISION = "general_ood_v123_source_extension"


def stage_spec(smoke=False):
    source = v122.stage_spec()
    spec = v121.stage_spec(smoke)
    spec.update(steps=source["steps"]+(512 if smoke else 200000), study_revision=REVISION,
                intervention="unchanged-v122-training-budget-extension", source_steps=source["steps"])
    return spec


def continue_training(agent, env, seed, spec, source_checkpoint, checkpoint, resume_from=None):
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if resume_from is None:
        saved = agent.load(str(source_checkpoint))
        assert (saved["revision"], saved["seed"], saved["spec"]) == (v121.REVISION, seed, v122.stage_spec())
        loop = saved["loop"]
        assert loop["step"] == spec["source_steps"] == agent.context_tracker._count
        assert agent.training_steps == loop["step"]-agent.config.batch_size+1
        assert agent.config.weight_reg == 0.
        env.reset(seed=0)
        v121.restore_environment(env, saved["environment"])
        if loop["episode_age"] == 0:
            loop["state"], _ = env.reset(seed=seed*10000+loop["episodes_completed"])
            agent.reset_context()
        loop["evaluations"].append(dict(step=loop["step"], **v121.evaluate(agent, spec["evaluation_seed"],
            spec["evaluation_episodes"], spec["evaluation_max_steps"])))
        v121.save_progress(agent, env, loop, spec, seed, checkpoint)
        resume_from = checkpoint
        print(f"[CONTINUE] seed{seed} source_steps={loop['step']} target_steps={spec['steps']} "
              f"episode_age={loop['episode_age']} replay={agent.replay_buffer.size}", flush=True)
    return v121.train(agent, env, seed, spec, checkpoint, resume_from)


def run(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    env = gym.make("Hopper-v4")
    agent = v122.make_agent(env, args.device)
    source_checkpoint = args.source_runtime/f"run_checkpoints/seed{args.seed}/latest.pt"
    source_result = json.loads((args.source_runtime/f"run_results/seed{args.seed}/source.json").read_text())
    assert source_result["revision"] == v122.REVISION and not source_result["smoke"]
    assert source_result["training_seed"] == args.seed
    expected_config = dict(source_result["config"], device=args.device)
    assert asdict(agent.config) == expected_config
    spec, started = stage_spec(args.smoke), time.monotonic()
    training = continue_training(agent, env, args.seed, spec, source_checkpoint,
                                 args.checkpoint_dir/"latest.pt", args.resume_from)
    final = args.checkpoint_dir/"final.pt"
    agent.save(str(final))
    indices = np.linspace(0, agent.replay_buffer.size-1, 32 if args.smoke else 512).astype(int)
    target_scale = v122.probe.decompose(agent, indices, 12200)
    restored = v122.make_agent(env, args.device)
    restored.load(str(final))
    env.close()
    result = dict(revision=REVISION, training_driver_revision=v121.REVISION, training_seed=args.seed,
        smoke=args.smoke, spec=spec, config=asdict(agent.config), environment="Hopper-v4",
        training_mode="normal-only", effective_beta=-2., trained_from_scratch=False,
        source_revision=v122.REVISION, source_checkpoint=str(source_checkpoint),
        source_interactions=spec["source_steps"], actual_new_training_interactions=spec["steps"]-spec["source_steps"],
        actual_new_optimizer_updates=training["optimizer_updates"]-source_result["training"]["optimizer_updates"],
        reward_settings=source_result["reward_settings"], checkpoint_selection="final-not-best", checkpoint=str(final),
        periodic_evaluation_selects_checkpoint=False, optimizer_updates_during_evaluation=0,
        source_readout_refitted=False, training=training, final_target_scale=target_scale,
        final_evaluation=v121.evaluate(restored, spec["final_evaluation_seed"], spec["final_evaluation_episodes"],
                                       spec["evaluation_max_steps"]),
        runtime=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
            gymnasium=gym.__version__, mujoco=__import__("mujoco").__version__,
            gpu=torch.cuda.get_device_name() if args.device == "cuda" else None), seconds=time.monotonic()-started)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    (args.result_dir/"source.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] seed{args.seed} total_steps={spec['steps']} "
          f"new_steps={result['actual_new_training_interactions']} "
          f"final_velocity={result['final_evaluation']['mean_episode_velocity']:.4f} "
          f"length={result['final_evaluation']['mean_episode_length']:.1f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=v121.SEEDS, required=True)
    parser.add_argument("--source-runtime", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
