"""Single-factor source pilot: remove target L1 uplift, retain beta=-2."""
import argparse
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

import run_general_ood_v121_locomotion_source as v121
import probe_general_ood_v122_target_scale as probe

REVISION = "general_ood_v122_target_l1"


def make_agent(env, device):
    agent = v121.v39.make_agent("Hopper-v4", "fixed-minus2", device, env)
    agent.config.weight_reg = 0.
    return agent


def stage_spec(smoke=False):
    return dict(v121.stage_spec(smoke), study_revision=REVISION, intervention="weight_reg=0-only")


def run(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env = gym.make("Hopper-v4")
    agent = make_agent(env, args.device)
    raw = env.unwrapped
    reward_settings = dict(healthy_reward=float(raw._healthy_reward), ctrl_cost_weight=float(raw._ctrl_cost_weight),
                          forward_reward_weight=float(raw._forward_reward_weight))
    spec, started = stage_spec(args.smoke), time.monotonic()
    training = v121.train(agent, env, args.seed, spec, args.checkpoint_dir/"latest.pt", args.resume_from)
    final = args.checkpoint_dir/"final.pt"
    agent.save(str(final))
    indices = np.linspace(0, agent.replay_buffer.size-1, 32 if args.smoke else 512).astype(int)
    target_scale = probe.decompose(agent, indices, 12200)
    restored = make_agent(env, args.device)
    restored.load(str(final))
    env.close()
    result = dict(revision=REVISION, training_driver_revision=v121.REVISION, training_seed=args.seed,
        smoke=args.smoke, spec=spec, config=asdict(agent.config), environment="Hopper-v4",
        training_mode="normal-only", effective_beta=-2., trained_from_scratch=True,
        intervention=dict(parameter="weight_reg", control_value=.01, pilot_value=0.),
        control_revision=v121.REVISION, reward_settings=reward_settings,
        checkpoint_selection="final-not-best", checkpoint=str(final),
        periodic_evaluation_selects_checkpoint=False, optimizer_updates_during_evaluation=0,
        source_readout_refitted=False, training=training, final_target_scale=target_scale,
        final_evaluation=v121.evaluate(restored, spec["final_evaluation_seed"], spec["final_evaluation_episodes"],
                                       spec["evaluation_max_steps"]),
        runtime=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
            gymnasium=gym.__version__, mujoco=__import__("mujoco").__version__,
            gpu=torch.cuda.get_device_name() if args.device == "cuda" else None), seconds=time.monotonic()-started)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    (args.result_dir/"source.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] seed{args.seed} weight_reg=0 steps={spec['steps']} "
          f"final_velocity={result['final_evaluation']['mean_episode_velocity']:.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=v121.SEEDS, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
