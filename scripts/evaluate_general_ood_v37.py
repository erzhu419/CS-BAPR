"""Read-only cross-environment OOD evaluation for the v37 route-B matrix."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from csbapr.envs.extreme_ood import ExtremeOODConfig, ExtremeOODWrapper
from scripts.train_csbapr import _evaluate_agent, make_config
from csbapr.agent import CSBAPRAgent


REVISION = "general_ood_v37_cross_env"
METHODS = (
    "csbapr-stable-context",
    "csbapr-no-sindy-stable",
    "bapr-stable-context",
    "resac",
)
ENVS = ("Pendulum-v1", "Hopper-v4")
OOD_INTENSITY = 50.0
EVAL_SEED_START = 70000
EVAL_EPISODES = 20


def make_ood_env(env_name: str, schedule: str, seed: int, static_profile: str | None = None):
    base = gym.make(env_name)
    config = ExtremeOODConfig(
        intensity=OOD_INTENSITY,
        schedule=schedule,
        profile_family="structured",
        seed=seed,
        static_profile=static_profile,
        switch_every_steps=50 if env_name == "Pendulum-v1" else 200,
        switch_probability=1.0,
    )
    return ExtremeOODWrapper(base, config=config)


def evaluate_agent(agent, env_name: str, mode: str, seed_base: int):
    max_steps = int(agent.config.max_steps_per_episode)
    if mode == "id":
        env = gym.make(env_name)
        result = _evaluate_agent(agent, env, max_steps, EVAL_EPISODES, seed_base)
        env.close()
        return result
    if mode == "markov":
        env = make_ood_env(env_name, "markov", seed_base)
        result = _evaluate_agent(agent, env, max_steps, EVAL_EPISODES, seed_base)
        env.close()
        return result
    raise ValueError(mode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, choices=ENVS)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--seed", type=int, required=True, choices=(0, 1, 2))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    assert args.checkpoint.exists(), f"missing checkpoint: {args.checkpoint}"

    config = make_config(args.env, args.method)
    config.device = args.device
    base = gym.make(args.env)
    agent = CSBAPRAgent(base.observation_space.shape[0], base.action_space.shape[0], config)
    agent.load(str(args.checkpoint))
    agent.actor.eval()
    agent.actor.requires_grad_(False)

    modes = {
        "id": evaluate_agent(agent, args.env, "id", EVAL_SEED_START),
        "markov_50x": evaluate_agent(agent, args.env, "markov", EVAL_SEED_START),
    }

    # Fixed-profile evaluations expose whether a random-schedule mean hides a
    # single catastrophic regime. They use the same paired seed block.
    fixed = {}
    probe = make_ood_env(args.env, "static", EVAL_SEED_START)
    profiles = [str(p["name"]) for p in probe.profiles if str(p.get("name")) != "id"]
    probe.close()
    for index, profile in enumerate(profiles):
        env = make_ood_env(args.env, "static", EVAL_SEED_START + index, profile)
        fixed[profile] = _evaluate_agent(
            agent, env, int(config.max_steps_per_episode), 5,
            EVAL_SEED_START + index * 100,
        )
        env.close()

    output = {
        "revision": REVISION,
        "environment": args.env,
        "method": args.method,
        "training_seed": args.seed,
        "checkpoint": args.checkpoint.name,
        "evaluation_only": True,
        "optimizer_updates_during_evaluation": 0,
        "evaluation_seed_start": EVAL_SEED_START,
        "evaluation_episodes": EVAL_EPISODES,
        "ood_profile_family": "structured",
        "ood_intensity": OOD_INTENSITY,
        "ood_schedule": "markov",
        "ood_switch_probability": 1.0,
        "ood_switch_steps": 50 if args.env == "Pendulum-v1" else 200,
        "results": modes,
        "fixed_profile_results": fixed,
        "runtime": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "python": sys.version,
            "device": str(agent.device),
            "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        },
    }
    args.result_dir.mkdir(parents=True, exist_ok=True)
    path = args.result_dir / f"{REVISION}_{args.env}_{args.method}_seed{args.seed}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)
    print(f"v37 evaluation saved to {path}", flush=True)


if __name__ == "__main__":
    main()
