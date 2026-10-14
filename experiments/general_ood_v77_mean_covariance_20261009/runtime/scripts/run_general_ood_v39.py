"""Hopper seed1 surprise-scale diagnostic; paired arms share one runtime."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import numpy as np
import torch

from csbapr.belief.surprise import RelativeSurpriseComputer
from csbapr.belief.tracker import BeliefTracker
from scripts import run_general_ood_v38 as v38

REVISION = "general_ood_v39_surprise"
VARIANTS = ("raw-adaptive", "relative-adaptive", "fixed-minus2", "fixed-minus3")
BUDGET = 100000
METRICS = v38.METRICS + (
    "surprise_reward_zscore", "surprise_q_std_spike", "surprise_reg_absolute",
    "surprise_reg_relative", "surprise_reg_selected", "surprise_raw",
    "surprise_reward_dominant", "surprise_q_std_dominant", "surprise_reg_dominant",
    "signal_q_std", "critic_reg_norm_current", "critic_reg_norm_target",
    "reward_normalization_std", "policy_entropy", "q_mean",
)


class ConstantPenaltyBelief(BeliefTracker):
    """A delta at age zero yields lambda=1; the existing warmup still applies."""

    def reset(self):
        self.belief[:] = 0.0
        self.belief[0] = 1.0

    def update(self, surprise):
        self.reset()


def make_agent(environment, variant, device, env):
    preset = "fixed-penalty" if variant == "fixed-minus2" else "corrected-belief"
    agent = v38.make_agent(environment, preset, device, env)
    if variant != "raw-adaptive":
        agent.surprise_computer = RelativeSurpriseComputer(ema_alpha=agent.config.surprise_ema)
    if variant == "fixed-minus3":
        config = agent.config
        agent.belief_tracker = ConstantPenaltyBelief(
            config.max_run_length, config.hazard_rate, config.base_variance)
        agent.belief_tracker.reset()
    return agent


def run_worker(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env = gym.make(args.env)
    agent = make_agent(args.env, args.variant, "cuda", env)
    budget = 512 if args.stage == "preflight" else BUDGET
    training = v38.train(agent, env, args.env, args.seed, budget, args.variant,
                         metric_keys=METRICS, diagnostic_interval=1000)
    env.close()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint_dir / f"{REVISION}_{args.env}_{args.variant}_seed{args.seed}_final.pt"
    agent.save(str(checkpoint))
    fresh = gym.make(args.env)
    restored = make_agent(args.env, args.variant, "cuda", fresh)
    fresh.close()
    restored.load(str(checkpoint))
    assert restored.context_tracker._count == budget == training["interactions"]
    primary_count, fixed_count = (2, 1) if args.stage == "preflight" else (20, 5)
    modes, fixed = v38.evaluate(restored, args.env, primary_count, fixed_count)
    ood = modes["markov_50x"]
    output = {
        "revision": REVISION, "stage": args.stage, "environment": args.env,
        "variant": args.variant, "training_seed": args.seed, "config": asdict(agent.config),
        "surprise_model": "absolute-L1-drift" if args.variant == "raw-adaptive" else "relative-L1-drift",
        "penalty_mode": args.variant, "checkpoint": checkpoint.name, "checkpoint_selection": "final",
        "loaded_training_context_count": int(restored.context_tracker._count),
        "restore_training_context_before_each_evaluation_block": True,
        "training_budget": budget, "training": training,
        "training_switch_steps": v38.TRAIN_SWITCH_STEPS, "diagnostic_interval": 1000,
        "evaluation_seed_start": v38.EVALUATION_SEED_START,
        "evaluation_episodes": primary_count, "fixed_profile_episodes": fixed_count,
        "ood_intensity": 50.0, "ood_profile_family": "structured",
        "ood_schedule": "markov", "ood_switch_probability": 1.0, "ood_switch_steps": 200,
        "evaluation_only": True, "optimizer_updates_during_evaluation": 0,
        "ood_non_id_transition_count": sum(n for name, n in ood["profile_counts"].items() if name != "id"),
        "ood_transition_count": sum(ood["lengths"]), "results": modes,
        "fixed_profile_results": fixed,
        "runtime": {"python": sys.version, "torch": torch.__version__,
                    "numpy": np.__version__, "gymnasium": gym.__version__,
                    "mujoco": __import__("mujoco").__version__, "gpu": torch.cuda.get_device_name()},
    }
    args.result_dir.mkdir(parents=True, exist_ok=True)
    path = args.result_dir / f"{REVISION}_{args.env}_{args.variant}_seed{args.seed}.json"
    path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(f"[DONE] {args.env} {args.variant} seed{args.seed}: steps={budget}, result={path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=("Hopper-v4",), default="Hopper-v4")
    parser.add_argument("--seed", type=int, choices=(1,), default=1)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--stage", choices=("preflight", "formal"), default="formal")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.variant:
        run_worker(args)
        return
    common = ["--env", args.env, "--seed", str(args.seed), "--stage", args.stage,
              "--checkpoint-dir", str(args.checkpoint_dir), "--result-dir", str(args.result_dir)]
    print(f"[BUNDLE] {args.env} seed{args.seed}: {','.join(VARIANTS)}", flush=True)
    processes = {v: subprocess.Popen([sys.executable, "-u", __file__, "--variant", v, *common])
                 for v in VARIANTS}
    codes = {v: process.wait() for v, process in processes.items()}
    print(f"[BUNDLE END] {args.env} seed{args.seed}: {codes}", flush=True)
    raise SystemExit(0 if all(code == 0 for code in codes.values()) else 1)


if __name__ == "__main__":
    main()
