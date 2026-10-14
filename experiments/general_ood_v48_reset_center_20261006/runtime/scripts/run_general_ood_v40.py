"""Strength-matched Hopper diagnostic with switch-aligned belief measurements."""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import numpy as np
import torch

from scripts import run_general_ood_v38 as v38
from scripts import run_general_ood_v39 as v39

REVISION = "general_ood_v40_strength"
VARIANTS = ("relative-adaptive", "fixed-minus2", "fixed-minus2p3")
BUDGET = 100000
PREFLIGHT_BUDGET = 6000
METRICS = v39.METRICS + ("belief_entropy",)
SWITCH_METRICS = (
    "surprise", "surprise_raw", "surprise_reward_zscore", "surprise_q_std_spike",
    "surprise_reg_relative", "surprise_reward_dominant", "surprise_q_std_dominant",
    "surprise_reg_dominant", "belief_reset_probability", "belief_effective_window",
    "belief_entropy", "weighted_lambda_effective", "effective_beta",
    "alpha", "q_std", "policy_loss", "q_loss", "policy_entropy",
)
PEAK_METRICS = ("surprise", "belief_reset_probability", "weighted_lambda_effective", "effective_beta")


class SwitchDiagnostics:
    """Observe training only; no random draws, policy calls or extra updates."""

    bin_interactions = 50
    pre_interactions = 250
    post_interactions = 500

    def __init__(self):
        self.history = deque(maxlen=self.pre_interactions)
        self.events = []
        self.active = None
        self.post_rows = []

    def _bin(self, rows, event_step):
        updates = [row[3] for row in rows if row[3] is not None]
        return {
            "relative_start": rows[0][0] - event_step,
            "relative_end": rows[-1][0] + 1 - event_step,
            "interactions": len(rows), "updates": len(updates),
            "episode_endings": sum(row[2] for row in rows),
            "reward_mean": float(np.mean([row[1] for row in rows])),
            "means": {key: float(np.mean([m[key] for m in updates]))
                      for key in SWITCH_METRICS} if updates else None,
            "extrema": {key: {"min": min(m[key] for m in updates),
                              "max": max(m[key] for m in updates)}
                        for key in PEAK_METRICS} if updates else None,
        }

    def _flush(self):
        if self.post_rows:
            self.active["bins"].append(self._bin(self.post_rows, self.active["step"]))
            self.post_rows = []

    def on_mode(self, step, previous_mode, mode):
        if step == 0:
            return
        self._flush()
        rows = list(self.history)
        self.active = {
            "step": step, "previous_mode": previous_mode, "mode": mode,
            "actual_change": mode != previous_mode,
            "bins": [self._bin(rows[i:i + self.bin_interactions], step)
                     for i in range(0, len(rows), self.bin_interactions)],
        }
        self.events.append(self.active)

    def observe(self, step, reward, done, metrics):
        selected = {key: float(metrics[key]) for key in SWITCH_METRICS} if metrics is not None else None
        row = (step, float(reward), int(done), selected)
        self.history.append(row)
        if self.active is not None:
            offset = step - self.active["step"]
            if 0 <= offset < self.post_interactions:
                self.post_rows.append(row)
                if len(self.post_rows) == self.bin_interactions or offset + 1 == self.post_interactions:
                    self._flush()
                if offset + 1 == self.post_interactions:
                    self.active = None

    def finish(self):
        self._flush()
        return {
            "coordinate": "Zero-based interaction index; a tick at s applies before transition s (the s+1-th collected transition). Bins are half-open offsets from s; metrics follow its optimizer update.",
            "bin_interactions": self.bin_interactions,
            "pre_interactions": self.pre_interactions, "post_interactions": self.post_interactions,
            "metric_keys": list(SWITCH_METRICS), "events": self.events,
        }


def make_agent(environment, variant, device, env):
    agent = v39.make_agent(environment, "fixed-minus2" if variant == "fixed-minus2" else "relative-adaptive",
                           device, env)
    if variant == "fixed-minus2p3":
        agent.config.penalty_scale = 0.3
        config = agent.config
        agent.belief_tracker = v39.ConstantPenaltyBelief(
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
    budget = PREFLIGHT_BUDGET if args.stage == "preflight" else BUDGET
    training = v38.train(agent, env, args.env, args.seed, budget, args.variant,
                         metric_keys=METRICS, diagnostic_interval=1000,
                         switch_diagnostics=SwitchDiagnostics())
    env.close()
    assert training["interactions"] == budget
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint_dir / f"{REVISION}_{args.env}_{args.variant}_seed{args.seed}_final.pt"
    agent.save(str(checkpoint))
    fresh = gym.make(args.env)
    restored = make_agent(args.env, args.variant, "cuda", fresh)
    fresh.close()
    restored.load(str(checkpoint))
    assert restored.context_tracker._count == budget
    primary_count, fixed_count = (2, 1) if args.stage == "preflight" else (20, 5)
    modes, fixed = v38.evaluate(restored, args.env, primary_count, fixed_count)
    ood = modes["markov_50x"]
    output = {
        "revision": REVISION, "stage": args.stage, "environment": args.env,
        "variant": args.variant, "training_seed": args.seed, "config": asdict(agent.config),
        "surprise_model": "relative-L1-drift", "penalty_mode": args.variant,
        "fixed_beta_after_warmup": {"fixed-minus2": -2.0, "fixed-minus2p3": -2.3}.get(args.variant),
        "checkpoint": checkpoint.name, "checkpoint_selection": "final",
        "loaded_training_context_count": int(restored.context_tracker._count),
        "restore_training_context_before_each_evaluation_block": True,
        "training_budget": budget, "training": training,
        "training_switch_steps": v38.TRAIN_SWITCH_STEPS, "diagnostic_interval": 1000,
        "evaluation_seed_start": v38.EVALUATION_SEED_START,
        "evaluation_episodes": primary_count, "fixed_profile_episodes": fixed_count,
        "ood_intensity": 50.0, "ood_profile_family": "structured",
        "ood_schedule": "markov", "ood_switch_probability": 1.0, "ood_switch_steps": 200,
        "ood_rng_reseeded_each_episode": True,
        "evaluation_only": True, "optimizer_updates_during_evaluation": 0,
        "ood_non_id_transition_count": sum(n for name, n in ood["profile_counts"].items() if name != "id"),
        "ood_transition_count": sum(ood["lengths"]), "results": modes, "fixed_profile_results": fixed,
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
