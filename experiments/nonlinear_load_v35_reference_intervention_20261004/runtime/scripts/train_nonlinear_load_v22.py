"""Matched entropy-floor intervention with full Jacobian and cubic feedback fixed."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pysindy
import scipy
import sklearn
import torch

from csbapr.envs.lqr_env import EP_LEN
from scripts import test_lqr_convergence as training
from scripts.train_nonlinear_load_v15 import CANDIDATE, RADII, SEVERITIES, evaluate_final, train_episode
from scripts.train_nonlinear_load_v17 import SAMPLER_SEED_OFFSET
from scripts.train_nonlinear_load_v19 import identify_reference
from scripts.train_nonlinear_load_v20 import REDUCTION, ScopedReferenceAgent

REVISION = "nonlinear_load_v22_entropy_floor"
MODES = ("floor", "no-floor")
TRAINING_SEEDS = (17, 18)
INPUT_COLUMNS = list(range(6))


def make_config(mode, device):
    assert mode in MODES
    config = training.make_config(CANDIDATE, device, "nonlinear_load")
    config.max_episodes = 500
    config.enable_entropy_floor = mode == "floor"
    assert config.auto_alpha and config.entropy_floor == .01
    return config


class EntropyFloorAgent(ScopedReferenceAgent):
    def __init__(self, config, seed, episode_length=EP_LEN):
        super().__init__(config, seed, "full", episode_length)
        self.below_nominal_floor_updates = 0
        self.floor_binding_updates = 0
        self.temperature_trace = []

    def update(self):
        before = self.training_steps
        metrics = super().update()
        if self.training_steps > before:
            raw = self.log_alpha.detach().exp().item()
            self.below_nominal_floor_updates += int(raw < self.config.entropy_floor)
            self.floor_binding_updates += int(self.config.enable_entropy_floor and raw < self.config.entropy_floor)
        return metrics

    def record_temperature(self, episode):
        self.temperature_trace.append({
            "episode": episode + 1, "training_updates": self.training_steps,
            "alpha": self.alpha, "raw_alpha": self.log_alpha.detach().exp().item(),
            "reward_std": getattr(self, "_reward_ema_var", 1.) ** .5,
            "below_nominal_floor_updates": self.below_nominal_floor_updates,
            "floor_binding_updates": self.floor_binding_updates})

    def save(self, path, extra_state=None):
        super().save(path, {**extra_state, "entropy_enabled": self.config.enable_entropy_floor,
                           "temperature_trace": self.temperature_trace,
                           "below_nominal_floor_updates": self.below_nominal_floor_updates,
                           "floor_binding_updates": self.floor_binding_updates})

    def load(self, path):
        extra = super().load(path)
        if extra["entropy_enabled"] != self.config.enable_entropy_floor:
            raise ValueError("Checkpoint entropy-floor setting differs from the requested arm")
        self.temperature_trace = extra["temperature_trace"]
        self.below_nominal_floor_updates = extra["below_nominal_floor_updates"]
        self.floor_binding_updates = extra["floor_binding_updates"]
        return extra


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--seed", type=int, choices=TRAINING_SEEDS, required=True)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--resume")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    versions = {"torch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
                "pysindy": pysindy.__version__, "sklearn": sklearn.__version__}
    manifest = json.loads((Path(__file__).resolve().parents[2] / "dispatch_manifest.json").read_text())
    assert (manifest["revision"], manifest["training_seeds"], manifest["modes"], manifest["reference"],
            manifest["jacobian_input_columns"], manifest["jacobian_reduction"]) == (
        REVISION, list(TRAINING_SEEDS), list(MODES), "cubic-local", INPUT_COLUMNS, REDUCTION)
    assert (manifest["radii"], manifest["severities"], manifest["reset_cycle"],
            manifest["evaluation_seed_start"], manifest["evaluation_episodes_per_severity"],
            manifest["sampler_seed_offset"], manifest["reset_sample_fraction"]) == (
        list(RADII), list(SEVERITIES), list(RADII), 60000, 100, SAMPLER_SEED_OFFSET, .5)
    for package, version in manifest["runtime_requirement"].items():
        if package != "python_minor_versions":
            assert versions[package] == version
    assert f"{sys.version_info.major}.{sys.version_info.minor}" in manifest["runtime_requirement"]["python_minor_versions"]
    training.seed_training(args.seed)
    config = make_config(args.mode, args.device)
    assert {**asdict(config), "device": "cuda"} == manifest["configs"][args.mode]
    assert config.buffer_size > 500 * EP_LEN and config.batch_size == 256
    episodes, count = (6, 2) if args.preflight_only else (500, 100)
    config.max_episodes = episodes
    agent = EntropyFloorAgent(config, args.seed)
    env = training.make_env("nonlinear_load", 1, args.seed)
    env.action_space.seed(args.seed)
    assert env.specification() == manifest["environment_settings"]
    protocol = {"revision": REVISION, "method": CANDIDATE, "training_seed": args.seed,
                "reference": "cubic-local", "entropy_mode": args.mode, "jacobian_scope": "full",
                "jacobian_input_columns": INPUT_COLUMNS, "jacobian_reduction": REDUCTION,
                "sampling": "reset-balanced", "training_episodes": episodes, "config": asdict(config),
                "environment": "nonlinear_load", "environment_settings": env.specification(),
                "training_load_severity": 1, "reset_cycle": list(RADII), "reset_index": "absolute_episode",
                "identification_reset_radius": .2, "reset_sample_fraction": .5,
                "sampler_seed": SAMPLER_SEED_OFFSET + args.seed,
                "radii": list(RADII), "severities": list(SEVERITIES),
                "evaluation_seed_start": 60000, "evaluation_episodes_per_severity": count,
                "preflight_only": args.preflight_only}
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{REVISION}_{args.mode}_seed{args.seed}"
    last = str(args.ckpt_dir / f"{tag}_last.pt")
    resume = args.resume or (last if Path(last).exists() else None)
    if resume:
        start_ep, env_steps, history = training.restore_training_checkpoint(agent, env, resume, protocol)
        print(f"[RESUME] episode={start_ep} steps={env_steps} entropy_mode={args.mode}", flush=True)
    else:
        identify_env = training.make_env("nonlinear_load", 1, args.seed)
        identify_env.action_space.seed(args.seed)
        identify_reference(agent, identify_env, "cubic-local")
        training.seed_training(args.seed)
        start_ep, env_steps, history = 0, 0, []
        training.save_training_checkpoint(agent, env, last, next_ep=0, env_steps=0, history=history, protocol=protocol)
    assert agent.reference_identification["selected_reference"] == "cubic-local"
    print(f"Training: {CANDIDATE} seed={args.seed} Jacobian entropy_mode={args.mode}", flush=True)
    started = time.time()
    for episode in range(start_ep, episodes):
        reward, env_steps = train_episode(env, agent, "balanced", episode, env_steps)
        assert np.isfinite(reward)
        history.append(reward)
        agent.record_temperature(episode)
        if (episode + 1) % 20 == 0 or episode + 1 == episodes or (args.preflight_only and episode == 4):
            training.save_training_checkpoint(agent, env, last, next_ep=episode + 1, env_steps=env_steps,
                                              history=history, protocol=protocol)
        if args.preflight_only and episode == 4:
            agent = EntropyFloorAgent(config, args.seed)
            restored_ep, restored_steps, restored_history = training.restore_training_checkpoint(agent, env, last, protocol)
            assert (restored_ep, restored_steps, restored_history) == (5, env_steps, history)
            print(f"[PREFLIGHT RESUME] episode5 steps={env_steps}; entropy, reference and sampler restored", flush=True)
        if (episode + 1) % 25 == 0 or episode + 1 == episodes:
            print(f"ep {episode + 1}/{episodes}: recent_25={np.mean(history[-25:]):.6f} "
                  f"updates={agent.training_steps} t={time.time() - started:.0f}s "
                  f"metrics={getattr(agent, '_last_update_metrics', None)}", flush=True)
    assert (len(history), env_steps, agent.training_steps) == (episodes, episodes * EP_LEN, episodes * EP_LEN - 999)
    assert len(agent.temperature_trace) == episodes
    sampler = agent.sampling_state()
    assert sampler["batches"] == agent.training_steps
    assert sampler["reset_samples"] == sampler["replay_samples"] == agent.training_steps * 128
    assert sum(sampler["radius_samples"].values()) == sampler["reset_samples"]
    assert (agent.reference_identification["identification_interactions"],
            agent.reference_identification["validation_interactions"]) == (10000, 1000)
    updates_before_eval = agent.training_steps
    evaluated = evaluate_final(agent, list(range(60000, 60000 + count)))
    assert agent.training_steps == updates_before_eval and sampler == agent.sampling_state()
    assert evaluated["feedback_gains"]["fitted"] == agent.reference_identification["gains"]["cubic-local"]
    output = {**protocol, **evaluated, "checkpoint_selection": "last", "environment_steps": env_steps,
              "training_updates": agent.training_steps, "training_returns": history,
              "temperature_trace": agent.temperature_trace,
              "below_nominal_floor_updates": agent.below_nominal_floor_updates,
              "floor_binding_updates": agent.floor_binding_updates,
              "reset_counts": dict(Counter(str(RADII[e % len(RADII)]) for e in range(episodes))),
              "jacobian_sampler": sampler, "sindy": agent._sindy_report,
              "reference_identification": agent.reference_identification, "reference_is_optimal": False,
              "symbolic_identification_interactions": 10000, "symbolic_validation_interactions": 1000,
              "optimizer_updates_during_evaluation": 0, "early_steps": 50, "sample_interval": 10,
              "runtime": {**versions, "python": sys.version, "device": str(agent.device),
                          "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None}}
    path = args.result_dir / f"{tag}_evaluation.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)
    print(f"Training complete: {path}; elapsed={time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
