"""Matched full versus load-input Jacobian constraints with a fixed cubic reference."""
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
from csbapr.losses.jacobian import compute_jacobian_loss
from scripts import test_lqr_convergence as training
from scripts.train_nonlinear_load_v15 import CANDIDATE, RADII, SEVERITIES, evaluate_final, train_episode
from scripts.train_nonlinear_load_v17 import SAMPLER_SEED_OFFSET
from scripts.train_nonlinear_load_v19 import ReferenceAgent, identify_reference

REVISION = "nonlinear_load_v20_jacobian_scope"
SCOPES = ("full", "load-only")
TRAINING_SEEDS = (15, 16)
INPUT_COLUMNS = {"full": list(range(6)), "load-only": [4, 5]}
REDUCTION = "mean_of_per_state_sum_squared_retained_entries"


class ScopedReferenceAgent(ReferenceAgent):
    def __init__(self, config, seed, scope, episode_length=EP_LEN):
        assert scope in SCOPES
        super().__init__(config, seed, episode_length)
        self.jacobian_scope = scope

    def _compute_jacobian_loss(self, states):
        columns = None if self.jacobian_scope == "full" else INPUT_COLUMNS[self.jacobian_scope]
        return compute_jacobian_loss(self.actor, self.pi_sym_torch, states, input_columns=columns)

    def save(self, path, extra_state=None):
        assert extra_state is not None
        super().save(path, {**extra_state, "jacobian_scope": self.jacobian_scope})

    def load(self, path):
        extra = super().load(path)
        if extra["jacobian_scope"] != self.jacobian_scope:
            raise ValueError("Checkpoint Jacobian scope differs from the requested scope")
        return extra


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=SCOPES, required=True)
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
    assert (manifest["revision"], manifest["training_seeds"], manifest["scopes"], manifest["reference"],
            manifest["jacobian_input_columns"], manifest["jacobian_reduction"]) == (
        REVISION, list(TRAINING_SEEDS), list(SCOPES), "cubic-local", INPUT_COLUMNS, REDUCTION)
    assert (manifest["radii"], manifest["severities"], manifest["reset_cycle"],
            manifest["evaluation_seed_start"], manifest["evaluation_episodes_per_severity"],
            manifest["sampler_seed_offset"], manifest["reset_sample_fraction"]) == (
        list(RADII), list(SEVERITIES), list(RADII), 60000, 100, SAMPLER_SEED_OFFSET, .5)
    for package, version in manifest["runtime_requirement"].items():
        if package != "python_minor_versions":
            assert versions[package] == version
    assert f"{sys.version_info.major}.{sys.version_info.minor}" in manifest["runtime_requirement"]["python_minor_versions"]
    training.seed_training(args.seed)
    config = training.make_config(CANDIDATE, args.device, "nonlinear_load")
    config.max_episodes = 500
    assert {**asdict(config), "device": "cuda"} == manifest["config"]
    assert config.buffer_size > 500 * EP_LEN and config.batch_size == 256
    episodes, count = (6, 2) if args.preflight_only else (500, 100)
    config.max_episodes = episodes
    agent = ScopedReferenceAgent(config, args.seed, args.scope)
    env = training.make_env("nonlinear_load", 1, args.seed)
    env.action_space.seed(args.seed)
    assert env.specification() == manifest["environment_settings"]
    protocol = {"revision": REVISION, "method": CANDIDATE, "training_seed": args.seed,
                "reference": "cubic-local", "jacobian_scope": args.scope,
                "jacobian_input_columns": INPUT_COLUMNS[args.scope], "jacobian_reduction": REDUCTION,
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
    tag = f"{REVISION}_{args.scope}_seed{args.seed}"
    last = str(args.ckpt_dir / f"{tag}_last.pt")
    resume = args.resume or (last if Path(last).exists() else None)
    if resume:
        start_ep, env_steps, history = training.restore_training_checkpoint(agent, env, resume, protocol)
        print(f"[RESUME] episode={start_ep} steps={env_steps} scope={args.scope}", flush=True)
    else:
        identify_env = training.make_env("nonlinear_load", 1, args.seed)
        identify_env.action_space.seed(args.seed)
        identify_reference(agent, identify_env, "cubic-local")
        training.seed_training(args.seed)
        start_ep, env_steps, history = 0, 0, []
        training.save_training_checkpoint(agent, env, last, next_ep=0, env_steps=0, history=history, protocol=protocol)
    assert agent.reference_identification["selected_reference"] == "cubic-local"
    print(f"Training: {CANDIDATE} seed={args.seed} Jacobian scope={args.scope}", flush=True)
    started = time.time()
    for episode in range(start_ep, episodes):
        reward, env_steps = train_episode(env, agent, "balanced", episode, env_steps)
        assert np.isfinite(reward)
        history.append(reward)
        if (episode + 1) % 20 == 0 or episode + 1 == episodes or (args.preflight_only and episode == 4):
            training.save_training_checkpoint(agent, env, last, next_ep=episode + 1, env_steps=env_steps,
                                              history=history, protocol=protocol)
        if args.preflight_only and episode == 4:
            agent = ScopedReferenceAgent(config, args.seed, args.scope)
            restored_ep, restored_steps, restored_history = training.restore_training_checkpoint(agent, env, last, protocol)
            assert (restored_ep, restored_steps, restored_history) == (5, env_steps, history)
            print(f"[PREFLIGHT RESUME] episode5 steps={env_steps}; scope, reference and sampler restored", flush=True)
        if (episode + 1) % 25 == 0 or episode + 1 == episodes:
            print(f"ep {episode + 1}/{episodes}: recent_25={np.mean(history[-25:]):.6f} "
                  f"updates={agent.training_steps} t={time.time() - started:.0f}s "
                  f"metrics={getattr(agent, '_last_update_metrics', None)}", flush=True)
    assert (len(history), env_steps, agent.training_steps) == (episodes, episodes * EP_LEN, episodes * EP_LEN - 999)
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
