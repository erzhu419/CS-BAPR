"""Paired critic-only replay intervention from one complete warmup checkpoint."""
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

from scripts import test_lqr_convergence as training
from scripts.diagnose_nonlinear_load_v24 import policy_sample
from scripts.train_nonlinear_load_v22 import (
    CANDIDATE, EP_LEN, INPUT_COLUMNS, RADII, REDUCTION, SEVERITIES,
    EntropyFloorAgent, evaluate_final, identify_reference, make_config, train_episode,
)

REVISION = "nonlinear_load_v25_critic_replay"
MODES = ("uniform", "early")
TRAINING_SEEDS = (19, 20)
CRITIC_SEED_OFFSET = 200000
EARLY_STEPS = 50
PREFIX_EPISODES = 5


class CriticReplayAgent(EntropyFloorAgent):
    def __init__(self, config, seed, mode, episode_length=EP_LEN):
        super().__init__(config, seed, episode_length)
        assert mode in MODES and not config.use_context
        self.critic_sampling = mode
        self.critic_rng = np.random.default_rng(CRITIC_SEED_OFFSET + seed)
        self.critic_batches = 0
        self.critic_counts = Counter(uniform=0, reset=0, early=0)
        self.critic_radius_counts = {k: Counter() for k in ("reset", "early")}

    def replacement_indices(self, count, *, rng=None):
        replay, length = self.replay_buffer, self.episode_length
        rng = self.critic_rng if rng is None else rng
        assert replay.position == replay.size < replay.capacity and length >= EARLY_STEPS
        full, partial = divmod(replay.size, length)
        resets = rng.integers(0, full + bool(partial), size=count) * length
        width = EARLY_STEPS - 1
        eligible = full * width + max(0, min(partial, EARLY_STEPS) - 1)
        ranks = rng.integers(0, eligible, size=count)
        early = (ranks // width) * length + ranks % width + 1
        return resets, early

    def _critic_loss_batch(self, batch, next_noise):
        size = batch[0].shape[0]
        self.critic_batches += 1
        if self.critic_sampling == "uniform":
            self.critic_counts["uniform"] += size
            return batch
        assert size % 4 == 0
        half, quarter = size // 2, size // 4
        indices = self.replacement_indices(quarter)
        self.critic_counts.update(uniform=half, reset=quarter, early=quarter)
        for kind, selected in zip(("reset", "early"), indices):
            self.critic_radius_counts[kind].update(str(RADII[int(i) // self.episode_length % 3]) for i in selected)
        rows = [self.replay_buffer.buffer[int(i)] for i in np.concatenate(indices)]
        state, action, reward, next_state, done = [
            torch.as_tensor(np.stack(values), dtype=batch[0].dtype, device=self.device)
            for values in zip(*rows)
        ]
        reward = reward.unsqueeze(1) / max(self._reward_ema_var ** .5, 1e-6)
        with torch.no_grad():
            next_action, next_log_prob = policy_sample(self.actor, next_state, next_noise[half:])
        replacement = (state, action, reward, next_state, done.unsqueeze(1), next_action, next_log_prob)
        return tuple(torch.cat((original[:half], extra), dim=0) for original, extra in zip(batch, replacement))

    def critic_sampler_state(self):
        return {"mode": self.critic_sampling, "rng": self.critic_rng.bit_generator.state,
                "batches": self.critic_batches, "counts": dict(self.critic_counts),
                "radius_counts": {k: dict(v) for k, v in self.critic_radius_counts.items()}}

    def save(self, path, extra_state=None):
        super().save(path, {**extra_state, "critic_sampler": self.critic_sampler_state()})

    def load(self, path):
        extra = super().load(path)
        state = extra["critic_sampler"]
        if state["mode"] != self.critic_sampling:
            raise ValueError("Checkpoint critic sampling differs from the requested arm")
        self.critic_rng.bit_generator.state = state["rng"]
        self.critic_batches = state["batches"]
        self.critic_counts = Counter(state["counts"])
        self.critic_radius_counts = {k: Counter(v) for k, v in state["radius_counts"].items()}
        return extra


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=TRAINING_SEEDS, required=True)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, help="Runner resumes each arm from its own last checkpoint")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    versions = {"torch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
                "pysindy": pysindy.__version__, "sklearn": sklearn.__version__}
    manifest = json.loads((Path(__file__).resolve().parents[2] / "dispatch_manifest.json").read_text())
    assert (manifest["revision"], manifest["training_seeds"], manifest["modes"], manifest["critic_sampling"],
            manifest["shared_prefix_episodes"]) == (REVISION, list(TRAINING_SEEDS), list(MODES),
                {"uniform": [1., 0., 0.], "early": [.5, .25, .25]}, PREFIX_EPISODES)
    for package, version in manifest["runtime_requirement"].items():
        if package != "python_minor_versions":
            assert versions[package] == version
    assert f"{sys.version_info.major}.{sys.version_info.minor}" in manifest["runtime_requirement"]["python_minor_versions"]
    config = make_config("floor", args.device)
    assert {**asdict(config), "device": "cuda"} == manifest["config"]
    episodes, count = (6, 2) if args.preflight_only else (500, 100)
    config.max_episodes = episodes
    assert config.buffer_size > episodes * EP_LEN and config.batch_size == 256
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    if args.resume:
        assert args.resume.exists() and args.resume.parent.resolve() == args.ckpt_dir.resolve()
    env = training.make_env("nonlinear_load", 1, args.seed)
    env.action_space.seed(args.seed)
    assert env.specification() == manifest["environment_settings"]
    common = {"revision": REVISION, "method": CANDIDATE, "training_seed": args.seed, "reference": "cubic-local",
              "entropy_mode": "floor", "jacobian_scope": "full", "jacobian_input_columns": INPUT_COLUMNS,
              "jacobian_reduction": REDUCTION, "sampling": "reset-balanced", "training_episodes": episodes,
              "config": asdict(config), "environment": "nonlinear_load", "environment_settings": env.specification(),
              "training_load_severity": 1, "reset_cycle": list(RADII), "reset_index": "absolute_episode",
              "identification_reset_radius": .2, "reset_sample_fraction": .5,
              "sampler_seed": 100000 + args.seed, "critic_sampler_seed": CRITIC_SEED_OFFSET + args.seed,
              "critic_early_steps": EARLY_STEPS, "shared_prefix_episodes": PREFIX_EPISODES,
              "radii": list(RADII), "severities": list(SEVERITIES), "evaluation_seed_start": 60000,
              "evaluation_episodes_per_severity": count, "preflight_only": args.preflight_only}
    prefix_protocol = {**common, "phase": "shared-prefix", "critic_sampling": "uniform"}
    prefix = str(args.ckpt_dir / "shared_prefix.pt")
    if not Path(prefix).exists():
        training.seed_training(args.seed)
        agent = CriticReplayAgent(config, args.seed, "uniform")
        identify_env = training.make_env("nonlinear_load", 1, args.seed)
        identify_env.action_space.seed(args.seed)
        identify_reference(agent, identify_env, "cubic-local")
        training.seed_training(args.seed)
        history, steps = [], 0
        for episode in range(PREFIX_EPISODES):
            reward, steps = train_episode(env, agent, "balanced", episode, steps)
            history.append(reward)
            agent.record_temperature(episode)
        training.save_training_checkpoint(agent, env, prefix, next_ep=PREFIX_EPISODES,
                                          env_steps=steps, history=history, protocol=prefix_protocol)
        del agent
    outputs = {}
    started = time.time()
    for mode in MODES:
        protocol = {**common, "phase": "arm", "critic_sampling": mode}
        tag = f"{REVISION}_{mode}_seed{args.seed}"
        last = str(args.ckpt_dir / f"{tag}_last.pt")
        agent = CriticReplayAgent(config, args.seed, mode if Path(last).exists() else "uniform")
        start, steps, history = training.restore_training_checkpoint(
            agent, env, last if Path(last).exists() else prefix,
            protocol if Path(last).exists() else prefix_protocol)
        agent.critic_sampling = mode
        print(f"[ARM] {mode} seed={args.seed} resume_episode={start} updates={agent.training_steps}", flush=True)
        for episode in range(start, episodes):
            reward, steps = train_episode(env, agent, "balanced", episode, steps)
            assert np.isfinite(reward)
            history.append(reward)
            agent.record_temperature(episode)
            if (episode + 1) % 20 == 0 or episode + 1 == episodes:
                training.save_training_checkpoint(agent, env, last, next_ep=episode + 1,
                                                  env_steps=steps, history=history, protocol=protocol)
            if (episode + 1) % 25 == 0 or episode + 1 == episodes:
                print(f"ep {episode + 1}/{episodes} arm={mode}: recent_25={np.mean(history[-25:]):.6f} "
                      f"updates={agent.training_steps} t={time.time() - started:.0f}s", flush=True)
        assert (len(history), steps, agent.training_steps) == (episodes, episodes * EP_LEN, episodes * EP_LEN - 999)
        jacobian, critic = agent.sampling_state(), agent.critic_sampler_state()
        assert jacobian["batches"] == critic["batches"] == agent.training_steps
        assert jacobian["reset_samples"] == jacobian["replay_samples"] == agent.training_steps * 128
        updates = agent.training_steps
        replaced = updates - 1 if mode == "early" else 0
        assert critic["counts"] == {"uniform": (updates - replaced) * 256 + replaced * 128,
                                    "reset": replaced * 64, "early": replaced * 64}
        evaluated = evaluate_final(agent, list(range(60000, 60000 + count)))
        assert agent.training_steps == updates and jacobian == agent.sampling_state() and critic == agent.critic_sampler_state()
        outputs[mode] = {**protocol, **evaluated, "checkpoint_selection": "last", "environment_steps": steps,
                         "training_updates": updates, "training_returns": history, "temperature_trace": agent.temperature_trace,
                         "below_nominal_floor_updates": agent.below_nominal_floor_updates,
                         "floor_binding_updates": agent.floor_binding_updates, "jacobian_sampler": jacobian,
                         "critic_sampler": critic, "reset_counts": dict(Counter(str(RADII[e % 3]) for e in range(episodes))),
                         "sindy": agent._sindy_report, "reference_identification": agent.reference_identification,
                         "reference_is_optimal": False, "symbolic_identification_interactions": 10000,
                         "symbolic_validation_interactions": 1000, "optimizer_updates_during_evaluation": 0,
                         "early_steps": 50, "sample_interval": 10,
                         "runtime": {**versions, "python": sys.version, "device": str(agent.device),
                                     "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None}}
        path = args.result_dir / f"{tag}_evaluation.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(outputs[mode], indent=2, allow_nan=False) + "\n")
        os.replace(temporary, path)
        del agent
    assert outputs["uniform"]["reference_identification"] == outputs["early"]["reference_identification"]
    assert outputs["uniform"]["training_returns"][:PREFIX_EPISODES] == outputs["early"]["training_returns"][:PREFIX_EPISODES]
    assert outputs["uniform"]["temperature_trace"][:PREFIX_EPISODES] == outputs["early"]["temperature_trace"][:PREFIX_EPISODES]
    print(f"Paired training complete; both arms shared full prefix state; elapsed={time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
