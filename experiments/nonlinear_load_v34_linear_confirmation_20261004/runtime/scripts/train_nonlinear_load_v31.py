"""Matched symbolic Jacobian on/off training in the qualified sustained-load process."""
from collections import Counter
from dataclasses import asdict
import argparse
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

from csbapr.envs.sustained_load_env import SustainedLoadEnv
from scripts import test_lqr_convergence as training
from scripts.train_nonlinear_load_v29 import (
    ACTOR_SEED_OFFSET, CRITIC_SEED_OFFSET, EARLY_STEPS, EP_LEN, INPUT_COLUMNS,
    PREFIX_EPISODES, RADII, REDUCTION, SEVERITIES, ActorReplayAgent,
    evaluate_final, identify_reference, make_config as candidate_config, train_episode,
)

REVISION = "nonlinear_load_v31_symbolic"
STUDY = "nonlinear_load_v31_symbolic_20261004"
TRAINING_SEEDS = (21, 22)
MODES = ("off", "on")
JAC_WEIGHTS = {"off": 0., "on": .01}
EVALUATION_SEED_START = 62000
METHOD = "centered-mlp-symbolic-jacobian-ablation"


def make_config(mode, device):
    config = candidate_config("floor", device)
    config.env_name = "SustainedNonlinearLoad-v0"
    config.jac_weight = JAC_WEIGHTS[mode]
    return config


def make_env(severity, seed, radius=.2):
    env = SustainedLoadEnv(severity, seed)
    env.amplitude = radius
    return env


class SymbolicConstraintAgent(ActorReplayAgent):
    def __init__(self, config, seed, mode, critic_mode="early", episode_length=EP_LEN):
        assert config.jac_weight == JAC_WEIGHTS[mode]
        super().__init__(config, seed, "uniform", critic_mode, episode_length)
        self.symbolic_constraint = mode

    def _actor_loss_batch(self, states, actions, log_probs, noise):
        # The off arm advances the same private sampler, without a Jacobian loss.
        if self.symbolic_constraint == "off":
            self._jacobian_loss_states(states)
        return super()._actor_loss_batch(states, actions, log_probs, noise)

    def set_constraint(self, mode):
        self.symbolic_constraint = mode
        self.config.jac_weight = JAC_WEIGHTS[mode]

    def save(self, path, extra_state=None):
        super().save(path, {**(extra_state or {}), "symbolic_constraint": self.symbolic_constraint})

    def load(self, path):
        extra = super().load(path)
        if extra["symbolic_constraint"] != self.symbolic_constraint:
            raise ValueError("Checkpoint symbolic constraint differs from the requested arm")
        return extra


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=TRAINING_SEEDS, required=True)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, help="Resume each arm from its own full last checkpoint")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    manifest = json.loads((Path(__file__).resolve().parents[2] / "dispatch_manifest.json").read_text())
    assert (manifest["revision"], manifest["training_seeds"], manifest["modes"],
            manifest["jacobian_weights"], manifest["shared_prefix_episodes"],
            manifest["critic_sampling"], manifest["actor_sampling"], manifest["evaluation_seed_start"]) == (
        REVISION, list(TRAINING_SEEDS), list(MODES), JAC_WEIGHTS, PREFIX_EPISODES,
        "early-after-prefix", "uniform", EVALUATION_SEED_START)
    versions = {"torch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
                "pysindy": pysindy.__version__, "sklearn": sklearn.__version__}
    for package, version in manifest["runtime_requirement"].items():
        if package != "python_minor_versions":
            assert versions[package] == version
    assert f"{sys.version_info.major}.{sys.version_info.minor}" in manifest["runtime_requirement"]["python_minor_versions"]
    episodes, count = (6, 2) if args.preflight_only else (500, 100)
    configs = {mode: make_config(mode, args.device) for mode in MODES}
    for mode, config in configs.items():
        assert {**asdict(config), "device": "cuda"} == manifest["configs"][mode]
        config.max_episodes = episodes
        assert config.buffer_size > episodes * EP_LEN and config.batch_size == 256
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    if args.resume:
        assert args.resume.exists() and args.resume.parent.resolve() == args.ckpt_dir.resolve()
    env = make_env(1, args.seed)
    env.action_space.seed(args.seed)
    assert env.specification() == manifest["environment_settings"]
    common = {"revision": REVISION, "method": METHOD, "training_seed": args.seed,
              "reference": "cubic-local", "entropy_mode": "floor", "jacobian_scope": "full",
              "jacobian_input_columns": INPUT_COLUMNS, "jacobian_reduction": REDUCTION,
              "sampling": "reset-balanced", "training_episodes": episodes,
              "environment": "sustained_load", "environment_settings": env.specification(),
              "training_load_severity": 1, "reset_cycle": list(RADII), "reset_index": "absolute_episode",
              "identification_reset_radius": .2, "reset_sample_fraction": .5,
              "sampler_seed": 100000 + args.seed, "critic_sampler_seed": CRITIC_SEED_OFFSET + args.seed,
              "critic_early_steps": EARLY_STEPS, "actor_sampler_seed": ACTOR_SEED_OFFSET + args.seed,
              "actor_sampling": "uniform", "shared_prefix_episodes": PREFIX_EPISODES,
              "radii": list(RADII), "severities": list(SEVERITIES),
              "evaluation_seed_start": EVALUATION_SEED_START, "evaluation_episodes_per_severity": count,
              "preflight_only": args.preflight_only}
    prefix_protocol = {**common, "phase": "shared-prefix", "critic_sampling": "uniform",
                       "symbolic_constraint": "off", "config": asdict(configs["off"])}
    prefix = str(args.ckpt_dir / "shared_prefix.pt")
    if not Path(prefix).exists():
        training.seed_training(args.seed)
        agent = SymbolicConstraintAgent(configs["off"], args.seed, "off", "uniform")
        identify_env = make_env(1, args.seed)
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
    outputs, started = {}, time.time()
    for mode in MODES:
        protocol = {**common, "phase": "arm", "critic_sampling": "early",
                    "symbolic_constraint": mode, "config": asdict(configs[mode])}
        tag = f"{REVISION}_{mode}_seed{args.seed}"
        last = str(args.ckpt_dir / f"{tag}_last.pt")
        exists = Path(last).exists()
        agent = SymbolicConstraintAgent(make_config(mode if exists else "off", args.device), args.seed,
                                        mode if exists else "off", "early" if exists else "uniform")
        agent.config.max_episodes = episodes
        start, steps, history = training.restore_training_checkpoint(
            agent, env, last if exists else prefix, protocol if exists else prefix_protocol)
        agent.critic_sampling = "early"
        agent.set_constraint(mode)
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
        jacobian, critic, actor = agent.sampling_state(), agent.critic_sampler_state(), agent.actor_sampler_state()
        updates = agent.training_steps
        assert jacobian["batches"] == critic["batches"] == actor["batches"] == updates
        assert jacobian["reset_samples"] == jacobian["replay_samples"] == updates * 128
        assert critic["counts"] == {"uniform": 256 + (updates - 1) * 128,
                                    "reset": (updates - 1) * 64, "early": (updates - 1) * 64}
        assert actor["counts"] == {"uniform": updates * 256, "reset": 0, "early": 0}
        evaluated = evaluate_final(agent, list(range(EVALUATION_SEED_START, EVALUATION_SEED_START + count)),
                                   env_factory=make_env)
        assert agent.training_steps == updates and jacobian == agent.sampling_state()
        assert critic == agent.critic_sampler_state() and actor == agent.actor_sampler_state()
        outputs[mode] = {**protocol, **evaluated, "checkpoint_selection": "last", "environment_steps": steps,
                         "training_updates": updates, "training_returns": history, "temperature_trace": agent.temperature_trace,
                         "below_nominal_floor_updates": agent.below_nominal_floor_updates,
                         "floor_binding_updates": agent.floor_binding_updates, "jacobian_sampler": jacobian,
                         "critic_sampler": critic, "actor_sampler": actor,
                         "reset_counts": dict(Counter(str(RADII[e % 3]) for e in range(episodes))),
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
    assert outputs["off"]["reference_identification"] == outputs["on"]["reference_identification"]
    for key in ("training_returns", "temperature_trace"):
        assert outputs["off"][key][:PREFIX_EPISODES] == outputs["on"][key][:PREFIX_EPISODES]
    print(f"Paired symbolic constraint training complete; full prefix shared; elapsed={time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
