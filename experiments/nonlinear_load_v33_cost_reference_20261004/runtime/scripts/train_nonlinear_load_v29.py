"""Paired actor-only coverage intervention; both critics retain v25 early replay."""
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

from scripts import test_lqr_convergence as training
from scripts.diagnose_nonlinear_load_v24 import policy_sample
from scripts.train_nonlinear_load_v25 import (
    CANDIDATE, CRITIC_SEED_OFFSET, EARLY_STEPS, EP_LEN, INPUT_COLUMNS,
    MODES, PREFIX_EPISODES, RADII, REDUCTION, SEVERITIES, TRAINING_SEEDS,
    CriticReplayAgent, evaluate_final, identify_reference, make_config, train_episode,
)

REVISION = "nonlinear_load_v29_actor_replay"
ACTOR_SEED_OFFSET = 300000


class ActorReplayAgent(CriticReplayAgent):
    def __init__(self, config, seed, mode, critic_mode="early", episode_length=EP_LEN):
        super().__init__(config, seed, critic_mode, episode_length)
        assert mode in MODES and not config.use_context and config.beta_bc == 0
        self.actor_sampling = mode
        self.actor_rng = np.random.default_rng(ACTOR_SEED_OFFSET + seed)
        self.actor_batches = 0
        self.actor_counts = Counter(uniform=0, reset=0, early=0)
        self.actor_radius_counts = {k: Counter() for k in ("reset", "early")}

    def _actor_loss_batch(self, states, actions, log_probs, noise):
        size = states.shape[0]
        self.actor_batches += 1
        if self.actor_sampling == "uniform":
            self.actor_counts["uniform"] += size
            return states, actions, log_probs
        assert size % 4 == 0
        half, quarter = size // 2, size // 4
        indices = self.replacement_indices(quarter, rng=self.actor_rng)
        self.actor_counts.update(uniform=half, reset=quarter, early=quarter)
        for kind, selected in zip(("reset", "early"), indices):
            self.actor_radius_counts[kind].update(str(RADII[int(i) // self.episode_length % 3]) for i in selected)
        replacement = torch.as_tensor(np.stack([self.replay_buffer.buffer[int(i)][0]
                                               for i in np.concatenate(indices)]), dtype=states.dtype, device=self.device)
        new_action, new_log_prob = policy_sample(self.actor, replacement, noise[half:])
        return (torch.cat((states[:half], replacement)), torch.cat((actions[:half], new_action)),
                torch.cat((log_probs[:half], new_log_prob)))

    def actor_sampler_state(self):
        return {"mode": self.actor_sampling, "rng": self.actor_rng.bit_generator.state,
                "batches": self.actor_batches, "counts": dict(self.actor_counts),
                "radius_counts": {k: dict(v) for k, v in self.actor_radius_counts.items()}}

    def save(self, path, extra_state=None):
        super().save(path, {**extra_state, "actor_sampler": self.actor_sampler_state()})

    def load(self, path):
        extra = super().load(path)
        state = extra["actor_sampler"]
        if state["mode"] != self.actor_sampling:
            raise ValueError("Checkpoint actor sampling differs from the requested arm")
        self.actor_rng.bit_generator.state = state["rng"]
        self.actor_batches = state["batches"]
        self.actor_counts = Counter(state["counts"])
        self.actor_radius_counts = {k: Counter(v) for k, v in state["radius_counts"].items()}
        return extra


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=TRAINING_SEEDS, required=True)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, help="Resume each arm from its own complete last checkpoint")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    manifest = json.loads((Path(__file__).resolve().parents[2] / "dispatch_manifest.json").read_text())
    assert (manifest["revision"], manifest["training_seeds"], manifest["modes"], manifest["actor_sampling"],
            manifest["shared_prefix_episodes"], manifest["critic_sampling"], manifest["actor_sampler_seed_offset"]) == (
                REVISION, list(TRAINING_SEEDS), list(MODES), {"uniform": [1., 0., 0.], "early": [.5, .25, .25]},
                PREFIX_EPISODES, "early-after-prefix", ACTOR_SEED_OFFSET)
    versions = {"torch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
                "pysindy": pysindy.__version__, "sklearn": sklearn.__version__}
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
              "identification_reset_radius": .2, "reset_sample_fraction": .5, "sampler_seed": 100000 + args.seed,
              "critic_sampler_seed": CRITIC_SEED_OFFSET + args.seed, "critic_early_steps": EARLY_STEPS,
              "actor_sampler_seed": ACTOR_SEED_OFFSET + args.seed, "actor_early_steps": EARLY_STEPS,
              "shared_prefix_episodes": PREFIX_EPISODES, "radii": list(RADII), "severities": list(SEVERITIES),
              "evaluation_seed_start": 60000, "evaluation_episodes_per_severity": count, "preflight_only": args.preflight_only}
    prefix_protocol = {**common, "phase": "shared-prefix", "critic_sampling": "uniform", "actor_sampling": "uniform"}
    prefix = str(args.ckpt_dir / "shared_prefix.pt")
    if not Path(prefix).exists():
        training.seed_training(args.seed)
        agent = ActorReplayAgent(config, args.seed, "uniform", "uniform")
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
    outputs, started = {}, time.time()
    for mode in MODES:
        protocol = {**common, "phase": "arm", "critic_sampling": "early", "actor_sampling": mode}
        tag = f"{REVISION}_{mode}_seed{args.seed}"
        last = str(args.ckpt_dir / f"{tag}_last.pt")
        exists = Path(last).exists()
        agent = ActorReplayAgent(config, args.seed, mode if exists else "uniform", "early" if exists else "uniform")
        start, steps, history = training.restore_training_checkpoint(
            agent, env, last if exists else prefix, protocol if exists else prefix_protocol)
        agent.critic_sampling, agent.actor_sampling = "early", mode
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
        for sampler, replacement in ((critic, updates - 1), (actor, updates - 1 if mode == "early" else 0)):
            assert sampler["counts"] == {"uniform": (updates - replacement) * 256 + replacement * 128,
                                         "reset": replacement * 64, "early": replacement * 64}
        evaluated = evaluate_final(agent, list(range(60000, 60000 + count)))
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
    assert outputs["uniform"]["reference_identification"] == outputs["early"]["reference_identification"]
    for key in ("training_returns", "temperature_trace"):
        assert outputs["uniform"][key][:PREFIX_EPISODES] == outputs["early"][key][:PREFIX_EPISODES]
    print(f"Paired actor replay training complete; full prefix shared; elapsed={time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
