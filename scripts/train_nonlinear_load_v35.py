"""Paired learner intervention changing only the action-valued reference."""
from collections import Counter
from copy import deepcopy
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

from csbapr.sindy.feedback import LQRFeedbackReference
from scripts import test_lqr_convergence as training
from scripts.diagnose_nonlinear_load_v33 import CostReference
from scripts.train_nonlinear_load_v31 import (
    ACTOR_SEED_OFFSET, CRITIC_SEED_OFFSET, EARLY_STEPS, EP_LEN, INPUT_COLUMNS,
    PREFIX_EPISODES, RADII, REDUCTION, SEVERITIES, TRAINING_SEEDS,
    SymbolicConstraintAgent, evaluate_final, identify_reference, make_config, make_env, train_episode,
)

REVISION = "nonlinear_load_v35_reference_intervention"
STUDY = "nonlinear_load_v35_reference_intervention_20261004"
METHOD = "centered-mlp-reference-intervention"
REFERENCES = ("cubic-local", "cost-linear")
MAPPINGS = {"cubic-local": "tanh", "cost-linear": "clip"}
EVALUATION_SEED_START = 66000
PREFLIGHT_SEED_START = 66500


class ReferenceInterventionAgent(SymbolicConstraintAgent):
    def __init__(self, config, seed, reference="cubic-local", constraint="on", critic_mode="early"):
        super().__init__(config, seed, constraint, critic_mode)
        self.reference_arm = reference

    def set_reference(self, reference):
        self.reference_arm = reference
        identification = self.reference_identification
        identification["selected_reference"] = reference
        self.pi_sym_torch = LQRFeedbackReference(
            identification["gains"][reference], MAPPINGS[reference]
        ).to(self.device).eval().requires_grad_(False)

    def save(self, path, extra_state=None):
        super().save(path, {**(extra_state or {}), "reference_arm": self.reference_arm})

    def load(self, path):
        extra = super().load(path)
        if extra["reference_arm"] != self.reference_arm:
            raise ValueError("Checkpoint reference differs from the requested arm")
        assert self.pi_sym_torch.action_mapping == MAPPINGS[self.reference_arm]
        return extra


def prepare_references(agent, source):
    fit = agent.reference_identification["fits"]["cubic"]
    expected = source["reference_identification"]["fits"]["cubic"]
    for key in expected:
        np.testing.assert_allclose(fit[key], expected[key], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(CostReference(fit, source["environment_settings"]).gain,
                               source["source_candidate_gain"], rtol=1e-6, atol=1e-8)
    # Use the confirmed gain itself, not a newly selected/tuned controller.
    agent.reference_identification["gains"]["cost-linear"] = (
        torch.as_tensor(source["source_candidate_gain"], dtype=torch.float32).tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=TRAINING_SEEDS, required=True)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    study = Path(__file__).resolve().parents[2]
    manifest = json.loads((study / "dispatch_manifest.json").read_text())
    source = json.loads((study / f"inputs/source_seed{args.seed}.json").read_text())
    assert (manifest["revision"], manifest["references"], manifest["training_seeds"],
            manifest["evaluation_seed_start"], manifest["preflight_seed_start"]) == (
        REVISION, list(REFERENCES), list(TRAINING_SEEDS), EVALUATION_SEED_START, PREFLIGHT_SEED_START)
    assert manifest["source_decision"]["reference_intervention_admitted"]
    assert source["training_seed"] == args.seed
    versions = {"torch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
                "pysindy": pysindy.__version__, "sklearn": sklearn.__version__}
    for package, version in manifest["runtime_requirement"].items():
        if package != "python_minor_versions":
            assert versions[package] == version
    assert f"{sys.version_info.major}.{sys.version_info.minor}" in manifest["runtime_requirement"]["python_minor_versions"]
    episodes, count = (6, 2) if args.preflight_only else (500, 100)
    seed_start = PREFLIGHT_SEED_START if args.preflight_only else EVALUATION_SEED_START
    config = make_config("on", args.device)
    assert {**asdict(config), "device": "cuda"} == manifest["config"]
    config.max_episodes = episodes
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    if args.resume:
        assert args.resume.exists() and args.resume.parent.resolve() == args.ckpt_dir.resolve()
    env = make_env(1, args.seed)
    env.action_space.seed(args.seed)
    assert env.specification() == source["environment_settings"] == manifest["environment_settings"]
    common = {"revision": REVISION, "method": METHOD, "training_seed": args.seed,
              "entropy_mode": "floor", "jacobian_scope": "full", "jacobian_input_columns": INPUT_COLUMNS,
              "jacobian_reduction": REDUCTION, "sampling": "reset-balanced", "training_episodes": episodes,
              "environment": "sustained_load", "environment_settings": env.specification(),
              "training_load_severity": 1, "reset_cycle": list(RADII), "reset_index": "absolute_episode",
              "identification_reset_radius": .2, "reset_sample_fraction": .5,
              "sampler_seed": 100000 + args.seed, "critic_sampler_seed": CRITIC_SEED_OFFSET + args.seed,
              "critic_early_steps": EARLY_STEPS, "actor_sampler_seed": ACTOR_SEED_OFFSET + args.seed,
              "actor_sampling": "uniform", "shared_prefix_episodes": PREFIX_EPISODES,
              "radii": list(RADII), "severities": list(SEVERITIES), "evaluation_seed_start": seed_start,
              "evaluation_episodes_per_severity": count, "preflight_only": args.preflight_only}
    prefix_config = make_config("off", args.device)
    prefix_config.max_episodes = episodes
    prefix_protocol = {**common, "phase": "shared-prefix", "critic_sampling": "uniform",
                       "reference": "cubic-local", "reference_mapping": "tanh", "symbolic_constraint": "off",
                       "config": asdict(prefix_config)}
    prefix = str(args.ckpt_dir / "shared_prefix.pt")
    if not Path(prefix).exists():
        training.seed_training(args.seed)
        agent = ReferenceInterventionAgent(deepcopy(prefix_config), args.seed, constraint="off", critic_mode="uniform")
        identify_env = make_env(1, args.seed)
        identify_env.action_space.seed(args.seed)
        identify_reference(agent, identify_env, "cubic-local")
        prepare_references(agent, source)
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
    for reference in REFERENCES:
        protocol = {**common, "phase": "arm", "critic_sampling": "early", "symbolic_constraint": "on",
                    "reference": reference, "reference_mapping": MAPPINGS[reference], "config": asdict(config)}
        tag = f"{REVISION}_{reference}_seed{args.seed}"
        last = str(args.ckpt_dir / f"{tag}_last.pt")
        exists = Path(last).exists()
        agent = ReferenceInterventionAgent(deepcopy(config if exists else prefix_config), args.seed,
                                           reference if exists else "cubic-local",
                                           "on" if exists else "off", "early" if exists else "uniform")
        start, steps, history = training.restore_training_checkpoint(
            agent, env, last if exists else prefix, protocol if exists else prefix_protocol)
        agent.critic_sampling = "early"
        agent.set_constraint("on")
        agent.set_reference(reference)
        print(f"[ARM] {reference} seed={args.seed} resume_episode={start} updates={agent.training_steps}", flush=True)
        for episode in range(start, episodes):
            reward, steps = train_episode(env, agent, "balanced", episode, steps)
            assert np.isfinite(reward)
            history.append(reward)
            agent.record_temperature(episode)
            if (episode + 1) % 20 == 0 or episode + 1 == episodes:
                training.save_training_checkpoint(agent, env, last, next_ep=episode + 1,
                                                  env_steps=steps, history=history, protocol=protocol)
            if (episode + 1) % 25 == 0 or episode + 1 == episodes:
                elapsed = time.time() - started
                print(f"ep {episode + 1}/{episodes} arm={reference}: recent_25={np.mean(history[-25:]):.6f} "
                      f"updates={agent.training_steps} t={elapsed:.0f}s", flush=True)
        assert (len(history), steps, agent.training_steps) == (episodes, episodes * EP_LEN, episodes * EP_LEN - 999)
        jacobian, critic, actor = agent.sampling_state(), agent.critic_sampler_state(), agent.actor_sampler_state()
        updates = agent.training_steps
        assert jacobian["batches"] == critic["batches"] == actor["batches"] == updates
        if args.preflight_only:
            restored = ReferenceInterventionAgent(config, args.seed, reference)
            restored_start, restored_steps, restored_history = training.restore_training_checkpoint(
                restored, env, last, protocol)
            assert (restored_start, restored_steps, restored_history) == (episodes, steps, history)
            assert restored.sampling_state() == jacobian and restored.critic_sampler_state() == critic
            assert restored.actor_sampler_state() == actor
            for a, b in zip(agent.actor.parameters(), restored.actor.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            del agent
            agent = restored
        evaluated = evaluate_final(agent, list(range(seed_start, seed_start + count)), env_factory=make_env)
        assert agent.training_steps == updates and jacobian == agent.sampling_state()
        assert critic == agent.critic_sampler_state() and actor == agent.actor_sampler_state()
        outputs[reference] = {**protocol, **evaluated, "checkpoint_selection": "last", "environment_steps": steps,
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
        temporary.write_text(json.dumps(outputs[reference], indent=2, allow_nan=False) + "\n")
        os.replace(temporary, path)
        del agent
    for key in ("training_returns", "temperature_trace"):
        assert outputs[REFERENCES[0]][key][:PREFIX_EPISODES] == outputs[REFERENCES[1]][key][:PREFIX_EPISODES]
    print(f"Paired reference intervention complete; full prefix shared; elapsed={time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
