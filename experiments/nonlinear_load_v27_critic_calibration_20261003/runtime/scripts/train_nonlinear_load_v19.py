"""Matched actor training with only the fitted feedback reference changed."""
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
from csbapr.sindy.data_collector import collect_trajectories
from csbapr.sindy.feedback import LQRFeedbackReference
from csbapr.sindy.load_feedback import fitted_load_reference
from csbapr.sindy.torch_wrapper import SINDyTorchWrapper
from scripts import test_lqr_convergence as training
from scripts.diagnose_nonlinear_load_v18 import fit_models, fit_report, local_gain, physical_metrics
from scripts.train_nonlinear_load_v15 import CANDIDATE, RADII, SEVERITIES, evaluate_final, train_episode
from scripts.train_nonlinear_load_v17 import ResetBalancedJacobianAgent, SAMPLER_SEED_OFFSET

REVISION = "nonlinear_load_v19_reference"
REFERENCES = ("linear", "cubic-local")
TRAINING_SEEDS = (13, 14)


class ReferenceAgent(ResetBalancedJacobianAgent):
    def __init__(self, config, seed, episode_length=EP_LEN):
        super().__init__(6, 2, config, seed, episode_length)
        self.reference_identification = None

    def save(self, path, extra_state=None):
        assert self.reference_identification is not None and extra_state is not None
        super().save(path, {**extra_state, "reference_identification": self.reference_identification})

    def load(self, path):
        extra = super().load(path)
        self.reference_identification = extra["reference_identification"]
        return extra


def identify_reference(agent, env, reference):
    assert reference in REFERENCES
    config = agent.config
    assert (config.sindy_lib_degree, config.sindy_threshold, config.sindy_discrete_time,
            config.sindy_with_control, config.weight_sym, config.weight_reg,
            config.action_anchor_weight) == (1, .001, True, True, 0., 0., 0.)
    states, actions, _ = collect_trajectories(
        env, n_episodes=config.sindy_n_explore_episodes, max_steps=config.max_steps_per_episode)
    models = fit_models(states, actions)
    original = fitted_load_reference(models["linear"])
    corrected = LQRFeedbackReference(local_gain(models["cubic"]))
    references = {"linear": original, "cubic-local": corrected}
    agent.pi_sym_torch = references[reference].to(agent.device).eval()
    agent.pi_sym_torch.requires_grad_(False)
    # Preserve the unchanged auxiliary world model; this candidate uses only feedback.
    agent.sindy_model = models["linear"]
    agent.f_sym_torch = SINDyTorchWrapper(models["linear"]).to(agent.device).eval()
    agent._irm_report = None
    validation_states, validation_actions, _ = collect_trajectories(
        env, n_episodes=5, max_steps=config.max_steps_per_episode)
    x, u, y = models["linear"].training_arrays(validation_states, validation_actions, multiple_trajectories=True)
    residual = models["linear"].predict(x, u) - y
    agent._sindy_report = {
        "sparsity": models["linear"].sparsity, "n_coeffs": int(models["linear"].coeffs.size),
        "n_nonzero": int((np.abs(models["linear"].coeffs) > 1e-6).sum()), "n_control": 2,
        "r_squared": float(1 - np.sum(residual ** 2) / max(np.sum((y - y.mean(axis=0)) ** 2), 1e-10)),
        "action_reference": "LQRFeedbackReference"}
    agent.reference_identification = {
        "selected_reference": reference,
        "identification_interactions": sum(len(a) for a in actions),
        "validation_interactions": len(x), "auxiliary_world_model": "linear-unchanged",
        "both_models_use_same_transitions": True,
        "gains": {name: module.gain.detach().cpu().tolist() for name, module in references.items()},
        "fits": {name: fit_report(model, local_gain(model)) for name, model in models.items()},
        "validation": {name: physical_metrics(model, x, u, y) for name, model in models.items()}}
    print(f"[Phase 0] reference={reference}: {agent._sindy_report}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", choices=REFERENCES, required=True)
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
    assert manifest["revision"] == REVISION and manifest["training_seeds"] == list(TRAINING_SEEDS)
    assert (manifest["references"], manifest["radii"], manifest["severities"], manifest["reset_cycle"],
            manifest["evaluation_seed_start"], manifest["evaluation_episodes_per_severity"],
            manifest["sampler_seed_offset"], manifest["reset_sample_fraction"]) == (
        list(REFERENCES), list(RADII), list(SEVERITIES), list(RADII), 60000, 100, SAMPLER_SEED_OFFSET, .5)
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
    agent = ReferenceAgent(config, args.seed)
    env = training.make_env("nonlinear_load", 1, args.seed)
    env.action_space.seed(args.seed)
    assert env.specification() == manifest["environment_settings"]
    protocol = {"revision": REVISION, "method": CANDIDATE, "training_seed": args.seed,
                "reference": args.reference, "sampling": "reset-balanced", "training_episodes": episodes,
                "config": asdict(config), "environment": "nonlinear_load", "environment_settings": env.specification(),
                "training_load_severity": 1, "reset_cycle": list(RADII), "reset_index": "absolute_episode",
                "identification_reset_radius": .2, "reset_sample_fraction": .5,
                "sampler_seed": SAMPLER_SEED_OFFSET + args.seed,
                "radii": list(RADII), "severities": list(SEVERITIES),
                "evaluation_seed_start": 60000, "evaluation_episodes_per_severity": count,
                "preflight_only": args.preflight_only}
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{REVISION}_{args.reference}_seed{args.seed}"
    last = str(args.ckpt_dir / f"{tag}_last.pt")
    resume = args.resume or (last if Path(last).exists() else None)
    if resume:
        start_ep, env_steps, history = training.restore_training_checkpoint(agent, env, resume, protocol)
        print(f"[RESUME] episode={start_ep} steps={env_steps}", flush=True)
    else:
        identify_env = training.make_env("nonlinear_load", 1, args.seed)
        identify_env.action_space.seed(args.seed)
        identify_reference(agent, identify_env, args.reference)
        training.seed_training(args.seed)
        start_ep, env_steps, history = 0, 0, []
        training.save_training_checkpoint(agent, env, last, next_ep=0, env_steps=0, history=history, protocol=protocol)
    assert agent.reference_identification["selected_reference"] == args.reference
    print(f"Training: {CANDIDATE} seed={args.seed} reference={args.reference}; Jacobian sampling=reset-balanced", flush=True)
    started = time.time()
    for episode in range(start_ep, episodes):
        reward, env_steps = train_episode(env, agent, "balanced", episode, env_steps)
        assert np.isfinite(reward)
        history.append(reward)
        if (episode + 1) % 20 == 0 or episode + 1 == episodes or (args.preflight_only and episode == 4):
            training.save_training_checkpoint(agent, env, last, next_ep=episode + 1, env_steps=env_steps,
                                              history=history, protocol=protocol)
        if args.preflight_only and episode == 4:
            agent = ReferenceAgent(config, args.seed)
            restored_ep, restored_steps, restored_history = training.restore_training_checkpoint(agent, env, last, protocol)
            assert (restored_ep, restored_steps, restored_history) == (5, env_steps, history)
            print(f"[PREFLIGHT RESUME] episode5 steps={env_steps}; reference and sampler restored", flush=True)
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
    assert evaluated["feedback_gains"]["fitted"] == agent.reference_identification["gains"][args.reference]
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
