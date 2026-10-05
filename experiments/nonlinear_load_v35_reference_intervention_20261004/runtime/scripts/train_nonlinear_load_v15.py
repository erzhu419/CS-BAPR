"""Matched reset-coverage intervention with unchanged v11 learner presets."""
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

from csbapr.agent import CSBAPRAgent
from csbapr.envs.lqr_env import EP_LEN, get_optimal_gain
from scripts import test_lqr_convergence as training
from scripts.diagnose_nonlinear_load_v9 import alignment
from scripts.diagnose_nonlinear_load_v14 import EARLY_STEPS, SAMPLE_INTERVAL, paired_excess_costs, rollout
from scripts.evaluate_nonlinear_load_v13 import RADII, SEVERITIES

REVISION = "nonlinear_load_v15_coverage"
CANDIDATE = "csbapr-centered-mlp-jacobian-only-no-target-shift"
METHODS = (CANDIDATE, "csbapr-centered-mlp-no-sindy-no-target-shift",
           "bapr-no-target-shift", "resac-no-target-shift")
RESET_CYCLES = {"balanced": RADII, "narrow": (0.2,)}


def reset_radius(condition, episode):
    cycle = RESET_CYCLES[condition]
    return cycle[episode % len(cycle)]


def train_episode(env, agent, condition, episode, env_steps):
    env.amplitude = reset_radius(condition, episode)
    return training.run_episode(env, agent, train=True, env_steps=env_steps)


def evaluate_final(agent, seeds, *, env_factory=None):
    agent.actor.eval()
    agent.actor.requires_grad_(False)
    gains = {"known_local": training.known_reference("nonlinear_load").to(agent.device).gain}
    if agent.pi_sym_torch is not None:
        gains["fitted"] = agent.pi_sym_torch.gain
    references = {"fitted": agent.pi_sym_torch} if agent.pi_sym_torch is not None else {}
    modes = ["actor", *gains, "known_inverse"]
    results = {}
    for radius in RADII:
        results[str(radius)] = {}
        for severity in SEVERITIES:
            controllers, samples = {}, None
            for mode in modes:
                controllers[mode], observations = rollout(
                    agent.actor, gains, mode, radius, severity, seeds, env_factory=env_factory,
                    references=references)
                if mode == "actor":
                    samples = observations
            block = {"controllers": controllers,
                     "paired_excess_costs": {mode: paired_excess_costs(controllers["actor"], controllers[mode], seeds)
                                             for mode in modes[1:]}}
            if "fitted" in gains:
                block["alignment"] = {
                    "initial_states": alignment(agent.actor, gains, samples[0], references=references),
                    "actor_trajectory": alignment(agent.actor, gains, samples.reshape(-1, 6), references=references),
                }
            results[str(radius)][str(severity)] = block
            print(f"Eval radius={radius} load={severity}: " +
                  str({m: round(v["mean_cost"], 6) for m, v in controllers.items()}), flush=True)
    return {"radius_results": results,
            "feedback_gains": {**{m: g.detach().cpu().tolist() for m, g in gains.items()},
                               "known_inverse_physical": get_optimal_gain().tolist()},
            "evaluation_episodes_including_controls": len(RADII) * len(SEVERITIES) * len(seeds) * len(modes)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, choices=[9, 10], required=True)
    parser.add_argument("--condition", choices=RESET_CYCLES, required=True)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--resume")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    assert args.condition == "balanced" or args.method == CANDIDATE
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    versions = {"torch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
                "pysindy": pysindy.__version__, "sklearn": sklearn.__version__}
    manifest = json.loads((Path(__file__).resolve().parents[2] / "dispatch_manifest.json").read_text())
    assert manifest["revision"] == REVISION
    assert (manifest["training_seeds"], manifest["radii"], manifest["severities"],
            manifest["evaluation_seed_start"], manifest["evaluation_episodes_per_severity"]) == (
                [9, 10], list(RADII), list(SEVERITIES), 60000, 100)
    assert manifest["reset_cycles"] == {k: list(v) for k, v in RESET_CYCLES.items()}
    for package, version in manifest["runtime_requirement"].items():
        if package != "python_minor_versions":
            assert versions[package] == version
    assert f"{sys.version_info.major}.{sys.version_info.minor}" in manifest["runtime_requirement"]["python_minor_versions"]
    training.seed_training(args.seed)
    config = training.make_config(args.method, args.device, "nonlinear_load")
    config.max_episodes = 500
    arm = next(a for a in manifest["arms"] if a["method"] == args.method)
    assert {**asdict(config), "device": "cuda"} == arm["config"]
    episodes, count = (6, 2) if args.preflight_only else (500, 100)
    config.max_episodes = episodes
    agent = CSBAPRAgent(6, 2, config)
    env = training.make_env("nonlinear_load", 1, args.seed)
    env.action_space.seed(args.seed)
    assert env.specification() == manifest["environment_settings"]
    protocol = {"revision": REVISION, "method": args.method, "training_seed": args.seed,
                "condition": args.condition, "training_episodes": episodes, "config": asdict(config),
                "environment": "nonlinear_load", "environment_settings": env.specification(),
                "training_load_severity": 1, "reset_cycle": list(RESET_CYCLES[args.condition]),
                "reset_index": "absolute_episode", "identification_reset_radius": 0.2,
                "radii": list(RADII), "severities": list(SEVERITIES),
                "evaluation_seed_start": 60000, "evaluation_episodes_per_severity": count,
                "preflight_only": args.preflight_only}
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{REVISION}_{args.condition}_{args.method}_seed{args.seed}"
    last = str(args.ckpt_dir / f"{tag}_last.pt")
    resume = args.resume or (last if Path(last).exists() else None)
    if resume:
        start_ep, env_steps, history = training.restore_training_checkpoint(agent, env, resume, protocol)
        print(f"[RESUME] episode={start_ep} steps={env_steps} next_radius={reset_radius(args.condition, start_ep)}", flush=True)
    else:
        if args.method == CANDIDATE:
            identify_env = training.make_env("nonlinear_load", 1, args.seed)
            identify_env.action_space.seed(args.seed)
            agent.sindy_preidentify(identify_env, reference_builder=lambda model: training.fitted_reference(model, "nonlinear_load"))
        training.seed_training(args.seed)
        start_ep, env_steps, history = 0, 0, []
        training.save_training_checkpoint(agent, env, last, next_ep=0, env_steps=0, history=history, protocol=protocol)
    print(f"Training: {args.method} seed={args.seed} condition={args.condition} reset_cycle={protocol['reset_cycle']}", flush=True)
    started = time.time()
    for episode in range(start_ep, episodes):
        reward, env_steps = train_episode(env, agent, args.condition, episode, env_steps)
        assert np.isfinite(reward)
        history.append(reward)
        if (episode + 1) % 20 == 0 or episode + 1 == episodes:
            training.save_training_checkpoint(agent, env, last, next_ep=episode + 1, env_steps=env_steps,
                                              history=history, protocol=protocol)
        if (episode + 1) % 25 == 0 or episode + 1 == episodes:
            print(f"ep {episode + 1}/{episodes}: radius={env.amplitude} recent_25={np.mean(history[-25:]):.6f} "
                  f"updates={agent.training_steps} t={time.time() - started:.0f}s "
                  f"metrics={getattr(agent, '_last_update_metrics', None)}", flush=True)
    assert (len(history), env_steps, agent.training_steps) == (episodes, episodes * EP_LEN, episodes * EP_LEN - 999)
    if args.method == CANDIDATE:
        assert agent.pi_sym_torch is not None and agent._sindy_report is not None
    updates_before_eval = agent.training_steps
    evaluated = evaluate_final(agent, list(range(60000, 60000 + count)))
    assert agent.training_steps == updates_before_eval
    output = {**protocol, **evaluated, "checkpoint_selection": "last", "environment_steps": env_steps,
              "training_updates": agent.training_steps, "training_returns": history,
              "reset_counts": dict(Counter(str(reset_radius(args.condition, e)) for e in range(episodes))),
              "sindy": agent._sindy_report if args.method == CANDIDATE else None, "reference_is_optimal": False,
              "symbolic_identification_interactions": 10000 if args.method == CANDIDATE else 0,
              "symbolic_validation_interactions": 1000 if args.method == CANDIDATE else 0,
              "optimizer_updates_during_evaluation": 0, "early_steps": EARLY_STEPS, "sample_interval": SAMPLE_INTERVAL,
              "runtime": {**versions, "python": sys.version, "device": str(agent.device),
                          "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None}}
    path = args.result_dir / f"{tag}_evaluation.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)
    print(f"Training complete: saved evaluation to {path}; elapsed={time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
