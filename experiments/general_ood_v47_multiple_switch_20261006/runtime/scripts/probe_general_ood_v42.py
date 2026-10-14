"""Past-only dynamics residuals on paired, frozen-policy Hopper probes."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import numpy as np
import torch

from csbapr.utils import compute_reg_norm
from scripts import probe_general_ood_v41 as v41
from scripts import run_general_ood_v40 as v40
from scripts.train_csbapr import MUJOCO_MODE_PROFILES

REVISION = "general_ood_v42_frozen_dynamics"
PREDICTORS = {"linear-state-action": (1, True), "quadratic-state-action": (2, True),
              "quadratic-state-only": (2, False)}
FIT_END, CALIBRATION_END = 512, 768
RIDGE = 1e-3
SCALE_FLOOR = 1e-6
PHASES = {"validation256": (768, 1024), "first_post16": (1024, 1040),
          "first_post256": (1024, 1280), "second_post256": (1280, 1536)}


class DynamicsResidual:
    def __init__(self, degree, use_action):
        self.degree, self.use_action = degree, use_action

    def inputs(self, trace):
        states = np.asarray(trace["states"], dtype=np.float64)
        return (np.column_stack((states, trace["executed_actions"])) if self.use_action else states)

    def features(self, inputs):
        x = (inputs - self.input_mean) / self.input_scale
        parts = [np.ones((len(x), 1)), x]
        if self.degree == 2:
            i, j = np.triu_indices(x.shape[1])
            parts.append(x[:, i] * x[:, j])
        return np.column_stack(parts)

    def errors(self, trace):
        target = np.asarray(trace["next_states"], dtype=np.float64) - trace["states"]
        prediction = self.features(self.inputs(trace)) @ self.coefficients * self.target_scale
        return target - prediction

    def fit(self, trace):
        fitting = {k: v[:FIT_END] for k, v in trace.items()}
        x = self.inputs(fitting)
        y = np.asarray(fitting["next_states"], dtype=np.float64) - fitting["states"]
        self.input_mean = x.mean(axis=0)
        self.input_scale = np.maximum(x.std(axis=0), SCALE_FLOOR)
        self.target_scale = np.maximum(y.std(axis=0), SCALE_FLOOR)
        design = self.features(x)
        penalty = FIT_END * RIDGE * np.eye(design.shape[1])
        penalty[0, 0] = 0.
        self.coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ (y / self.target_scale))
        # Neither feature fitting nor residual calibration reads the validation/test suffix.
        prefix = {k: v[:CALIBRATION_END] for k, v in trace.items()}
        calibration = self.errors(prefix)[FIT_END:CALIBRATION_END]
        self.residual_scale = np.maximum(np.sqrt(np.mean(calibration ** 2, axis=0)),
                                         SCALE_FLOOR * self.target_scale)
        calibration_scores = np.mean((calibration / self.residual_scale) ** 2, axis=1)
        self.threshold95 = float(np.quantile(calibration_scores, .95))
        self.fit_mse_in_target_units = float(np.mean((self.errors(prefix)[:FIT_END] / self.target_scale) ** 2))
        self.calibration_mse_in_target_units = float(np.mean((calibration / self.target_scale) ** 2))
        return self

    def scores(self, trace):
        return np.mean((self.errors(trace) / self.residual_scale) ** 2, axis=1)

    def metadata(self):
        return {"degree": self.degree, "use_action": self.use_action,
                "feature_count": len(self.coefficients), "ridge_mean_loss_coefficient": RIDGE,
                "input_scale": self.input_scale.tolist(), "target_scale": self.target_scale.tolist(),
                "calibration_residual_rms": self.residual_scale.tolist(),
                "calibration_threshold95": self.threshold95,
                "fit_mse_in_target_units": self.fit_mse_in_target_units,
                "calibration_mse_in_target_units": self.calibration_mse_in_target_units}


def phase_statistics(trace, predictor):
    scores = predictor.scores(trace)
    result = {}
    for phase, (start, end) in PHASES.items():
        values = scores[start:end]
        ongoing = ~trace["dones"][start:end]
        result[phase] = {"transitions": end - start, "mean_score": float(values.mean()),
                         "median_score": float(np.median(values)), "p95_score": float(np.quantile(values, .95)),
                         "first_transition_score": float(values[0]),
                         "fraction_above_calibration95": float(np.mean(values > predictor.threshold95)),
                         "nonterminal_transitions": int(ongoing.sum()),
                         "nonterminal_mean_score": float(values[ongoing].mean()) if ongoing.any() else None,
                         "terminated": int(trace["terminated"][start:end].sum()),
                         "truncated": int(trace["truncated"][start:end].sum())}
    return result


def reward_reference(trace, config, reg_signal):
    detector = v41.PreviousChunkReference(config.surprise_ema)
    result = []
    for start in range(0, v41.PROBE_STEPS, v41.WINDOW):
        rewards = torch.as_tensor(trace["rewards"][start:start + v41.WINDOW], dtype=torch.float32)
        q_mean = torch.as_tensor(trace["q_spread"][start:start + v41.WINDOW]).mean().item()
        signal = detector.compute(rewards, q_mean, reg_signal)
        result.append(None if signal is None else {"surprise": signal, **detector.last_components})
    return result


def run(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(1)
    np.random.seed(1)
    seeds = list(range(99000, 99002)) if args.stage == "preflight" else list(range(90000, 90020))
    targets = ("normal", "high_grav") if args.stage == "preflight" else tuple(MUJOCO_MODE_PROFILES["Hopper-v4"])
    args.result_dir.mkdir(parents=True, exist_ok=True)
    for policy in v41.POLICIES:
        started = time.monotonic()
        fresh = gym.make("Hopper-v4")
        agent = v40.make_agent("Hopper-v4", policy, "cuda", fresh)
        fresh.close()
        checkpoint = args.checkpoint_dir / f"{v40.REVISION}_Hopper-v4_{policy}_seed1_final.pt"
        agent.load(str(checkpoint))
        assert agent.training_steps == 99745 and agent.context_tracker._count == 100000
        for model in (agent.actor, agent.critic, agent.target_critic):
            model.eval()
            model.requires_grad_(False)
        context_snapshot = agent.snapshot_context()
        current, target = compute_reg_norm(agent.critic).cpu(), compute_reg_norm(agent.target_critic).cpu()
        reg_signal = .5 * ((current - target).abs() / target.abs().clamp_min(1e-6)).mean().item()
        records = []
        for seed in seeds:
            control, control_scores, predictors = None, None, None
            for mode in targets:
                env = gym.make("Hopper-v4", max_episode_steps=v41.PROBE_STEPS)
                trace = v41.collect_trajectory(agent, env, seed, mode, context_snapshot)
                env.close()
                if mode == "normal":
                    control = trace
                    predictors = {name: DynamicsResidual(*spec).fit(control) for name, spec in PREDICTORS.items()}
                assert v41.prefix_equal(control, trace), (policy, seed, mode, "unpaired-prefix")
                observed = {name: phase_statistics(trace, model) for name, model in predictors.items()}
                if mode == "normal":
                    control_scores = observed
                contrasts = {name: {phase: {key: observed[name][phase][key] - control_scores[name][phase][key]
                                           for key in ("mean_score", "first_transition_score", "fraction_above_calibration95")}
                                    for phase in PHASES} for name in PREDICTORS}
                records.append({"seed": seed, "target_mode": mode, "actual_change": mode != "normal",
                                "interactions": len(trace["rewards"]), "paired_prefix_equal": True,
                                "predictors": observed, "change_minus_nochange": contrasts,
                                "reward_reference_by_256": reward_reference(trace, agent.config, reg_signal),
                                "reward_means_by_256": [float(trace["rewards"][i:i + 256].mean())
                                                        for i in range(0, v41.PROBE_STEPS, 256)],
                                "termination_counts_by_256": [int(trace["terminated"][i:i + 256].sum())
                                                              for i in range(0, v41.PROBE_STEPS, 256)],
                                "truncation_counts_by_256": [int(trace["truncated"][i:i + 256].sum())
                                                            for i in range(0, v41.PROBE_STEPS, 256)],
                                "fit_metadata": {name: model.metadata() for name, model in predictors.items()}
                                                if mode == "normal" else None})
            print(f"[PROBE] {policy} seed={seed}: {len(targets)} modes, past-only predictors", flush=True)
        agent.restore_context(deepcopy(context_snapshot))
        assert agent.context_tracker._count == 100000 and agent.training_steps == 99745
        assert agent.replay_buffer.size == 0
        output = {"revision": REVISION, "stage": args.stage, "environment": "Hopper-v4", "source_policy": policy,
                  "source_checkpoint": checkpoint.name, "source_training_seed": 1,
                  "source_training_interactions": 100000, "loaded_training_update_steps": 99745,
                  "loaded_training_context_count": 100000, "config": asdict(agent.config),
                  "evaluation_seeds": seeds, "target_modes": list(targets), "predictor_names": list(PREDICTORS),
                  "fit_range": [0, FIT_END], "calibration_range": [FIT_END, CALIBRATION_END],
                  "validation_range": [CALIBRATION_END, v41.SWITCH_STEP], "phases": PHASES,
                  "ridge_mean_loss_coefficient": RIDGE, "scale_floor": SCALE_FLOOR,
                  "probe_interactions_per_rollout": v41.PROBE_STEPS, "switch_step": v41.SWITCH_STEP,
                  "probe_episode_time_limit": v41.PROBE_STEPS,
                  "optimizer_updates_during_probe": 0, "replay_insertions": 0,
                  "actor_critic_target_frozen": True, "restore_training_context_before_each_rollout": True,
                  "auxiliary_signal_controls_actions": False, "predictors_updated_after_calibration": False,
                  "total_probe_interactions": len(records) * v41.PROBE_STEPS, "records": records,
                  "seconds": time.monotonic() - started,
                  "runtime": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                              "gymnasium": gym.__version__, "mujoco": __import__("mujoco").__version__,
                              "gpu": torch.cuda.get_device_name()}}
        path = args.result_dir / f"{REVISION}_{policy}.json"
        path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
        print(f"[DONE] frozen dynamics {policy}: {len(records)} rollouts, result={path}", flush=True)
        del agent
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "formal"), default="formal")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
