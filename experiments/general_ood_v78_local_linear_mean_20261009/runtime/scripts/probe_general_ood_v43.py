"""Independent normal references and fixed residual calibrations; no RL updates."""
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

from scripts import probe_general_ood_v41 as v41
from scripts import probe_general_ood_v42 as v42
from scripts import run_general_ood_v40 as v40
from scripts.train_csbapr import MUJOCO_MODE_PROFILES

REVISION = "general_ood_v43_normal_reference"
REFERENCE_STEPS = 3072
AGE_BIN_WIDTH, AGE_GROUPS = 256, 4
CALIBRATORS = ("global-rms", "global-centered", "episode-centered")
KEYS = ("mean_score", "first_transition_score", "fraction_above_calibration95", "nonterminal_mean_score")


def episode_steps(dones):
    ages, age = [], 0
    for done in dones:
        ages.append(age)
        age = 0 if done else age + 1
    return np.asarray(ages, dtype=np.int64)


def concatenate_references(traces):
    result = {k: np.concatenate([t[k] for t in traces])
              for k in ("states", "next_states", "executed_actions")}
    # Each independent rollout starts a fresh episode, even if its budget ends mid-episode.
    result["episode_steps"] = np.concatenate([episode_steps(t["dones"]) for t in traces])
    return result


class ResidualCalibration:
    def __init__(self, name, errors, ages, target_scale):
        self.name = name
        group_ids = self.groups(ages)
        count = AGE_GROUPS if name == "episode-centered" else 1
        self.means, self.scales, self.thresholds, self.counts = [], [], [], []
        for group in range(count):
            values = errors[group_ids == group]
            assert len(values), (name, group, "normal-reference-age-group-empty")
            center = np.zeros(errors.shape[1]) if name == "global-rms" else values.mean(axis=0)
            scale = np.maximum(np.sqrt(np.mean((values - center) ** 2, axis=0)),
                               v42.SCALE_FLOOR * target_scale)
            scores = np.mean(((values - center) / scale) ** 2, axis=1)
            self.means.append(center)
            self.scales.append(scale)
            self.thresholds.append(float(np.quantile(scores, .95)))
            self.counts.append(len(values))
        self.means, self.scales = np.asarray(self.means), np.asarray(self.scales)
        self.thresholds = np.asarray(self.thresholds)

    def groups(self, ages):
        return (np.minimum(ages // AGE_BIN_WIDTH, AGE_GROUPS - 1) if self.name == "episode-centered"
                else np.zeros(len(ages), dtype=np.int64))

    def scores(self, errors, ages):
        groups = self.groups(ages)
        scores = np.mean(((errors - self.means[groups]) / self.scales[groups]) ** 2, axis=1)
        return scores, self.thresholds[groups]

    def metadata(self):
        return {"counts": self.counts, "residual_centers": self.means.tolist(),
                "residual_scales": self.scales.tolist(), "threshold95": self.thresholds.tolist()}


class ReferenceDynamics(v42.DynamicsResidual):
    def fit_references(self, fitting, calibration):
        x = self.inputs(fitting)
        y = np.asarray(fitting["next_states"], dtype=np.float64) - fitting["states"]
        self.input_mean = x.mean(axis=0)
        self.input_scale = np.maximum(x.std(axis=0), v42.SCALE_FLOOR)
        self.target_scale = np.maximum(y.std(axis=0), v42.SCALE_FLOOR)
        design = self.features(x)
        penalty = len(x) * v42.RIDGE * np.eye(design.shape[1])
        penalty[0, 0] = 0.
        self.coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ (y / self.target_scale))
        errors = self.errors(calibration)
        self.calibrations = {name: ResidualCalibration(name, errors, calibration["episode_steps"], self.target_scale)
                             for name in CALIBRATORS}
        self.fit_mse = float(np.mean((self.errors(fitting) / self.target_scale) ** 2))
        self.calibration_mse = float(np.mean((errors / self.target_scale) ** 2))
        return self

    def metadata(self):
        return {"degree": self.degree, "use_action": self.use_action, "feature_count": len(self.coefficients),
                "fit_mse_in_target_units": self.fit_mse, "calibration_mse_in_target_units": self.calibration_mse,
                "input_mean": self.input_mean.tolist(), "input_scale": self.input_scale.tolist(),
                "target_scale": self.target_scale.tolist(),
                "calibrations": {name: cal.metadata() for name, cal in self.calibrations.items()}}


def observe_trace(trace, models):
    ages, observed, forecasts = episode_steps(trace["dones"]), {}, {}
    for name, model in models.items():
        errors = model.errors(trace)
        target_mse = np.mean((errors / model.target_scale) ** 2, axis=1)
        forecasts[name] = {phase: float(target_mse[start:end].mean())
                           for phase, (start, end) in v42.PHASES.items()}
        observed[name] = {}
        for channel, calibration in model.calibrations.items():
            scores, thresholds = calibration.scores(errors, ages)
            phases = {}
            for phase, (start, end) in v42.PHASES.items():
                values, ongoing = scores[start:end], ~trace["dones"][start:end]
                phases[phase] = {"mean_score": float(values.mean()), "first_transition_score": float(values[0]),
                                 "fraction_above_calibration95": float(np.mean(values > thresholds[start:end])),
                                 "nonterminal_mean_score": float(values[ongoing].mean()) if ongoing.any() else None}
            observed[name][channel] = phases
    return observed, forecasts


def reference_record(trace, seed, split):
    ends = np.flatnonzero(trace["dones"])
    ages = episode_steps(trace["dones"])
    return {"seed": seed, "split": split, "mode": "normal", "interactions": len(trace["dones"]),
            "physical_terminations": int(trace["terminated"].sum()), "time_limit_truncations": int(trace["truncated"].sum()),
            "completed_episode_lengths": (ages[ends] + 1).tolist(),
            "episode_age_group_counts": np.bincount(np.minimum(ages // AGE_BIN_WIDTH, AGE_GROUPS - 1), minlength=AGE_GROUPS).tolist()}


def stage_seeds(stage):
    if stage == "preflight":
        return list(range(97900, 97902)), list(range(97910, 97912)), list(range(99000, 99002))
    return list(range(98000, 98008)), list(range(98010, 98014)), list(range(90000, 90020))


def run(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(1)
    np.random.seed(1)
    fit_seeds, calibration_seeds, seeds = stage_seeds(args.stage)
    assert len(set(fit_seeds + calibration_seeds + seeds)) == len(fit_seeds + calibration_seeds + seeds)
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
        snapshot = agent.snapshot_context()
        references, datasets = [], {}
        for split, ref_seeds in (("fit", fit_seeds), ("calibration", calibration_seeds)):
            traces = []
            for seed in ref_seeds:
                env = gym.make("Hopper-v4", max_episode_steps=v41.PROBE_STEPS)
                trace = v41.collect_trajectory(agent, env, seed, "normal", snapshot, budget=REFERENCE_STEPS)
                env.close()
                traces.append(trace)
                references.append(reference_record(trace, seed, split))
            datasets[split] = concatenate_references(traces)
            print(f"[REFERENCE] {policy} {split}: {len(ref_seeds)} normal rollouts", flush=True)
        models = {name: ReferenceDynamics(*spec).fit_references(datasets["fit"], datasets["calibration"])
                  for name, spec in v42.PREDICTORS.items()}
        model_metadata = {name: model.metadata() for name, model in models.items()}
        del datasets, traces, trace
        records = []
        for seed in seeds:
            control, control_observers = None, None
            for mode in targets:
                env = gym.make("Hopper-v4", max_episode_steps=v41.PROBE_STEPS)
                trace = v41.collect_trajectory(agent, env, seed, mode, snapshot)
                env.close()
                if mode == "normal":
                    control = trace
                assert v41.prefix_equal(control, trace), (policy, seed, mode, "unpaired-prefix")
                assert all(np.array_equal(control[k][v41.SWITCH_STEP], trace[k][v41.SWITCH_STEP])
                           for k in ("states", "executed_actions", "contexts")), "unpaired-switch-inputs"
                observers, forecasts = observe_trace(trace, models)
                if mode == "normal":
                    control_observers = observers
                contrasts = {name: {channel: {phase: {key: observers[name][channel][phase][key] - control_observers[name][channel][phase][key]
                                                    for key in KEYS[:3]} for phase in v42.PHASES}
                                    for channel in CALIBRATORS} for name in v42.PREDICTORS}
                endings = {phase: {"terminated": int(trace["terminated"][start:end].sum()),
                                   "truncated": int(trace["truncated"][start:end].sum()),
                                   "nonterminal_transitions": int((~trace["dones"][start:end]).sum())}
                           for phase, (start, end) in v42.PHASES.items()}
                records.append({"seed": seed, "target_mode": mode, "actual_change": mode != "normal",
                                "interactions": len(trace["dones"]), "paired_prefix_equal": True,
                                "paired_switch_inputs_equal": True, "predictors": observers,
                                "forecast_mse_in_target_units": forecasts, "change_minus_nochange": contrasts,
                                "phase_endings": endings,
                                "reward_means_by_256": [float(trace["rewards"][i:i + 256].mean()) for i in range(0, 1536, 256)],
                                "termination_counts_by_256": [int(trace["terminated"][i:i + 256].sum()) for i in range(0, 1536, 256)],
                                "truncation_counts_by_256": [int(trace["truncated"][i:i + 256].sum()) for i in range(0, 1536, 256)]})
            print(f"[PROBE] {policy} seed={seed}: {len(targets)} modes, frozen reference models", flush=True)
        assert {name: model.metadata() for name, model in models.items()} == model_metadata
        agent.restore_context(deepcopy(snapshot))
        assert agent.context_tracker._count == 100000 and agent.training_steps == 99745 and agent.replay_buffer.size == 0
        output = {"revision": REVISION, "stage": args.stage, "environment": "Hopper-v4", "source_policy": policy,
                  "source_checkpoint": checkpoint.name, "source_training_seed": 1,
                  "source_training_interactions": 100000, "loaded_training_update_steps": 99745,
                  "loaded_training_context_count": 100000, "config": asdict(agent.config),
                  "fit_seeds": fit_seeds, "calibration_seeds": calibration_seeds, "evaluation_seeds": seeds,
                  "target_modes": list(targets), "predictor_names": list(v42.PREDICTORS), "calibrator_names": list(CALIBRATORS),
                  "reference_interactions_per_rollout": REFERENCE_STEPS,
                  "probe_interactions_per_rollout": v41.PROBE_STEPS, "switch_step": v41.SWITCH_STEP,
                  "episode_time_limit": v41.PROBE_STEPS, "phases": v42.PHASES,
                  "episode_age_bin_width": AGE_BIN_WIDTH, "episode_age_groups": AGE_GROUPS,
                  "ridge_mean_loss_coefficient": v42.RIDGE, "scale_floor": v42.SCALE_FLOOR,
                  "optimizer_updates_during_probe": 0, "replay_insertions": 0,
                  "actor_critic_target_frozen": True, "restore_training_context_before_each_rollout": True,
                  "auxiliary_signal_controls_actions": False, "predictors_updated_after_calibration": False,
                  "normal_only_reference_modes": True, "independent_reference_and_test_seeds": True,
                  "references": references, "model_metadata": model_metadata,
                  "total_reference_interactions": len(references) * REFERENCE_STEPS,
                  "total_probe_interactions": len(records) * v41.PROBE_STEPS, "records": records,
                  "seconds": time.monotonic() - started,
                  "runtime": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                              "gymnasium": gym.__version__, "mujoco": __import__("mujoco").__version__,
                              "gpu": torch.cuda.get_device_name()}}
        path = args.result_dir / f"{REVISION}_{policy}.json"
        path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
        print(f"[DONE] normal reference {policy}: {len(records)} tests, result={path}", flush=True)
        del agent, models
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "formal"), default="formal")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
