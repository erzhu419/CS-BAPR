"""Cross calibration and predictive tails on unchanged frozen-policy traces."""
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
from scripts import probe_general_ood_v43 as v43
from scripts import probe_general_ood_v44 as v44
from scripts import run_general_ood_v40 as v40
from scripts import v44_belief as belief
from scripts.v46_belief import RobustAdaptiveStudent
from scripts.train_csbapr import MUJOCO_MODE_PROFILES

REVISION = "general_ood_v46_crossed_tails"
ARM_CALIBRATIONS = {"episode-original": "episode-centered", "episode-robust": "episode-centered",
                    "global-original": "global-centered", "global-robust": "global-centered"}
ARM_LIKELIHOODS = {name: ("robust-student-adaptive" if name.endswith("robust") else "student-adaptive")
                   for name in ARM_CALIBRATIONS}
PHASES = dict(v44.PHASES, full1536=(0, 1536), prefix1024=(0, 1024), all_post512=(1024, 1536))


def observe(trace, model):
    ages = v43.episode_steps(trace["dones"])
    errors = model.errors(trace)
    signals = {}
    for channel in set(ARM_CALIBRATIONS.values()):
        scores, thresholds = model.calibrations[channel].scores(errors, ages)
        signals[channel] = np.log1p(scores / thresholds)
    arms = {}
    for name, channel in ARM_CALIBRATIONS.items():
        tracker = (RobustAdaptiveStudent if name.endswith("robust") else belief.AdaptiveStudent)(**belief.SETTINGS)
        signal = signals[channel]
        metrics, events = [], []
        groups = model.calibrations[channel].groups(ages)
        for step, value in enumerate(signal):
            tracker.update(float(value))
            penalty = float(tracker.belief @ np.exp(-.1 * np.arange(tracker.max_H)))
            metrics.append((tracker.belief[0], tracker.effective_window, penalty, -2. - penalty, tracker.entropy))
            if tracker.belief[0] > .5:
                events.append(dict(step=step, episode_age=int(ages[step]), signal=float(value),
                                   reset_probability=float(tracker.belief[0]),
                                   calibration_changed=bool(step and groups[step] != groups[step - 1]),
                                   previous_transition_ended=bool(step and trace["dones"][step - 1]),
                                   terminated=bool(trace["terminated"][step]), truncated=bool(trace["truncated"][step])))
        metrics = np.asarray(metrics)
        arms[name] = dict(phases={phase: v44.phase_statistics(metrics[start:end]) for phase, (start, end) in PHASES.items()},
                          major_reset_transition_counts={phase: int(np.sum(metrics[start:end, 0] > .5))
                                                         for phase, (start, end) in PHASES.items()},
                          major_reset_events=events)
    scores, _ = v43.observe_trace(trace, {v44.PREDICTOR: model})
    signal_statistics = {channel: {phase: float(signal[start:end].mean()) for phase, (start, end) in PHASES.items()}
                         for channel, signal in signals.items()}
    return arms, signal_statistics, scores[v44.PREDICTOR][v44.CALIBRATOR]


def run(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(1)
    np.random.seed(1)
    fit_seeds, calibration_seeds, seeds = v43.stage_seeds(args.stage)
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
        for network in (agent.actor, agent.critic, agent.target_critic):
            network.eval()
            network.requires_grad_(False)
        snapshot = agent.snapshot_context()
        references, datasets = [], {}
        for split, ref_seeds in (("fit", fit_seeds), ("calibration", calibration_seeds)):
            traces = []
            for seed in ref_seeds:
                env = gym.make("Hopper-v4", max_episode_steps=v41.PROBE_STEPS)
                trace = v41.collect_trajectory(agent, env, seed, "normal", snapshot, budget=v43.REFERENCE_STEPS)
                env.close()
                traces.append(trace)
                references.append(v43.reference_record(trace, seed, split))
            datasets[split] = v43.concatenate_references(traces)
        model = v43.ReferenceDynamics(*v42.PREDICTORS[v44.PREDICTOR]).fit_references(datasets["fit"], datasets["calibration"])
        metadata = model.metadata()
        del datasets, traces, trace
        print(f"[REFERENCE] {policy}: unchanged normal fit/calibration", flush=True)
        records = []
        for seed in seeds:
            control = None
            for mode in targets:
                env = gym.make("Hopper-v4", max_episode_steps=v41.PROBE_STEPS)
                trace = v41.collect_trajectory(agent, env, seed, mode, snapshot)
                env.close()
                if mode == "normal":
                    control = trace
                assert v41.prefix_equal(control, trace), (policy, seed, mode, "unpaired-prefix")
                assert all(np.array_equal(control[k][v41.SWITCH_STEP], trace[k][v41.SWITCH_STEP])
                           for k in ("states", "executed_actions", "contexts")), "unpaired-switch-inputs"
                arms, signal, score_check = observe(trace, model)
                endings = {phase: {"terminated": int(trace["terminated"][start:end].sum()),
                                   "truncated": int(trace["truncated"][start:end].sum())}
                           for phase, (start, end) in PHASES.items()}
                records.append(dict(seed=seed, target_mode=mode, paired_prefix_equal=True, paired_switch_inputs_equal=True,
                                    interactions=len(trace["dones"]), arms=arms, signal=signal,
                                    v43_score_check=score_check, phase_endings=endings,
                                    reward_means_by_256=[float(trace["rewards"][i:i + 256].mean()) for i in range(0, 1536, 256)],
                                    termination_counts_by_256=[int(trace["terminated"][i:i + 256].sum()) for i in range(0, 1536, 256)],
                                    truncation_counts_by_256=[int(trace["truncated"][i:i + 256].sum()) for i in range(0, 1536, 256)]))
            print(f"[CROSSED] {policy} seed={seed}: {len(targets)} modes, four fixed arms", flush=True)
        assert model.metadata() == metadata
        agent.restore_context(deepcopy(snapshot))
        assert agent.context_tracker._count == 100000 and agent.training_steps == 99745 and agent.replay_buffer.size == 0
        output = dict(revision=REVISION, stage=args.stage, environment="Hopper-v4", source_policy=policy,
                      source_checkpoint=checkpoint.name, source_training_seed=1, config=asdict(agent.config),
                      loaded_training_update_steps=agent.training_steps, loaded_training_context_count=agent.context_tracker._count,
                      fit_seeds=fit_seeds, calibration_seeds=calibration_seeds, evaluation_seeds=seeds, target_modes=list(targets),
                      predictor=v44.PREDICTOR, arm_calibrations=ARM_CALIBRATIONS, arm_likelihoods=ARM_LIKELIHOODS,
                      filter_settings=belief.SETTINGS, robust_student_degrees=4, phases=PHASES, model_metadata=metadata,
                      reset_filters_at_episode_end=False, references=references, records=records,
                      optimizer_updates_during_probe=0, replay_insertions=0, auxiliary_signal_controls_actions=False,
                      predictors_updated_after_calibration=False, actor_critic_target_frozen=True,
                      total_reference_interactions=len(references) * v43.REFERENCE_STEPS,
                      total_probe_interactions=len(records) * v41.PROBE_STEPS, seconds=time.monotonic() - started,
                      runtime=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__, gymnasium=gym.__version__,
                                   mujoco=__import__("mujoco").__version__, gpu=torch.cuda.get_device_name()))
        path = args.result_dir / f"{REVISION}_{policy}.json"
        path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
        print(f"[DONE] crossed calibration/tails {policy}: result={path}", flush=True)
        del agent, model
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "formal"), default="formal")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
