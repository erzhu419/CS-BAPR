"""Localize frozen residual-belief resets; export small event windows only."""
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

REVISION = "general_ood_v45_reset_localization"
CHANNELS = ("episode-centered", "global-centered")
EVENT_RADIUS = 1


def diagnose(trace, model):
    errors = model.errors(trace)
    ages = v43.episode_steps(trace["dones"])
    calibration = model.calibrations["episode-centered"]
    groups = calibration.groups(ages)
    scores, thresholds = calibration.scores(errors, ages)
    signal = np.log1p(scores / thresholds)
    previous_groups = np.r_[groups[0], groups[:-1]]
    held_scores = np.mean(((errors - calibration.means[previous_groups]) / calibration.scales[previous_groups]) ** 2, axis=1)
    held_signal = np.log1p(held_scores / calibration.thresholds[previous_groups])
    tracker = belief.AdaptiveStudent(**belief.SETTINGS)
    rows, metrics, last_boundary = [], [], None
    for step, value in enumerate(signal):
        changed = step > 0 and groups[step] != groups[step - 1]
        location, scale2, df = tracker.predictive_parameters()
        log_likelihood = tracker._compute_log_likelihood(float(value))
        prior_cap = float(tracker.belief[-1])
        if changed:
            last_boundary = step
            alternative = deepcopy(tracker)
            alternative.update(float(held_signal[step]))
            held_reset = float(alternative.belief[0])
        tracker.update(float(value))
        reset = float(tracker.belief[0])
        penalty = float(tracker.belief @ np.exp(-.1 * np.arange(tracker.max_H)))
        metrics.append((reset, tracker.effective_window, penalty, -2. - penalty, tracker.entropy))
        rows.append(dict(step=step, episode_age=int(ages[step]), calibration_group=int(groups[step]),
                         calibration_changed=bool(changed), steps_since_calibration_change=None if last_boundary is None else step - last_boundary,
                         previous_transition_ended=bool(step and trace["dones"][step - 1]),
                         terminated=bool(trace["terminated"][step]), truncated=bool(trace["truncated"][step]),
                         score=float(scores[step]), threshold95=float(thresholds[step]), signal=float(value),
                         previous_group_current_error_signal=float(held_signal[step]),
                         previous_group_current_error_reset=held_reset if changed else reset,
                         mature_predictive_mean=float(location[-1]),
                         mature_predictive_variance=float(scale2[-1] * df[-1] / (df[-1] - 2)),
                         previous_cap_probability=prior_cap,
                         log_likelihood_reset_minus_cap=float(log_likelihood[0] - log_likelihood[-1]),
                         target_scaled_residual_rms=float(np.sqrt(np.mean((errors[step] / model.target_scale) ** 2))),
                         reset_probability=reset, effective_age=float(tracker.effective_window), penalty=penalty))
    metrics = np.asarray(metrics)
    peaks = {phase: start + int(np.argmax(metrics[start:end, 0])) for phase, (start, end) in v44.PHASES.items()}
    major = np.flatnonzero(metrics[:, 0] > .5).tolist()
    boundaries = np.flatnonzero(groups != previous_groups).tolist()
    selected = sorted({t for event in set(major + boundaries + list(peaks.values()))
                       for t in range(max(0, event - EVENT_RADIUS), min(len(rows), event + EVENT_RADIUS + 1))})
    global_cal = model.calibrations["global-centered"]
    global_scores, global_thresholds = global_cal.scores(errors, ages)
    _, global_replay = belief.replay(global_scores, global_thresholds)
    channel_traces = {"episode-centered": metrics, "global-centered": global_replay["student-adaptive"]}
    statistics = {channel: {phase: v44.phase_statistics(values[start:end]) for phase, (start, end) in v44.PHASES.items()}
                  for channel, values in channel_traces.items()}
    major_counts = {channel: {phase: int(np.sum(values[start:end, 0] > .5)) for phase, (start, end) in v44.PHASES.items()}
                    for channel, values in channel_traces.items()}
    score_check, _ = v43.observe_trace(trace, {v44.PREDICTOR: model})
    return dict(channel_statistics=statistics, major_reset_transition_counts=major_counts,
                phase_peak_steps=peaks, major_reset_steps=major, calibration_boundary_steps=boundaries,
                event_window=[rows[i] for i in selected], v43_score_check=score_check[v44.PREDICTOR][v44.CALIBRATOR])


def run(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(1)
    np.random.seed(1)
    fit_seeds, calibration_seeds, seeds = v43.stage_seeds(args.stage)
    # Preflight reuses v44's existing normal/high-gravity fixtures; formal focuses on the remaining issue.
    targets = ("normal", "high_grav") if args.stage == "preflight" else ("normal", "heavy")
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
        print(f"[REFERENCE] {policy}: frozen normal references", flush=True)
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
                row = diagnose(trace, model)
                row.update(seed=seed, target_mode=mode, paired_prefix_equal=True, paired_switch_inputs_equal=True,
                           interactions=len(trace["dones"]),
                           reward_means_by_256=[float(trace["rewards"][i:i + 256].mean()) for i in range(0, 1536, 256)],
                           termination_counts_by_256=[int(trace["terminated"][i:i + 256].sum()) for i in range(0, 1536, 256)],
                           truncation_counts_by_256=[int(trace["truncated"][i:i + 256].sum()) for i in range(0, 1536, 256)])
                records.append(row)
            print(f"[LOCALIZE] {policy} seed={seed}: {len(targets)} modes", flush=True)
        assert model.metadata() == metadata
        agent.restore_context(deepcopy(snapshot))
        assert agent.context_tracker._count == 100000 and agent.training_steps == 99745 and agent.replay_buffer.size == 0
        output = dict(revision=REVISION, stage=args.stage, environment="Hopper-v4", source_policy=policy,
                      source_checkpoint=checkpoint.name, source_training_seed=1, config=asdict(agent.config),
                      loaded_training_update_steps=agent.training_steps, loaded_training_context_count=agent.context_tracker._count,
                      fit_seeds=fit_seeds, calibration_seeds=calibration_seeds, evaluation_seeds=seeds, target_modes=list(targets),
                      predictor=v44.PREDICTOR, channel_names=list(CHANNELS), filter_settings=belief.SETTINGS,
                      event_radius=EVENT_RADIUS, phases=v44.PHASES, model_metadata=metadata, references=references, records=records,
                      optimizer_updates_during_probe=0, replay_insertions=0, auxiliary_signal_controls_actions=False,
                      predictors_updated_after_calibration=False, actor_critic_target_frozen=True,
                      total_reference_interactions=len(references) * v43.REFERENCE_STEPS,
                      total_probe_interactions=len(records) * v41.PROBE_STEPS, seconds=time.monotonic() - started,
                      runtime=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__, gymnasium=gym.__version__,
                                   mujoco=__import__("mujoco").__version__, gpu=torch.cuda.get_device_name()))
        path = args.result_dir / f"{REVISION}_{policy}.json"
        path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
        print(f"[DONE] reset localization {policy}: result={path}", flush=True)
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
