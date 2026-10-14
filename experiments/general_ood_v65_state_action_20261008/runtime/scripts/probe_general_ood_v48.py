"""Normal-only startup bias correction and paired first50 response curves."""
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
from scripts import probe_general_ood_v47 as v47
from scripts import run_general_ood_v40 as v40
from scripts import v44_belief as belief
from scripts.v46_belief import RobustAdaptiveStudent

REVISION = "general_ood_v48_reset_center"
ARMS = ("global-robust", "reset-centered-robust")
PHASES = dict(v47.PHASES, second_first50=(1280, 1330), restart_first16=(1536, 1552), restart_first50=(1536, 1586))


def correct_startup(errors, ages, offset):
    corrected = errors.copy()
    corrected[ages == 0] -= offset
    return corrected


def fit_startup_offset(model, calibration):
    errors, ages = model.errors(calibration), calibration["episode_steps"]
    starts = errors[ages == 0]
    assert len(starts), "normal-startup-reference-empty"
    global_cal = model.calibrations["global-centered"]
    offset = starts.mean(axis=0) - global_cal.means[0]
    before, thresholds = global_cal.scores(errors, ages)
    after, _ = global_cal.scores(correct_startup(errors, ages, offset), ages)
    metadata = dict(normal_start_count=len(starts), residual_offset=offset.tolist(),
                    original_start_mean_signal=float(np.log1p(before[ages == 0] / thresholds[ages == 0]).mean()),
                    corrected_start_mean_signal=float(np.log1p(after[ages == 0] / thresholds[ages == 0]).mean()),
                    ongoing_scores_unchanged=bool(np.array_equal(before[ages > 0], after[ages > 0])),
                    global_scales_and_thresholds_unchanged=True)
    return offset, metadata


def observe(trace, model, offset):
    ages, errors = v43.episode_steps(trace["dones"]), model.errors(trace)
    calibration = model.calibrations["global-centered"]
    signals = {}
    for arm, residuals in zip(ARMS, (errors, correct_startup(errors, ages, offset))):
        scores, thresholds = calibration.scores(residuals, ages)
        signals[arm] = np.log1p(scores / thresholds)
    trackers = {arm: RobustAdaptiveStudent(**belief.SETTINGS) for arm in ARMS}
    metrics, events, startups = {a: [] for a in ARMS}, {a: [] for a in ARMS}, []
    for step in range(len(ages)):
        counterfactual = None
        if ages[step] == 0:
            counterfactual = deepcopy(trackers["global-robust"])
            counterfactual.update(float(signals["reset-centered-robust"][step]))
        for arm, tracker in trackers.items():
            value = float(signals[arm][step])
            tracker.update(value)
            penalty = float(tracker.belief @ np.exp(-.1 * np.arange(tracker.max_H)))
            metrics[arm].append((tracker.belief[0], tracker.effective_window, penalty, -2. - penalty, tracker.entropy))
            if tracker.belief[0] > .5:
                events[arm].append(dict(step=step, episode_age=int(ages[step]), signal=value,
                    reset_probability=float(tracker.belief[0]), calibration_changed=False,
                    previous_transition_ended=bool(step and trace["dones"][step - 1]),
                    terminated=bool(trace["terminated"][step]), truncated=bool(trace["truncated"][step])))
        if counterfactual is not None:
            startups.append(dict(step=step, previous_terminated=bool(step and trace["terminated"][step - 1]),
                previous_truncated=bool(step and trace["truncated"][step - 1]),
                original_signal=float(signals["global-robust"][step]), corrected_signal=float(signals["reset-centered-robust"][step]),
                original_reset_probability=float(trackers["global-robust"].belief[0]),
                same_past_corrected_reset_probability=float(counterfactual.belief[0]),
                corrected_reset_probability=float(trackers["reset-centered-robust"].belief[0])))
    histories = {a: np.asarray(v) for a, v in metrics.items()}
    arms = {a: dict(phases={p: v44.phase_statistics(v[s:e]) for p, (s, e) in PHASES.items()},
                    major_reset_transition_counts={p: int(np.sum(v[s:e, 0] > .5)) for p, (s, e) in PHASES.items()},
                    major_reset_events=events[a]) for a, v in histories.items()}
    checks, _ = v43.observe_trace(trace, {v44.PREDICTOR: model})
    return arms, histories, signals, startups, checks[v44.PREDICTOR][v44.CALIBRATOR]


def run(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(1)
    np.random.seed(1)
    fit_seeds, calibration_seeds, seeds = v43.stage_seeds(args.stage)
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
        snapshot, references, datasets = agent.snapshot_context(), [], {}
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
        offset, startup_metadata = fit_startup_offset(model, datasets["calibration"])
        metadata = model.metadata()
        del datasets, traces, trace
        print(f"[REFERENCE] {policy}: {startup_metadata['normal_start_count']} normal startup samples, mean-only correction", flush=True)
        records = []
        for seed in seeds:
            for initial in ("normal", *v47.INITIAL_MODES):
                targets = ("normal",) if initial == "normal" else (initial, "normal", next(m for m in v47.INITIAL_MODES if m != initial))
                for target in targets:
                    switches = {} if initial == "normal" else {v47.FIRST_SWITCH: initial}
                    if target != initial:
                        switches[v47.SECOND_SWITCH] = target
                    env = gym.make("Hopper-v4", max_episode_steps=v41.PROBE_STEPS)
                    trace = v47.collect_trajectory(agent, env, seed, switches, snapshot)
                    env.close()
                    arms, histories, signals, startups, check = observe(trace, model, offset)
                    if initial == "normal":
                        normal = trace
                    assert v41.prefix_equal(normal, trace, v47.FIRST_SWITCH), "unpaired-normal-prefix"
                    if target == initial:
                        control, control_histories, control_signals = trace, histories, signals
                    assert v41.prefix_equal(control, trace, v47.SECOND_SWITCH), "unpaired-second-prefix"
                    assert all(np.array_equal(control[k][v47.SECOND_SWITCH], trace[k][v47.SECOND_SWITCH])
                               for k in ("states", "executed_actions", "contexts")), "unpaired-switch-inputs"
                    assert all(np.array_equal(control_histories[k][:v47.SECOND_SWITCH], histories[k][:v47.SECOND_SWITCH]) for k in ARMS)
                    assert all(np.array_equal(control_signals[k][:v47.SECOND_SWITCH], signals[k][:v47.SECOND_SWITCH]) for k in ARMS)
                    records.append(dict(seed=seed, initial_mode=initial, target_mode=target,
                        paired_prefix_equal=True, paired_switch_inputs_equal=True, paired_belief_prefix_equal=True,
                        interactions=len(trace["dones"]), arms=arms, startup_observations=startups, v43_score_check=check,
                        signal={a: {p: float(s[b:e].mean()) for p, (b, e) in PHASES.items()} for a, s in signals.items()},
                        paired_first50_penalty={a: (histories[a][1280:1330, 2] - control_histories[a][1280:1330, 2]).tolist() for a in ARMS},
                        phase_endings={p: {k: int(trace[k][b:e].sum()) for k in ("terminated", "truncated")} for p, (b, e) in PHASES.items()},
                        reward_means_by_256=[float(trace["rewards"][i:i + 256].mean()) for i in range(0, v47.PROBE_STEPS, 256)],
                        termination_counts_by_256=[int(trace["terminated"][i:i + 256].sum()) for i in range(0, v47.PROBE_STEPS, 256)],
                        truncation_counts_by_256=[int(trace["truncated"][i:i + 256].sum()) for i in range(0, v47.PROBE_STEPS, 256)]))
            print(f"[RESET-CENTER] {policy} seed={seed}: normal + six matched branches, first50 retained", flush=True)
        assert model.metadata() == metadata
        agent.restore_context(deepcopy(snapshot))
        assert agent.context_tracker._count == 100000 and agent.training_steps == 99745 and agent.replay_buffer.size == 0
        output = dict(revision=REVISION, stage=args.stage, source_policy=policy, source_checkpoint=checkpoint.name,
            source_training_seed=1, config=asdict(agent.config), references=references, model_metadata=metadata, startup_metadata=startup_metadata,
            fit_seeds=fit_seeds, calibration_seeds=calibration_seeds, evaluation_seeds=seeds, arms=ARMS, phases=PHASES,
            startup_rule="Subtract normal startup residual mean minus global center only at observable episode age0",
            first_switch=v47.FIRST_SWITCH, second_switch=v47.SECOND_SWITCH, probe_steps=v47.PROBE_STEPS,
            episode_time_limit=v41.PROBE_STEPS, filter_settings=belief.SETTINGS, robust_student_degrees=4,
            optimizer_updates_during_probe=0, replay_insertions=0, auxiliary_signal_controls_actions=False,
            reset_filters_at_episode_end=False, predictors_updated_after_calibration=False, actor_critic_target_frozen=True,
            loaded_training_update_steps=agent.training_steps, loaded_training_context_count=agent.context_tracker._count,
            records=records, total_reference_interactions=len(references) * v43.REFERENCE_STEPS,
            total_probe_interactions=len(records) * v47.PROBE_STEPS, seconds=time.monotonic() - started,
            runtime=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__, gymnasium=gym.__version__,
                         mujoco=__import__("mujoco").__version__, gpu=torch.cuda.get_device_name()))
        path = args.result_dir / f"{REVISION}_{policy}.json"
        path.write_text(json.dumps(output, separators=(",", ":"), allow_nan=False) + "\n")
        print(f"[DONE] reset-center {policy}: result={path}", flush=True)
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
