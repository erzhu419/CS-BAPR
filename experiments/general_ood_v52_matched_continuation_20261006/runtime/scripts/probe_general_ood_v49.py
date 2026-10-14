"""Known-startup uncertainty and true switches at the first observable reset."""
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
from scripts import probe_general_ood_v48 as v48
from scripts import run_general_ood_v40 as v40
from scripts import v44_belief as belief
from scripts.v46_belief import RobustAdaptiveStudent
from scripts.v49_belief import NoiseLikelihoodStudent, NoiseWeightedStudent

REVISION = "general_ood_v49_startup_noise"
ARMS = ("global-robust", "noise-likelihood", "noise-weighted-history")
PHASES = v48.PHASES
ASSESSMENT = {"first16": (0, 16), "first50": (0, 50), "first256": (0, 256), "last64": (192, 256)}
NOISE_RULE = "max(normal startup signal MSE about ongoing mean minus ongoing variance,0)"


def fit_startup_noise(model, calibration):
    ages = calibration["episode_steps"]
    scores, thresholds = model.calibrations["global-centered"].scores(model.errors(calibration), ages)
    signal = np.log1p(scores / thresholds)
    startup, ongoing = signal[ages == 0], signal[ages > 0]
    assert len(startup) and len(ongoing)
    reference_mean, reference_variance = float(ongoing.mean()), float(ongoing.var())
    startup_mse = float(np.mean((startup - reference_mean) ** 2))
    noise = max(startup_mse - reference_variance, 0.)
    return noise, dict(normal_start_count=len(startup), ongoing_count=len(ongoing),
        ongoing_mean_signal=reference_mean, ongoing_signal_variance=reference_variance,
        startup_mean_signal=float(startup.mean()), startup_discrepancy_mse=startup_mse,
        startup_extra_variance=noise, startup_history_precision=belief.SETTINGS["base_variance"] / (belief.SETTINGS["base_variance"] + noise))


def window_statistics(histories, start):
    return {a: {p: v44.phase_statistics(v[start + b:start + e]) for p, (b, e) in ASSESSMENT.items()}
            for a, v in histories.items()}


def observe(trace, model, startup_noise):
    ages = v43.episode_steps(trace["dones"])
    scores, thresholds = model.calibrations["global-centered"].scores(model.errors(trace), ages)
    signal = np.log1p(scores / thresholds)
    trackers = dict(zip(ARMS, (RobustAdaptiveStudent(**belief.SETTINGS),
        NoiseLikelihoodStudent(**belief.SETTINGS), NoiseWeightedStudent(**belief.SETTINGS))))
    metrics, events, startups = {a: [] for a in ARMS}, {a: [] for a in ARMS}, []
    for step, value in enumerate(signal):
        noise = startup_noise if ages[step] == 0 else 0.
        cf = None
        if ages[step] == 0:
            original = trackers["global-robust"]
            cf = NoiseLikelihoodStudent(**belief.SETTINGS)
            cf.belief = original.belief.copy()
            cf.history.extend(original.history)
            cf.update(float(value), noise)
        for arm, tracker in trackers.items():
            if arm == "global-robust":
                tracker.update(float(value))
            else:
                tracker.update(float(value), noise)
            penalty = float(tracker.belief @ np.exp(-.1 * np.arange(tracker.max_H)))
            metrics[arm].append((tracker.belief[0], tracker.effective_window, penalty, -2. - penalty, tracker.entropy))
            if tracker.belief[0] > .5:
                events[arm].append(dict(step=step, episode_age=int(ages[step]), signal=float(value),
                    reset_probability=float(tracker.belief[0]), calibration_changed=False,
                    previous_transition_ended=bool(step and trace["dones"][step - 1]),
                    terminated=bool(trace["terminated"][step]), truncated=bool(trace["truncated"][step])))
        if cf is not None:
            startups.append(dict(step=step, signal=float(value), observation_variance=float(noise),
                previous_terminated=bool(step and trace["terminated"][step - 1]), previous_truncated=bool(step and trace["truncated"][step - 1]),
                original_reset_probability=float(trackers["global-robust"].belief[0]), same_past_noisy_reset_probability=float(cf.belief[0]),
                arm_reset_probabilities={a: float(t.belief[0]) for a, t in trackers.items()}))
    histories = {a: np.asarray(v) for a, v in metrics.items()}
    arms = {a: dict(phases={p: v44.phase_statistics(v[s:e]) for p, (s, e) in PHASES.items()},
        major_reset_transition_counts={p: int(np.sum(v[s:e, 0] > .5)) for p, (s, e) in PHASES.items()},
        major_reset_events=events[a]) for a, v in histories.items()}
    checks, _ = v43.observe_trace(trace, {v44.PREDICTOR: model})
    return arms, histories, signal, startups, checks[v44.PREDICTOR][v44.CALIBRATOR]


def make_record(trace, observed, control, control_histories, switch_step, scenario, seed, initial, target, normal_restart):
    arms, histories, signal, startups, check = observed
    assert v41.prefix_equal(control, trace, switch_step), "unpaired-assessment-prefix"
    assert all(np.array_equal(control[k][switch_step], trace[k][switch_step])
               for k in ("states", "executed_actions", "contexts")), "unpaired-switch-inputs"
    assert all(np.array_equal(control_histories[a][:switch_step], histories[a][:switch_step]) for a in ARMS), "unpaired-filter-prefix"
    return dict(seed=seed, scenario=scenario, initial_mode=initial, target_mode=target, assessed_switch_step=switch_step,
        normal_first_restart_step=normal_restart, paired_prefix_equal=True, paired_switch_inputs_equal=True, paired_belief_prefix_equal=True,
        interactions=len(trace["dones"]), arms=arms, startup_observations=startups, v43_score_check=check,
        signal={p: float(signal[s:e].mean()) for p, (s, e) in PHASES.items()},
        assessment=window_statistics(histories, switch_step), control_assessment=window_statistics(control_histories, switch_step),
        normal_restart_assessment=window_statistics(histories, normal_restart) if scenario == "normal->normal" else None,
        paired_first50_penalty={a: (histories[a][switch_step:switch_step + 50, 2] - control_histories[a][switch_step:switch_step + 50, 2]).tolist() for a in ARMS},
        assessment_endings={p: {k: int(trace[k][switch_step + b:switch_step + e].sum()) for k in ("terminated", "truncated")} for p, (b, e) in ASSESSMENT.items()},
        control_assessment_endings={p: {k: int(control[k][switch_step + b:switch_step + e].sum()) for k in ("terminated", "truncated")} for p, (b, e) in ASSESSMENT.items()},
        reward_means_by_256=[float(trace["rewards"][i:i + 256].mean()) for i in range(0, v47.PROBE_STEPS, 256)],
        termination_counts_by_256=[int(trace["terminated"][i:i + 256].sum()) for i in range(0, v47.PROBE_STEPS, 256)],
        truncation_counts_by_256=[int(trace["truncated"][i:i + 256].sum()) for i in range(0, v47.PROBE_STEPS, 256)])


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
        noise, noise_metadata = fit_startup_noise(model, datasets["calibration"])
        metadata = model.metadata()
        del datasets, traces, trace
        print(f"[REFERENCE] {policy}: startup variance budget={noise:.4f}, precision={noise_metadata['startup_history_precision']:.4f}", flush=True)
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
                    observed = observe(trace, model, noise)
                    if initial == "normal":
                        normal, normal_histories = trace, observed[1]
                        normal_restart = int(np.flatnonzero(normal["dones"])[0]) + 1
                        assert normal_restart + 256 <= v47.PROBE_STEPS
                    assert v41.prefix_equal(normal, trace, v47.FIRST_SWITCH), "unpaired-normal-prefix"
                    if target == initial:
                        control, control_histories = trace, observed[1]
                    records.append(make_record(trace, observed, control, control_histories, v47.SECOND_SWITCH,
                        f"{initial}->{target}", seed, initial, target, normal_restart))
            for target in v47.INITIAL_MODES:
                env = gym.make("Hopper-v4", max_episode_steps=v41.PROBE_STEPS)
                trace = v47.collect_trajectory(agent, env, seed, {normal_restart: target}, snapshot)
                env.close()
                assert normal["dones"][normal_restart - 1] and trace["dones"][normal_restart - 1]
                records.append(make_record(trace, observe(trace, model, noise), normal, normal_histories, normal_restart,
                    f"reset-normal->{target}", seed, "normal", target, normal_restart))
            print(f"[STARTUP-NOISE] {policy} seed={seed}: seven legacy + two reset-switch branches", flush=True)
        assert model.metadata() == metadata
        agent.restore_context(deepcopy(snapshot))
        assert agent.context_tracker._count == 100000 and agent.training_steps == 99745 and agent.replay_buffer.size == 0
        output = dict(revision=REVISION, stage=args.stage, source_policy=policy, source_checkpoint=checkpoint.name,
            source_training_seed=1, config=asdict(agent.config), references=references, model_metadata=metadata, noise_metadata=noise_metadata,
            fit_seeds=fit_seeds, calibration_seeds=calibration_seeds, evaluation_seeds=seeds, arms=ARMS, phases=PHASES, assessment_offsets=ASSESSMENT,
            noise_rule=NOISE_RULE, first_switch=v47.FIRST_SWITCH, second_switch=v47.SECOND_SWITCH, probe_steps=v47.PROBE_STEPS,
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
        print(f"[DONE] startup-noise {policy}: result={path}", flush=True)
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
