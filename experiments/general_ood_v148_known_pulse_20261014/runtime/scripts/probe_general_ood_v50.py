"""Separate normal-departure residuals from capped change-point readouts."""
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
from scripts import probe_general_ood_v49 as v49
from scripts import run_general_ood_v40 as v40
from scripts import v44_belief as belief
from scripts.v46_belief import RobustAdaptiveStudent
from scripts.v49_belief import NoiseLikelihoodStudent

REVISION = "general_ood_v50_residual_readout"
ARMS = ("global-robust", "noise-likelihood")
ASSESSMENT = v49.ASSESSMENT
SCENARIOS = {"normal->normal": "normal->normal", "clock-normal->heavy": "heavy->heavy",
             "clock-normal->slippery": "slippery->slippery", "reset-normal->heavy": "reset-normal->heavy",
             "reset-normal->slippery": "reset-normal->slippery"}
def observe(trace, model, startup_noise):
    ages, calibration = v43.episode_steps(trace["dones"]), model.calibrations["global-centered"]
    errors = model.errors(trace)
    scores, thresholds = calibration.scores(errors, ages)
    z = (errors - calibration.means[0]) / calibration.scales[0]
    ratio, signal = scores / thresholds, np.log1p(scores / thresholds)
    trackers = {"global-robust": RobustAdaptiveStudent(**belief.SETTINGS),
                "noise-likelihood": NoiseLikelihoodStudent(**belief.SETTINGS)}
    histories, events = {a: [] for a in ARMS}, {a: [] for a in ARMS}
    for step, value in enumerate(signal):
        for arm, tracker in trackers.items():
            if arm == "global-robust":
                tracker.update(float(value))
            else:
                tracker.update(float(value), startup_noise if ages[step] == 0 else 0.)
            penalty = float(tracker.belief @ np.exp(-.1 * np.arange(tracker.max_H)))
            histories[arm].append((tracker.belief[0], tracker.effective_window, penalty, -2. - penalty, tracker.entropy))
            if tracker.belief[0] > .5:
                events[arm].append(dict(step=step, episode_age=int(ages[step]), signal=float(value),
                    reset_probability=float(tracker.belief[0]), calibration_changed=False,
                    previous_transition_ended=bool(step and trace["dones"][step - 1]),
                    terminated=bool(trace["terminated"][step]), truncated=bool(trace["truncated"][step])))
    histories = {a: np.asarray(h) for a, h in histories.items()}
    arms = {a: dict(phases={p: v44.phase_statistics(h[s:e]) for p, (s, e) in v49.PHASES.items()},
        major_reset_transition_counts={p: int(np.sum(h[s:e, 0] > .5)) for p, (s, e) in v49.PHASES.items()},
        major_reset_events=events[a]) for a, h in histories.items()}
    source = dict(arms=arms, signal={p: float(signal[s:e].mean()) for p, (s, e) in v49.PHASES.items()},
        reward_means_by_256=[float(trace["rewards"][i:i + 256].mean()) for i in range(0, v47.PROBE_STEPS, 256)],
        termination_counts_by_256=[int(trace["terminated"][i:i + 256].sum()) for i in range(0, v47.PROBE_STEPS, 256)],
        truncation_counts_by_256=[int(trace["truncated"][i:i + 256].sum()) for i in range(0, v47.PROBE_STEPS, 256)])
    return dict(z=z, ratio=ratio, signal=signal, ages=ages, histories=histories, source=source)


def windows(trace, observed, switch):
    result = {}
    for phase, (b, e) in ASSESSMENT.items():
        section = slice(switch + b, switch + e)
        z, ratio = observed["z"][section], observed["ratio"][section]
        ongoing = observed["ages"][section] > 0
        assert len(z) == e - b and ongoing.any()
        signed_mean = z.mean(axis=0)
        result[phase] = dict(mean_signal=float(observed["signal"][section].mean()), mean_score_ratio=float(ratio.mean()),
            exceedance_fraction=float(np.mean(ratio > 1)), nonstartup_score_ratio=float(ratio[ongoing].mean()),
            signed_mean_energy=float(np.mean(signed_mean ** 2)), signed_mean=signed_mean.tolist(),
            squared_mean=np.mean(z ** 2, axis=0).tolist(), startup_count=int((~ongoing).sum()),
            terminated=int(trace["terminated"][section].sum()), truncated=int(trace["truncated"][section].sum()),
            beliefs={a: v44.phase_statistics(h[section]) for a, h in observed["histories"].items()})
    return result


def curves(observed, switch):
    section = slice(switch, switch + 50)
    return dict(signal=observed["signal"][section].tolist(), score_ratio=observed["ratio"][section].tolist(),
        penalty={a: h[section, 2].tolist() for a, h in observed["histories"].items()})


def make_record(trace, observed, normal, control, seed, scenario, restart):
    timing = "reset" if scenario.startswith("reset-") else "clock"
    switch = restart if timing == "reset" else v47.FIRST_SWITCH
    assert v41.prefix_equal(normal, trace, switch), "unpaired-physical-prefix"
    assert all(np.array_equal(normal[k][switch], trace[k][switch]) for k in ("states", "executed_actions", "contexts")), "unpaired-switch-inputs"
    assert all(np.array_equal(control["histories"][a][:switch], observed["histories"][a][:switch]) for a in ARMS), "unpaired-belief-prefix"
    diagnostics = ({t: windows(trace, observed, s) for t, s in (("clock", v47.FIRST_SWITCH), ("reset", restart))}
                   if scenario == "normal->normal" else {timing: windows(trace, observed, switch)})
    first50 = ({t: curves(observed, s) for t, s in (("clock", v47.FIRST_SWITCH), ("reset", restart))}
               if scenario == "normal->normal" else {timing: curves(observed, switch)})
    return dict(seed=seed, scenario=scenario, source_v49_scenario=SCENARIOS[scenario], normal_first_restart_step=restart,
        timing=timing, assessed_switch_step=switch, paired_prefix_equal=True, paired_switch_inputs_equal=True,
        paired_belief_prefix_equal=True, interactions=len(trace["dones"]), source_v49=observed["source"],
        diagnostics=diagnostics, first50=first50,
        restart_previous_terminated=bool(normal["terminated"][restart - 1]),
        restart_previous_truncated=bool(normal["truncated"][restart - 1]))


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
        noise, noise_metadata = v49.fit_startup_noise(model, datasets["calibration"])
        metadata = model.metadata()
        del datasets, traces, trace
        print(f"[REFERENCE] {policy}: unchanged normal-only model and startup variance={noise:.4f}", flush=True)
        records = []
        for seed in seeds:
            for scenario in SCENARIOS:
                if scenario == "normal->normal":
                    switches = {}
                else:
                    switches = {restart if scenario.startswith("reset-") else v47.FIRST_SWITCH: scenario.split("->")[1]}
                env = gym.make("Hopper-v4", max_episode_steps=v41.PROBE_STEPS)
                trace = v47.collect_trajectory(agent, env, seed, switches, snapshot)
                env.close()
                observed = observe(trace, model, noise)
                if scenario == "normal->normal":
                    normal, control = trace, observed
                    restart = int(np.flatnonzero(normal["dones"])[0]) + 1
                    assert restart + 256 <= v47.PROBE_STEPS
                records.append(make_record(trace, observed, normal, control, seed, scenario, restart))
            print(f"[RESIDUAL-READOUT] {policy} seed={seed}: normal + clock/reset heavy/slippery", flush=True)
        assert model.metadata() == metadata
        agent.restore_context(deepcopy(snapshot))
        assert agent.context_tracker._count == 100000 and agent.training_steps == 99745 and agent.replay_buffer.size == 0
        output = dict(revision=REVISION, stage=args.stage, source_policy=policy, source_checkpoint=checkpoint.name,
            source_training_seed=1, config=asdict(agent.config), references=references, model_metadata=metadata, noise_metadata=noise_metadata,
            fit_seeds=fit_seeds, calibration_seeds=calibration_seeds, evaluation_seeds=seeds, arms=ARMS, scenarios=SCENARIOS,
            assessment_offsets=ASSESSMENT, clock_switch_step=v47.FIRST_SWITCH, probe_steps=v47.PROBE_STEPS,
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
        print(f"[DONE] residual readout {policy}: result={path}", flush=True)
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
