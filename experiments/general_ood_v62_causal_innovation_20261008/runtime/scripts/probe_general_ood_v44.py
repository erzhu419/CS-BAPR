"""Frozen-policy replay: fixed versus learned residual-level belief likelihoods."""
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
from scripts import run_general_ood_v40 as v40
from scripts import v44_belief as belief
from scripts.train_csbapr import MUJOCO_MODE_PROFILES

REVISION = "general_ood_v44_frozen_belief"
PREDICTOR, CALIBRATOR = "quadratic-state-action", "episode-centered"
PHASES = dict(v42.PHASES, early_post64=(1024, 1088), late_first64=(1216, 1280), late_second64=(1472, 1536))
METRICS = ("reset_probability", "effective_age", "penalty", "hypothetical_beta", "entropy")


def phase_statistics(values):
    return {key: {"mean": float(values[:, i].mean()), "peak": float(values[:, i].max()),
                  "end": float(values[-1, i])} for i, key in enumerate(METRICS)}


def observe(trace, model):
    scores, thresholds = model.calibrations[CALIBRATOR].scores(model.errors(trace), v43.episode_steps(trace["dones"]))
    signal, traces = belief.replay(scores, thresholds)
    observers = {name: {phase: phase_statistics(values[start:end]) for phase, (start, end) in PHASES.items()}
                 for name, values in traces.items()}
    bins = {name: [{key: float(values[i:i + 32, j].mean()) for j, key in enumerate(METRICS)}
                   for i in range(1024, 1536, 32)]
            for name, values in traces.items()}
    signal_stats = {phase: {"mean": float(signal[start:end].mean()), "peak": float(signal[start:end].max()),
                            "end": float(signal[end - 1])} for phase, (start, end) in PHASES.items()}
    # Retain v43's exact score summaries to detect source/model drift without pulling trajectories.
    score_check, _ = v43.observe_trace(trace, {PREDICTOR: model})
    return observers, bins, signal_stats, score_check[PREDICTOR][CALIBRATOR]


def run(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(1)
    np.random.seed(1)
    fitting_seeds, calibration_seeds, seeds = v43.stage_seeds(args.stage)
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
        for split, ref_seeds in (("fit", fitting_seeds), ("calibration", calibration_seeds)):
            traces = []
            for seed in ref_seeds:
                env = gym.make("Hopper-v4", max_episode_steps=v41.PROBE_STEPS)
                trace = v41.collect_trajectory(agent, env, seed, "normal", snapshot, budget=v43.REFERENCE_STEPS)
                env.close()
                traces.append(trace)
                references.append(v43.reference_record(trace, seed, split))
            datasets[split] = v43.concatenate_references(traces)
        model = v43.ReferenceDynamics(*v42.PREDICTORS[PREDICTOR]).fit_references(datasets["fit"], datasets["calibration"])
        metadata = model.metadata()
        del datasets, traces, trace
        print(f"[REFERENCE] {policy}: frozen normal dynamics calibrated", flush=True)
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
                observers, bins, signal, score_check = observe(trace, model)
                endings = {phase: {"terminated": int(trace["terminated"][start:end].sum()),
                                   "truncated": int(trace["truncated"][start:end].sum())}
                           for phase, (start, end) in PHASES.items()}
                records.append(dict(seed=seed, target_mode=mode, actual_change=mode != "normal",
                                    interactions=len(trace["dones"]), paired_prefix_equal=True, paired_switch_inputs_equal=True,
                                    observers=observers, post_bins32=bins, signal=signal, v43_score_check=score_check,
                                    phase_endings=endings,
                                    reward_means_by_256=[float(trace["rewards"][i:i + 256].mean()) for i in range(0, 1536, 256)],
                                    termination_counts_by_256=[int(trace["terminated"][i:i + 256].sum()) for i in range(0, 1536, 256)],
                                    truncation_counts_by_256=[int(trace["truncated"][i:i + 256].sum()) for i in range(0, 1536, 256)]))
            print(f"[REPLAY] {policy} seed={seed}: {len(targets)} modes, three likelihoods", flush=True)
        assert model.metadata() == metadata
        agent.restore_context(deepcopy(snapshot))
        assert agent.context_tracker._count == 100000 and agent.training_steps == 99745 and agent.replay_buffer.size == 0
        output = dict(revision=REVISION, stage=args.stage, environment="Hopper-v4", source_policy=policy,
                      source_checkpoint=checkpoint.name, source_training_seed=1, config=asdict(agent.config),
                      loaded_training_update_steps=agent.training_steps, loaded_training_context_count=agent.context_tracker._count,
                      fit_seeds=fitting_seeds, calibration_seeds=calibration_seeds, evaluation_seeds=seeds, target_modes=list(targets),
                      predictor=PREDICTOR, calibrator=CALIBRATOR, filter_names=list(belief.FILTERS), filter_settings=belief.SETTINGS,
                      signal_transform="log1p(score / calibration95)", phases=PHASES, post_bin_width=32,
                      reset_filters_at_episode_end=False, model_metadata=metadata, references=references, records=records,
                      optimizer_updates_during_probe=0, replay_insertions=0, auxiliary_signal_controls_actions=False,
                      predictors_updated_after_calibration=False, actor_critic_target_frozen=True,
                      total_reference_interactions=len(references) * v43.REFERENCE_STEPS,
                      total_probe_interactions=len(records) * v41.PROBE_STEPS, seconds=time.monotonic() - started,
                      runtime=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__, gymnasium=gym.__version__,
                                   mujoco=__import__("mujoco").__version__, gpu=torch.cuda.get_device_name()))
        path = args.result_dir / f"{REVISION}_{policy}.json"
        path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
        print(f"[DONE] belief replay {policy}: result={path}", flush=True)
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
