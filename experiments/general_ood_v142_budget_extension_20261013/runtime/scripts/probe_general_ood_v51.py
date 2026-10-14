"""Age-calibrated departure readout with new normal nulls; no RL feedback."""
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

from csbapr.belief.departure import DepartureReadout, NormalAgeReference, SETTINGS
from scripts import probe_general_ood_v41 as v41
from scripts import probe_general_ood_v42 as v42
from scripts import probe_general_ood_v43 as v43
from scripts import probe_general_ood_v44 as v44
from scripts import probe_general_ood_v47 as v47
from scripts import probe_general_ood_v49 as v49
from scripts import probe_general_ood_v50 as v50
from scripts import run_general_ood_v40 as v40
from scripts import v44_belief as belief

REVISION = "general_ood_v51_departure_penalty"
READOUTS = ("global-departure", "age-departure")


def observe_readouts(observed, reference):
    trackers = {"global-departure": DepartureReadout(NormalAgeReference.global95()),
                "age-departure": DepartureReadout(reference)}
    rows = {a: [] for a in READOUTS}
    for signal, age in zip(observed["signal"], observed["ages"]):
        for arm, tracker in trackers.items():
            tracker.update(signal, age)
            rows[arm].append((tracker.risk, tracker.instant_risk, tracker.threshold, tracker.beta))
    return {a: np.asarray(v) for a, v in rows.items()}


def statistics(values):
    return dict(mean_risk=float(values[:, 0].mean()), peak_risk=float(values[:, 0].max()),
        positive_instant_fraction=float(np.mean(values[:, 1] > 0)), mean_instant_risk=float(values[:, 1].mean()),
        mean_threshold=float(values[:, 2].mean()), mean_beta=float(values[:, 3].mean()))


def window_readouts(histories, switch):
    return {a: {p: statistics(h[switch + b:switch + e]) for p, (b, e) in v50.ASSESSMENT.items()}
            for a, h in histories.items()}


def null_record(trace, observed, histories, seed):
    restarts = (np.flatnonzero(trace["dones"]) + 1).tolist()
    return dict(seed=seed, interactions=len(trace["dones"]),
        full={a: statistics(h) for a, h in histories.items()},
        initial=window_readouts(histories, 0),
        restarts=[dict(step=s, previous_terminated=bool(trace["terminated"][s - 1]),
                      previous_truncated=bool(trace["truncated"][s - 1]),
                      first50={a: statistics(h[s:s + 50]) for a, h in histories.items()})
                  for s in restarts if s + 50 <= len(trace["dones"])],
        physical_terminations=int(trace["terminated"].sum()), time_limit_truncations=int(trace["truncated"].sum()),
        original_age_full_penalty=observed["source"]["arms"]["noise-likelihood"]["phases"]["full1792"]["penalty"]["mean"])


def make_record(source, histories):
    switches = {"clock": v47.FIRST_SWITCH, "reset": source["normal_first_restart_step"]}
    timings = ("clock", "reset") if source["scenario"] == "normal->normal" else (source["timing"],)
    return dict(source_v50=source,
        readouts={t: window_readouts(histories, switches[t]) for t in timings},
        first50={t: {a: h[switches[t]:switches[t] + 50, 0].tolist() for a, h in histories.items()} for t in timings},
        full={a: statistics(h) for a, h in histories.items()})


def run(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(1)
    np.random.seed(1)
    fit_seeds, calibration_seeds, seeds = v43.stage_seeds(args.stage)
    null_seeds = list(range(99300, 99302)) if args.stage == "preflight" else list(range(90100, 90120))
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
        calibration = datasets["calibration"]
        scores, thresholds = model.calibrations["global-centered"].scores(model.errors(calibration), calibration["episode_steps"])
        reference = NormalAgeReference.fit_normal(np.log1p(scores / thresholds), calibration["episode_steps"])
        metadata, risk_metadata = model.metadata(), reference.metadata()
        del datasets, traces, trace, calibration
        print(f"[REFERENCE] {policy}: normal age knots={len(reference.ages)}, starts={reference.normal_start_count}", flush=True)
        records, old_nulls, new_nulls = [], [], []
        for seed in seeds:
            for scenario in v50.SCENARIOS:
                switches = {} if scenario == "normal->normal" else {
                    restart if scenario.startswith("reset-") else v47.FIRST_SWITCH: scenario.split("->")[1]}
                env = gym.make("Hopper-v4", max_episode_steps=v41.PROBE_STEPS)
                trace = v47.collect_trajectory(agent, env, seed, switches, snapshot)
                env.close()
                observed = v50.observe(trace, model, noise)
                histories = observe_readouts(observed, reference)
                if scenario == "normal->normal":
                    normal, control, normal_readouts = trace, observed, histories
                    restart = int(np.flatnonzero(normal["dones"])[0]) + 1
                    assert restart + 256 <= v47.PROBE_STEPS
                    old_nulls.append(null_record(trace, observed, histories, seed))
                source = v50.make_record(trace, observed, normal, control, seed, scenario, restart)
                switch = source["assessed_switch_step"]
                assert all(np.array_equal(normal_readouts[a][:switch], histories[a][:switch]) for a in READOUTS), "unpaired-readout-prefix"
                records.append(make_record(source, histories))
            print(f"[DEPARTURE-PENALTY] {policy} seed={seed}: paired clock/reset readouts", flush=True)
        for seed in null_seeds:
            env = gym.make("Hopper-v4", max_episode_steps=v41.PROBE_STEPS)
            trace = v47.collect_trajectory(agent, env, seed, {}, snapshot)
            env.close()
            observed = v50.observe(trace, model, noise)
            new_nulls.append(null_record(trace, observed, observe_readouts(observed, reference), seed))
            print(f"[NEW-NORMAL] {policy} seed={seed}: untouched null trajectory", flush=True)
        assert model.metadata() == metadata and reference.metadata() == risk_metadata
        agent.restore_context(deepcopy(snapshot))
        assert agent.context_tracker._count == 100000 and agent.training_steps == 99745 and agent.replay_buffer.size == 0
        output = dict(revision=REVISION, stage=args.stage, source_policy=policy, source_checkpoint=checkpoint.name,
            source_training_seed=1, config=asdict(agent.config), references=references, model_metadata=metadata,
            noise_metadata=noise_metadata, risk_metadata=risk_metadata, readout_settings=SETTINGS, readouts=READOUTS,
            fit_seeds=fit_seeds, calibration_seeds=calibration_seeds, evaluation_seeds=seeds, null_seeds=null_seeds,
            scenarios=v50.SCENARIOS, assessment_offsets=v50.ASSESSMENT, probe_steps=v47.PROBE_STEPS, episode_time_limit=v41.PROBE_STEPS,
            filter_settings=belief.SETTINGS, robust_student_degrees=4, optimizer_updates_during_probe=0, replay_insertions=0,
            auxiliary_signal_controls_actions=False, reset_filters_at_episode_end=False, predictors_updated_after_calibration=False,
            actor_critic_target_frozen=True, risk_reference_updated_after_calibration=False, risk_history_reset_at_episode_end=False,
            loaded_training_update_steps=agent.training_steps, loaded_training_context_count=agent.context_tracker._count,
            records=records, reused_normal=old_nulls, new_normal=new_nulls,
            total_reference_interactions=len(references) * v43.REFERENCE_STEPS,
            total_probe_interactions=(len(records) + len(new_nulls)) * v47.PROBE_STEPS, seconds=time.monotonic() - started,
            runtime=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__, gymnasium=gym.__version__,
                         mujoco=__import__("mujoco").__version__, gpu=torch.cuda.get_device_name()))
        path = args.result_dir / f"{REVISION}_{policy}.json"
        path.write_text(json.dumps(output, separators=(",", ":"), allow_nan=False) + "\n")
        print(f"[DONE] departure penalty {policy}: result={path}", flush=True)
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
