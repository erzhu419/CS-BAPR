"""Learning-matched age calibration with fixed v54 scores and covariance."""
from copy import deepcopy
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import numpy as np
import torch

from csbapr.belief.departure import NormalAgeReference, SETTINGS
from scripts import probe_general_ood_v53_behavior_coverage as v53
from scripts.v52_agent import ContinuationAgent
from scripts.v54_window_readout import normalized_residuals, observe_trace, rolling_means

REVISION = "general_ood_v55_learning_calibration"
OBSERVERS = ("v54_pooled", "learning_calibrated")


def stage_spec(stage):
    if stage == "preflight":
        return dict(audit=[98410], calibration=[98600, 98601], test=[98620, 98621], steps=1200, switch_step=800)
    return dict(audit=[], calibration=[98700, 98701], test=list(range(98720, 98724)), steps=1200, switch_step=800)


def trajectory_specs(spec):
    roles = [set(spec[k]) for k in ("audit", "calibration", "test")]
    assert not (roles[0] & roles[1] or roles[0] & roles[2] or roles[1] & roles[2])
    return ([dict(split="audit", seed=seed, updating=updating, mode="normal")
             for seed in spec["audit"] for updating in (False, True)] +
            [dict(split="calibration", seed=seed, updating=True, mode="normal") for seed in spec["calibration"]] +
            [dict(split="test", seed=seed, updating=True, mode=mode) for seed in spec["test"] for mode in v53.MODES])


def signals_and_ages(model, reference, trace):
    scores = reference.scores(rolling_means(normalized_residuals(model, trace)))
    return np.log1p(scores / reference.score95), v53.v43.episode_steps(trace["dones"])


def calibrate_learning(model, original, traces):
    reference = deepcopy(original)
    values = [signals_and_ages(model, original, trace) for trace in traces]
    signal, ages = (np.concatenate([v[i] for v in values]) for i in (0, 1))
    reference.age_reference = NormalAgeReference.fit_normal(signal, ages)
    reference.calibration_transitions = len(signal)
    return reference


def audit_pooled_calibration(model, reference, traces):
    values = [signals_and_ages(model, reference, trace) for trace in traces]
    signal, ages = (np.concatenate([v[i] for v in values]) for i in (0, 1))
    reproduced = NormalAgeReference.fit_normal(signal, ages)
    np.testing.assert_array_equal(reproduced.ages, reference.age_reference.ages)
    np.testing.assert_allclose(reproduced.upper, reference.age_reference.upper, rtol=0., atol=1e-12)
    records = []
    for age in (800, 816, 832, 848):
        groups = [s[np.abs(a - age) <= SETTINGS["age_radius"]] for s, a in values]
        records.append(dict(age=age, pooled_threshold=float(reference.age_reference.threshold(age)),
            frozen_q95=float(np.quantile(groups[0], .95)), learning_q95=float(np.quantile(groups[1], .95)),
            frozen_max=float(groups[0].max()), learning_max=float(groups[1].max()),
            pooled_threshold_exceeds_learning_max=bool(reference.age_reference.threshold(age) > groups[1].max())))
    return dict(old_calibration_seed=98410, age_reference_reproduced_atol=1e-12,
                data_used_in_new_calibration=False, first50_age_knots=records)


def summarize(records):
    normal = [r for r in records if r["split"] == "test" and r["mode"] == "normal"]
    changed = [r for r in records if r["split"] == "test" and r["mode"] != "normal"]
    null_pass = all(r["observers"]["learning_calibrated"][w]["mean_coefficient"] <= .1
                    for r in normal for w in ("full", "last64"))
    separation = all(r["change_minus_nochange"]["learning_calibrated"][w] > 0
                     for r in changed for w in ("first_post50", "last64"))
    return dict(usable_learning_normal_null=null_pass, all_paired_first50_and_tail_positive=separation,
        normal_means={name: {w: [r["observers"][name][w]["mean_coefficient"] for r in normal]
                            for w in ("full", "last64")} for name in OBSERVERS},
        decision="learning-calibration-diagnostic-pass-not-RL-confirmation" if null_pass and separation
                 else "hold-learning-calibration")


def run(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    spec, source = stage_spec(args.stage), args.source_dir / v53.SOURCE_NAME
    cells = trajectory_specs(spec)
    shared = torch.load(args.shared_dir / "shared.pt", map_location="cpu", weights_only=False)
    dynamics = torch.load(args.dynamics_checkpoint, map_location="cpu", weights_only=False)["extra_state"]["progress"]
    signed = torch.load(args.signed_checkpoint, map_location="cpu", weights_only=False)["extra_state"]["progress"]
    assert dynamics["stage"] == signed["stage"] == "preflight"
    assert dynamics["next_cell"] == 9 and signed["next_cell"] == 14
    model, pooled_reference = dynamics["models"]["broad"][0], signed["reference"]
    del dynamics, signed
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint_dir / f"{args.stage}_latest.pt"
    extra = torch.load(args.resume_from, map_location="cpu", weights_only=False)["extra_state"] if args.resume_from else None
    progress = extra["progress"] if extra else dict(stage=args.stage, next_cell=0, audit=[], calibration=[],
        audit_metadata=None, reference=None, records=[], control=None)
    assert progress["stage"] == args.stage
    for index in range(progress["next_cell"], len(cells)):
        cell = cells[index]
        if cell["split"] != "audit" and progress["audit"]:
            progress["audit_metadata"] = audit_pooled_calibration(model, pooled_reference, progress["audit"])
            progress["audit"] = []
            print("[AUDIT] Old mixed calibration reproduced; new calibration remains separate", flush=True)
        if cell["split"] == "test" and progress["reference"] is None:
            progress["reference"] = calibrate_learning(model, pooled_reference, progress["calibration"])
            progress["calibration"] = []
            print("[REFERENCE] Learning-normal age thresholds frozen; covariance and score scale unchanged", flush=True)
        env = gym.make("Hopper-v4")
        agent = ContinuationAgent(env, "fixed-minus2")
        agent.start_shared(source, shared)
        assert agent.training_steps == 99745 and agent.context_tracker._count == 100000
        loop = None
        if extra and extra["loop"] is not None:
            agent.load(str(args.resume_from))
            loop = extra["loop"]
            mode = cell["mode"] if cell["split"] == "test" and loop["step"] > spec["switch_step"] else "normal"
            v53.apply_mode_to_env(env, "Hopper-v4", v53.MUJOCO_MODE_PROFILES["Hopper-v4"][mode])
            env.reset(seed=cell["seed"])
            v53.restore_environment(env, extra["environment"])
        trace, record = v53.collect(agent, env, cell, spec, loop,
            lambda active: v53.save_progress(agent, env, progress, active, checkpoint))
        if cell["split"] != "test":
            progress[cell["split"]].append(trace)
            record["reference_record"] = v53.v43.reference_record(trace, cell["seed"], cell["split"])
        else:
            record["observers"] = dict(v54_pooled=observe_trace(trace, model, pooled_reference, spec["switch_step"]),
                learning_calibrated=observe_trace(trace, model, progress["reference"], spec["switch_step"]))
            if cell["mode"] == "normal":
                progress["control"] = (trace, deepcopy(record))
            control, baseline = progress["control"]
            assert baseline["seed"] == cell["seed"] and v53.paired_prefix_equal(trace, control, spec["switch_step"])
            record["paired_prefix_equal"] = True
            record["change_minus_nochange"] = {name: {w: values["mean_coefficient"] -
                baseline["observers"][name][w]["mean_coefficient"] for w, values in windows.items()}
                for name, windows in record["observers"].items()}
        progress["records"].append(record)
        progress["next_cell"] = index + 1
        v53.save_progress(agent, env, progress, None, checkpoint)
        print(f"[CELL] {index+1}/{len(cells)}: {cell}, updates={record['optimizer_updates']}", flush=True)
        extra = None
        env.close()
        del agent
        torch.cuda.empty_cache()
    reference = progress["reference"]
    for name in ("center", "covariance", "precision"):
        np.testing.assert_array_equal(getattr(reference, name), getattr(pooled_reference, name))
    assert reference.score95 == pooled_reference.score95 and reference.ridge == pooled_reference.ridge
    records = progress["records"]
    output = dict(revision=REVISION, stage=args.stage, stage_spec=spec, source_checkpoint=v53.SOURCE_NAME,
        source_training_seed=1, source_updates=99745, shared_replay_size=shared["replay"].size,
        action_sampling="stochastic", beta=-2., auxiliary_controls_actions=False, episode_time_limit=1000,
        dynamics_refitted=False, moments_covariance_and_global_score95_changed=False,
        calibration_population="normal-online-learning-only", old_calibration_audit=progress["audit_metadata"],
        reference_frozen_during_test=True, environment_interactions=sum(r["interactions"] for r in records),
        optimizer_updates=sum(r["optimizer_updates"] for r in records),
        dynamics_metadata=model.metadata(), pooled_reference_metadata=pooled_reference.metadata(),
        learning_reference_metadata=reference.metadata(), records=records, summary=summarize(records),
        limitations=["Calibration targets online learning, not a frozen-policy deployment detector.",
                     "One source training seed; conditional detector probes under fixed beta, not adaptive RL or 50x OOD return confirmation."])
    args.result_dir.mkdir(parents=True, exist_ok=True)
    destination = args.result_dir / "learning_calibration.json"
    destination.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(f"[DONE] {output['summary']['decision']}: {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "formal"), required=True)
    for name in ("source-dir", "shared-dir", "dynamics-checkpoint", "signed-checkpoint", "checkpoint-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
