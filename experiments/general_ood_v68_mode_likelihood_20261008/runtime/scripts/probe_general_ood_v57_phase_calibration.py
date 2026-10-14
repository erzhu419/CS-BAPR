"""Phase-conditioned normal calibration against same-data pooled controls."""
from copy import deepcopy
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import numpy as np
import torch

from scripts import probe_general_ood_v53_behavior_coverage as v53
from scripts.v52_agent import ContinuationAgent
from scripts.v54_window_readout import observe_trace
from scripts.v57_phase_readout import fit_references, observe_trace as observe_phase

REVISION = "general_ood_v57_phase_calibration"
OBSERVERS = ("v55_frozen", "matched_pooled", "phase_calibrated")


def stage_spec(stage):
    if stage == "preflight":
        return dict(calibration=[98800, 98801], test=[98820, 98821], calibration_steps=2000, steps=1200, switch_step=800)
    return dict(calibration=[98900, 98901], test=list(range(98920, 98924)), calibration_steps=2000, steps=1200, switch_step=800)


def trajectory_specs(spec):
    assert not set(spec["calibration"]) & set(spec["test"])
    return ([dict(split="calibration", seed=seed, updating=True, mode="normal") for seed in spec["calibration"]] +
            [dict(split="test", seed=seed, updating=True, mode=mode) for seed in spec["test"] for mode in v53.MODES])


def summarize(records):
    normal = [r for r in records if r["split"] == "test" and r["mode"] == "normal"]
    changed = [r for r in records if r["split"] == "test" and r["mode"] != "normal"]
    null_pass = all(r["observers"]["phase_calibrated"][w]["mean_coefficient"] <= .1
                    for r in normal for w in ("full", "last64"))
    separation = all(r["change_minus_nochange"]["phase_calibrated"][w] > 0
                     for r in changed for w in ("first_post50", "last64"))
    return dict(usable_learning_normal_null=null_pass, all_paired_first50_and_tail_positive=separation,
        normal_means={name: {w: [r["observers"][name][w]["mean_coefficient"] for r in normal]
                            for w in ("full", "last64")} for name in OBSERVERS},
        positive_pair_counts={name: {w: sum(r["change_minus_nochange"][name][w] > 0 for r in changed)
                                    for w in ("first_post50", "last64")} for name in OBSERVERS},
        changed_pairs=len(changed),
        decision="phase-calibration-diagnostic-pass-not-RL-confirmation" if null_pass and separation
                 else "hold-phase-calibration")


def run(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    spec, source = stage_spec(args.stage), args.source_dir / v53.SOURCE_NAME
    cells = trajectory_specs(spec)
    shared = torch.load(args.shared_dir / "shared.pt", map_location="cpu", weights_only=False)
    dynamics = torch.load(args.dynamics_checkpoint, map_location="cpu", weights_only=False)["extra_state"]["progress"]
    completed = torch.load(args.reference_checkpoint, map_location="cpu", weights_only=False)["extra_state"]["progress"]
    assert dynamics["stage"] == "preflight" and dynamics["next_cell"] == 9
    assert completed["stage"] == "formal" and completed["next_cell"] == 22
    model, frozen = dynamics["models"]["broad"][0], completed["reference"]
    del dynamics, completed
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint_dir / f"{args.stage}_latest.pt"
    extra = torch.load(args.resume_from, map_location="cpu", weights_only=False)["extra_state"] if args.resume_from else None
    progress = extra["progress"] if extra else dict(stage=args.stage, next_cell=0, calibration=[],
        references=None, records=[], control=None)
    assert progress["stage"] == args.stage
    for index in range(progress["next_cell"], len(cells)):
        cell = cells[index]
        if cell["split"] == "test" and progress["references"] is None:
            pooled, phase = fit_references(model, frozen, progress["calibration"])
            progress["references"] = dict(matched_pooled=pooled, phase_calibrated=phase)
            progress["calibration"] = []
            print("[REFERENCE] Same-data pooled and phase curves frozen; both phases cover ages0--999", flush=True)
        cell_spec = dict(spec, steps=spec["calibration_steps"] if cell["split"] == "calibration" else spec["steps"])
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
        trace, record = v53.collect(agent, env, cell, cell_spec, loop,
            lambda active: v53.save_progress(agent, env, progress, active, checkpoint))
        if cell["split"] == "calibration":
            progress["calibration"].append(trace)
            record["reference_record"] = v53.v43.reference_record(trace, cell["seed"], cell["split"])
        else:
            refs = progress["references"]
            pooled = refs["matched_pooled"]
            record["observers"] = dict(v55_frozen=observe_trace(trace, model, frozen, spec["switch_step"]),
                matched_pooled=observe_trace(trace, model, pooled, spec["switch_step"]),
                phase_calibrated=observe_phase(trace, model, pooled, refs["phase_calibrated"], spec["switch_step"]))
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
    pooled = progress["references"]["matched_pooled"]
    for name in ("center", "covariance", "precision"):
        np.testing.assert_array_equal(getattr(pooled, name), getattr(frozen, name))
    assert pooled.score95 == frozen.score95 and pooled.ridge == frozen.ridge
    records = progress["records"]
    output = dict(revision=REVISION, stage=args.stage, stage_spec=spec, source_checkpoint=v53.SOURCE_NAME,
        source_training_seed=1, source_updates=99745, shared_replay_size=shared["replay"].size,
        action_sampling="stochastic", beta=-2., auxiliary_controls_actions=False, episode_time_limit=1000,
        dynamics_refitted=False, moments_covariance_and_global_score95_changed=False,
        matched_control_uses_same_calibration=True, reference_frozen_during_test=True,
        environment_interactions=sum(r["interactions"] for r in records),
        optimizer_updates=sum(r["optimizer_updates"] for r in records),
        reference_metadata=dict(v55_frozen=frozen.metadata(), matched_pooled=pooled.metadata(),
            phase_calibrated=progress["references"]["phase_calibrated"].metadata()),
        records=records, summary=summarize(records),
        limitations=["Initial-versus-later episode calibration under one source seed and fixed beta; later learning lifetimes remain untested.",
                     "No adaptive learning/action effect or 50x OOD return confirmation."])
    args.result_dir.mkdir(parents=True, exist_ok=True)
    destination = args.result_dir / "phase_calibration.json"
    destination.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(f"[DONE] {output['summary']['decision']}: {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "formal"), required=True)
    for name in ("source-dir", "shared-dir", "dynamics-checkpoint", "reference-checkpoint", "checkpoint-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
