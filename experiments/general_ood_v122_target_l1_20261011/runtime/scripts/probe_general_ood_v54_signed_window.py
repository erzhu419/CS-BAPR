"""Held-out learning diagnostics for a frozen signed-window reference."""
from copy import deepcopy
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import torch

from scripts import probe_general_ood_v53_behavior_coverage as v53
from scripts.v52_agent import ContinuationAgent
from scripts.v54_window_readout import SignedWindowReference, observe_trace

REVISION = "general_ood_v54_signed_window"


def stage_spec(stage):
    if stage == "preflight":
        return dict(fit=[98400], calibration=[98410], test=[98420, 98421], steps=1200, switch_step=800)
    return dict(fit=[98500, 98501], calibration=[98510, 98511], test=list(range(98520, 98524)),
                steps=1200, switch_step=800)


def summarize(records):
    normal = [r for r in records if r["split"] == "test" and r["mode"] == "normal"]
    changes = [r for r in records if r["split"] == "test" and r["mode"] != "normal"]
    null_pass = all(r["observers"]["signed_window"][w]["mean_coefficient"] <= .1
                    for r in normal for w in ("full", "last64"))
    separation = all(r["change_minus_nochange"]["signed_window"][w] > 0
                     for r in changes for w in ("first_post50", "last64"))
    return dict(usable_learning_normal_null=null_pass, all_paired_first50_and_tail_positive=separation,
        normal_means={name: {w: [r["observers"][name][w]["mean_coefficient"] for r in normal]
                            for w in ("full", "last64")} for name in ("v53_scalar", "signed_window")},
        decision="signed-window-diagnostic-pass-not-RL-confirmation" if null_pass and separation else "hold-signed-window")


def run(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    spec = stage_spec(args.stage)
    cells = v53.trajectory_specs(spec)
    shared = torch.load(args.shared_dir / "shared.pt", map_location="cpu", weights_only=False)
    completed = torch.load(args.dynamics_checkpoint, map_location="cpu", weights_only=False)["extra_state"]["progress"]
    assert completed["stage"] == "preflight" and completed["next_cell"] == 9
    model, scalar_reference = completed["models"]["broad"]
    del completed
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint_dir / f"{args.stage}_latest.pt"
    extra = torch.load(args.resume_from, map_location="cpu", weights_only=False)["extra_state"] if args.resume_from else None
    progress = extra["progress"] if extra else dict(stage=args.stage, next_cell=0, fitting=[], calibration=[],
        reference=None, records=[], control=None)
    assert progress["stage"] == args.stage
    for index in range(progress["next_cell"], len(cells)):
        cell = cells[index]
        if cell["split"] == "test" and progress["reference"] is None:
            progress["reference"] = SignedWindowReference.fit_normal(model, progress["fitting"], progress["calibration"])
            progress["fitting"], progress["calibration"] = [], []
            print("[REFERENCE] Normal-only moments and separate threshold calibration complete; dynamics unchanged", flush=True)
        env = gym.make("Hopper-v4")
        agent = ContinuationAgent(env, "fixed-minus2")
        agent.start_shared(args.source_dir / v53.SOURCE_NAME, shared)
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
            progress["fitting" if cell["split"] == "fit" else "calibration"].append(trace)
            record["reference_record"] = v53.v43.reference_record(trace, cell["seed"], cell["split"])
        else:
            record["observers"] = dict(v53_scalar=v53.observe(trace, model, scalar_reference, spec["switch_step"]),
                signed_window=observe_trace(trace, model, progress["reference"], spec["switch_step"]))
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
    records = progress["records"]
    output = dict(revision=REVISION, stage=args.stage, stage_spec=spec, source_checkpoint=v53.SOURCE_NAME,
        source_training_seed=1, source_updates=99745, shared_replay_size=shared["replay"].size,
        dynamics_origin="v53 preflight broad normal model; no dynamics refit",
        action_sampling="stochastic", beta=-2., auxiliary_controls_actions=False, episode_time_limit=1000,
        reference_frozen_during_test=True, environment_interactions=sum(r["interactions"] for r in records),
        optimizer_updates=sum(r["optimizer_updates"] for r in records),
        dynamics_metadata=model.metadata(), scalar_reference_metadata=scalar_reference.metadata(),
        signed_window_metadata=progress["reference"].metadata(), records=records, summary=summarize(records),
        limitations=["One source training checkpoint; continuation seeds are conditional detector probes, not independent training seeds.",
                     "Fixed beta, no readout control of actions; no adaptive RL return confirmation or 50x OOD comparison."])
    args.result_dir.mkdir(parents=True, exist_ok=True)
    destination = args.result_dir / "signed_window.json"
    destination.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(f"[DONE] {output['summary']['decision']}: {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "formal"), required=True)
    for name in ("source-dir", "shared-dir", "dynamics-checkpoint", "checkpoint-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
