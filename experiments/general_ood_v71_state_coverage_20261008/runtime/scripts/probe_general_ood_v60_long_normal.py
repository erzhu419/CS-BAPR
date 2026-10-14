"""Long normal coverage versus calibration-only repair on matched late changes."""
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
from scripts.probe_general_ood_v55_learning_calibration import signals_and_ages
from scripts.v52_agent import ContinuationAgent
from scripts.v54_window_readout import SignedWindowReference
from scripts.v57_phase_readout import PhaseDepartureReadout, fit_references

REVISION = "general_ood_v60_long_normal"
SPEC = dict(fit=[132000], calibration=[132010], test=[132020, 132021], steps=6000, switch_step=3800)
OBSERVERS = ("v57_frozen", "long_calibration_only", "long_normal_refit")
NORMAL_WINDOWS = ("first800", "full", "last1000", "last64")
CHANGE_WINDOWS = ("first_post50", "last64")


def trajectory_specs(spec):
    assert len(set(spec["fit"] + spec["calibration"] + spec["test"])) == sum(len(spec[k]) for k in ("fit", "calibration", "test"))
    cells = [dict(split=split, seed=seed, updating=updating, mode="normal")
             for split in ("fit", "calibration") for seed in spec[split] for updating in (False, True)]
    for seed in spec["test"]:
        cells.append(dict(split="test", seed=seed, updating=False, mode="normal"))
        cells.extend(dict(split="test", seed=seed, updating=True, mode=mode) for mode in v53.MODES)
    return cells


def build_references(old_model, old_pooled, old_phase, fitting, calibration):
    age_pooled, age_phase = fit_references(old_model, old_pooled, calibration)
    model, _ = v53.fit_reference(fitting, calibration)
    signed = SignedWindowReference.fit_normal(model, fitting, calibration)
    pooled, phase = fit_references(model, signed, calibration)
    return dict(v57_frozen=(old_model, old_pooled, old_phase),
                long_calibration_only=(old_model, age_pooled, age_phase), long_normal_refit=(model, pooled, phase))


def reference_metadata(references):
    return {name: dict(model=model.metadata(), pooled=pooled.metadata(), phase=phase.metadata())
            for name, (model, pooled, phase) in references.items()}


def observe(trace, model, pooled, phase, switch):
    signals, ages = signals_and_ages(model, pooled, trace)
    sensor = PhaseDepartureReadout(phase)
    risks = np.asarray([sensor.update(s, int(a)) for s, a in zip(signals, ages)])
    assert np.isfinite(signals).all() and np.isfinite(risks).all()
    n = len(ages)
    windows = dict(full=(0,n), prefix=(0,switch), first800=(0,min(800,n)), first_post50=(switch,min(switch+50,n)),
                   post=(switch,n), last1000=(max(0,n-1000),n), last64=(max(0,n-64),n))
    return {name: dict(mean_coefficient=float(risks[start:end].mean()), mean_signal=float(signals[start:end].mean()))
            for name, (start,end) in windows.items()}


def summarize(records):
    normal = [r for r in records if r["split"] == "test" and r["mode"] == "normal"]
    changed = [r for r in records if r["split"] == "test" and r["mode"] != "normal"]
    summaries = {}
    for name in OBSERVERS:
        null = all(r["observers"][name][w]["mean_coefficient"] <= .1 for r in normal for w in NORMAL_WINDOWS)
        counts = {w: sum(r["change_minus_nochange"][name][w] > 0 for r in changed) for w in CHANGE_WINDOWS}
        summaries[name] = dict(usable_normal_null=null, positive_pairs=counts, changed_pairs=len(changed),
            decision="long-normal-detector-preflight-pass-not-RL-confirmation"
                     if null and all(v == len(changed) for v in counts.values()) else "hold-long-normal-readout")
    return summaries


def run(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cells = trajectory_specs(SPEC)
    shared = torch.load(args.shared_dir / "shared.pt", map_location="cpu", weights_only=False)
    dynamics = torch.load(args.dynamics_checkpoint, map_location="cpu", weights_only=False)["extra_state"]["progress"]
    frozen = torch.load(args.reference_checkpoint, map_location="cpu", weights_only=False)["extra_state"]["progress"]
    assert dynamics["stage"] == "preflight" and dynamics["next_cell"] == 9
    assert frozen["stage"] == "formal" and frozen["next_cell"] == 22
    old_model, refs = dynamics["models"]["broad"][0], frozen["references"]
    old_pooled, old_phase = refs["matched_pooled"], refs["phase_calibrated"]
    old_metadata = reference_metadata(dict(v57_frozen=(old_model, old_pooled, old_phase)))
    del dynamics, frozen
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint_dir / "preflight_latest.pt"
    extra = torch.load(args.resume_from, map_location="cpu", weights_only=False)["extra_state"] if args.resume_from else None
    progress = extra["progress"] if extra else dict(revision=REVISION, next_cell=0, fitting=[], calibration=[],
        references=None, frozen_metadata=None, records=[], control=None)
    assert progress["revision"] == REVISION
    for index in range(progress["next_cell"], len(cells)):
        cell = cells[index]
        if cell["split"] == "test" and progress["references"] is None:
            progress["references"] = build_references(old_model, old_pooled, old_phase, progress["fitting"], progress["calibration"])
            progress["frozen_metadata"] = reference_metadata(progress["references"])
            progress["fitting"], progress["calibration"] = [], []
            print("[REFERENCE] Long-normal dynamics/calibration frozen before held-out tests", flush=True)
        env = gym.make("Hopper-v4")
        agent = ContinuationAgent(env, "fixed-minus2")
        agent.start_shared(args.source_dir / v53.SOURCE_NAME, shared)
        assert agent.training_steps == 99745 and agent.context_tracker._count == 100000 and agent.replay_buffer.size == 6144
        loop = None
        if extra and extra["loop"] is not None:
            agent.load(str(args.resume_from))
            loop = extra["loop"]
            mode = cell["mode"] if cell["split"] == "test" and loop["step"] > SPEC["switch_step"] else "normal"
            v53.apply_mode_to_env(env, "Hopper-v4", v53.MUJOCO_MODE_PROFILES["Hopper-v4"][mode])
            env.reset(seed=cell["seed"])
            v53.restore_environment(env, extra["environment"])
        trace, record = v53.collect(agent, env, cell, SPEC, loop,
            lambda active: v53.save_progress(agent, env, progress, active, checkpoint))
        if cell["split"] != "test":
            progress["fitting" if cell["split"] == "fit" else "calibration"].append(trace)
            record["reference_record"] = v53.v43.reference_record(trace, cell["seed"], cell["split"])
        else:
            record["observers"] = {name: observe(trace, *reference, SPEC["switch_step"])
                                   for name, reference in progress["references"].items()}
            if cell["updating"]:
                if cell["mode"] == "normal":
                    progress["control"] = (trace, deepcopy(record))
                control, baseline = progress["control"]
                assert baseline["seed"] == cell["seed"] and v53.paired_prefix_equal(trace, control, SPEC["switch_step"])
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
    assert reference_metadata(progress["references"]) == progress["frozen_metadata"]
    assert reference_metadata(dict(v57_frozen=(old_model, old_pooled, old_phase))) == old_metadata
    records = progress["records"]
    output = dict(revision=REVISION, stage="preflight", stage_spec=SPEC, source_checkpoint=v53.SOURCE_NAME,
        source_training_seed=1, source_updates=99745, shared_replay_size=6144, beta=-2., readout_controls_actor=False,
        normal_reference_population="6000-step frozen and updating normal branches, separate fit/calibration seeds",
        reference_frozen_during_test=True, reference_metadata=progress["frozen_metadata"], records=records,
        environment_interactions=sum(r["interactions"] for r in records),
        optimizer_updates=sum(r["optimizer_updates"] for r in records), summary=summarize(records),
        limitations=["One source policy and development detector preflight; not adaptive RL or independent-source confirmation.",
                     "Normal coverage is tested over6000 steps; later lifetimes and extreme50x returns remain untested."])
    args.result_dir.mkdir(parents=True, exist_ok=True)
    destination = args.result_dir / "long_normal.json"
    destination.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(f"[DONE] {output['summary']}: {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ("source-dir", "shared-dir", "dynamics-checkpoint", "reference-checkpoint", "checkpoint-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
