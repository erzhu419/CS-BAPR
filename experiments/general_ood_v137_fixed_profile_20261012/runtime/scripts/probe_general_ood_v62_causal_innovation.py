"""New-seed fixed-beta preflight for level plus causal residual innovation."""
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
from scripts import probe_general_ood_v60_long_normal as v60
from scripts.v52_agent import ContinuationAgent
from scripts.v54_window_readout import normalized_residuals
from scripts.v57_phase_readout import PhaseDepartureReadout
from scripts.v62_innovation_readout import InnovationReference, NAMES, window_features

REVISION = "general_ood_v62_causal_innovation"
SPEC = dict(fit=[134000], calibration=[134010], test=[134020, 134021], steps=6000, switch_step=3800)
OBSERVERS = ("v60_frozen", *NAMES)
NORMAL_WINDOWS, CHANGE_WINDOWS = v60.NORMAL_WINDOWS, v60.CHANGE_WINDOWS


def observe(trace, model, reference, switch):
    signals = reference.signals(*window_features(normalized_residuals(model, trace)))
    ages = v53.v43.episode_steps(trace["dones"])
    n = len(ages)
    windows = dict(full=(0,n), prefix=(0,switch), first800=(0,min(800,n)), first_post50=(switch,min(switch+50,n)),
                   post=(switch,n), last1000=(max(0,n-1000),n), last64=(max(0,n-64),n))
    output = {}
    for name in NAMES:
        sensor = PhaseDepartureReadout(reference.phases[name])
        risks, instant = [], []
        for s, age in zip(signals[name], ages):
            risks.append(sensor.update(s, int(age)))
            instant.append(sensor.departure.instant_risk)
        risks, instant = np.asarray(risks), np.asarray(instant)
        assert np.isfinite(risks).all() and np.isfinite(signals[name]).all()
        output[name] = {w: dict(mean_coefficient=float(risks[a:b].mean()), mean_signal=float(signals[name][a:b].mean()),
            mean_instant_coefficient=float(instant[a:b].mean()), mean_level_ratio=float(signals["level_ratio"][a:b].mean()),
            mean_innovation_ratio=float(signals["innovation_ratio"][a:b].mean()),
            innovation_dominant_fraction=float((signals["innovation_ratio"][a:b]>signals["level_ratio"][a:b]).mean()))
            for w,(a,b) in windows.items()}
    return output


def summarize(records):
    normal = [r for r in records if r["split"] == "test" and r["mode"] == "normal"]
    changed = [r for r in records if r["split"] == "test" and r["mode"] != "normal"]
    output = {}
    for name in OBSERVERS:
        null = all(r["observers"][name][w]["mean_coefficient"] <= .1 for r in normal for w in NORMAL_WINDOWS)
        counts = {w: sum(r["change_minus_nochange"][name][w] > 0 for r in changed) for w in CHANGE_WINDOWS}
        output[name] = dict(usable_normal_null=null, positive_pairs=counts, changed_pairs=len(changed),
            decision="causal-innovation-preflight-pass-not-RL-confirmation"
                     if null and all(c == len(changed) for c in counts.values()) else "hold-causal-innovation")
    return output


def run(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cells = v60.trajectory_specs(SPEC)
    shared = torch.load(args.shared_dir/"shared.pt", map_location="cpu", weights_only=False)
    previous = torch.load(args.reference_checkpoint, map_location="cpu", weights_only=False)["extra_state"]["progress"]
    assert previous["revision"] == v60.REVISION and previous["next_cell"] == 16
    frozen = previous["references"]["long_normal_refit"]
    model, pooled, phase = frozen
    inherited_metadata = v60.reference_metadata(dict(v60_frozen=frozen))
    del previous
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    traces = args.checkpoint_dir/"traces"
    traces.mkdir(exist_ok=True)
    checkpoint = args.checkpoint_dir/"preflight_latest.pt"
    extra = torch.load(args.resume_from, map_location="cpu", weights_only=False)["extra_state"] if args.resume_from else None
    progress = extra["progress"] if extra else dict(revision=REVISION, spec=SPEC, next_cell=0, fitting=[],
        calibration=[], reference=None, reference_metadata=None, records=[], control=None)
    assert progress["revision"] == REVISION and progress["spec"] == SPEC
    for index in range(progress["next_cell"], len(cells)):
        cell = cells[index]
        if cell["split"] == "test" and progress["reference"] is None:
            progress["reference"] = InnovationReference.fit_normal(model, pooled, progress["fitting"], progress["calibration"])
            progress["reference_metadata"] = progress["reference"].metadata()
            progress["fitting"], progress["calibration"] = [], []
            print("[REFERENCE] Innovation covariance/scale and joint normal calibration frozen before test", flush=True)
        env = gym.make("Hopper-v4")
        agent = ContinuationAgent(env, "fixed-minus2")
        agent.start_shared(args.source_dir/v53.SOURCE_NAME, shared)
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
            record["observers"] = dict(v60_frozen=v60.observe(trace, *frozen, SPEC["switch_step"]),
                **observe(trace, model, progress["reference"], SPEC["switch_step"]))
            if cell["updating"]:
                if cell["mode"] == "normal":
                    progress["control"] = (trace, deepcopy(record))
                control, baseline = progress["control"]
                assert baseline["seed"] == cell["seed"] and v53.paired_prefix_equal(trace, control, SPEC["switch_step"])
                record["paired_prefix_equal"] = True
                record["change_minus_nochange"] = {name: {w: value["mean_coefficient"]-
                    baseline["observers"][name][w]["mean_coefficient"] for w,value in windows.items()}
                    for name,windows in record["observers"].items()}
        path = traces/f"{cell['split']}_seed{cell['seed']}_{'updating' if cell['updating'] else 'frozen'}_{cell['mode']}.npz"
        np.savez_compressed(path, **trace)
        record["server_trace"] = str(path)
        progress["records"].append(record)
        progress["next_cell"] = index+1
        v53.save_progress(agent, env, progress, None, checkpoint)
        print(f"[CELL] {index+1}/{len(cells)}: {cell}, updates={record['optimizer_updates']}", flush=True)
        extra = None
        env.close()
        del agent
        torch.cuda.empty_cache()
    assert inherited_metadata == v60.reference_metadata(dict(v60_frozen=frozen))
    assert progress["reference_metadata"] == progress["reference"].metadata()
    records = progress["records"]
    output = dict(revision=REVISION, stage="preflight", stage_spec=SPEC, source_training_seed=1,
        source_checkpoint=v53.SOURCE_NAME, source_updates=99745, shared_replay_size=6144,
        beta=-2., readout_controls_actor=False, known_change_boundary_used_by_readout=False,
        reference_frozen_during_test=True, inherited_reference_metadata=inherited_metadata,
        reference_metadata=progress["reference_metadata"], records=records, summary=summarize(records),
        environment_interactions=sum(r["interactions"] for r in records),
        optimizer_updates=sum(r["optimizer_updates"] for r in records),
        limitations=["New continuation seeds share one previously trained source policy; development preflight, not independent-source confirmation.",
                     "Fixed beta and mild physical changes establish neither adaptive return gains nor extreme50x performance."])
    args.result_dir.mkdir(parents=True, exist_ok=True)
    destination = args.result_dir/"causal_innovation.json"
    destination.write_text(json.dumps(output, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] {output['summary']}: {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ("source-dir", "shared-dir", "reference-checkpoint", "checkpoint-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
