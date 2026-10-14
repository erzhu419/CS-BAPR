"""Offline signed-moment geometry and smoothing diagnosis for v60 negatives."""
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
from scripts.v54_window_readout import normalized_residuals, rolling_means
from scripts.v57_phase_readout import PhaseDepartureReadout

REVISION = "general_ood_v61_early_response"
SEED = 132020
SPEC = dict(steps=6000, switch_step=3800)
ANCHOR_STEPS = 64
REPRODUCTION_TOLERANCE = 1e-8
SCALARS = ("level_score", "signal", "threshold", "instant_risk", "risk", "cross_term", "displacement")


def geometry(moments, reference, anchor):
    delta, offset = moments - anchor, anchor - reference.center
    dimension = moments.shape[1]
    base = float(offset @ reference.precision @ offset / dimension)
    cross = 2 * np.einsum("i,ij,nj->n", offset, reference.precision, delta) / dimension
    displacement = np.einsum("ni,ij,nj->n", delta, reference.precision, delta) / dimension
    scores = reference.scores(moments)
    # The cross term distinguishes movement towards the global center from no movement.
    error = float(np.max(np.abs(scores - (base + cross + displacement))))
    np.testing.assert_allclose(scores, base + cross + displacement, rtol=1e-10, atol=1e-10)
    return dict(level_score=scores, cross_term=cross, displacement=displacement), base, error


def diagnose(trace, model, pooled, phase, anchor, switch):
    moments = rolling_means(normalized_residuals(model, trace))
    values, base, error = geometry(moments, pooled, anchor)
    signals = np.log1p(values["level_score"] / pooled.score95)
    sensor = PhaseDepartureReadout(phase)
    risks, instant, thresholds = [], [], []
    for signal, age in zip(signals, v53.v43.episode_steps(trace["dones"])):
        risks.append(sensor.update(signal, int(age)))
        instant.append(sensor.departure.instant_risk)
        thresholds.append(sensor.departure.threshold)
    values.update(signal=signals, risk=np.asarray(risks), instant_risk=np.asarray(instant), threshold=np.asarray(thresholds))
    assert all(np.isfinite(v).all() for v in values.values())
    n = len(signals)
    windows = dict(pre64=(max(0,switch-64),switch), first_post50=(switch,min(n,switch+50)), last64=(max(0,n-64),n))
    for start,end in ((0,10),(10,20),(20,30),(30,50),(50,100),(100,200)):
        if switch + start < n:
            windows[f"post{start}_{end}"] = (switch+start,min(n,switch+end))
    return dict(anchor_score=base, score_identity_max_abs_error=error,
        windows={name: dict(**{key:float(v[start:end].mean()) for key,v in values.items()},
                           mean_signed_moment=moments[start:end].mean(axis=0).tolist())
                 for name,(start,end) in windows.items()})


def paired_differences(record, baseline):
    differences = {}
    for window, values in record["diagnostic"]["windows"].items():
        control = baseline["diagnostic"]["windows"][window]
        delta = {key: values[key]-control[key] for key in SCALARS}
        assert np.isclose(delta["level_score"],delta["cross_term"]+delta["displacement"], rtol=1e-9,atol=1e-9)
        delta["mean_signed_moment"] = (np.asarray(values["mean_signed_moment"])-control["mean_signed_moment"]).tolist()
        differences[window] = delta
    return differences


def run(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    baseline = json.loads(args.baseline_result.read_text())
    assert baseline["revision"] == v60.REVISION and all(baseline["stage_spec"][k] == v for k,v in SPEC.items())
    completed = torch.load(args.reference_checkpoint,map_location="cpu",weights_only=False)["extra_state"]["progress"]
    assert completed["revision"] == v60.REVISION and completed["next_cell"] == 16
    reference = completed["references"]["long_normal_refit"]
    model, pooled, phase = reference
    metadata = v60.reference_metadata(dict(long_normal_refit=reference))
    assert metadata["long_normal_refit"] == baseline["reference_metadata"]["long_normal_refit"]
    del completed
    shared = torch.load(args.shared_dir/"shared.pt",map_location="cpu",weights_only=False)
    args.checkpoint_dir.mkdir(parents=True,exist_ok=True)
    traces = args.checkpoint_dir/"traces"
    traces.mkdir(exist_ok=True)
    checkpoint = args.checkpoint_dir/"preflight_latest.pt"
    extra = torch.load(args.resume_from,map_location="cpu",weights_only=False)["extra_state"] if args.resume_from else None
    progress = extra["progress"] if extra else dict(revision=REVISION,next_cell=0,records=[],control=None,anchor=None)
    assert progress["revision"] == REVISION
    for index in range(progress["next_cell"],len(v53.MODES)):
        mode = v53.MODES[index]
        cell = dict(split="test",seed=SEED,updating=True,mode=mode)
        env = gym.make("Hopper-v4")
        agent = ContinuationAgent(env,"fixed-minus2")
        agent.start_shared(args.source_dir/v53.SOURCE_NAME,shared)
        assert agent.training_steps == 99745 and agent.context_tracker._count == 100000 and agent.replay_buffer.size == 6144
        loop = None
        if extra and extra["loop"] is not None:
            agent.load(str(args.resume_from))
            loop = extra["loop"]
            v53.apply_mode_to_env(env,"Hopper-v4",v53.MUJOCO_MODE_PROFILES["Hopper-v4"][mode if loop["step"] > SPEC["switch_step"] else "normal"])
            env.reset(seed=SEED)
            v53.restore_environment(env,extra["environment"])
        trace,record = v53.collect(agent,env,cell,SPEC,loop,
            lambda active:v53.save_progress(agent,env,progress,active,checkpoint))
        if mode == "normal":
            moments = rolling_means(normalized_residuals(model,trace))
            progress["anchor"] = moments[max(0,SPEC["switch_step"]-ANCHOR_STEPS):SPEC["switch_step"]].mean(axis=0)
        record["diagnostic"] = diagnose(trace,*reference,progress["anchor"],SPEC["switch_step"])
        expected = next(r for r in baseline["records"] if r["split"] == "test" and r["seed"] == SEED and r["updating"] and r["mode"] == mode)
        discrepancy = max(abs(record["diagnostic"]["windows"][w][key]-expected["observers"]["long_normal_refit"][w][old_key])
            for w in ("first_post50","last64") for key,old_key in (("risk","mean_coefficient"),("signal","mean_signal")))
        record["reproduction"] = dict(max_abs_scalar_error=discrepancy,matched=discrepancy <= REPRODUCTION_TOLERANCE)
        if mode == "normal":
            progress["control"] = (trace,deepcopy(record))
        control,normal = progress["control"]
        assert v53.paired_prefix_equal(trace,control,SPEC["switch_step"])
        record["paired_prefix_equal"] = True
        record["change_minus_nochange"] = paired_differences(record,normal)
        path = traces/f"seed{SEED}_{mode}.npz"
        np.savez_compressed(path,**trace)
        record["server_trace"] = str(path)
        progress["records"].append(record)
        progress["next_cell"] = index+1
        v53.save_progress(agent,env,progress,None,checkpoint)
        print(f"[CELL] {index+1}/5: {mode}, reproduction={record['reproduction']}",flush=True)
        extra = None
        env.close()
        del agent
        torch.cuda.empty_cache()
    assert metadata == v60.reference_metadata(dict(long_normal_refit=reference))
    records = progress["records"]
    output = dict(revision=REVISION,continuation_seed=SEED,source_training_seed=1,source_checkpoint=v53.SOURCE_NAME,
        stage_spec=SPEC,beta=-2.,readout_controls_actor=False,reference_frozen=True,reference_metadata=metadata,
        diagnostic_anchor=progress["anchor"].tolist(),anchor_usage="known-boundary offline geometry only",
        environment_interactions=sum(r["interactions"] for r in records),optimizer_updates=sum(r["optimizer_updates"] for r in records),
        records=records,decision="reproduced-localization-ready" if all(r["reproduction"]["matched"] for r in records) else "hold-unmatched-v60-reproduction",
        limitations=["Targeted reused development seed, not independent confirmation.","Known-boundary diagnostic anchor is not a deployable online change detector."])
    args.result_dir.mkdir(parents=True,exist_ok=True)
    destination = args.result_dir/"early_response.json"
    destination.write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
    print(f"[DONE] {output['decision']}: {destination}",flush=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ("source-dir","shared-dir","reference-checkpoint","baseline-result","checkpoint-dir","result-dir"):
        parser.add_argument(f"--{name}",type=Path,required=True)
    parser.add_argument("--resume-from",type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
