"""Normal-only phase-vector fitting and retrospective fixed-beta replay screen."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from scripts import analyze_general_ood_v63_readout_attribution as v63
from scripts.v54_window_readout import normalized_residuals, rolling_means
from scripts.v64_phase_vector_readout import NAMES, PhaseVectorReference

REVISION = "general_ood_v64_phase_vector"
OBSERVERS = (*v63.v62.OBSERVERS,*NAMES)


def window_ranges(n,switch):
    windows = dict(first800=(0,min(800,n)),full=(0,n),last1000=(max(0,n-1000),n),
        pre64=(max(0,switch-64),switch),first_post50=(switch,min(n,switch+50)),last64=(max(0,n-64),n))
    for a,b in ((0,20),(20,50),(50,100),(100,200)):
        if switch+a<n:
            windows[f"post{a}_{b}"] = (switch+a,min(n,switch+b))
    return windows


def observe(trace,model,reference,switch):
    moments = rolling_means(normalized_residuals(model,trace))
    ages = v63.v62.v53.v43.episode_steps(trace["dones"])
    signals = reference.signals(moments,ages,v63.episode_ordinals(trace["dones"])>0)
    output = {}
    for name in NAMES:
        sensor = v63.v62.PhaseDepartureReadout(reference.phases[name])
        risks,instant,thresholds = [],[],[]
        for signal,age in zip(signals[name],ages):
            risks.append(sensor.update(signal,int(age)))
            instant.append(sensor.departure.instant_risk)
            thresholds.append(sensor.departure.threshold)
        risks,instant,thresholds = np.asarray(risks),np.asarray(instant),np.asarray(thresholds)
        output[name] = {w:dict(mean_coefficient=float(risks[a:b].mean()),mean_signal=float(signals[name][a:b].mean()),
            mean_instant_coefficient=float(instant[a:b].mean()),mean_threshold=float(thresholds[a:b].mean()),
            max_margin=float((signals[name][a:b]-thresholds[a:b]).max()),
            above_threshold_fraction=float((signals[name][a:b]>thresholds[a:b]).mean()))
            for w,(a,b) in window_ranges(len(ages),switch).items()}
    return output


def summarize(records):
    output = {}
    for group in ("v62_test","v61_development_replay"):
        normal = [r for r in records if r["group"] == group and r["mode"] == "normal"]
        changed = [r for r in records if r["group"] == group and r["mode"] != "normal"]
        output[group] = {}
        for name in OBSERVERS:
            null = all(r["observers"][name][w]["mean_coefficient"]<=.1 for r in normal for w in v63.v62.NORMAL_WINDOWS)
            counts = {w:sum(r["change_minus_nochange"][name][w]>0 for r in changed) for w in v63.v62.CHANGE_WINDOWS}
            output[group][name] = dict(usable_normal_null=null,positive_pairs=counts,changed_pairs=len(changed),
                largest_gated_normal_mean=max(r["observers"][name][w]["mean_coefficient"] for r in normal for w in v63.v62.NORMAL_WINDOWS),
                decision="retrospective-screen-pass-not-confirmation" if null and all(v==len(changed) for v in counts.values())
                         else "hold-phase-vector-screen")
    return output


def run(args):
    torch.set_num_threads(2)
    baseline = json.loads((args.v62_dir/"run_results/preflight/causal_innovation.json").read_text())
    replay = json.loads((args.v63_dir/"run_results/attribution/readout_attribution.json").read_text())
    previous = torch.load(args.v60_dir/"run_checkpoints/preflight/preflight_latest.pt",map_location="cpu",weights_only=False)["extra_state"]["progress"]
    assert baseline["revision"] == v63.v62.REVISION and baseline["stage_spec"] == v63.v62.SPEC
    assert replay["revision"] == v63.REVISION and replay["maximum_reproduction_error"] == 0.
    assert previous["revision"] == v63.v62.v60.REVISION and previous["next_cell"] == 16
    model,pooled,phase = previous["references"]["long_normal_refit"]
    assert v63.v62.v60.reference_metadata(dict(v60_frozen=(model,pooled,phase))) == baseline["inherited_reference_metadata"]
    old_model_metadata = deepcopy(model.metadata())
    del previous
    stored = args.v62_dir/"run_checkpoints/preflight/traces"
    def load_v62(cell):
        kind = "updating" if cell["updating"] else "frozen"
        with np.load(stored/f"{cell['split']}_seed{cell['seed']}_{kind}_{cell['mode']}.npz") as f:
            return {k:f[k] for k in f.files}
    fitting = [load_v62(r) for r in baseline["records"] if r["split"] == "fit"]
    calibration = [load_v62(r) for r in baseline["records"] if r["split"] == "calibration"]
    assert len(fitting) == len(calibration) == 2
    reference = PhaseVectorReference.fit_normal(model,pooled,fitting,calibration)
    metadata = deepcopy(reference.metadata())
    args.checkpoint_dir.mkdir(parents=True,exist_ok=True)
    path = args.checkpoint_dir/"phase_vector_reference.pt"
    torch.save(dict(revision=REVISION,model=model,reference=reference,reference_metadata=metadata,
        source_training_seed=1,normal_fit_seeds=baseline["stage_spec"]["fit"],normal_calibration_seeds=baseline["stage_spec"]["calibration"]),path)
    del fitting,calibration
    print("[REFERENCE] Normal-only phase centers/covariances and calibration frozen before replay",flush=True)
    records,normal = [],None
    for original in replay["records"]:
        cell = {k:original[k] for k in ("group","seed","updating","mode")}
        if cell["group"] == "v62_test":
            trace = load_v62(dict(split="test",**cell))
        else:
            assert cell["group"] == "v61_development_replay"
            with np.load(args.v61_dir/f"run_checkpoints/preflight/traces/seed{cell['seed']}_{cell['mode']}.npz") as f:
                trace = {k:f[k] for k in f.files}
        observers = {name:{w:{key:d["observers"][name][key] for key in ("mean_coefficient","mean_signal")}
            for w,d in original["diagnostic"].items()} for name in v63.v62.OBSERVERS}
        observers.update(observe(trace,model,reference,baseline["stage_spec"]["switch_step"]))
        record = dict(**cell,saved_transitions=len(trace["dones"]),observers=observers)
        if cell["updating"]:
            if cell["mode"] == "normal":
                normal = record
            assert normal["group"] == cell["group"] and normal["seed"] == cell["seed"]
            record["change_minus_nochange"] = {name:{w:d["mean_coefficient"]-normal["observers"][name][w]["mean_coefficient"]
                for w,d in values.items()} for name,values in observers.items()}
            assert all(abs(record["change_minus_nochange"][name][w])<1e-10 for name in NAMES for w in ("first800","pre64"))
        records.append(record)
    assert old_model_metadata == model.metadata() and metadata == reference.metadata()
    summary = summarize(records)
    main = [s["phase_center_covariance"] for s in summary.values()]
    decision = "retrospective-phase-vector-screen-pass-not-confirmation" if all(s["decision"].startswith("retrospective-screen-pass") for s in main) else "hold-phase-vector-screen"
    output = dict(revision=REVISION,environment_interactions=0,optimizer_updates=0,source_training_seed=1,
        reused_normal_fit_seeds=baseline["stage_spec"]["fit"],reused_normal_calibration_seeds=baseline["stage_spec"]["calibration"],
        inherited_dynamics_unchanged=True,reference_frozen_during_replay=True,known_boundary_used_by_readout=False,
        beta=-2.,readout_controls_actor=False,reference_metadata=metadata,server_reference_checkpoint=str(path),
        records=records,summary=summary,decision=decision,
        replayed_saved_transitions=sum(r["saved_transitions"] for r in records),
        limitations=["All evaluation trajectories are reused development data; screening is not independent confirmation.",
                     "Phase/age conditioning is observable but does not condition directly on state/action or establish adaptive RL returns."])
    args.result_dir.mkdir(parents=True,exist_ok=True)
    destination = args.result_dir/"phase_vector.json"
    destination.write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
    print(f"[DONE] {decision}: {destination}",flush=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ("v60-dir","v61-dir","v62-dir","v63-dir","checkpoint-dir","result-dir"):
        parser.add_argument(f"--{name}",type=Path,required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
