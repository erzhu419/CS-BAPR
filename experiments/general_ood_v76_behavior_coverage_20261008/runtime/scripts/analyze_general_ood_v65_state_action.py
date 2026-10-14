"""Normal-only state/action reference and retrospective fixed-beta replay screen."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from scripts import analyze_general_ood_v64_phase_vector as v64
from scripts.v54_window_readout import normalized_residuals, rolling_means
from scripts.v65_state_action_readout import NAMES, StateActionReference

REVISION = "general_ood_v65_state_action"
OBSERVERS = (*v64.OBSERVERS,*NAMES)
MAIN = "state_action_center_covariance"


def observe(trace,model,reference,switch):
    ages = v64.v63.v62.v53.v43.episode_steps(trace["dones"])
    signals,geometry = reference.evaluate(rolling_means(normalized_residuals(model,trace)),rolling_means(model.inputs(trace)))
    output = {}
    for name in NAMES:
        sensor = v64.v63.v62.PhaseDepartureReadout(reference.phases[name])
        risks,instant,thresholds = [],[],[]
        for signal,age in zip(signals[name],ages):
            risks.append(sensor.update(signal,int(age)))
            instant.append(sensor.departure.instant_risk)
            thresholds.append(sensor.departure.threshold)
        risks,instant,thresholds = np.asarray(risks),np.asarray(instant),np.asarray(thresholds)
        output[name] = {w:dict(mean_coefficient=float(risks[a:b].mean()),mean_signal=float(signals[name][a:b].mean()),
            mean_instant_coefficient=float(instant[a:b].mean()),mean_threshold=float(thresholds[a:b].mean()),
            max_margin=float((signals[name][a:b]-thresholds[a:b]).max()),
            above_threshold_fraction=float((signals[name][a:b]>thresholds[a:b]).mean()),
            mean_shrinkage=float(geometry["shrinkage"][a:b].mean()),
            mean_covariance_trace=float(geometry["covariance_trace"][a:b].mean()),
            mean_neighbor_radius=float(geometry["neighbor_radius"][a:b].mean()),
            outside_calibration_radius95_fraction=float((geometry["neighbor_radius"][a:b]>reference.neighbor_radius95).mean()))
            for w,(a,b) in v64.window_ranges(len(ages),switch).items()}
    return output


def summarize(records):
    output = {}
    for group in ("v62_test","v61_development_replay"):
        normal = [r for r in records if r["group"]==group and r["mode"]=="normal"]
        changed = [r for r in records if r["group"]==group and r["mode"]!="normal"]
        output[group] = {}
        for name in OBSERVERS:
            null = all(r["observers"][name][w]["mean_coefficient"]<=.1 for r in normal for w in v64.v63.v62.NORMAL_WINDOWS)
            counts = {w:sum(r["change_minus_nochange"][name][w]>0 for r in changed) for w in v64.v63.v62.CHANGE_WINDOWS}
            output[group][name] = dict(usable_normal_null=null,positive_pairs=counts,changed_pairs=len(changed),
                largest_gated_normal_mean=max(r["observers"][name][w]["mean_coefficient"] for r in normal for w in v64.v63.v62.NORMAL_WINDOWS),
                decision="retrospective-screen-pass-not-confirmation" if null and all(v==len(changed) for v in counts.values())
                         else "hold-state-action-screen")
    return output


def decision(summary):
    return ("retrospective-state-action-screen-pass-not-confirmation"
            if all(s[MAIN]["decision"].startswith("retrospective-screen-pass") for s in summary.values())
            else "hold-state-action-screen")


def run(args):
    torch.set_num_threads(2)
    baseline = json.loads((args.v62_dir/"run_results/preflight/causal_innovation.json").read_text())
    replay = json.loads((args.v64_dir/"run_results/screen/phase_vector.json").read_text())
    previous = torch.load(args.v60_dir/"run_checkpoints/preflight/preflight_latest.pt",map_location="cpu",weights_only=False)["extra_state"]["progress"]
    assert baseline["revision"]==v64.v63.v62.REVISION and baseline["stage_spec"]==v64.v63.v62.SPEC
    assert replay["revision"]==v64.REVISION and replay["environment_interactions"]==replay["optimizer_updates"]==0
    assert previous["revision"]==v64.v63.v62.v60.REVISION and previous["next_cell"]==16
    model,pooled,phase = previous["references"]["long_normal_refit"]
    assert v64.v63.v62.v60.reference_metadata(dict(v60_frozen=(model,pooled,phase)))==baseline["inherited_reference_metadata"]
    model_metadata = deepcopy(model.metadata())
    del previous
    stored = args.v62_dir/"run_checkpoints/preflight/traces"
    def load_v62(cell):
        kind = "updating" if cell["updating"] else "frozen"
        with np.load(stored/f"{cell['split']}_seed{cell['seed']}_{kind}_{cell['mode']}.npz") as f:
            return {k:f[k] for k in f.files}
    fit_cells = [r for r in baseline["records"] if r["split"]=="fit"]
    cal_cells = [r for r in baseline["records"] if r["split"]=="calibration"]
    assert len(fit_cells)==len(cal_cells)==2 and all(r["mode"]=="normal" for r in fit_cells+cal_cells)
    reference = StateActionReference.fit_normal(model,pooled,[load_v62(r) for r in fit_cells],[load_v62(r) for r in cal_cells])
    metadata = deepcopy(reference.metadata())
    args.checkpoint_dir.mkdir(parents=True,exist_ok=True)
    path = args.checkpoint_dir/"state_action_reference.pt"
    torch.save(dict(revision=REVISION,model=model,reference=reference,reference_metadata=metadata,
        source_training_seed=1,normal_fit_seeds=baseline["stage_spec"]["fit"],normal_calibration_seeds=baseline["stage_spec"]["calibration"]),path)
    print("[REFERENCE] Normal-only state/action centers and OAS covariance frozen before replay",flush=True)
    records,normal = [],None
    for original in replay["records"]:
        cell = {k:original[k] for k in ("group","seed","updating","mode")}
        if cell["group"]=="v62_test":
            trace = load_v62(dict(split="test",**cell))
        else:
            assert cell["group"]=="v61_development_replay"
            with np.load(args.v61_dir/f"run_checkpoints/preflight/traces/seed{cell['seed']}_{cell['mode']}.npz") as f:
                trace = {k:f[k] for k in f.files}
        observers = deepcopy(original["observers"])
        observers.update(observe(trace,model,reference,baseline["stage_spec"]["switch_step"]))
        record = dict(**cell,saved_transitions=len(trace["dones"]),observers=observers)
        if cell["updating"]:
            if cell["mode"]=="normal":
                normal = record
            assert normal["group"]==cell["group"] and normal["seed"]==cell["seed"]
            record["change_minus_nochange"] = {name:{w:d["mean_coefficient"]-normal["observers"][name][w]["mean_coefficient"]
                for w,d in values.items()} for name,values in observers.items()}
            assert all(abs(record["change_minus_nochange"][name][w])<1e-10 for name in NAMES for w in ("first800","pre64"))
        records.append(record)
        print(f"[REPLAY] {cell['group']} seed{cell['seed']} {cell['mode']} updating={cell['updating']}",flush=True)
    assert model_metadata==model.metadata() and metadata==reference.metadata()
    summary = summarize(records)
    output = dict(revision=REVISION,environment_interactions=0,optimizer_updates=0,source_training_seed=1,
        reused_normal_fit_seeds=baseline["stage_spec"]["fit"],reused_normal_calibration_seeds=baseline["stage_spec"]["calibration"],
        inherited_dynamics_unchanged=True,reference_frozen_during_replay=True,known_boundary_used_by_readout=False,
        beta=-2.,readout_controls_actor=False,reference_metadata=metadata,server_reference_checkpoint=str(path),
        records=records,summary=summary,decision=decision(summary),
        replayed_saved_transitions=sum(r["saved_transitions"] for r in records),
        limitations=["All evaluation trajectories are reused development data; screening is not independent confirmation.",
                     "Normal neighbor windows do not imply independent samples or guarantee conditional support.",
                     "The readout does not control actions and this screen does not establish adaptive RL returns."])
    args.result_dir.mkdir(parents=True,exist_ok=True)
    destination = args.result_dir/"state_action.json"
    destination.write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
    print(f"[DONE] {output['decision']}: {destination}",flush=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ("v60-dir","v61-dir","v62-dir","v64-dir","checkpoint-dir","result-dir"):
        parser.add_argument(f"--{name}",type=Path,required=True)
    run(parser.parse_args())


if __name__=="__main__":
    main()
