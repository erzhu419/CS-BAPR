"""Normal-only covariance mixing and retrospective fixed-beta readout replay."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from scripts import analyze_general_ood_v65_state_action as v65
from scripts.v54_window_readout import normalized_residuals, rolling_means
from scripts.v66_covariance_mix_readout import NAME, MixedStateActionReference

REVISION = "general_ood_v66_covariance_mix"
OBSERVERS = (*v65.OBSERVERS,NAME)


def observe(trace,model,reference,switch):
    ages = v65.v64.v63.v62.v53.v43.episode_steps(trace["dones"])
    signals,geometry = reference.evaluate(rolling_means(normalized_residuals(model,trace)),rolling_means(model.inputs(trace)))
    sensor = v65.v64.v63.v62.PhaseDepartureReadout(reference.phases[NAME])
    risks,instant,thresholds = [],[],[]
    for signal,age in zip(signals[NAME],ages):
        risks.append(sensor.update(signal,int(age)))
        instant.append(sensor.departure.instant_risk)
        thresholds.append(sensor.departure.threshold)
    risks,instant,thresholds = np.asarray(risks),np.asarray(instant),np.asarray(thresholds)
    return {NAME:{w:dict(mean_coefficient=float(risks[a:b].mean()),mean_signal=float(signals[NAME][a:b].mean()),
        mean_instant_coefficient=float(instant[a:b].mean()),mean_threshold=float(thresholds[a:b].mean()),
        max_margin=float((signals[NAME][a:b]-thresholds[a:b]).max()),
        above_threshold_fraction=float((signals[NAME][a:b]>thresholds[a:b]).mean()),
        mean_shrinkage=float(geometry["shrinkage"][a:b].mean()),mean_covariance_trace=float(geometry["covariance_trace"][a:b].mean()),
        mean_neighbor_radius=float(geometry["neighbor_radius"][a:b].mean()),
        outside_calibration_radius95_fraction=float((geometry["neighbor_radius"][a:b]>reference.neighbor_radius95).mean()))
        for w,(a,b) in v65.v64.window_ranges(len(ages),switch).items()}}


def summarize(records):
    output = {}
    for group in ("v62_test","v61_development_replay"):
        normal = [r for r in records if r["group"]==group and r["mode"]=="normal"]
        changed = [r for r in records if r["group"]==group and r["mode"]!="normal"]
        output[group] = {}
        for name in OBSERVERS:
            null = all(r["observers"][name][w]["mean_coefficient"]<=.1 for r in normal for w in v65.v64.v63.v62.NORMAL_WINDOWS)
            counts = {w:sum(r["change_minus_nochange"][name][w]>0 for r in changed) for w in v65.v64.v63.v62.CHANGE_WINDOWS}
            output[group][name] = dict(usable_normal_null=null,positive_pairs=counts,changed_pairs=len(changed),
                largest_gated_normal_mean=max(r["observers"][name][w]["mean_coefficient"] for r in normal for w in v65.v64.v63.v62.NORMAL_WINDOWS),
                decision="retrospective-screen-pass-not-confirmation" if null and all(v==len(changed) for v in counts.values())
                         else "hold-covariance-mix-screen")
    return output


def decision(summary):
    return ("retrospective-covariance-mix-screen-pass-not-confirmation"
            if all(s[NAME]["decision"].startswith("retrospective-screen-pass") for s in summary.values())
            else "hold-covariance-mix-screen")


def run(args):
    torch.set_num_threads(2)
    baseline = json.loads((args.v62_dir/"run_results/preflight/causal_innovation.json").read_text())
    replay = json.loads((args.v65_dir/"run_results/screen/state_action.json").read_text())
    previous = torch.load(args.v65_dir/"run_checkpoints/reference/state_action_reference.pt",map_location="cpu",weights_only=False)
    assert baseline["revision"]==v65.v64.v63.v62.REVISION and baseline["stage_spec"]==v65.v64.v63.v62.SPEC
    assert replay["revision"]==previous["revision"]==v65.REVISION
    assert replay["environment_interactions"]==replay["optimizer_updates"]==0
    model,original = previous["model"],previous["reference"]
    assert original.metadata()==previous["reference_metadata"]==replay["reference_metadata"]
    assert previous["normal_fit_seeds"]==baseline["stage_spec"]["fit"]
    assert previous["normal_calibration_seeds"]==baseline["stage_spec"]["calibration"]
    model_metadata,original_metadata = deepcopy(model.metadata()),deepcopy(original.metadata())
    stored = args.v62_dir/"run_checkpoints/preflight/traces"
    def load_v62(cell):
        kind = "updating" if cell["updating"] else "frozen"
        with np.load(stored/f"{cell['split']}_seed{cell['seed']}_{kind}_{cell['mode']}.npz") as f:
            return {k:f[k] for k in f.files}
    fit_cells = [r for r in baseline["records"] if r["split"]=="fit"]
    cal_cells = [r for r in baseline["records"] if r["split"]=="calibration"]
    assert len(fit_cells)==len(cal_cells)==2 and all(r["mode"]=="normal" for r in fit_cells+cal_cells)
    reference = MixedStateActionReference.fit_normal(model,original,[load_v62(r) for r in fit_cells],[load_v62(r) for r in cal_cells])
    metadata = deepcopy(reference.metadata())
    args.checkpoint_dir.mkdir(parents=True,exist_ok=True)
    path = args.checkpoint_dir/"covariance_mix_reference.pt"
    torch.save(dict(revision=REVISION,model=model,reference=reference,reference_metadata=metadata,
        source_training_seed=1,normal_fit_seeds=baseline["stage_spec"]["fit"],normal_calibration_seeds=baseline["stage_spec"]["calibration"]),path)
    print(f"[REFERENCE] Normal-block-selected global weight={reference.global_weight:.8f}; reference frozen",flush=True)
    records,normal = [],None
    for original_record in replay["records"]:
        cell = {k:original_record[k] for k in ("group","seed","updating","mode")}
        if cell["group"]=="v62_test":
            trace = load_v62(dict(split="test",**cell))
        else:
            assert cell["group"]=="v61_development_replay"
            with np.load(args.v61_dir/f"run_checkpoints/preflight/traces/seed{cell['seed']}_{cell['mode']}.npz") as f:
                trace = {k:f[k] for k in f.files}
        observers = deepcopy(original_record["observers"])
        observers.update(observe(trace,model,reference,baseline["stage_spec"]["switch_step"]))
        record = dict(**cell,saved_transitions=len(trace["dones"]),observers=observers)
        if cell["updating"]:
            if cell["mode"]=="normal":
                normal = record
            assert normal["group"]==cell["group"] and normal["seed"]==cell["seed"]
            record["change_minus_nochange"] = {name:{w:d["mean_coefficient"]-normal["observers"][name][w]["mean_coefficient"]
                for w,d in values.items()} for name,values in observers.items()}
            assert all(abs(record["change_minus_nochange"][NAME][w])<1e-10 for w in ("first800","pre64"))
        records.append(record)
        print(f"[REPLAY] {cell['group']} seed{cell['seed']} {cell['mode']} updating={cell['updating']}",flush=True)
    assert model_metadata==model.metadata() and original_metadata==original.metadata() and metadata==reference.metadata()
    summary = summarize(records)
    output = dict(revision=REVISION,environment_interactions=0,optimizer_updates=0,source_training_seed=1,
        reused_normal_fit_seeds=baseline["stage_spec"]["fit"],reused_normal_calibration_seeds=baseline["stage_spec"]["calibration"],
        inherited_dynamics_and_centers_unchanged=True,reference_frozen_during_replay=True,known_boundary_used_by_readout=False,
        beta=-2.,readout_controls_actor=False,reference_metadata=metadata,server_reference_checkpoint=str(path),
        records=records,summary=summary,decision=decision(summary),
        replayed_saved_transitions=sum(r["saved_transitions"] for r in records),
        limitations=["Reused development trajectories from one source training seed; not independent confirmation.",
                     "Blocked normal predictive likelihood is not an OOD detection objective or a coverage guarantee.",
                     "This fixed-beta readout replay does not establish adaptive RL returns."])
    args.result_dir.mkdir(parents=True,exist_ok=True)
    destination = args.result_dir/"covariance_mix.json"
    destination.write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
    print(f"[DONE] {output['decision']}: {destination}",flush=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ("v61-dir","v62-dir","v65-dir","checkpoint-dir","result-dir"):
        parser.add_argument(f"--{name}",type=Path,required=True)
    run(parser.parse_args())


if __name__=="__main__":
    main()
