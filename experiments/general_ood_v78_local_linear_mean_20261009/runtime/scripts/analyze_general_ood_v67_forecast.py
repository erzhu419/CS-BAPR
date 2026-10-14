"""Normal-only causal residual forecast and retrospective v66-plus-forecast replay."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from scripts import analyze_general_ood_v66_covariance_mix as v66
from scripts.v67_forecast_readout import ForecastReference, LEVEL_NAME, MAIN, NAMES, observe_series

REVISION = "general_ood_v67_forecast"
OBSERVERS = (*v66.OBSERVERS,*NAMES)


def observe(trace,model,reference,switch):
    series = observe_series(trace,model,reference)
    windows = v66.v65.v64.window_ranges(len(trace["dones"]),switch)
    output = {}
    for name,key in (("forecast_only","forecast_risk"),(MAIN,"combined_risk")):
        output[name] = {w:dict(mean_coefficient=float(series[key][a:b].mean()),
            mean_level_coefficient=float(series["level_risk"][a:b].mean()),
            mean_forecast_coefficient=float(series["forecast_risk"][a:b].mean()),
            mean_level_signal=float(series["level_signal"][a:b].mean()),
            mean_forecast_signal=float(series["forecast_signal"][a:b].mean()),
            mean_forecast_threshold=float(series["forecast_threshold"][a:b].mean()),
            forecast_dominant_fraction=float((series["forecast_risk"][a:b]>series["level_risk"][a:b]).mean()),
            max_forecast_margin=float((series["forecast_signal"][a:b]-series["forecast_threshold"][a:b]).max()),
            forecast_above_threshold_fraction=float((series["forecast_signal"][a:b]>series["forecast_threshold"][a:b]).mean()))
            for w,(a,b) in windows.items()}
    level = {w:dict(mean_coefficient=float(series["level_risk"][a:b].mean()),mean_signal=float(series["level_signal"][a:b].mean()))
             for w,(a,b) in windows.items()}
    return output,level


def summarize(records):
    output = {}
    for group in ("v62_test","v61_development_replay"):
        normal = [r for r in records if r["group"]==group and r["mode"]=="normal"]
        changed = [r for r in records if r["group"]==group and r["mode"]!="normal"]
        output[group] = {}
        for name in OBSERVERS:
            null = all(r["observers"][name][w]["mean_coefficient"]<=.1 for r in normal for w in v66.v65.v64.v63.v62.NORMAL_WINDOWS)
            counts = {w:sum(r["change_minus_nochange"][name][w]>0 for r in changed) for w in v66.v65.v64.v63.v62.CHANGE_WINDOWS}
            output[group][name] = dict(usable_normal_null=null,positive_pairs=counts,changed_pairs=len(changed),
                largest_gated_normal_mean=max(r["observers"][name][w]["mean_coefficient"] for r in normal for w in v66.v65.v64.v63.v62.NORMAL_WINDOWS),
                decision="retrospective-screen-pass-not-confirmation" if null and all(v==len(changed) for v in counts.values())
                         else "hold-forecast-screen")
    return output


def decision(summary):
    return ("retrospective-forecast-screen-pass-not-confirmation"
            if all(s[MAIN]["decision"].startswith("retrospective-screen-pass") for s in summary.values())
            else "hold-forecast-screen")


def run(args):
    torch.set_num_threads(2)
    baseline = json.loads((args.v62_dir/"run_results/preflight/causal_innovation.json").read_text())
    replay = json.loads((args.v66_dir/"run_results/screen/covariance_mix.json").read_text())
    previous = torch.load(args.v66_dir/"run_checkpoints/reference/covariance_mix_reference.pt",map_location="cpu",weights_only=False)
    assert baseline["revision"]==v66.v65.v64.v63.v62.REVISION and baseline["stage_spec"]==v66.v65.v64.v63.v62.SPEC
    assert replay["revision"]==previous["revision"]==v66.REVISION
    assert replay["environment_interactions"]==replay["optimizer_updates"]==0
    model,level = previous["model"],previous["reference"]
    assert level.metadata()==previous["reference_metadata"]==replay["reference_metadata"]
    assert previous["normal_fit_seeds"]==baseline["stage_spec"]["fit"]
    assert previous["normal_calibration_seeds"]==baseline["stage_spec"]["calibration"]
    model_metadata,level_metadata = deepcopy(model.metadata()),deepcopy(level.metadata())
    stored = args.v62_dir/"run_checkpoints/preflight/traces"
    def load_v62(cell):
        kind = "updating" if cell["updating"] else "frozen"
        with np.load(stored/f"{cell['split']}_seed{cell['seed']}_{kind}_{cell['mode']}.npz") as f:
            return {k:f[k] for k in f.files}
    fit_cells = [r for r in baseline["records"] if r["split"]=="fit"]
    cal_cells = [r for r in baseline["records"] if r["split"]=="calibration"]
    assert len(fit_cells)==len(cal_cells)==2 and all(r["mode"]=="normal" for r in fit_cells+cal_cells)
    reference = ForecastReference.fit_normal(model,level,[load_v62(r) for r in fit_cells],[load_v62(r) for r in cal_cells])
    metadata = deepcopy(reference.metadata())
    args.checkpoint_dir.mkdir(parents=True,exist_ok=True)
    path = args.checkpoint_dir/"forecast_reference.pt"
    torch.save(dict(revision=REVISION,model=model,reference=reference,reference_metadata=metadata,
        source_training_seed=1,normal_fit_seeds=baseline["stage_spec"]["fit"],normal_calibration_seeds=baseline["stage_spec"]["calibration"]),path)
    print("[REFERENCE] Normal-only residual forecaster and calibration frozen; v66 unchanged",flush=True)
    records,normal,reproduction_error = [],None,0.
    for original in replay["records"]:
        cell = {k:original[k] for k in ("group","seed","updating","mode")}
        if cell["group"]=="v62_test":
            trace = load_v62(dict(split="test",**cell))
        else:
            assert cell["group"]=="v61_development_replay"
            with np.load(args.v61_dir/f"run_checkpoints/preflight/traces/seed{cell['seed']}_{cell['mode']}.npz") as f:
                trace = {k:f[k] for k in f.files}
        observers = deepcopy(original["observers"])
        added,reconstructed = observe(trace,model,reference,baseline["stage_spec"]["switch_step"])
        reproduction_error = max(reproduction_error,max(abs(value-observers[LEVEL_NAME][w][k])
            for w,values in reconstructed.items() for k,value in values.items()))
        assert reproduction_error<1e-10, "v66-level-reproduction-failed"
        observers.update(added)
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
    assert model_metadata==model.metadata() and level_metadata==level.metadata() and metadata==reference.metadata()
    summary = summarize(records)
    output = dict(revision=REVISION,environment_interactions=0,optimizer_updates=0,source_training_seed=1,
        reused_normal_fit_seeds=baseline["stage_spec"]["fit"],reused_normal_calibration_seeds=baseline["stage_spec"]["calibration"],
        inherited_dynamics_and_level_unchanged=True,inherited_reference_metadata=level_metadata,
        reference_frozen_during_replay=True,known_boundary_used_by_readout=False,
        beta=-2.,readout_controls_actor=False,reference_metadata=metadata,server_reference_checkpoint=str(path),
        maximum_v66_reproduction_error=reproduction_error,records=records,summary=summary,decision=decision(summary),
        replayed_saved_transitions=sum(r["saved_transitions"] for r in records),
        limitations=["Reused development trajectories from one source training seed; not independent confirmation.",
                     "Risk union preserves the v66 per-transition risk floor, not its paired contrasts or normal bounds.",
                     "This fixed-beta readout replay does not establish adaptive RL returns."])
    args.result_dir.mkdir(parents=True,exist_ok=True)
    destination = args.result_dir/"forecast.json"
    destination.write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
    print(f"[DONE] {output['decision']}: {destination}",flush=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ("v61-dir","v62-dir","v66-dir","checkpoint-dir","result-dir"):
        parser.add_argument(f"--{name}",type=Path,required=True)
    run(parser.parse_args())


if __name__=="__main__":
    main()
