"""Server-only replay attribution; no fitting, environment interaction or updates."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from csbapr.belief.departure import SETTINGS
from scripts import probe_general_ood_v61_early_response as v61
from scripts import probe_general_ood_v62_causal_innovation as v62
from scripts.probe_general_ood_v56_reset_calibration import episode_ordinals

REVISION = "general_ood_v63_readout_attribution"
TOLERANCE = 1e-8


def arrays(trace, model, reference, old_phase):
    moments, innovations = v62.window_features(v62.normalized_residuals(model, trace))
    values = reference.signals(moments, innovations)
    ages = v62.v53.v43.episode_steps(trace["dones"])
    observers = {}
    for name in v62.OBSERVERS:
        signal = values["level_recalibrated" if name == "v60_frozen" else name]
        sensor = v62.PhaseDepartureReadout(old_phase if name == "v60_frozen" else reference.phases[name])
        risks, instant, thresholds = [], [], []
        for s, age in zip(signal, ages):
            risks.append(sensor.update(s, int(age)))
            instant.append(sensor.departure.instant_risk)
            thresholds.append(sensor.departure.threshold)
        observers[name] = dict(signal=signal, risk=np.asarray(risks), instant=np.asarray(instant),
                               threshold=np.asarray(thresholds))
    inputs = (model.inputs(trace)-model.input_mean)/model.input_scale
    return dict(observers=observers, moments=moments, ages=ages,
                level=values["level_ratio"], innovation=values["innovation_ratio"],
                input_energy=np.mean(inputs**2,axis=1))


def window_statistics(data, start, end, support95):
    result = {}
    for name, values in data["observers"].items():
        signal, threshold = values["signal"][start:end], values["threshold"][start:end]
        margin = signal-threshold
        result[name] = dict(mean_coefficient=float(values["risk"][start:end].mean()),
            mean_signal=float(signal.mean()), mean_threshold=float(threshold.mean()),
            mean_instant_coefficient=float(values["instant"][start:end].mean()),
            mean_margin=float(margin.mean()), max_margin=float(margin.max()),
            above_threshold_fraction=float((margin>0).mean()),
            threshold_min=float(threshold.min()), threshold_max=float(threshold.max()))
    common = dict(mean_signed_moment=data["moments"][start:end].mean(axis=0).tolist(),
        mean_level_ratio=float(data["level"][start:end].mean()),
        mean_innovation_ratio=float(data["innovation"][start:end].mean()),
        innovation_dominant_fraction=float((data["innovation"][start:end]>data["level"][start:end]).mean()),
        mean_input_energy=float(data["input_energy"][start:end].mean()),
        input_above_normal_fit95_fraction=float((data["input_energy"][start:end]>support95).mean()),
        episode_age_min=int(data["ages"][start:end].min()), episode_age_max=int(data["ages"][start:end].max()))
    return dict(observers=result, common=common)


def diagnose(data, switch, support95):
    n = len(data["ages"])
    windows = dict(first800=(0,min(800,n)), full=(0,n), last1000=(max(0,n-1000),n),
        pre64=(max(0,switch-64),switch), first_post50=(switch,min(n,switch+50)), last64=(max(0,n-64),n))
    for a,b in ((0,20),(20,50),(50,100),(100,200)):
        if switch+a<n:
            windows[f"post{a}_{b}"] = (switch+a,min(n,switch+b))
    return {w:window_statistics(data,a,b,support95) for w,(a,b) in windows.items()}


def reproduction_error(diagnostic, expected):
    return max(abs(diagnostic[w]["observers"][name][key]-values[w][key])
        for name,values in expected.items() for w in ("first_post50","last64")
        for key in ("mean_coefficient","mean_signal"))


def paired_differences(record, normal, pooled):
    output = {}
    for window, current in record["diagnostic"].items():
        control = normal["diagnostic"][window]
        delta = np.asarray(current["common"]["mean_signed_moment"])-control["common"]["mean_signed_moment"]
        output[window] = dict(observers={name:{key:value-control["observers"][name][key]
            for key,value in values.items()} for name,values in current["observers"].items()},
            signed_mean_displacement=delta.tolist(),
            signed_mean_displacement_energy=float(delta@pooled.precision@delta/len(delta)))
    return output


def calibration_knots(calibration, model, reference, old_phase):
    data = [arrays(t,model,reference,old_phase) for t in calibration]
    ages = np.concatenate([d["ages"] for d in data])
    ordinal = np.concatenate([episode_ordinals(t["dones"]) for t in calibration])
    output = []
    for age in (800,816,832,848,944,960,976,992,999):
        mask = (np.abs(ages-age)<=SETTINGS["age_radius"])&(ordinal>0)
        entry = dict(age=age, count=int(mask.sum()), observers={})
        for name in v62.OBSERVERS:
            signals = np.concatenate([d["observers"][name]["signal"] for d in data])
            threshold = (old_phase if name == "v60_frozen" else reference.phases[name]).after_reset.threshold(age)
            empirical = float(np.quantile(signals[mask],SETTINGS["quantile"]))
            if name != "v60_frozen":
                assert abs(empirical-threshold)<1e-10
            entry["observers"][name] = dict(threshold=float(threshold), normal_calibration95=empirical,
                normal_calibration_mean=float(signals[mask].mean()), normal_calibration_max=float(signals[mask].max()))
        output.append(entry)
    return output


def run(args):
    torch.set_num_threads(2)
    baseline = json.loads((args.v62_dir/"run_results/preflight/causal_innovation.json").read_text())
    legacy = json.loads((args.v61_dir/"run_results/preflight/early_response.json").read_text())
    progress = torch.load(args.v62_dir/"run_checkpoints/preflight/preflight_latest.pt",
                          map_location="cpu",weights_only=False)["extra_state"]["progress"]
    previous = torch.load(args.v60_dir/"run_checkpoints/preflight/preflight_latest.pt",
                          map_location="cpu",weights_only=False)["extra_state"]["progress"]
    assert baseline["revision"] == v62.REVISION and legacy["revision"] == v61.REVISION
    assert progress["revision"] == v62.REVISION and progress["next_cell"] == previous["next_cell"] == 16
    model, pooled, old_phase = previous["references"]["long_normal_refit"]
    reference = progress["reference"]
    assert v62.v60.reference_metadata(dict(v60_frozen=(model,pooled,old_phase))) == baseline["inherited_reference_metadata"]
    assert reference.metadata() == baseline["reference_metadata"]
    frozen_metadata = deepcopy(reference.metadata())
    del progress, previous
    stored = args.v62_dir/"run_checkpoints/preflight/traces"
    def load_v62(cell):
        kind = 'updating' if cell['updating'] else 'frozen'
        with np.load(stored/f"{cell['split']}_seed{cell['seed']}_{kind}_{cell['mode']}.npz") as f:
            return {k:f[k] for k in f.files}
    fit_cells = [r for r in baseline["records"] if r["split"] == "fit"]
    support = [arrays(load_v62(r),model,reference,old_phase)["input_energy"] for r in fit_cells]
    support95 = float(np.quantile(np.concatenate(support),.95))
    calibration = [load_v62(r) for r in baseline["records"] if r["split"] == "calibration"]
    knots = calibration_knots(calibration,model,reference,old_phase)
    records, control, normal = [], None, None
    for group, originals in (("v62_test",[r for r in baseline["records"] if r["split"] == "test"]),
                             ("v61_development_replay",legacy["records"])):
        for cell in originals:
            if group == "v62_test":
                trace = load_v62(cell)
                expected = cell["observers"]
                seed, updating = cell["seed"], cell["updating"]
            else:
                with np.load(args.v61_dir/f"run_checkpoints/preflight/traces/seed{v61.SEED}_{cell['mode']}.npz") as f:
                    trace = {k:f[k] for k in f.files}
                expected = dict(v60_frozen={w:dict(mean_coefficient=cell['diagnostic']['windows'][w]['risk'],
                    mean_signal=cell['diagnostic']['windows'][w]['signal']) for w in ("first_post50","last64")})
                seed, updating = v61.SEED, True
            diagnostic = diagnose(arrays(trace,model,reference,old_phase),v62.SPEC["switch_step"],support95)
            error = reproduction_error(diagnostic,expected)
            assert error <= TOLERANCE, f"unmatched replay: {group}/{seed}/{cell['mode']}: {error}"
            record = dict(group=group,seed=seed,updating=updating,mode=cell["mode"],diagnostic=diagnostic,
                          reproduction_max_abs_error=error,saved_transitions=len(trace["dones"]))
            if updating:
                if cell["mode"] == "normal":
                    control,normal = trace,record
                assert normal["seed"] == seed and normal["group"] == group
                assert v62.v53.paired_prefix_equal(trace,control,v62.SPEC["switch_step"])
                record["change_minus_nochange"] = paired_differences(record,normal,pooled)
            records.append(record)
    assert frozen_metadata == reference.metadata()
    summaries = {}
    for group in ("v62_test","v61_development_replay"):
        converted = []
        for r in records:
            if r["group"] != group:
                continue
            converted.append(dict(split="test",mode=r["mode"],
                observers={name:{w:d["observers"][name] for w,d in r["diagnostic"].items()} for name in v62.OBSERVERS},
                change_minus_nochange={name:{w:d["observers"][name]["mean_coefficient"]
                    for w,d in r.get("change_minus_nochange",{}).items()} for name in v62.OBSERVERS}))
        summaries[group] = v62.summarize(converted)
    assert summaries["v62_test"] == baseline["summary"]
    output = dict(revision=REVISION,environment_interactions=0,optimizer_updates=0,reference_refitted=False,
        threshold_modified=False,readout_controls_actor=False,known_boundary_used_by_readout=False,
        input_support_normal_fit95=support95,calibration_knots=knots,records=records,summary=summaries,
        replayed_saved_transitions=sum(r["saved_transitions"] for r in records),
        maximum_reproduction_error=max(r["reproduction_max_abs_error"] for r in records),
        limitations=["Reused saved trajectories are retrospective diagnostics, not new independent repetitions.",
                     "Matched signed displacement and input support are descriptive, not a calibrated online decision rule."])
    args.result_dir.mkdir(parents=True,exist_ok=True)
    destination = args.result_dir/"readout_attribution.json"
    destination.write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
    print(f"[DONE] {len(records)} exact readout replays; new interactions=0, updates=0: {destination}",flush=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ("v60-dir","v61-dir","v62-dir","result-dir"):
        parser.add_argument(f"--{name}",type=Path,required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
