"""Replay a failed v55 tail to separate initial and reset calibration sources."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import numpy as np
import torch

from csbapr.belief.departure import DepartureReadout, SETTINGS
from scripts import probe_general_ood_v53_behavior_coverage as v53
from scripts import probe_general_ood_v55_learning_calibration as v55
from scripts.v52_agent import ContinuationAgent
from scripts.v54_window_readout import observe_trace

REVISION = "general_ood_v56_reset_calibration"
AUDIT_AGES = (128, 144, 160, 176, 192, 800, 816, 832, 848)


def episode_ordinals(dones):
    return np.cumsum(np.r_[0, np.asarray(dones, dtype=int)[:-1]])


def replay_cells(original):
    assert original["stage"] == "formal"
    failed = [r for r in original["records"] if r["split"] == "test" and r["mode"] != "normal"
              and any(r["change_minus_nochange"]["learning_calibrated"][w] <= 0
                      for w in ("first_post50", "last64"))]
    assert len(failed) == 1 and failed[0]["mode"] == "high_grav"
    cells = [c for c in v55.trajectory_specs(original["stage_spec"]) if c["split"] == "calibration"]
    return cells + [{k: failed[0][k] for k in ("split", "seed", "updating", "mode")}]


def population_windows(values, reference):
    signals, ages, ordinals = (np.concatenate([v[i] for v in values]) for i in range(3))
    records = []
    for age in AUDIT_AGES:
        selected = np.abs(ages - age) <= SETTINGS["age_radius"]
        groups = {}
        for name, mask in (("initial", ordinals == 0), ("after_reset", ordinals > 0)):
            group = signals[selected & mask]
            groups[name] = dict(count=len(group), q95=float(np.quantile(group, .95)) if len(group) else None,
                maximum=float(group.max()) if len(group) else None)
        threshold = float(reference.age_reference.threshold(age))
        records.append(dict(age=age, pooled_threshold=threshold, **groups,
            threshold_exceeds_after_reset_max=(threshold > groups["after_reset"]["maximum"])
            if groups["after_reset"]["count"] else None))
    return records


def window_details(signals, ages, dones, reference, switch_step):
    signals, ages = np.asarray(signals), np.asarray(ages)
    ordinals = episode_ordinals(dones)
    departure = DepartureReadout(reference.age_reference)
    coefficients = np.asarray([departure.update(s, a) for s, a in zip(signals, ages)])
    thresholds = np.asarray([reference.age_reference.threshold(a) for a in ages])
    excess = np.maximum(signals - thresholds, 0.)
    length = len(signals)
    windows = dict(full=(0, length), prefix=(0, switch_step), first_post50=(switch_step, switch_step + 50),
                   post=(switch_step, length), last64=(length - 64, length))
    for index, reset in enumerate(np.flatnonzero(dones) + 1):
        if reset < length:
            windows[f"reset{index + 1}_before64"] = (max(0, reset - 64), reset)
            windows[f"reset{index + 1}_first50"] = (reset, min(length, reset + 50))
    return {name: dict(start=int(start), end=int(end),
        episode_ordinal_min=int(ordinals[start:end].min()), episode_ordinal_max=int(ordinals[start:end].max()),
        age_min=int(ages[start:end].min()), age_max=int(ages[start:end].max()),
        mean_signal=float(signals[start:end].mean()), min_signal=float(signals[start:end].min()),
        max_signal=float(signals[start:end].max()), mean_threshold=float(thresholds[start:end].mean()),
        min_threshold=float(thresholds[start:end].min()), max_threshold=float(thresholds[start:end].max()),
        above_threshold_fraction=float(np.mean(excess[start:end] > 0)),
        mean_instant_risk=float(np.mean(-np.expm1(-excess[start:end]))),
        mean_coefficient=float(coefficients[start:end].mean())) for name, (start, end) in windows.items()}


def run(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    original = json.loads(args.completed_result.read_text())
    completed = torch.load(args.completed_checkpoint, map_location="cpu", weights_only=False)["extra_state"]["progress"]
    assert completed["stage"] == original["stage"] == "formal"
    assert completed["next_cell"] == len(v55.trajectory_specs(original["stage_spec"]))
    control, baseline = completed["control"]
    reference = completed["reference"]
    assert reference.metadata() == original["learning_reference_metadata"]
    del completed
    dynamics = torch.load(args.dynamics_checkpoint, map_location="cpu", weights_only=False)["extra_state"]["progress"]
    model = dynamics["models"]["broad"][0]
    del dynamics
    shared = torch.load(args.shared_dir / "shared.pt", map_location="cpu", weights_only=False)
    spec, cells = original["stage_spec"], replay_cells(original)
    assert baseline["seed"] == cells[-1]["seed"]
    assert observe_trace(control, model, reference, spec["switch_step"]) == baseline["observers"]["learning_calibrated"]
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint_dir / "localization_latest.pt"
    extra = torch.load(args.resume_from, map_location="cpu", weights_only=False)["extra_state"] if args.resume_from else None
    progress = extra["progress"] if extra else dict(stage=REVISION, next_cell=0, calibration=[], records=[], gravity=None)
    assert progress["stage"] == REVISION
    for index in range(progress["next_cell"], len(cells)):
        cell = cells[index]
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
        expected = next(r for r in original["records"] if all(r[k] == cell[k] for k in cell))
        assert record == {k: expected[k] for k in record}, "nonmatching-replay-of-completed-cell"
        if cell["split"] == "calibration":
            progress["calibration"].append(trace)
        else:
            assert v53.paired_prefix_equal(trace, control, spec["switch_step"])
            assert observe_trace(trace, model, reference, spec["switch_step"]) == expected["observers"]["learning_calibrated"]
            progress["gravity"] = trace
        progress["records"].append(record)
        progress["next_cell"] = index + 1
        v53.save_progress(agent, env, progress, None, checkpoint)
        print(f"[CELL] Exact replay {index + 1}/{len(cells)}: {cell}", flush=True)
        extra = None
        env.close()
        del agent
        torch.cuda.empty_cache()
    reproduced = v55.calibrate_learning(model, reference, progress["calibration"])
    np.testing.assert_array_equal(reproduced.age_reference.ages, reference.age_reference.ages)
    np.testing.assert_allclose(reproduced.age_reference.upper, reference.age_reference.upper, rtol=0., atol=1e-12)
    assert reproduced.age_reference.counts == reference.age_reference.counts
    values = [(*v55.signals_and_ages(model, reference, trace), episode_ordinals(trace["dones"]))
              for trace in progress["calibration"]]
    windows = {name: window_details(*v55.signals_and_ages(model, reference, trace), trace["dones"], reference,
                                  spec["switch_step"]) for name, trace in (("normal", control), ("high_grav", progress["gravity"]))}
    output = dict(revision=REVISION, reused_calibration_seeds=spec["calibration"], reused_test_seed=baseline["seed"],
        exact_reproduction_of_completed_cells=True, old_age_calibration_reproduced_atol=1e-12,
        readout_reference_changed=False, gate_changed=False, auxiliary_controls_actions=False, beta=-2.,
        added_interactions=sum(r["interactions"] for r in progress["records"]),
        added_optimizer_updates=sum(r["optimizer_updates"] for r in progress["records"]),
        calibration_populations=population_windows(values, reference), windows=windows,
        change_minus_nochange={w: {k: windows["high_grav"][w][k] - windows["normal"][w][k]
            for k in ("mean_signal", "mean_coefficient", "above_threshold_fraction")} for w in windows["normal"]},
        decision="localization-only-v55-remains-hold",
        limitations=["Reused development trajectories, not independent confirmation or an RL return comparison.",
                     "Population statistics localize calibration; no new conditional reference is fitted or evaluated."])
    args.result_dir.mkdir(parents=True, exist_ok=True)
    destination = args.result_dir / "reset_calibration_localization.json"
    destination.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(f"[DONE] {output['decision']}: {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ("source-dir", "shared-dir", "dynamics-checkpoint", "completed-checkpoint", "completed-result", "checkpoint-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
