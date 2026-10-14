"""Localize weak gravity response without fitting or changing the readout."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import numpy as np
import torch

from scripts import probe_general_ood_v53_behavior_coverage as v53
from scripts.v52_agent import ContinuationAgent


def residual_windows(trace, model, switch_step):
    calibration = model.calibrations["global-centered"]
    normalized = (model.errors(trace) - calibration.means[0]) / calibration.scales[0]
    # The same causal history spans physical transitions and episode resets.
    rolling = np.asarray([normalized[max(0, i - 15):i + 1].mean(axis=0) for i in range(len(normalized))])
    windows = dict(prefix=(0, switch_step), first_post50=(switch_step, switch_step + 50),
                   post=(switch_step, len(normalized)), last64=(len(normalized) - 64, len(normalized)))
    return {name: dict(signed_normalized_mean=normalized[start:end].mean(axis=0).tolist(),
        mean_normalized_square=(normalized[start:end] ** 2).mean(axis=0).tolist(),
        rolling16_mean_squared=(rolling[start:end] ** 2).mean(axis=0).tolist())
        for name, (start, end) in windows.items()}


def run(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    completed = torch.load(args.completed_checkpoint, map_location="cpu", weights_only=False)["extra_state"]["progress"]
    original = json.loads(args.completed_result.read_text())
    assert completed["stage"] == original["stage"] == "preflight"
    spec = original["stage_spec"]
    assert completed["next_cell"] == len(v53.trajectory_specs(spec))
    control, baseline = completed["control"]
    source_record = next(r for r in original["records"] if r["split"] == "test" and r["mode"] == "high_grav")
    cell = {k: source_record[k] for k in ("split", "seed", "updating", "mode")}
    shared = torch.load(args.shared_dir / "shared.pt", map_location="cpu", weights_only=False)
    env = gym.make("Hopper-v4")
    agent = ContinuationAgent(env, "fixed-minus2")
    agent.start_shared(args.source_dir / v53.SOURCE_NAME, shared)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint_dir / "gravity_latest.pt"
    loop = None
    if args.resume_from:
        extra = agent.load(str(args.resume_from))
        loop = extra["loop"]
        if loop is not None:
            mode = "high_grav" if loop["step"] > spec["switch_step"] else "normal"
            v53.apply_mode_to_env(env, "Hopper-v4", v53.MUJOCO_MODE_PROFILES["Hopper-v4"][mode])
            env.reset(seed=cell["seed"])
            v53.restore_environment(env, extra["environment"])
    if args.resume_from and loop is None:
        trace, record = extra["progress"]["trace"], extra["progress"]["record"]
    else:
        trace, record = v53.collect(agent, env, cell, spec, loop,
            lambda active: v53.save_progress(agent, env, dict(stage="gravity-localization"), active, checkpoint))
    assert baseline["seed"] == cell["seed"]
    assert v53.paired_prefix_equal(trace, control, spec["switch_step"])
    assert record["mean_q_loss"] == source_record["mean_q_loss"], "nonmatching-replay-of-completed-cell"
    for name, (model, reference) in completed["models"].items():
        assert v53.observe(trace, model, reference, spec["switch_step"]) == source_record["observers"][name]
    model, reference = completed["models"]["broad"]
    normal = residual_windows(control, model, spec["switch_step"])
    gravity = residual_windows(trace, model, spec["switch_step"])
    assert normal["prefix"] == gravity["prefix"]
    output = dict(revision="general_ood_v53_gravity_localization", stage="preflight",
        reused_test_seed=cell["seed"], source_checkpoint=v53.SOURCE_NAME, exact_reproduction_of_completed_gravity_cell=True,
        added_interactions=spec["steps"], added_optimizer_updates=record["optimizer_updates"], beta=-2.,
        reference_refitted=False, readout_or_threshold_changed=False, auxiliary_controls_actions=False,
        coordinate_order="Hopper-v4 observation indices, 0-based", normal=normal, high_grav=gravity,
        change_minus_nochange={phase: {key: (np.asarray(values) - normal[phase][key]).tolist()
            for key, values in dimensions.items()} for phase, dimensions in gravity.items()},
        coefficient_contrasts=source_record["change_minus_nochange"]["broad"],
        limitations=["Replays one development seed to inspect previously unreported residual coordinates; not independent confirmation.",
                     "Rolling signed residual statistics have no normal calibration or action effect in this diagnostic."])
    args.result_dir.mkdir(parents=True, exist_ok=True)
    destination = args.result_dir / "gravity_localization.json"
    destination.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    v53.save_progress(agent, env, dict(stage="gravity-localization", trace=trace, record=record), None, checkpoint)
    env.close()
    print(f"[DONE] Exact gravity replay with residual localization: {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ("source-dir", "shared-dir", "completed-checkpoint", "completed-result", "checkpoint-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
