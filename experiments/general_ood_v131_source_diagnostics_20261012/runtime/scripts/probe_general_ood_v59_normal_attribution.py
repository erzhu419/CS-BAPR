"""Six-thousand-step normal nulls with and without optimizer updates."""
from copy import deepcopy
from dataclasses import asdict
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import numpy as np
import torch

from csbapr.belief.departure import DepartureReadout
from scripts import probe_general_ood_v53_behavior_coverage as v53
from scripts.probe_general_ood_v55_learning_calibration import signals_and_ages
from scripts.v52_agent import ContinuationAgent
from scripts.v57_phase_readout import PhaseDepartureReadout

REVISION = "general_ood_v59_normal_attribution"
BUDGET = 6000
SEEDS = (116000, 116001)
ARMS = ("normal-frozen", "normal-updating")


def first_transition(trace):
    return {**{name: trace[key][0].tolist() for name, key in
        (("state", "states"), ("action", "actions"), ("executed_action", "executed_actions"),
         ("next_state", "next_states"))}, "reward": float(trace["rewards"][0]), "done": bool(trace["dones"][0])}


def observe(trace, model, pooled, phase):
    signals, ages = signals_and_ages(model, pooled, trace)
    readouts = dict(pooled=DepartureReadout(pooled.age_reference), phase=PhaseDepartureReadout(phase))
    values = {f"{name}_coefficient": np.asarray([sensor.update(s, int(a)) for s, a in zip(signals, ages)])
              for name, sensor in readouts.items()}
    values.update(signal=signals, action_rms=np.sqrt(np.mean(trace["actions"] ** 2, axis=1)), reward=trace["rewards"])
    assert all(np.isfinite(v).all() for v in values.values())
    def means(start, end):
        return {key: float(value[start:end].mean()) for key, value in values.items()}
    windows = dict(first800=(0, min(800, len(ages))), full=(0, len(ages)),
                   last1000=(max(0, len(ages) - 1000), len(ages)), last64=(max(0, len(ages) - 64), len(ages)))
    return dict(windows={name: means(*bounds) for name, bounds in windows.items()},
        bins250=[dict(start=i, end=min(i+250, len(ages)), **means(i, min(i+250, len(ages))),
                      mean_episode_age=float(ages[i:i+250].mean()), boundaries=int(trace["dones"][i:i+250].sum()))
                 for i in range(0, len(ages), 250)])


def run(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    baseline = json.loads(args.v58_result.read_text())
    assert baseline["continuation_seed"] == args.seed and baseline["variant"] == "fixed-minus2"
    shared = torch.load(args.shared_dir / "shared.pt", map_location="cpu", weights_only=False)
    dynamics = torch.load(args.dynamics_checkpoint, map_location="cpu", weights_only=False)["extra_state"]["progress"]
    frozen = torch.load(args.reference_checkpoint, map_location="cpu", weights_only=False)["extra_state"]["progress"]
    assert dynamics["stage"] == "preflight" and dynamics["next_cell"] == 9
    assert frozen["stage"] == "formal" and frozen["next_cell"] == 22
    model, refs = dynamics["models"]["broad"][0], frozen["references"]
    pooled, phase = refs["matched_pooled"], refs["phase_calibrated"]
    metadata = deepcopy(dict(pooled=pooled.metadata(), phase=phase.metadata(), model=model.metadata()))
    assert metadata == baseline["readout_metadata"]
    del dynamics, frozen
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint_dir / "normal_attribution_latest.pt"
    extra = torch.load(args.resume_from, map_location="cpu", weights_only=False)["extra_state"] if args.resume_from else None
    progress = extra["progress"] if extra else dict(seed=args.seed, next_cell=0, records=[])
    assert progress["seed"] == args.seed
    for index in range(progress["next_cell"], len(ARMS)):
        arm = ARMS[index]
        env = gym.make("Hopper-v4")
        agent = ContinuationAgent(env, "fixed-minus2")
        agent.start_shared(args.source_dir / v53.SOURCE_NAME, shared)
        assert agent.training_steps == 99745 and agent.context_tracker._count == 100000 and agent.replay_buffer.size == 6144
        assert asdict(agent.config) == baseline["config"]
        assert float(getattr(agent, "_reward_ema_var", 1.)) == baseline["initial_reward_ema_var"]
        before = [p.detach().clone() for p in agent.actor.parameters()]
        loop = None
        if extra and extra["loop"] is not None:
            agent.load(str(args.resume_from))
            loop = extra["loop"]
            v53.apply_mode_to_env(env, "Hopper-v4", v53.MUJOCO_MODE_PROFILES["Hopper-v4"]["normal"])
            env.reset(seed=args.seed)
            v53.restore_environment(env, extra["environment"])
        cell = dict(split=arm, seed=args.seed, updating=arm == "normal-updating", mode="normal")
        trace, record = v53.collect(agent, env, cell, dict(steps=BUDGET), loop,
            lambda active: v53.save_progress(agent, env, progress, active, checkpoint))
        first = first_transition(trace)
        assert first == baseline["training"]["first_transition"], "mismatched-v58-initial-transition"
        movement = float(torch.sqrt(sum((p.detach()-q).square().sum() for p,q in zip(agent.actor.parameters(), before))).item())
        if not cell["updating"]:
            assert movement == 0.
        record.update(first_transition=first, observers=observe(trace, model, pooled, phase),
            physical_parameters_unchanged=True, actor_parameter_l2_change=movement,
            final_replay_size=agent.replay_buffer.size, final_context_count=agent.context_tracker._count)
        progress["records"].append(record)
        progress["next_cell"] = index + 1
        v53.save_progress(agent, env, progress, None, checkpoint)
        print(f"[CELL] {arm}/seed{args.seed}: {record['observers']['windows']['last1000']}", flush=True)
        extra = None
        env.close()
        del agent
        torch.cuda.empty_cache()
    assert metadata == dict(pooled=pooled.metadata(), phase=phase.metadata(), model=model.metadata())
    assert progress["records"][0]["first_transition"] == progress["records"][1]["first_transition"]
    output = dict(revision=REVISION, continuation_seed=args.seed, source_training_seed=1,
        source_checkpoint=v53.SOURCE_NAME, mode="normal-only", beta=-2., readout_controls_actor=False,
        reference_frozen=True, reference_metadata=metadata, comparison_v58=str(args.v58_result),
        environment_interactions=sum(r["interactions"] for r in progress["records"]),
        optimizer_updates=sum(r["optimizer_updates"] for r in progress["records"]), records=progress["records"],
        limitations=["Paired reused continuation seeds from one trained source; attribution diagnostic, not independent RL confirmation.",
                     "Optimizer updates change parameters and stochastic sampling paths; only the initial transition is shared across arms."])
    args.result_dir.mkdir(parents=True, exist_ok=True)
    destination = args.result_dir / "normal_attribution.json"
    destination.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(f"[DONE] seed{args.seed}: {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    for name in ("source-dir", "shared-dir", "dynamics-checkpoint", "reference-checkpoint", "v58-result", "checkpoint-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
