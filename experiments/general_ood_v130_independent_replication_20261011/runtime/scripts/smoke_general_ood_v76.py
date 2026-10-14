"""Native source/shared replay, GPU updates and exact-resume preflight for v76."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import numpy as np
import torch

from scripts import probe_general_ood_v76_behavior_coverage as worker


def run(args):
    torch.set_num_threads(2)
    shared = torch.load(args.shared_dir/"shared.pt", map_location="cpu", weights_only=False)
    spec = dict(steps=64, switch_step=32)
    cell = dict(role="smoke", seed=135179, mode="compound", updating=True)
    def fresh():
        env = gym.make("Hopper-v4")
        shadows = worker.v69.make_shadows(env)
        agent = worker.v69.ContinuationAgent(env, "fixed-minus2", args.device)
        agent.start_shared(args.source_dir/worker.v69.SOURCE_NAME, shared)
        assert agent.training_steps == 99745 and agent.replay_buffer.size == 6144
        return agent, env, shadows
    def close(env, shadows):
        for physical in (env, *shadows.values()):
            physical.close()
    direct, env, shadows = fresh()
    trace, record = worker.collect_updating(direct, env, shadows, cell, spec)
    close(env, shadows)
    normal, env, shadows = fresh()
    baseline, normal_record = worker.collect_updating(normal, env, shadows, dict(cell, mode="normal"), spec)
    close(env, shadows)
    assert worker.v68.v53.paired_prefix_equal(trace, baseline, spec["switch_step"])
    frozen, env, shadows = fresh()
    frozen_trace, frozen_record = worker.v70.collect(frozen, env, shadows, dict(cell, updating=False), spec)
    close(env, shadows)
    for key in ("states", "actions", "executed_actions"):
        np.testing.assert_array_equal(trace[key][0], frozen_trace[key][0])
    partial, env, shadows = fresh()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = args.checkpoint_dir/"resume_test.pt"
    worker.CHECKPOINT_STEPS = 48
    def interrupt(loop):
        worker.v68.v53.save_progress(partial, env, {}, loop, path)
        raise InterruptedError("preflight interruption")
    try:
        worker.collect_updating(partial, env, shadows, cell, spec, checkpoint=interrupt)
        raise AssertionError("preflight interruption missing")
    except InterruptedError:
        pass
    close(env, shadows)
    restored, env, shadows = fresh()
    extra = restored.load(str(path))
    worker.v70.apply_mode(env, worker.v70.active_mode(cell, extra["loop"]["step"]-1, spec))
    env.reset(seed=cell["seed"])
    worker.v69.restore_step_state(env, extra["environment"])
    recovered, resumed_record = worker.collect_updating(restored, env, shadows, cell, spec, extra["loop"])
    close(env, shadows)
    assert record == resumed_record
    for key in trace:
        np.testing.assert_array_equal(trace[key], recovered[key])
    for name in ("actor", "critic", "target_critic"):
        for key, value in getattr(direct, name).state_dict().items():
            assert torch.equal(value, getattr(restored, name).state_dict()[key])
    audited = (record, normal_record, frozen_record, resumed_record)
    output = dict(revision=worker.REVISION, device=args.device, source_shared_loaded=True,
        direct_trace_steps=64, actual_transitions=sum(r["actual_transitions"] for r in audited),
        optimizer_updates=sum(r["optimizer_updates"] for r in audited), frozen_optimizer_updates=frozen_record["optimizer_updates"],
        exact_true_mode_replays=sum(r["exact_true_mode_replays"] for r in audited), paired_prefix=True,
        frozen_updating_first_action_equal=True, exact_resume=True, result="pass")
    args.result_dir.mkdir(parents=True, exist_ok=True)
    (args.result_dir/"native_smoke.json").write_text(json.dumps(output, indent=2)+"\n")
    print(f"[DONE] Native v76 smoke PASS: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("source-dir", "shared-dir", "checkpoint-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    run(parser.parse_args())
