"""Frozen v133 final policies: anchor, new-reset and common-noise CPU episodes."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import numpy as np
import torch

import analyze_general_ood_v126_noise_sensitivity as evaluation
import run_general_ood_v133_source_extension as v133

REVISION = "general_ood_v134_frozen_source_robustness"


def stage_spec(seed, arm, smoke=False):
    return dict(training_seed=seed, arm=arm, smoke=smoke, device="cpu", source_steps=300000,
        anchor_reset_seeds=list(range(150000, 150001 if smoke else 150005)),
        new_reset_seeds=list(range(151000, 151001 if smoke else 151020)),
        action_noise_seeds=[134000] if smoke else [134000, 134001, 134002],
        sampled_noise_scale=1., max_steps=64 if smoke else 1000,
        context_protocol="restore trained normalizer per episode; reset short EMA; update actual transitions",
        noise_schedule="v126 CPU torch.Generator; one float32 (1,3) draw per step")


def cells(spec):
    anchors = [dict(group="anchor", mode="deterministic", reset_seed=reset, noise_seed=None)
               for reset in spec["anchor_reset_seeds"]]
    resets = [dict(group="new-reset", mode="deterministic", reset_seed=reset, noise_seed=None)
              for reset in spec["new_reset_seeds"]]
    sampled = [dict(group="new-reset", mode="sampled", reset_seed=reset, noise_seed=noise)
               for reset in spec["new_reset_seeds"] for noise in spec["action_noise_seeds"]]
    return anchors+resets+sampled


def evaluate_cell(agent, cell, max_steps):
    result = evaluation.evaluate_episode(agent, cell["reset_seed"],
        0. if cell["mode"] == "deterministic" else 1., cell["noise_seed"], max_steps)
    return dict({k: v for k, v in result.items() if k not in ("tail", "same_input_parity")},
                group=cell["group"], mode=cell["mode"])


def collect(agent, spec, progress, save):
    expected = cells(spec)
    assert [{k: e[k] for k in expected[0]} for e in progress["evaluations"]] == expected[:len(progress["evaluations"])]
    updates, size = agent.training_steps, agent.replay_buffer.size
    for cell in expected[len(progress["evaluations"]):]:
        episode = evaluate_cell(agent, cell, spec["max_steps"])
        progress["evaluations"].append(episode)
        save()
        print(f"[EVAL] seed{spec['training_seed']} {spec['arm']} "
              f"cell={len(progress['evaluations'])}/{len(expected)} {cell['group']} "
              f"mode={cell['mode']} length={episode['length']}", flush=True)
    assert (agent.training_steps, agent.replay_buffer.size) == (updates, size)


def load_source(source, seed, arm, versions):
    assert (source["revision"], source["training_seed"], source["arm"], source["smoke"]) == (v133.REVISION, seed, arm, False)
    assert source["spec"] == v133.stage_spec(arm)
    assert source["training"]["interactions"] == 300000 and source["training"]["optimizer_updates"] == 299745
    assert source["checkpoint_selection"] == "final-not-best"
    assert not source["source_readout_refitted"] and not source["periodic_evaluation_selects_checkpoint"]
    assert all(versions[k] == source["runtime"][k] for k in versions)
    env = gym.make("Hopper-v4")
    try:
        agent = v133.v132.make_agent(env, "cpu", arm)
        assert asdict(agent.config) == dict(source["config"], device="cpu")
        agent.load(source["checkpoint"])
    finally:
        env.close()
    assert agent.training_steps == 299745 and agent.context_tracker._count == 300000
    assert agent.replay_buffer.size == 0
    for model in (agent.actor, agent.critic, agent.target_critic):
        model.eval().requires_grad_(False)
    return agent


def run(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    versions = dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
                    gymnasium=gym.__version__, mujoco=__import__("mujoco").__version__)
    source = json.loads(args.source_json.read_text())
    agent = load_source(source, args.seed, args.arm, versions)
    spec, started = stage_spec(args.seed, args.arm, args.smoke), time.monotonic()
    normalizer = dict(count=agent.context_tracker._count, mean=agent.context_tracker._mean.tolist(),
                      m2=agent.context_tracker._m2.tolist())
    source_metadata = dict(revision=source["revision"], checkpoint=source["checkpoint"],
        training_steps=agent.training_steps, context_count=agent.context_tracker._count,
        config=asdict(agent.config), runtime=source["runtime"], gpu_anchor_evaluation=source["final_evaluation"],
        initial_context_normalizer=normalizer)
    progress = json.loads(args.resume_from.read_text()) if args.resume_from else dict(
        revision=REVISION, spec=spec, runtime=versions, source=source_metadata, evaluations=[])
    assert (progress["revision"], progress["spec"], progress["runtime"], progress["source"]) == (
        REVISION, spec, versions, source_metadata)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    def save():
        path = args.checkpoint_dir/"progress.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(progress, indent=2, allow_nan=False)+"\n")
        temporary.replace(path)
    collect(agent, spec, progress, save)
    assert not torch.cuda.is_initialized()
    result = dict(**progress, training_seed=args.seed, arm=args.arm, smoke=args.smoke,
        evaluation_device="cpu", cuda_initialized=False, optimizer_updates=0, new_training_interactions=0,
        source_readout_refitted=False, checkpoint_selection="predeclared-final-only", replay_loaded=False,
        reward_or_physics_changes=[], cpu_anchor_requires_gpu_trajectory_parity=False,
        seconds=time.monotonic()-started, evaluation_interactions=sum(e["length"] for e in progress["evaluations"]))
    args.result_dir.mkdir(parents=True, exist_ok=True)
    (args.result_dir/"robustness.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] seed{args.seed} {args.arm} episodes={len(cells(spec))} "
          f"optimizer_updates=0 evaluation_steps={result['evaluation_interactions']}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=v133.v132.SEEDS, required=True)
    parser.add_argument("--arm", choices=v133.v132.ARMS, required=True)
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
