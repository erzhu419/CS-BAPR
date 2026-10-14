"""Common frozen evaluation for the original and two independent training seeds."""
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

import analyze_general_ood_v129_heldout_robustness as v129

v128 = v129.v128
REVISION = "general_ood_v130_independent_replication"
SEEDS = (0, 1, 2, 3, 4)
MODELS = v129.MODELS


def stage_spec(seed, smoke=False):
    spec = v129.stage_spec(seed, smoke)
    spec.update(reset_seeds=list(range(151000, 151001 if smoke else 151005)),
                noise_seeds=list(range(14000, 14001 if smoke else 14020)))
    return spec


def source_entries(args):
    source_runtime = args.v123_runtime if args.seed < 3 else args.replication_runtime
    continuation_runtime = args.v128_runtime if args.seed < 3 else args.replication_runtime
    return [(MODELS[0], 300000, source_runtime/f"run_results/seed{args.seed}/source.json", None)] + [
        (model, 350000, continuation_runtime/f"run_results/seed{args.seed}_{arm}/mean_retention.json", arm)
        for model, arm in zip(MODELS[1:], v128.ARMS)]


def load_sources(args, versions):
    agents, sources = {}, []
    env = gym.make("Hopper-v4")
    try:
        for model, age, path, arm in source_entries(args):
            source = json.loads(path.read_text())
            revision = v128.v123.REVISION if arm is None else v128.REVISION
            assert (source["revision"], source["training_seed"], source["smoke"]) == (revision, args.seed, False)
            assert source["training"]["interactions"] == age and source["effective_beta"] == -2.
            assert source["config"]["weight_reg"] == 0. and source["checkpoint_selection"] == "final-not-best"
            assert not source["source_readout_refitted"] and not source["periodic_evaluation_selects_checkpoint"]
            assert all(versions[k] == source["runtime"][k] for k in versions)
            if arm is None:
                assert source["spec"] == v128.v123.stage_spec()
            else:
                assert source["arm"] == arm and source["spec"] == v128.stage_spec(arm)
                assert source["mean_retention_coefficient"] == v128.ARMS[arm]
                assert source["reference_checkpoint"] == sources[0]["checkpoint"]
                assert source["reference_frozen"] and not source["current_std_directly_regularized"]
                assert not source["initial_normalizer_swapped"]
            agent = v128.v125.make_agent(env, "cpu", -2.)
            assert asdict(agent.config) == dict(source["config"], device="cpu")
            agent.load(source["checkpoint"])
            assert agent.training_steps == age-agent.config.batch_size+1 and agent.context_tracker._count == age
            agent.actor.eval().requires_grad_(False)
            sources.append(dict(model=model, age=age, revision=revision, arm=arm,
                checkpoint=source["checkpoint"], config=asdict(agent.config),
                initial_context_normalizer=dict(count=agent.context_tracker._count,
                    mean=agent.context_tracker._mean.tolist(), m2=agent.context_tracker._m2.tolist())))
            agents[model] = agent
    finally:
        env.close()
    assert all(s["config"] == sources[0]["config"] for s in sources)
    return agents, sources


def collect(agents, spec, progress, save):
    expected = v129.cells(spec)
    assert [{k: e[k] for k in expected[0]} for e in progress["evaluations"]] == expected[:len(progress["evaluations"])]
    for cell in expected[len(progress["evaluations"]):]:
        progress["evaluations"].append(v129.evaluate_cell(agents, cell, spec["max_steps"]))
        save()
        print(f"[EVAL] seed{spec['training_seed']} cell={len(progress['evaluations'])}/{len(expected)} "
              f"model={cell['model']} mode={cell['mode']} noise={cell['noise_seed']} "
              f"length={progress['evaluations'][-1]['length']}", flush=True)


def run(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    spec, started = stage_spec(args.seed, args.smoke), time.monotonic()
    versions = dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
        gymnasium=gym.__version__, mujoco=__import__("mujoco").__version__)
    agents, sources = load_sources(args, versions)
    progress = json.loads(args.resume_from.read_text()) if args.resume_from else dict(
        revision=REVISION, spec=spec, runtime=versions, sources=sources, evaluations=[])
    assert (progress["revision"], progress["spec"], progress["runtime"], progress["sources"]) == (
        REVISION, spec, versions, sources)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    def save():
        path = args.checkpoint_dir/"progress.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(progress, indent=2, allow_nan=False)+"\n")
        temporary.replace(path)
    collect(agents, spec, progress, save)
    result = dict(**progress, training_seed=args.seed, evaluation_device="cpu",
        optimizer_updates=0, new_training_interactions=0, source_readout_refitted=False,
        checkpoint_selection="predeclared-final-only", reward_or_physics_changes=[],
        seconds=time.monotonic()-started,
        evaluation_interactions=sum(e["length"] for e in progress["evaluations"]))
    args.result_dir.mkdir(parents=True, exist_ok=True)
    (args.result_dir/"replication.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] seed{args.seed} episodes={len(progress['evaluations'])} "
          f"optimizer_updates=0 evaluation_steps={result['evaluation_interactions']}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--v123-runtime", type=Path, required=True)
    parser.add_argument("--v128-runtime", type=Path, required=True)
    parser.add_argument("--replication-runtime", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
