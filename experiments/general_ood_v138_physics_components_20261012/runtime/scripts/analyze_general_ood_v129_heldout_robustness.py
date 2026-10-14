"""Frozen reference/control/retention evaluation on twenty new noise seeds."""
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

import analyze_general_ood_v127_component_attribution as v127
import run_general_ood_v128_mean_retention as v128

v126 = v127.v126
REVISION = "general_ood_v129_heldout_robustness"
MODELS = ("reference-300k", "control-350k", "retention-350k")
NOISE_SEEDS = list(range(13000, 13020))


def stage_spec(seed, smoke=False):
    return dict(training_seed=seed, smoke=smoke, device="cpu", models=list(MODELS),
        reset_seeds=list(range(150000, 150001 if smoke else 150005)),
        noise_seeds=NOISE_SEEDS[:1] if smoke else NOISE_SEEDS,
        sampled_noise_scale=1., max_steps=1000,
        noise_schedule="v126 CPU torch.Generator, one float32 (1,3) draw per step",
        context_protocol="native source normalizer; reset EMA; update on actual executed transitions")


def cells(spec):
    controls = [dict(model=model, mode="deterministic", reset_seed=reset, noise_seed=None)
                for model in MODELS for reset in spec["reset_seeds"]]
    heldout = [dict(model=model, mode="sampled", reset_seed=reset, noise_seed=noise)
               for model in MODELS for reset in spec["reset_seeds"] for noise in spec["noise_seeds"]]
    return controls+heldout


def evaluate_cell(agents, cell, max_steps):
    episode = v126.evaluate_episode(agents[cell["model"]], cell["reset_seed"],
        0. if cell["mode"] == "deterministic" else 1., cell["noise_seed"], max_steps)
    # This round needs outcomes, not per-step traces; keep the synced JSON small.
    compact = {k: v for k, v in episode.items() if k not in ("tail", "same_input_parity")}
    return dict(compact, model=cell["model"], mode=cell["mode"])


def collect(agents, baselines, spec, progress, save):
    expected = cells(spec)
    assert [{k: e[k] for k in expected[0]} for e in progress["evaluations"]] == expected[:len(progress["evaluations"])]
    for cell in expected[len(progress["evaluations"]):]:
        episode = evaluate_cell(agents, cell, spec["max_steps"])
        episode["baseline_match"] = None
        if cell["mode"] == "deterministic":
            v127.verify_baseline(episode, baselines[(cell["model"], cell["reset_seed"])])
            episode["baseline_match"] = True
        progress["evaluations"].append(episode)
        save()
        print(f"[EVAL] seed{spec['training_seed']} cell={len(progress['evaluations'])}/{len(expected)} "
              f"model={cell['model']} mode={cell['mode']} noise={cell['noise_seed']} "
              f"length={episode['length']}", flush=True)


def load_sources(args, versions):
    seed = args.seed
    old = json.loads((args.v126_runtime/f"run_results/seed{seed}_cpu/noise.json").read_text())
    assert old["revision"] == v126.REVISION and old["spec"] == v126.stage_spec(seed)
    assert old["runtime"] == versions and old["checkpoint_selection"] == "final-not-best"
    old_reference = next(m for m in old["models"] if m["age"] == 300000)
    entries = [(MODELS[0], 300000, args.v123_runtime/f"run_results/seed{seed}/source.json", None)] + [
        (model, 350000, args.v128_runtime/f"run_results/seed{seed}_{arm}/mean_retention.json", arm)
        for model, arm in zip(MODELS[1:], v128.ARMS)]
    agents, sources, baselines = {}, [], {}
    env = gym.make("Hopper-v4")
    try:
        for model, age, path, arm in entries:
            source = json.loads(path.read_text())
            revision = v128.v123.REVISION if arm is None else v128.REVISION
            assert (source["revision"], source["training_seed"], source["smoke"]) == (revision, seed, False)
            assert source["training"]["interactions"] == age and source["effective_beta"] == -2.
            assert source["config"]["weight_reg"] == 0. and source["checkpoint_selection"] == "final-not-best"
            assert not source["source_readout_refitted"] and not source["periodic_evaluation_selects_checkpoint"]
            assert all(versions[k] == source["runtime"][k] for k in versions)
            if arm is None:
                assert old_reference["checkpoint"] == source["checkpoint"]
                references = [e for e in old_reference["evaluations"] if e["noise_scale"] == 0.]
            else:
                assert source["arm"] == arm and source["spec"] == v128.stage_spec(arm)
                assert source["mean_retention_coefficient"] == v128.ARMS[arm]
                assert source["final_evaluation"]["protocol"] == v128.evaluation_spec()
                assert source["final_evaluation"]["device"] == "cpu"
                references = source["final_evaluation"]["groups"]["deterministic"]["episodes"]
            agent = v128.v125.make_agent(env, "cpu", -2.)
            assert asdict(agent.config) == dict(source["config"], device="cpu")
            agent.load(source["checkpoint"])
            assert agent.training_steps == age-agent.config.batch_size+1 and agent.context_tracker._count == age
            agent.actor.eval().requires_grad_(False)
            context = dict(count=agent.context_tracker._count,
                mean=agent.context_tracker._mean.tolist(), m2=agent.context_tracker._m2.tolist())
            if arm is None:
                assert old_reference["initial_context_normalizer"] == context
            sources.append(dict(model=model, age=age, revision=revision, arm=arm,
                checkpoint=source["checkpoint"], config=asdict(agent.config),
                initial_context_normalizer=context))
            agents[model] = agent
            for episode in references:
                baselines[(model, episode["reset_seed"])] = episode
    finally:
        env.close()
    assert all(s["config"] == sources[0]["config"] for s in sources)
    return agents, sources, baselines


def run(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    spec, started = stage_spec(args.seed, args.smoke), time.monotonic()
    versions = dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
        gymnasium=gym.__version__, mujoco=__import__("mujoco").__version__)
    agents, sources, baselines = load_sources(args, versions)
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
    collect(agents, baselines, spec, progress, save)
    result = dict(**progress, training_seed=args.seed, evaluation_device="cpu",
        optimizer_updates=0, new_training_interactions=0, source_readout_refitted=False,
        checkpoint_selection="predeclared-final-only", reward_or_physics_changes=[],
        deterministic_baseline_matches=sum(e["baseline_match"] is True for e in progress["evaluations"]),
        seconds=time.monotonic()-started,
        evaluation_interactions=sum(e["length"] for e in progress["evaluations"]))
    args.result_dir.mkdir(parents=True, exist_ok=True)
    (args.result_dir/"heldout.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] seed{args.seed} episodes={len(progress['evaluations'])} "
          f"baseline_matches={result['deterministic_baseline_matches']} optimizer_updates=0 "
          f"evaluation_steps={result['evaluation_interactions']}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--v123-runtime", type=Path, required=True)
    parser.add_argument("--v126-runtime", type=Path, required=True)
    parser.add_argument("--v128-runtime", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
