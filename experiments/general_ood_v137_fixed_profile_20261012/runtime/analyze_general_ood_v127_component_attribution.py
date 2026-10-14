"""Frozen crossed mean/std functions and context-normalizer attribution."""
import argparse
from dataclasses import asdict
from itertools import product
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import numpy as np
import torch

import analyze_general_ood_v126_noise_sensitivity as v126

REVISION = "general_ood_v127_component_attribution"
AGES = (300000, 350000)


def stage_spec(seed, smoke=False):
    return dict(training_seed=seed, smoke=smoke, device="cpu", component_ages=list(AGES),
        reset_seeds=list(range(150000, 150001 if smoke else 150005)),
        noise_groups={k: v[:1] if smoke else v for k, v in v126.NOISE_GROUPS.items()},
        sampled_noise_scale=1., max_steps=1000,
        mean_std_input="same current augmented state; complete frozen source actor functions",
        context_protocol="source initial count/mean/m2; reset EMA and update context on executed transitions")


def cases():
    return [dict(mode="deterministic", mean_age=m, std_age=None, normalizer_age=n)
            for m, n in product(AGES, repeat=2)] + [
        dict(mode="sampled", mean_age=m, std_age=s, normalizer_age=n)
        for m, s, n in product(AGES, repeat=3)]


def cells(spec):
    return [dict(case, reset_seed=reset, noise_seed=noise, noise_group=group)
        for case in cases()
        for group, noises in ({"deterministic": [None]} if case["mode"] == "deterministic"
                              else spec["noise_groups"]).items()
        for reset in spec["reset_seeds"] for noise in noises]


def coherent(cell):
    return cell["mean_age"] == cell["normalizer_age"] and (
        cell["std_age"] is None or cell["std_age"] == cell["mean_age"])


class CrossedPolicy(torch.nn.Module):
    def __init__(self, mean_actor, std_actor):
        super().__init__()
        assert mean_actor.action_range == std_actor.action_range
        self.mean_actor, self.std_actor = mean_actor, std_actor
        self.action_range = mean_actor.action_range

    def forward(self, state):
        mean, log_std = self.mean_actor(state)
        if self.std_actor is not self.mean_actor:
            _, log_std = self.std_actor(state)
        return mean, log_std


def evaluate_cell(agents, actors, cell, max_steps):
    normalizer = agents[cell["normalizer_age"]]
    original_actor = normalizer.actor
    normalizer.actor = CrossedPolicy(actors[cell["mean_age"]],
        actors[cell["std_age"] if cell["std_age"] is not None else cell["mean_age"]])
    try:
        result = v126.evaluate_episode(normalizer, cell["reset_seed"],
            0. if cell["mode"] == "deterministic" else 1., cell["noise_seed"], max_steps)
        return dict(result, **{k: v for k, v in cell.items() if k not in result})
    finally:
        normalizer.actor = original_actor


def verify_baseline(episode, reference):
    for key in ("length", "physical_termination", "time_limit_truncation", "diagnostic_cutoff"):
        assert episode[key] == reference[key], key
    for key in ("episode_return", "mean_velocity", "mean_policy_std",
                "mean_action_noise_l2", "mean_context_l2"):
        np.testing.assert_allclose(episode[key], reference[key], rtol=1e-7, atol=1e-5, err_msg=key)
    assert episode["final_physical_state"]["unhealthy_conditions"] == reference[
        "final_physical_state"]["unhealthy_conditions"]


def collect(agents, actors, baseline, spec, progress, save):
    expected = cells(spec)
    assert [{k: e[k] for k in expected[0]} for e in progress["evaluations"]] == expected[:len(progress["evaluations"])]
    for cell in expected[len(progress["evaluations"]):]:
        episode = evaluate_cell(agents, actors, cell, spec["max_steps"])
        episode["baseline_match"] = None
        if coherent(cell):
            model = next(m for m in baseline["models"] if m["age"] == cell["mean_age"])
            reference = next(e for e in model["evaluations"] if
                (e["reset_seed"], e["noise_seed"], e["noise_scale"]) ==
                (cell["reset_seed"], cell["noise_seed"], episode["noise_scale"]))
            verify_baseline(episode, reference)
            episode["baseline_match"] = True
        progress["evaluations"].append(episode)
        save()
        print(f"[EVAL] seed{spec['training_seed']} cell={len(progress['evaluations'])}/{len(expected)} "
              f"mode={cell['mode']} mean={cell['mean_age']} std={cell['std_age']} "
              f"context={cell['normalizer_age']} group={cell['noise_group']} length={episode['length']}", flush=True)


def run(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    spec, started = stage_spec(args.seed, args.smoke), time.monotonic()
    versions = dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
                    gymnasium=gym.__version__, mujoco=__import__("mujoco").__version__)
    baseline = json.loads((args.v126_runtime/f"run_results/seed{args.seed}_cpu/noise.json").read_text())
    assert baseline["revision"] == v126.REVISION and baseline["spec"] == v126.stage_spec(args.seed)
    assert baseline["runtime"] == versions and baseline["checkpoint_selection"] == "final-not-best"
    agents, sources = {}, []
    env = gym.make("Hopper-v4")
    try:
        for age, runtime, relative, revision in (
            (300000, args.v123_runtime, f"seed{args.seed}/source.json", v126.v125.v123.REVISION),
            (350000, args.v125_runtime, f"seed{args.seed}_beta-minus2/actor_beta.json", v126.v125.REVISION)):
            source = json.loads((runtime/"run_results"/relative).read_text())
            assert (source["revision"], source["training_seed"], source["smoke"]) == (revision, args.seed, False)
            assert source["effective_beta"] == -2. and source["checkpoint_selection"] == "final-not-best"
            assert source["training"]["interactions"] == age and source["config"]["weight_reg"] == 0.
            assert all(versions[k] == source["runtime"][k] for k in versions)
            agent = v126.v125.make_agent(env, "cpu", -2.)
            assert asdict(agent.config) == dict(source["config"], device="cpu")
            agent.load(source["checkpoint"])
            assert agent.training_steps == age-agent.config.batch_size+1 and agent.context_tracker._count == age
            context = dict(count=agent.context_tracker._count,
                mean=agent.context_tracker._mean.tolist(), m2=agent.context_tracker._m2.tolist())
            previous = next(m for m in baseline["models"] if m["age"] == age)
            assert previous["checkpoint"] == source["checkpoint"] and previous["initial_context_normalizer"] == context
            agents[age] = agent
            sources.append(dict(age=age, revision=revision, checkpoint=source["checkpoint"],
                config=asdict(agent.config), initial_context_normalizer=context))
    finally:
        env.close()
    assert sources[0]["config"] == sources[1]["config"]
    actors = {age: agent.actor for age, agent in agents.items()}
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
    collect(agents, actors, baseline, spec, progress, save)
    result = dict(**progress, training_seed=args.seed, evaluation_device="cpu",
        optimizer_updates=0, new_training_interactions=0, source_readout_refitted=False,
        checkpoint_selection="final-not-best", reward_or_physics_changes=[],
        coherent_baseline_matches=sum(e["baseline_match"] is True for e in progress["evaluations"]),
        seconds=time.monotonic()-started,
        evaluation_interactions=sum(e["length"] for e in progress["evaluations"]))
    args.result_dir.mkdir(parents=True, exist_ok=True)
    (args.result_dir/"components.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] seed{args.seed} episodes={len(progress['evaluations'])} "
          f"baseline_matches={result['coherent_baseline_matches']} optimizer_updates=0 "
          f"evaluation_steps={result['evaluation_interactions']}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--v123-runtime", type=Path, required=True)
    parser.add_argument("--v125-runtime", type=Path, required=True)
    parser.add_argument("--v126-runtime", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
