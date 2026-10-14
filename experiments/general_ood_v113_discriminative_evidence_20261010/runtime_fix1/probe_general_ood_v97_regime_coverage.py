"""Equal-budget old/diverse updating reference collection; no evaluation fitting."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

REVISION = "general_ood_v97_regime_coverage"
SEEDS = (135200, 135201)
SMOKE_SEED = 135199
SPEC = dict(seeds=list(SEEDS), steps=6000, switch_step=3800, stride=2)
RECIPES = dict(old_recipe=("normal", "slippery"), diverse_recipe=("high_grav", "compound"))
KEYS = ("states", "executed_actions", "next_states")
COUNTERS = ("actual_transitions", "counterfactual_step_calls", "environment_step_calls",
            "exact_true_mode_replays", "optimizer_updates")


def cells(seed):
    return [dict(role="reference_extension", seed=seed, mode=mode, updating=True,
                 group=recipe) for recipe, modes in RECIPES.items() for mode in modes]


def extensions(traces, spec, modes):
    indices = np.arange(spec["switch_step"], spec["steps"], spec["stride"])
    assert len(indices) > 0
    output, sources = {}, {}
    for recipe, physical_modes in RECIPES.items():
        selected = [traces[m] for m in physical_modes]
        output[recipe] = {mode: {key: np.concatenate([
            t["oracle_observations"][indices, i] if key == "next_states" else t[key][indices]
            for t in selected]) for key in KEYS} for i, mode in enumerate(modes)}
        sources[recipe] = [dict(source_mode=m, source_steps=[spec["switch_step"], spec["steps"]],
            stride=spec["stride"], input_row_range=[j*len(indices), (j+1)*len(indices)])
            for j, m in enumerate(physical_modes)]
        for m, trace in zip(physical_modes, selected):
            np.testing.assert_array_equal(trace["oracle_observations"][indices, modes.index(m)],
                                          trace["next_states"][indices])
        for mode in modes:
            for key in ("states", "executed_actions"):
                np.testing.assert_array_equal(output[recipe][mode][key], output[recipe]["normal"][key])
    assert len(output["old_recipe"]["normal"]["states"]) == len(output["diverse_recipe"]["normal"]["states"])
    return output, sources


def collect(args, source, spec):
    import gymnasium as gym

    worker = source.v76
    shared = torch.load(args.shared_dir/"shared.pt", map_location="cpu", weights_only=False)
    extra = torch.load(args.resume_from, map_location="cpu", weights_only=False)["extra_state"] if args.resume_from else None
    progress = extra["progress"] if extra else dict(revision=REVISION, seed=args.seed, spec=spec, next_cell=0, records=[])
    assert progress["revision"] == REVISION and progress["seed"] == args.seed and progress["spec"] == spec
    population = cells(args.seed)
    assert progress["next_cell"] == len(progress["records"])
    assert [(r["group"], r["mode"]) for r in progress["records"]] == [
        (c["group"], c["mode"]) for c in population[:progress["next_cell"]]]
    stored = args.checkpoint_dir/"traces"
    stored.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint_dir/"latest.pt"
    for i in range(progress["next_cell"], len(population)):
        cell = population[i]
        env = gym.make("Hopper-v4")
        shadows = worker.v69.make_shadows(env)
        agent = worker.v69.ContinuationAgent(env, "fixed-minus2", args.device)
        agent.start_shared(args.source_dir/worker.v69.SOURCE_NAME, shared)
        assert (agent.training_steps, agent.context_tracker._count, agent.replay_buffer.size) == (99745, 100000, 6144)
        loop = None
        if extra and extra["loop"] is not None:
            agent.load(str(args.resume_from))
            loop = extra["loop"]
            worker.v70.apply_mode(env, worker.v70.active_mode(cell, max(0, loop["step"]-1), spec))
            env.reset(seed=cell["seed"])
            worker.v69.restore_step_state(env, extra["environment"])
        trace, record = worker.collect_updating(agent, env, shadows, cell, spec, loop,
            lambda active: worker.v68.v53.save_progress(agent, env, progress, active, checkpoint))
        np.savez_compressed(worker.trace_path(stored, cell), **trace)
        if cell["mode"] != "normal":
            normal = worker.v68.load_trace(worker.trace_path(stored, dict(cell, mode="normal")))
            assert worker.v68.v53.paired_prefix_equal(trace, normal, spec["switch_step"]), "non-matched-prefix"
        progress["records"].append(record)
        progress["next_cell"] = i+1
        worker.v68.v53.save_progress(agent, env, progress, None, checkpoint)
        for physical in (env, *shadows.values()):
            physical.close()
        del agent
        torch.cuda.empty_cache()
        extra = None
        print(f"[PATH] {i+1}/4 seed{args.seed} {cell['group']} {cell['mode']}", flush=True)
    return progress["records"]


def check_seed(args, source):
    assert args.seed == SMOKE_SEED if args.smoke else args.seed in SEEDS
    selection = json.loads((args.v83_dir/"run_checkpoints/cross-seed-covariance/nested_selection.json").read_text())
    used = {b["seed"] for b in selection["reference_blocks"]}
    used.update(source.v76.SPEC["test"])
    for version, filename in (("v94", "suffix_replay.json"), ("v95", "fresh_seed.json")):
        directory = getattr(args, f"{version}_dir")
        paths = [directory/"run_results/replay"/filename] if version == "v94" else [
            directory/f"run_results/seed{seed}"/filename for seed in (135190, 135191, 135192)]
        for path in paths:
            used.update(r["seed"] for r in json.loads(path.read_text())["records"])
    used.update(source.v76.v69.SPEC[k] for k in ("fit_seed", "density_calibration_seed", "normal_null_seed"))
    assert not (set(SEEDS) | {SMOKE_SEED}) & used, "new reference seed overlaps existing reference/evaluation"


def run(args):
    sys.path.insert(0, str(args.v83_dir))
    from scripts import analyze_general_ood_v83_cross_seed_covariance as source

    torch.set_num_threads(2)
    check_seed(args, source)
    spec = dict(seeds=[SMOKE_SEED], steps=64, switch_step=32, stride=2) if args.smoke else SPEC
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    records = collect(args, source, spec)
    traces = {c["mode"]: source.v76.v68.load_trace(source.v76.trace_path(args.checkpoint_dir/"traces", c))
              for c in cells(args.seed)}
    payload, sources = extensions(traces, spec, source.MODES)
    counts = {recipe: len(data["normal"]["states"]) for recipe, data in payload.items()}
    assert all(n == (32 if args.smoke else 2200) for n in counts.values())
    totals = {k: sum(r[k] for r in records) for k in COUNTERS}
    assert totals["actual_transitions"] == totals["optimizer_updates"] == 4*spec["steps"]
    budgets = {recipe: {k: sum(r[k] for r in records if r["group"] == recipe) for k in COUNTERS}
               for recipe in RECIPES}
    assert budgets["old_recipe"] == budgets["diverse_recipe"]
    for recipe in sources:
        for row in sources[recipe]:
            row.update(seed=args.seed, role="reference_extension", updating=True)
    torch.save(dict(revision=REVISION, seed=args.seed, spec=spec, extensions=payload, sources=sources,
        groups={c: np.full(n, args.seed, dtype=np.int64) for c, n in counts.items()}),
        args.checkpoint_dir/"reference_inputs.pt")
    result = dict(revision=REVISION, stage="reference-collection", passed=True, seed=args.seed, spec=spec,
        smoke=args.smoke, collection_device=args.device, source_training_seed=1, source_updates=99745,
        recipes=RECIPES, collection_records=records, **totals, per_recipe_budgets=budgets,
        per_recipe_reference_inputs=counts, per_recipe_counterfactual_labels={c: n*5 for c, n in counts.items()},
        sources=sources, paired_prefixes=True, equal_recipe_budgets=True, exact_true_mode_replays_verified=True,
        counterfactuals_added_to_RL_replay=False, evaluation_paths_used_for_reference=False,
        reference_banks_fitted=False, normal_thresholds_recalibrated=False, coefficient_tuned=False,
        readout_controls_actor=False, readout_adopted=False,
        decision="reference-collection-only-pending-error-refit-calibration-development-replay",
        limitations=["Known-mode simulator counterfactual labels for offline reference construction.",
            "All collection policies adapt with fixed-minus2 from one source policy; not independent training seeds.",
            "The recipes change the post-switch trajectory distribution, not model form or observation representation.",
            "No recognition, risk, return or actor-integration claim from collection alone."])
    args.result_dir.mkdir(parents=True, exist_ok=True)
    path = args.result_dir/("native_smoke.json" if args.smoke else "reference_collection.json")
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] {REVISION}: {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("v83-dir", "v94-dir", "v95-dir", "source-dir", "shared-dir", "checkpoint-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
