"""Untouched environment-seed confirmation of the frozen v109 normal mixture."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_general_ood_v106_standard_mean_replay as v106
import analyze_general_ood_v109_density_mixture as v109

v98, v95, summary = v106.v98, v106.v98.v95, v109.summary
REVISION = "general_ood_v110_fresh_mixture"
SEEDS = (135250, 135251, 135252)
SMOKE_SEED = 135249
SPEC = dict(seeds=list(SEEDS), steps=6000, switch_step=3800)
ARMS = ("old", "new", "normal_density_mixture")


def cells(seed, smoke=False):
    assert seed == SMOKE_SEED if smoke else seed in SEEDS
    return [dict(role="native_smoke" if smoke else "confirmation", seed=seed, mode=m, updating=u,
                 group="v110_confirmation_updating" if u else "v110_confirmation_frozen")
            for u in (False, True) for m in v98.v94.MODES]


def frozen_references(saved, development, family):
    from scripts.v73_recursive_mode_readout import RecursiveModeReference

    assert saved["revision"] == development["revision"] == v109.REVISION
    assert development["passed"] and not development["smoke"]
    assert development["checked_paths"] == 95 and development["checked_transitions"] == 570000
    assert development["mixture_weights"] == [.5, .5] and not development["mixture_weights_fitted"]
    assert saved["normal_null_transitions"] == 12000
    assert saved["mode_density_sources"] == development["mode_density_sources"] == v109.SOURCES
    references = {}
    for arm in ARMS:
        references[arm] = {}
        for c in summary.CONDITIONS:
            metadata = development["reference_metadata"][arm][c]
            if arm in ("old", "new"):
                component = family.incumbent if arm == "old" else family
                assert component.metadata(c) == metadata["bank"], (arm, c)
            ref = RecursiveModeReference()
            ref.bank = SimpleNamespace(metadata=lambda m=metadata: deepcopy(m["bank"]))
            ref.phase = deepcopy(saved["phases"][arm][c])
            ref.normal_null_calibration_transitions = 12000
            assert ref.metadata() == metadata, (arm, c)
            references[arm][c] = ref
    return references


def load_readouts(args, source):
    family, _ = v106.load_family(args, source)
    development = json.loads((args.v109_dir/"run_results/formal/density_mixture.json").read_text())
    used = {r["seed"] for r in development["records"]} | set(map(int, family.mean_groups))
    used |= {source.v76.v69.SPEC[k] for k in ("fit_seed", "density_calibration_seed", "normal_null_seed")}
    assert not (set(SEEDS) | {SMOKE_SEED}) & used, "confirmation seed already used"
    saved = torch.load(args.v109_dir/"run_checkpoints/formal/phases.pt", map_location="cpu", weights_only=False)
    references = frozen_references(saved, development, family)
    print("[REFERENCE] v98/v106 component banks and all nine v109 phases loaded without fitting", flush=True)
    return family, references


def component_likelihoods(family, trace):
    inputs = family.bank.features.inputs(trace)
    # Both means share exactly one incumbent covariance, including its old moment center.
    old_fixed, _, covariance, _ = family.incumbent.geometry(inputs)
    prediction = family.correction.predict(inputs)
    new_fixed = np.stack([prediction[m] for m in v98.v94.MODES], axis=1)
    return {a: {c: t["likelihoods"] for c, t in v98.v94.terms_for_covariances(
        trace, family.bank, fixed, covariance).items()} for a, fixed in (("old", old_fixed), ("new", new_fixed))}


def cached_likelihoods(args, family, trace, cell):
    path = args.checkpoint_dir/"terms"/f"{cell['mode']}_{cell['updating']}.pt"
    if path.is_file():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        assert saved["revision"] == REVISION and saved["spec"] == SPEC
        assert tuple(saved["key"]) == summary.key(cell)
        values = saved["likelihoods"]
    else:
        values = component_likelihoods(family, trace)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(revision=REVISION, spec=SPEC, key=summary.key(cell), likelihoods=values), path)
    assert all(v.shape == (SPEC["steps"], 5) and np.isfinite(v).all()
               for component in values.values() for v in component.values())
    return values


def summarize(records):
    index = {summary.key(r): r for r in records}
    views = {}
    for arm in ARMS:
        views[arm] = []
        for r in records:
            normal = index[r["group"], r["seed"], "normal", r["updating"]]
            contrasts = {c: {w: r["arms"][arm][c][w]["mean_coefficient"]
                -normal["arms"][arm][c][w]["mean_coefficient"] for w in summary.WINDOWS}
                for c in summary.CONDITIONS}
            views[arm].append(dict(r, observers=r["arms"][arm], change_minus_nochange=contrasts))
    subsets = dict(all=lambda r: True, updating=lambda r: r["updating"], frozen=lambda r: not r["updating"])
    for seed in sorted({r["seed"] for r in records}):
        for updating in (False, True):
            subsets[f"seed{seed}_{'updating' if updating else 'frozen'}"] = (
                lambda r, s=seed, u=updating: r["seed"] == s and r["updating"] == u)
    return {name: {a: {c: dict(summary.aggregate([r for r in rows if select(r)], c),
        mean_normal_full_accuracy=float(np.mean([r["observers"][c]["full"]["mode_accuracy"]
            for r in rows if select(r) and r["mode"] == "normal"])))
        for c in summary.CONDITIONS} for a, rows in views.items()} for name, select in subsets.items()}


def evaluate_paths(args, source, population, family, references):
    records = []
    for cell in population:
        trace = source.v76.v68.load_trace(source.v76.trace_path(args.checkpoint_dir/"traces", cell))
        assert len(trace["dones"]) == SPEC["steps"]
        ll = cached_likelihoods(args, family, trace, cell)
        arms, episodes = {a: {} for a in ARMS}, {a: {} for a in ARMS}
        for c in summary.CONDITIONS:
            values = v109.compose(ll["old"][c], ll["new"][c])
            for a in ARMS:
                readout = source.v76.v73.observe_likelihoods(values[a], trace["dones"], references[a][c], "recursive")
                arms[a][c] = source.v76.v73.window_statistics(trace, values[a], readout, cell["mode"])
                if cell["mode"] == "normal" and c == v109.v108.PRIMARY:
                    episodes[a][c] = v109.v107.episode_diagnostics(trace, readout)
        records.append(dict(**cell, arms=arms, normal_episodes=episodes if cell["mode"] == "normal" else None))
        print(f"[EVAL] {len(records)}/{len(population)} seed{args.seed} {cell['mode']} updating={cell['updating']}", flush=True)
    return records


def run(args):
    sys.path.insert(0, str(args.v83_dir))
    from scripts import analyze_general_ood_v83_cross_seed_covariance as source

    torch.set_num_threads(2)
    population = cells(args.seed, args.smoke)
    spec = dict(seeds=[SMOKE_SEED], steps=64, switch_step=32) if args.smoke else SPEC
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    family, references = load_readouts(args, source)
    before = {a: {c: r.metadata() for c, r in refs.items()} for a, refs in references.items()}
    collected = v95.collect_paths(args, source, population, spec)
    totals = {k: sum(r[k] for r in collected) for k in v95.COUNTERS}
    steps = spec["steps"]
    assert totals == dict(actual_transitions=10*steps, counterfactual_step_calls=50*steps,
        environment_step_calls=60*steps, exact_true_mode_replays=10*steps, optimizer_updates=5*steps)
    if args.smoke:
        checked, comparisons = 0, 0
        for i, cell in enumerate(population):
            trace = source.v76.v68.load_trace(source.v76.trace_path(args.checkpoint_dir/"traces", cell))
            ll = component_likelihoods(family, trace)
            if i == 0:
                for arm, component in (("old", family.incumbent), ("new", family)):
                    direct, _ = component.terms(trace)
                    for c in summary.CONDITIONS:
                        np.testing.assert_allclose(ll[arm][c], direct[c]["likelihoods"], rtol=1e-10, atol=1e-8)
                        comparisons += 1
            for c in summary.CONDITIONS:
                values = v109.compose(ll["old"][c], ll["new"][c])
                for arm in ARMS:
                    series = source.v76.v73.observe_likelihoods(values[arm], trace["dones"], references[arm][c], "recursive")
                    assert values[arm].shape == (64, 5) and np.isfinite(values[arm]).all()
                    assert np.isfinite(series["mode_scores"]).all() and np.isfinite(series["risk"]).all()
                    checked += 1
        assert (checked, comparisons) == (90, 6)
        result = dict(checked_path_arm_conditions=checked, production_component_comparisons=comparisons)
        name = "native_smoke.json"
    else:
        records = evaluate_paths(args, source, population, family, references)
        result = dict(records=records, summary=summarize(records), fresh_paths=10, changed_windows=16)
        name = "fresh_seed.json"
    assert before == {a: {c: r.metadata() for c, r in refs.items()} for a, refs in references.items()}
    result.update(revision=REVISION, source_revision=v109.REVISION, collector_revision=v95.REVISION,
        passed=True, smoke=args.smoke, seed=args.seed, spec=spec, arms=list(ARMS), conditions=list(summary.CONDITIONS),
        primary_condition=v109.v108.PRIMARY, collection_records=collected, **totals, reference_metadata=before,
        source_training_seed=1, source_updates=99745, collection_device=args.device,
        mixture_weights=[.5, .5], mixture_weights_fitted=False, reference_refitted=False,
        normal_thresholds_recalibrated=False, covariance_reference_unchanged=True,
        temporal_rules_changed=False, complete_history_preserved=True, paired_prefixes=True,
        readout_controls_actor=False, readout_adopted=False, counterfactuals_added_to_RL_replay=False,
        fresh_environment_seeds=True, independent_training_seed_confirmation=False,
        reference_metadata_unchanged=True, decision="frozen-candidate-environment-seed-confirmation-not-adoption",
        limitations=["One source training seed and known mild Hopper modes; not cross-environment confirmation.",
            "Reference budgets differ; mixture components share reference inputs.",
            "Fixed-minus2 collection is not controlled by the candidate; no candidate return claim.",
            "Mean risk is not a false-positive rate; inspected confirmation seeds become development if reused for tuning."])
    path = args.result_dir/name
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] {REVISION}: {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("v109-dir", "v104-dir", "v98-dir", "v97-dir", "v95-dir", "v94-dir", "v83-dir", "v78-dir",
                 "v76-dir", "v74-dir", "v73-dir", "v72-dir", "v71-dir", "v69-dir", "v70-dir", "v61-dir",
                 "v62-dir", "source-dir", "shared-dir", "checkpoint-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
