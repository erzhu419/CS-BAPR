"""Fresh paired trajectories for the frozen v94 suffix uncertainty candidate."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_general_ood_v94_suffix_replay as v94

REVISION = "general_ood_v95_fresh_seed"
SEEDS = (135190, 135191, 135192)
SMOKE_SEED = 135189
SPEC = dict(seeds=list(SEEDS), steps=6000, switch_step=3800)
BASELINE = "linear_mean_updating_cov"
CONDITIONS = (BASELINE, *v94.CONDITIONS)
COUNTERS = ("actual_transitions", "counterfactual_step_calls", "environment_step_calls",
            "exact_true_mode_replays", "optimizer_updates")


def cells(seed, smoke=False):
    assert seed == SMOKE_SEED if smoke else seed in SEEDS
    return [dict(role="native_smoke" if smoke else "confirmation", seed=seed, mode=m, updating=u,
                 group="v95_fresh_updating" if u else "v95_fresh_frozen")
            for u in (False, True) for m in v94.MODES]


def load_readouts(args, source):
    from scripts.v73_recursive_mode_readout import RecursiveModeReference

    old = torch.load(args.v76_dir/"run_checkpoints/formal/behavior_coverage_references.pt",
                     map_location="cpu", weights_only=False)
    assert old["revision"] == source.v76.REVISION and old["spec"] == source.v76.SPEC
    raw = old["references"]["updating_mixed"].bank
    selection = json.loads((args.v83_dir/"run_checkpoints/cross-seed-covariance/nested_selection.json").read_text())
    groups = source.groups_from_blocks(selection["reference_blocks"], raw.tree.n)
    stored = args.v76_dir/"run_checkpoints/formal/traces"
    traces = {s: source.v76.v68.load_trace(stored/f"density_extension_seed{s}_updating_normal.npz")
              for s in v94.v89.SEEDS}
    expanded, mean_groups = v94.v89.expand_banks(raw, groups, traces)
    directory = args.v94_dir/"run_checkpoints/replay"
    errors = torch.load(directory/"full_suffix_errors.pt", map_location="cpu", weights_only=False)
    assert errors["revision"] == v94.REVISION and errors["ridge_alpha"] == .064
    assert errors["raw_rows"] == raw.tree.n == 23000 and errors["mean_rows"] == 25200
    np.testing.assert_array_equal(errors["raw_groups"], groups)
    np.testing.assert_array_equal(errors["mean_groups"], mean_groups)
    weight = selection["selection"]["raw_weight"]
    family = v94.PairedSuffixDensity(raw, expanded["suffix_mean_mix"], mean_groups, errors["errors"], weight)
    phases = torch.load(directory/"paired_phases.pt", map_location="cpu", weights_only=False)
    assert phases["revision"] == v94.REVISION and phases["raw_weight"] == weight
    assert phases["spec"] == source.v76.SPEC and phases["normal_null_transitions"] == 12000
    references = {}
    for c in v94.CONDITIONS:
        ref = RecursiveModeReference()
        ref.bank = SimpleNamespace(metadata=lambda condition=c: family.metadata(condition))
        ref.phase, ref.normal_null_calibration_transitions = deepcopy(phases["phases"][c]), 12000
        references[c] = ref
    baseline = source.v78.LocalLinearDensityBank(
        {"frozen": old["references"]["frozen_mixed"].bank, "updating": raw},
        source.v78.LocalLinearMean(raw), "updating")
    saved = torch.load(args.v78_dir/"run_checkpoints/mean-correction/linear_phases.pt",
                       map_location="cpu", weights_only=False)
    assert saved["revision"] == source.v78.REVISION and saved["spec"] == source.v76.SPEC
    ref = RecursiveModeReference()
    ref.bank, ref.phase = baseline, deepcopy(saved["phases"][BASELINE])
    ref.normal_null_calibration_transitions = saved["normal_null_transitions"]
    assert ref.normal_null_calibration_transitions == 12000
    references[BASELINE] = ref
    result = json.loads((args.v94_dir/"run_results/replay/suffix_replay.json").read_text())
    assert result["revision"] == v94.REVISION and len(result["records"]) == 65
    assert result["raw_weight"] == weight and result["all_five_reused_control_summaries_match_v90"]
    for c in v94.CONDITIONS:
        assert references[c].metadata() == result["reference_metadata"][c]
    used = {r["seed"] for r in result["records"]} | set(map(int, mean_groups))
    used |= {source.v76.v69.SPEC[k] for k in ("fit_seed", "density_calibration_seed", "normal_null_seed")}
    assert not (set(SEEDS) | {SMOKE_SEED}) & used, "confirmation seed already used"
    print("[REFERENCE] v78/v94 errors, weights and phases loaded before fresh collection; no fitting", flush=True)
    return family, baseline, references


def score_terms(family, baseline, trace):
    paired, geometry = family.terms(trace)
    return {BASELINE: baseline_terms(baseline, trace), **paired}, geometry


def baseline_terms(baseline, trace):
    from scripts.v78_local_linear_mean import correction_terms

    return correction_terms(trace, baseline.banks, baseline.correction)[BASELINE]


def collect_paths(args, source, population, spec):
    import gymnasium as gym

    worker = source.v76
    shared = torch.load(args.shared_dir/"shared.pt", map_location="cpu", weights_only=False)
    extra = torch.load(args.resume_from, map_location="cpu", weights_only=False)["extra_state"] if args.resume_from else None
    progress = extra["progress"] if extra else dict(revision=REVISION, spec=spec, seed=args.seed, next_cell=0, records=[])
    assert progress["revision"] == REVISION and progress["spec"] == spec and progress["seed"] == args.seed
    assert progress["next_cell"] == len(progress["records"])
    assert [source.v78.v77.cell_key(r) for r in progress["records"]] == [
        source.v78.v77.cell_key(c) for c in population[:progress["next_cell"]]]
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
        collector = worker.collect_updating if cell["updating"] else worker.v70.collect
        trace, record = collector(agent, env, shadows, cell, spec, loop,
            lambda active: worker.v68.v53.save_progress(agent, env, progress, active, checkpoint))
        np.savez_compressed(worker.trace_path(stored, cell), **trace)
        if cell["mode"] != "normal":
            normal = worker.v68.load_trace(worker.trace_path(stored, dict(cell, mode="normal")))
            assert worker.v68.v53.paired_prefix_equal(trace, normal, spec["switch_step"]), "non-matched-prefix"
        else:
            if cell["updating"]:
                frozen = worker.v68.load_trace(worker.trace_path(stored, dict(cell, updating=False)))
                for key in ("states", "actions", "executed_actions"):
                    np.testing.assert_array_equal(trace[key][0], frozen[key][0])
        progress["records"].append(record)
        progress["next_cell"] = i+1
        worker.v68.v53.save_progress(agent, env, progress, None, checkpoint)
        for physical in (env, *shadows.values()):
            physical.close()
        del agent
        torch.cuda.empty_cache()
        extra = None
        print(f"[PATH] {i+1}/{len(population)} seed{args.seed} {cell['mode']} updating={cell['updating']}", flush=True)
    return progress["records"]


def evaluate_paths(args, source, population, family, baseline, references):
    from scripts.v73_recursive_mode_readout import observe_likelihoods

    path = args.checkpoint_dir/"evaluation_progress.json"
    records = []
    if path.is_file():
        saved = json.loads(path.read_text())
        assert saved["revision"] == REVISION and saved["seed"] == args.seed and saved["spec"] == SPEC
        assert saved["conditions"] == list(CONDITIONS) and saved["raw_weight"] == family.raw_weight
        records = saved["records"]
        assert [source.v78.v77.cell_key(r) for r in records] == [source.v78.v77.cell_key(c) for c in population[:len(records)]]
    for cell in population[len(records):]:
        trace = source.v76.v68.load_trace(source.v76.trace_path(args.checkpoint_dir/"traces", cell))
        assert len(trace["states"]) == SPEC["steps"]
        paired, geometry = v94.cached_terms(args.checkpoint_dir/f"terms/{cell['mode']}_{cell['updating']}.pt",
            family, trace, source.v78.v77.cell_key(cell))
        terms = {BASELINE: baseline_terms(baseline, trace), **paired}
        record = dict(**cell, observers={}, attribution={}, geometry=geometry)
        for c in CONDITIONS:
            ll = terms[c]["likelihoods"]
            all_windows = source.v76.v73.window_statistics(trace, ll,
                observe_likelihoods(ll, trace["dones"], references[c], "recursive"), cell["mode"])
            record["observers"][c] = {w: all_windows[w] for w in source.v78.v77.WINDOWS}
            record["attribution"][c] = {w: source.v78.v77.attribution(trace, terms[c], family.bank, cell["mode"], a, b)
                                      for w, a, b in source.v78.v77.windows()}
        normal = record if cell["mode"] == "normal" else next(r for r in records
            if r["seed"] == cell["seed"] and r["updating"] == cell["updating"] and r["mode"] == "normal")
        record["change_minus_nochange"] = {c: {w: record["observers"][c][w]["mean_coefficient"]
            -normal["observers"][c][w]["mean_coefficient"] for w in source.v76.v68.BASE.CHANGE_WINDOWS} for c in CONDITIONS}
        records.append(record)
        path.write_text(json.dumps(dict(revision=REVISION, seed=args.seed, spec=SPEC, conditions=list(CONDITIONS),
                                       raw_weight=family.raw_weight, records=records), allow_nan=False)+"\n")
        print(f"[EVAL] {len(records)}/{len(population)} seed{args.seed} {cell['mode']} updating={cell['updating']}", flush=True)
    return records


def run(args):
    sys.path.insert(0, str(args.v83_dir))
    from scripts import analyze_general_ood_v83_cross_seed_covariance as source
    from scripts.v73_recursive_mode_readout import observe_likelihoods

    torch.set_num_threads(2)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    family, baseline, references = load_readouts(args, source)
    metadata = deepcopy({c: r.metadata() for c, r in references.items()})
    population = cells(args.seed, args.smoke)
    spec = dict(seeds=[SMOKE_SEED], steps=64, switch_step=32) if args.smoke else SPEC
    collected = collect_paths(args, source, population, spec)
    totals = {k: sum(r[k] for r in collected) for k in COUNTERS}
    assert totals["actual_transitions"] == len(population)*spec["steps"]
    assert totals["optimizer_updates"] == 5*spec["steps"]
    if args.smoke:
        checked = 0
        for cell in population:
            trace = source.v76.v68.load_trace(source.v76.trace_path(args.checkpoint_dir/"traces", cell))
            terms, _ = score_terms(family, baseline, trace)
            for c in CONDITIONS:
                assert all(a.shape == (64, 5) and np.isfinite(a).all() for a in terms[c].values())
                series = observe_likelihoods(terms[c]["likelihoods"], trace["dones"], references[c], "recursive")
                assert np.isfinite(series["mode_scores"]).all() and np.isfinite(series["risk"]).all()
                checked += 1
        result = dict(passed=True, checked_path_readouts=checked, paired_prefixes=True,
                      frozen_updating_first_action_equal=True, reference_metadata_unchanged=True)
        name = "native_smoke.json"
    else:
        records = evaluate_paths(args, source, population, family, baseline, references)
        result = dict(records=records, summary=source.v78.v77.summarize(records, CONDITIONS),
                      fresh_paths=10, changed_windows=16, decision="fresh-environment-seed-confirmation-not-readout-adoption")
        name = "fresh_seed.json"
    assert metadata == {c: r.metadata() for c, r in references.items()}, "reference changed during confirmation"
    result.update(revision=REVISION, source_revision=v94.REVISION, seed=args.seed, spec=spec, smoke=args.smoke,
        conditions=list(CONDITIONS), source_training_seed=1, source_updates=99745, collection_device=args.device,
        collection_records=collected, **totals, raw_weight=family.raw_weight, reference_metadata=metadata,
        reference_refitted=False, normal_thresholds_recalibrated=False, forward_models_refitted=False,
        scaling_refitted=False, coefficient_tuned=False, temporal_rules_changed=False,
        readout_controls_actor=False, counterfactuals_added_to_RL_replay=False,
        independent_training_seed_confirmation=False,
        limitations=["New environment seeds conditional on one source policy, known mild modes and inherited calibration.",
            "Updating collection uses the same fixed-minus2 continuation policy, not the candidate readout.",
            "Offline recognition and risk coefficients are not closed-loop algorithm returns or false-positive rates.",
            "Previously observed v61/v62/v76 failures remain separate development evidence."])
    path = args.result_dir/name
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] {REVISION}: {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("v94-dir", "v83-dir", "v76-dir", "v78-dir", "source-dir", "shared-dir", "checkpoint-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
