"""Fresh environment-seed confirmation of the unchanged v117 score readouts."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_general_ood_v117_temporal_classifier_replay as v117
import analyze_general_ood_v107_normal_attribution as v107

v98, v95 = v117.v98, v117.v98.v95
REVISION = "general_ood_v118_frozen_classifier_confirmation"
SEEDS, SMOKE_SEED = (135260, 135261, 135262), 135259
SPEC = dict(seeds=list(SEEDS), steps=6000, switch_step=3800)
ARMS, CONDITIONS, PRIMARY = v117.ARMS, v98.v94.CONDITIONS, v117.PRIMARY


def cells(seed, smoke=False):
    assert seed == SMOKE_SEED if smoke else seed in SEEDS
    return [dict(role="native_smoke" if smoke else "confirmation", seed=seed, mode=m, updating=u,
                 group="v118_confirmation_updating" if u else "v118_confirmation_frozen")
            for u in (False, True) for m in v117.MODES]


def restore_score_references(saved, development):
    from scripts.v73_recursive_mode_readout import RecursiveModeReference

    assert saved["revision"] == development["revision"] == v117.REVISION
    assert development["passed"] and not development["smoke"]
    assert (development["checked_paths"], development["checked_transitions"],
            development["unchanged_v116_score_windows_verified"]) == (125, 750000, 76)
    assert saved["normal_null_cells"] == development["normal_null_cells"]
    assert saved["reference_metadata"] == development["reference_metadata"]
    references = {}
    for arm in ARMS:
        metadata = development["reference_metadata"][arm]
        ref = RecursiveModeReference()
        ref.bank = SimpleNamespace(metadata=lambda m=metadata: deepcopy(m["bank"]))
        ref.phase = deepcopy(saved["phases"][arm])
        ref.normal_null_calibration_transitions = 12000
        references[arm] = ref
    assert v117.reference_metadata(references) == development["reference_metadata"]
    return references


def verify_unused_seeds(development, family, source):
    used = {r["seed"] for r in development["records"]} | set(map(int, family.mean_groups))
    used |= {source.v76.v69.SPEC[k] for k in ("fit_seed", "density_calibration_seed", "normal_null_seed")}
    assert not (set(SEEDS) | {SMOKE_SEED}) & used, "confirmation seed already used"


def load_readouts(args, source):
    from scripts.v73_recursive_mode_readout import RecursiveModeReference

    family, _ = v117.v113.v106.load_family(args, source)
    mean, classifiers = v117.load_models(args, family)
    development = json.loads((args.v117_dir/"run_results/formal/temporal_classifier_replay.json").read_text())
    saved = torch.load(args.v117_dir/"run_checkpoints/formal/phases.pt", map_location="cpu", weights_only=False)
    references = restore_score_references(saved, development)
    verify_unused_seeds(development, family, source)
    incumbent_data = json.loads((args.v98_dir/"run_results/replay_diverse_recipe/regime_replay.json").read_text())
    saved = torch.load(args.v98_dir/"run_checkpoints/diverse_recipe/phases.pt", map_location="cpu", weights_only=False)
    assert saved["revision"] == v98.REVISION and saved["recipe"] == "diverse_recipe"
    assert saved["normal_null_transitions"] == 12000 and saved["raw_weight"] == family.incumbent.raw_weight
    np.testing.assert_array_equal(saved["groups"], family.groups)
    incumbent = {}
    for c in CONDITIONS:
        ref = RecursiveModeReference()
        ref.bank = SimpleNamespace(metadata=lambda c=c: family.incumbent.metadata(c))
        ref.phase, ref.normal_null_calibration_transitions = deepcopy(saved["phases"][c]), 12000
        assert ref.metadata() == incumbent_data["reference_metadata"][c]
        incumbent[c] = ref
    print("[REFERENCE] formal means, classifiers and all six phases frozen before fresh collection", flush=True)
    return family, mean, classifiers, references, incumbent, development


def reference_metadata(references, incumbent):
    return dict(scores=v117.reference_metadata(references), incumbent={c: r.metadata() for c, r in incumbent.items()})


def score_trace(family, mean, classifiers, trace):
    query = {k: trace[k] for k in v98.v97.KEYS}
    scores = v117.transition_scores(family.bank, mean, classifiers, query)
    terms, _ = family.incumbent.terms(query)
    old = {c: t["likelihoods"] for c, t in terms.items()}
    return scores, old


def verify_development(args, source, family, mean, classifiers, references, incumbent, development):
    from scripts.v73_recursive_mode_readout import observe_likelihoods

    cell = dict(group="v95_development_updating", seed=135192, mode="high_grav", updating=True, role="confirmation")
    previous = next(r for r in development["records"] if v117.key(r) == v117.key(cell))
    trace = source.v76.v68.load_trace(source.v76.trace_path(args.v95_dir/"run_checkpoints/seed135192/traces", cell))
    scores, old = score_trace(family, mean, classifiers, trace)
    checked = 0
    for arm, values, ref, expected in (
            [(a, scores[a], references[a], previous["arms"][a]) for a in ARMS]
            +[(c, old[c], incumbent[c], previous["incumbent"][c]) for c in CONDITIONS]):
        series = observe_likelihoods(values, trace["dones"], ref, "recursive")
        actual = source.v76.v73.window_statistics(trace, values, series, cell["mode"])
        for window, row in expected.items():
            for field in ("mode_accuracy", "mean_coefficient", "mean_true_minus_best_other_score", "instantaneous_mode_accuracy"):
                np.testing.assert_allclose(actual[window][field], row[field], rtol=1e-9, atol=1e-8)
        checked += 1
    return checked


def summarize(records):
    result = v117.summarize(records)
    for name, arms in result.items():
        selected = [r for r in records if r["mode"] == "normal" and
            (name == "all" or name == "updating" and r["updating"] or name == "frozen" and not r["updating"] or r["group"] == name)]
        arms["incumbent_v98"].update(normal_full_above_threshold_fraction=float(np.mean([
            r["incumbent"][PRIMARY]["full"]["above_threshold_fraction"] for r in selected])),
            worst_normal_episode_mean_risk=max(e["mean_risk"] for r in selected for e in r["incumbent_normal_episodes"][PRIMARY]))
        for arm, entry in arms.items():
            windows = lambda r: r["incumbent"][PRIMARY] if arm == "incumbent_v98" else r["arms"][arm]
            entry["normal_full_mean_risk"] = float(np.mean([windows(r)["full"]["mean_coefficient"] for r in selected]))
            index = {v117.key(r): r for r in selected}
            changed = [r for r in records if r["mode"] != "normal" and (r["group"], r["seed"], "normal", r["updating"]) in index]
            entry["minimum_departure_contrast"] = min(windows(r)[w]["mean_coefficient"]-windows(
                index[r["group"], r["seed"], "normal", r["updating"]])[w]["mean_coefficient"]
                for r in changed for w in ("first_post50", "last64"))
    return result


def evaluate_paths(args, source, population, spec, family, mean, classifiers, references, incumbent):
    from scripts.v73_recursive_mode_readout import observe_likelihoods

    records, checked = [], 0
    for cell in population:
        trace = source.v76.v68.load_trace(source.v76.trace_path(args.checkpoint_dir/"traces", cell))
        assert len(trace["dones"]) == spec["steps"]
        scores, old = score_trace(family, mean, classifiers, trace)
        arms, old_stats, episodes, old_episodes = {}, {}, {}, {}
        for name, values, ref in ([(a, scores[a], references[a]) for a in ARMS]
                                  +[(c, old[c], incumbent[c]) for c in CONDITIONS]):
            assert values.shape == (spec["steps"], 5) and np.isfinite(values).all()
            series = observe_likelihoods(values, trace["dones"], ref, "recursive")
            assert all(np.isfinite(v).all() for v in series.values())
            checked += 1
            if args.smoke:
                prefix = observe_likelihoods(values[:48], trace["dones"][:48], ref, "recursive")
                for field in series:
                    np.testing.assert_array_equal(prefix[field], series[field][:48])
                continue
            stats = v117.window_statistics(trace, values, series, cell["mode"], source)
            (arms if name in ARMS else old_stats)[name] = stats
            if cell["mode"] == "normal":
                (episodes if name in ARMS else old_episodes)[name] = v107.episode_diagnostics(trace, series)
        if not args.smoke:
            records.append(dict(**cell, arms=arms, incumbent=old_stats, normal_episodes=episodes or None,
                                incumbent_normal_episodes=old_episodes or None))
        print(f"[EVAL] {checked//6}/{len(population)} seed{args.seed} {cell['mode']} updating={cell['updating']}", flush=True)
    return records, checked


def run(args):
    sys.path.insert(0, str(args.v83_dir))
    from scripts import analyze_general_ood_v83_cross_seed_covariance as source

    torch.set_num_threads(2)
    population = cells(args.seed, args.smoke)
    spec = dict(seeds=[SMOKE_SEED], steps=64, switch_step=32) if args.smoke else SPEC
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    family, mean, classifiers, references, incumbent, development = load_readouts(args, source)
    before = reference_metadata(references, incumbent)
    comparisons = verify_development(args, source, family, mean, classifiers, references, incumbent, development) if args.smoke else 0
    collected = v95.collect_paths(args, source, population, spec)
    totals = {k: sum(r[k] for r in collected) for k in v95.COUNTERS}
    steps = spec["steps"]
    assert totals == dict(actual_transitions=10*steps, counterfactual_step_calls=50*steps,
        environment_step_calls=60*steps, exact_true_mode_replays=10*steps, optimizer_updates=5*steps)
    records, checked = evaluate_paths(args, source, population, spec, family, mean, classifiers, references, incumbent)
    assert checked == 60 and before == reference_metadata(references, incumbent)
    result = dict(revision=REVISION, source_revision=v117.REVISION, collector_revision=v95.REVISION,
        passed=True, smoke=args.smoke, seed=args.seed, spec=spec, arms=list(ARMS), incumbent_conditions=list(CONDITIONS),
        primary_condition=PRIMARY, collection_records=collected, **totals,
        checked_path_readouts=checked, development_readout_comparisons=comparisons,
        reference_metadata=before, reference_metadata_unchanged=True, references_frozen_before_collection=True,
        reference_refitted=False, normal_thresholds_recalibrated=False, temporal_rules_changed=False,
        classifier_settings_changed=False, mean_settings_changed=False, candidate_selected=False,
        complete_history_preserved=True, paired_prefixes=True, readout_controls_actor=False, readout_adopted=False,
        oracle_observations_used_for_scores=False, counterfactuals_added_to_RL_replay=False,
        source_training_seed=1, source_updates=99745, collection_device=args.device,
        fresh_environment_seeds=True, independent_training_seed_confirmation=False,
        fresh_paths=0 if args.smoke else 10, changed_windows=0 if args.smoke else 16,
        records=records, summary=summarize(records) if records else None,
        decision="frozen-classifier-native-smoke" if args.smoke else "frozen-classifier-environment-seed-confirmation-not-adoption",
        limitations=["Three environment seeds conditional on one source policy and known mild Hopper modes.",
            "The candidate observes fixed-minus2 collection; its actor feedback and return are not tested.",
            "Empirical score weights are not calibrated physical-density posteriors; measured exceedance is not a guarantee.",
            "Reference budgets differ. Reusing these outcomes for redesign makes them development data."])
    path = args.result_dir/("native_smoke.json" if args.smoke else "fresh_seed.json")
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] {REVISION}: {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("v117-dir", "v116-dir", "v115-dir", "v113-dir", "v104-dir", "v98-dir", "v97-dir", "v95-dir",
                 "v94-dir", "v83-dir", "v78-dir", "v76-dir", "v74-dir", "v73-dir", "v72-dir", "v71-dir",
                 "v69-dir", "v70-dir", "v61-dir", "v62-dir", "source-dir", "shared-dir", "checkpoint-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
