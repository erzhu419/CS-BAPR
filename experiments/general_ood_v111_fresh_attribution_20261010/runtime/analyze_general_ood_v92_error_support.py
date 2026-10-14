"""Match error support to frozen v89 means, retaining original raw geometry."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import Covariance, multivariate_normal
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_general_ood_v91_matched_uncertainty as v91

v89, v84 = v91.v89, v91.v84
REVISION = "general_ood_v92_error_support"
CONDITIONS = {"prefix_support_mix": "prefix_mean_mix", "suffix_support_mix": "suffix_mean_mix"}


def added_errors(bank, groups, base_rows):
    from scripts.v83_cross_seed_covariance import LocalLinearMean, subset_bank

    rows = np.arange(base_rows, bank.tree.n)
    errors = {m: np.empty_like(bank.residuals[m][rows]) for m in v84.MODES}
    fitting = []
    for seed in np.unique(groups[rows]):
        selected = rows[groups[rows] == seed]
        keep = groups != seed
        reduced = subset_bank(bank, keep)
        inputs = bank.tree.data[selected]*bank.features.input_scale+bank.features.input_mean
        predicted = LocalLinearMean(reduced).predict(inputs)
        for mode in v84.MODES:
            errors[mode][selected-base_rows] = bank.residuals[mode][selected]-predicted[mode]
        original = int((groups[:base_rows] != seed).sum())
        fitting.append(dict(seed=int(seed), error_rows=len(selected), mean_training_rows=reduced.tree.n,
            mean_base_training_rows=original, mean_added_training_rows=reduced.tree.n-original,
            excluded_mean_rows=int((~keep).sum()), source_seed_excluded_from_all_mean_rows=True))
        print(f"[ADDED] seed{seed}: {len(selected)} errors from {reduced.tree.n} other-seed mean rows", flush=True)
    return errors, fitting


def load_errors(raw_bank, raw_groups, mean_bank, mean_groups, seed, condition, source, directory):
    from scripts.v83_cross_seed_covariance import ALPHA

    np.testing.assert_array_equal(mean_groups[:raw_bank.tree.n], raw_groups)
    np.testing.assert_array_equal(mean_bank.tree.data[:raw_bank.tree.n], raw_bank.tree.data)
    old = torch.load(source/f"outer_seed{seed}_{condition}.pt", map_location="cpu", weights_only=False)
    assert old["revision"] == v91.REVISION and old["outer_seed"] == seed and old["condition"] == condition
    assert old["ridge_alpha"] == ALPHA and old["mean_rows"] == mean_bank.tree.n
    np.testing.assert_array_equal(old["covariance_groups"], raw_groups)
    np.testing.assert_array_equal(old["mean_groups"], mean_groups)
    assert all(old["errors"][m].shape == raw_bank.residuals[m].shape and
               np.isfinite(old["errors"][m]).all() for m in v84.MODES)
    path = directory/f"outer_seed{seed}_{condition}_added.pt"
    if path.is_file():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        assert saved["revision"] == REVISION and saved["outer_seed"] == seed and saved["condition"] == condition
        assert saved["ridge_alpha"] == ALPHA and saved["base_rows"] == raw_bank.tree.n
        np.testing.assert_array_equal(saved["mean_groups"], mean_groups)
        extra, fitting = saved["errors"], saved["inner_fits"]
        print(f"[REUSE] seed{seed} {condition} added errors", flush=True)
    else:
        extra, fitting = added_errors(mean_bank, mean_groups, raw_bank.tree.n)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(dict(revision=REVISION, outer_seed=seed, condition=condition, ridge_alpha=ALPHA,
            base_rows=raw_bank.tree.n, mean_groups=mean_groups, errors=extra, inner_fits=fitting), path)
    assert all(extra[m].shape == mean_bank.residuals[m][raw_bank.tree.n:].shape and
               np.isfinite(extra[m]).all() for m in v84.MODES)
    errors = {m: np.concatenate((old["errors"][m], extra[m])) for m in v84.MODES}
    return errors, old["errors"], old["inner_fits"]+fitting


def verify_error_rows(bank, groups, errors, base_rows):
    from scripts.v83_cross_seed_covariance import LocalLinearMean, subset_bank

    checked = 0
    for seed in np.unique(groups):
        rows = np.concatenate([np.flatnonzero((groups == seed) & (np.arange(bank.tree.n) < base_rows))[:1],
                               np.flatnonzero((groups == seed) & (np.arange(bank.tree.n) >= base_rows))[:1]])
        inputs = bank.tree.data[rows]*bank.features.input_scale+bank.features.input_mean
        predicted = LocalLinearMean(subset_bank(bank, groups != seed)).predict(inputs)
        for mode in v84.MODES:
            np.testing.assert_allclose(errors[mode][rows], bank.residuals[mode][rows]-predicted[mode],
                                       rtol=1e-10, atol=1e-10)
        checked += len(rows)
    return checked


def support_geometry(raw_bank, mean_bank, errors, inputs, weight):
    from scripts.v83_cross_seed_covariance import LocalLinearMean, error_covariance

    centers = LocalLinearMean(mean_bank).predict(inputs)
    normalized = (inputs-mean_bank.features.input_mean)/mean_bank.features.input_scale
    _, indices = mean_bank.tree.query(normalized, k=64)
    covariance, diagnostic = {}, {}
    for mode in v84.MODES:
        neighbors = errors[mode][indices]
        error = np.asarray([error_covariance(e)[0] for e in neighbors])
        raw = raw_bank.geometry(inputs, mode)[1]
        covariance[mode] = weight*raw+(1-weight)*error
        diagnostic[mode] = dict(error_trace=float(np.trace(error, axis1=1, axis2=2).mean()),
            neighbor_cross_seed_error_rms=float(np.sqrt(np.mean(neighbors**2, axis=(1, 2))).mean()))
    return centers, covariance, dict(per_mode=diagnostic,
        error_added_neighbor_fraction=float(np.mean(indices >= raw_bank.tree.n)))


def verify_geometry(raw_bank, mean_bank, errors, inputs, weight, expected, shifted, raw, centers, covariance):
    from scripts.v83_cross_seed_covariance import error_covariance

    _, indices = mean_bank.tree.query((inputs-mean_bank.features.input_mean)/mean_bank.features.input_scale, k=64)
    terms = v84.scores(shifted, np.stack([covariance[m] for m in v84.MODES], axis=1))
    maximum_condition, maximum_difference = 0., 0.
    for c, mode in enumerate(v84.MODES):
        np.testing.assert_allclose(centers[mode], expected[:, c], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(raw_bank.geometry(inputs, mode)[1], raw[:, c], rtol=1e-10, atol=1e-10)
        component = np.asarray([error_covariance(e)[0] for e in errors[mode][indices]])
        np.testing.assert_allclose(covariance[mode], weight*raw[:, c]+(1-weight)*component, rtol=1e-10, atol=1e-10)
        physical = covariance[mode]*raw_bank.target_scale[None, :, None]*raw_bank.target_scale[None, None, :]
        for row in range(len(inputs)):
            maximum_condition = max(maximum_condition, float(np.linalg.cond(physical[row])))
            distribution = Covariance.from_cholesky(np.linalg.cholesky(physical[row]))
            for truth in range(len(v84.MODES)):
                target = shifted[row, truth, c]*raw_bank.target_scale
                reference = multivariate_normal.logpdf(target, cov=distribution)
                score = terms[0][row, truth, c]-.5*len(target)*np.log(2*np.pi)-np.log(raw_bank.target_scale).sum()
                maximum_difference = max(maximum_difference, abs(float(score-reference)))
                np.testing.assert_allclose(score, reference, rtol=1e-10, atol=1e-8)
    return dict(gaussian_checks=len(inputs)*len(v84.MODES)**2,
        maximum_physical_covariance_condition=maximum_condition, maximum_physical_score_difference=maximum_difference)


def run(args):
    sys.path.insert(0, str(args.v83_dir))
    from scripts import v83_cross_seed_covariance as v83

    torch.set_num_threads(2)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    source = args.v83_dir/"run_checkpoints/cross-seed-covariance"
    selection = json.loads((source/"nested_selection.json").read_text())
    previous = json.loads((args.v91_dir/"run_results/pilot/matched_uncertainty.json").read_text())
    frozen = json.loads((args.v89_dir/"run_results/pilot/reference_prefix.json").read_text())
    assert selection["revision"] == v83.REVISION and previous["revision"] == v91.REVISION and not previous["smoke"]
    assert frozen["revision"] == v89.REVISION and len(previous["folds"]) == len(frozen["folds"]) == 6
    bank = torch.load(args.v76_dir/"run_checkpoints/formal/behavior_coverage_references.pt",
                      map_location="cpu", weights_only=False)["references"]["updating_mixed"].bank
    assert bank.tree.n == 23000 and tuple(bank.residuals) == v84.MODES
    groups = v83.groups_from_blocks(selection["reference_blocks"], bank.tree.n)
    weight = selection["selection"]["raw_weight"]
    assert weight == previous["raw_weight"] == frozen["raw_weight"]
    traces = {}
    for seed in v89.SEEDS:
        with np.load(args.v76_dir/f"run_checkpoints/formal/traces/density_extension_seed{seed}_updating_normal.npz") as saved:
            traces[seed] = {k: saved[k] for k in (*v89.KEYS, "oracle_observations")}
    v89.verify_sources(bank, selection["reference_blocks"], traces)
    banks, expanded_groups = v89.expand_banks(bank, groups, traces)
    folds = selection["selection"]["folds"]
    assert len(folds) == 6
    if args.smoke:
        folds = [f for f in folds if f["seed"] == v89.SEEDS[0]]
    old_by_seed = {f["seed"]: f for f in previous["folds"]}
    frozen_by_seed = {f["seed"]: f for f in frozen["folds"]}
    weights = dict(raw=1., cross_seed=0., selected_mix=weight)
    collected = {c: [] for c in (*previous["conditions"], *CONDITIONS)}
    records, changes = [], []
    verification = dict(source_base_rows_checked=2200, fold_control_comparisons=0, direct_baseline_mean_queries=0,
        inner_seed_exclusion_checks=0, direct_error_rows_checked=0, gaussian_checks=0,
        maximum_physical_covariance_condition=0., maximum_physical_score_difference=0.)
    for fold in folds:
        seed = fold["seed"]
        print(f"[OUTER] seed{seed}", flush=True)
        cached = torch.load(source/f"validation/outer_seed{seed}.pt", map_location="cpu", weights_only=False)
        assert cached["revision"] == v83.REVISION and cached["ridge_alpha"] == v83.ALPHA
        values = cached["values"]
        assert values[3]["seed"] == seed and values[3]["validation_rows"] == 512
        assert values[3]["outer_seed_excluded_from_inner_fits"] and values[3]["inner_seed_excluded_from_its_error_fit"]
        inputs, baseline, offsets, cross, raw = v84.reconstruct_fold(bank, groups, values)
        rows = np.linspace(0, len(inputs)-1, 8, dtype=int)
        direct, _ = v89.predict_heldout(bank, groups, seed, inputs[rows], bank.tree.n)
        np.testing.assert_allclose(direct, baseline[rows], rtol=1e-10, atol=1e-10)
        verification["direct_baseline_mean_queries"] += len(rows)
        conditions, fitting, uncertainty, terms_by_name = {}, {}, {}, {}
        for name, w in weights.items():
            terms_by_name[name] = v84.scores(offsets, w*raw+(1-w)*cross)
        frozen_covariance = weight*raw+(1-weight)*cross
        for condition, mean_condition in CONDITIONS.items():
            old_condition = next(c for c, m in v91.CONDITIONS.items() if m == mean_condition)
            predicted, metadata = v89.predict_heldout(banks[mean_condition], expanded_groups, seed, inputs, bank.tree.n)
            assert metadata == old_by_seed[seed]["mean_fit"][old_condition]
            assert metadata == frozen_by_seed[seed]["mean_fit"][mean_condition]
            shifted = v89.coverage_offsets(offsets, baseline, predicted)
            terms_by_name[mean_condition] = v84.scores(shifted, frozen_covariance)
            raw_bank, raw_groups, mean_bank, mean_groups = v91.outer_banks(
                bank, groups, banks[mean_condition], expanded_groups, seed)
            assert seed not in raw_groups and seed not in mean_groups
            errors, old_errors, inner = load_errors(raw_bank, raw_groups, mean_bank, mean_groups, seed,
                mean_condition, args.v91_dir/"run_checkpoints/pilot", args.checkpoint_dir)
            assert all(f["source_seed_excluded_from_all_mean_rows"] for f in inner)
            assert {f["seed"] for f in inner} == set(np.unique(mean_groups))
            assert sum(f["error_rows"] for f in inner) == mean_bank.tree.n
            verification["inner_seed_exclusion_checks"] += len(inner)
            verification["direct_error_rows_checked"] += verify_error_rows(mean_bank, mean_groups, errors, raw_bank.tree.n)
            old_density = v91.make_density(raw_bank, mean_bank, old_errors, weight)
            old_centers, old_covariance, _ = old_density.geometry(inputs)
            terms_by_name[old_condition] = v84.scores(shifted, np.stack([old_covariance[m] for m in v84.MODES], axis=1))
            query_rows = rows if args.smoke else np.arange(len(inputs))
            centers, covariance, diagnostic = support_geometry(raw_bank, mean_bank, errors, inputs[query_rows], weight)
            for c, mode in enumerate(v84.MODES):
                np.testing.assert_allclose(centers[mode], old_centers[mode][query_rows], rtol=1e-10, atol=1e-10)
            expected = predicted[query_rows]-np.stack(
                [bank.features.features(inputs[query_rows])@bank.coefficients[m] for m in v84.MODES], axis=1)
            check_rows = np.arange(len(rows)) if args.smoke else rows
            check = verify_geometry(raw_bank, mean_bank, errors, inputs[query_rows][check_rows], weight,
                expected[check_rows], shifted[query_rows][check_rows], raw[query_rows][check_rows],
                {m: a[check_rows] for m, a in centers.items()}, {m: a[check_rows] for m, a in covariance.items()})
            verification["gaussian_checks"] += check["gaussian_checks"]
            for name in ("maximum_physical_covariance_condition", "maximum_physical_score_difference"):
                verification[name] = max(verification[name], check[name])
            mixed = np.stack([covariance[m] for m in v84.MODES], axis=1)
            terms_by_name[condition] = v84.scores(shifted[query_rows], mixed)
            fitting[condition] = metadata
            uncertainty[condition] = dict(raw_support_rows=raw_bank.tree.n, error_support_rows=mean_bank.tree.n,
                reused_original_error_rows=raw_bank.tree.n, added_error_rows=mean_bank.tree.n-raw_bank.tree.n,
                inner_fits=inner, **diagnostic,
                per_mode_traces={m: dict(frozen_trace=float(np.trace(frozen_covariance[query_rows, c], axis1=1, axis2=2).mean()),
                    original_support_trace=float(np.trace(old_covariance[m][query_rows], axis1=1, axis2=2).mean()),
                    expanded_support_trace=float(np.trace(mixed[:, c], axis1=1, axis2=2).mean())) for c, m in enumerate(v84.MODES)})
            stats = v84.summarize(*terms_by_name[condition])
            for control in (mean_condition, old_condition):
                before = v84.summarize(*(a[query_rows] for a in terms_by_name[control]))
                for mode in v84.MODES:
                    new, old = stats["per_mode"][mode], before["per_mode"][mode]
                    changes.append(dict(seed=seed, mode=mode, condition=condition, control=control,
                        classification_nll_delta=new["classification_nll"]-old["classification_nll"],
                        accuracy_delta=new["accuracy"]-old["accuracy"]))
            print(f"[FOLD] seed{seed} {condition}: NLL={stats['classification_nll']:.8f}, accuracy={stats['accuracy']:.6f}", flush=True)
        for name, terms in terms_by_name.items():
            stats = v84.summarize(*terms)
            if name in previous["conditions"]:
                v89.prior.verify_control(stats, old_by_seed[seed]["conditions"][name])
                verification["fold_control_comparisons"] += 1
            collected[name].append(terms)
            conditions[name] = v89.prior.compact(stats)
        records.append(dict(values[3], mean_fit=fitting, uncertainty=uncertainty, conditions=conditions))
    aggregate = {c: dict(raw_weight=weights.get(c, weight), **v89.prior.compact(v84.summarize(
        *(np.concatenate([v[i] for v in terms]) for i in range(3))))) for c, terms in collected.items()}
    if not args.smoke:
        for name in previous["conditions"]:
            v89.prior.verify_control(aggregate[name], previous["conditions"][name])
    result = dict(revision=REVISION, source_revision=v91.REVISION, smoke=args.smoke, passed=True,
        modes=list(v84.MODES), raw_weight=weight, conditions=aggregate, folds=records,
        new_conditions=CONDITIONS, fold_mode_changes=changes, verification=dict(verification,
            all_seven_controls_match_v91=True, expanded_means_unchanged=True, original_raw_geometry_unchanged=True,
            outer_seed_excluded_before_inner_fits=True, inner_seed_exclusion_includes_all_added_rows=True,
            reused_original_errors_directly_checked=True, physical_gaussian_scores_match=True),
        mean_fitting_rows=25200, raw_reference_rows=23000, error_reference_rows=25200,
        queries_per_new_condition=len(records)*(8 if args.smoke else 512),
        true_mode_labels_per_new_condition=len(records)*(8 if args.smoke else 512)*5,
        weighting="equal seed / true mode / row", prior="uniform, instantaneous posterior",
        covariance_formula="frozen v83 weight * original-bank raw covariance + complementary weight * expanded-bank 64-neighbor OAS second moment of expanded-mean whole-seed-held-out errors",
        covariance_support_changed=True, conditional_means_changed=False, raw_covariance_component_changed=False,
        weight_refitted=False, global_forward_refitted=False, scaling_refitted=False, temporal_rules_changed=False,
        evaluation_paths_loaded=False, normal_null_loaded=False, readout_controls_actor=False,
        environment_steps=0, optimizer_updates=0, additional_counterfactual_steps=0, independent_confirmation=False,
        decision="native-correspondence-only" if args.smoke else "reference-only-support-matching-pending-analysis",
        limitations=["Previously inspected development folds and globally selected frozen weight; no independent confirmation.",
            "Inner seed exclusions change the mean training population; support matching need not fix error transport.",
            "Smoke evaluates new conditions on eight inputs, old controls on all 512; only paired subset deltas are comparable.",
            "Instantaneous reference classification is not recursive trajectory or RL return evidence."])
    path = args.result_dir/("native_smoke.json" if args.smoke else "error_support.json")
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] {REVISION}: {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("v91-dir", "v89-dir", "v83-dir", "v76-dir", "checkpoint-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
