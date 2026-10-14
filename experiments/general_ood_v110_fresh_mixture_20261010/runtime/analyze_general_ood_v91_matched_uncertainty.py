"""Reference-only error covariances matched to the two frozen v89 means."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import Covariance, multivariate_normal
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_general_ood_v89_reference_prefix as v89

v84 = v89.v84
REVISION = "general_ood_v91_matched_uncertainty"
CONDITIONS = {"prefix_consistent_mix": "prefix_mean_mix", "suffix_consistent_mix": "suffix_mean_mix"}


def outer_banks(bank, groups, expanded, expanded_groups, seed):
    from scripts.v83_cross_seed_covariance import subset_bank

    keep, mean_keep = groups != seed, expanded_groups != seed
    return (subset_bank(bank, keep), groups[keep],
            subset_bank(expanded, mean_keep), expanded_groups[mean_keep])


def aligned_errors(bank, groups, mean_bank, mean_groups, rows=None):
    from scripts.v83_cross_seed_covariance import LocalLinearMean, subset_bank

    assert mean_bank.tree.n == len(mean_groups) and bank.tree.n == len(groups)
    np.testing.assert_array_equal(mean_groups[:bank.tree.n], groups)
    np.testing.assert_array_equal(mean_bank.tree.data[:bank.tree.n], bank.tree.data)
    rows = np.arange(bank.tree.n) if rows is None else np.asarray(rows)
    assert len(np.unique(rows)) == len(rows)
    errors = {m: np.full_like(bank.residuals[m], np.nan) for m in v84.MODES}
    fitting = []
    for seed in np.unique(groups):
        selected = rows[groups[rows] == seed]
        if not len(selected):
            continue
        keep = mean_groups != seed
        reduced = subset_bank(mean_bank, keep)
        inputs = bank.tree.data[selected]*bank.features.input_scale+bank.features.input_mean
        predictions = LocalLinearMean(reduced).predict(inputs)
        for mode in v84.MODES:
            errors[mode][selected] = bank.residuals[mode][selected]-predictions[mode]
        base_rows = int((groups != seed).sum())
        fitting.append(dict(seed=int(seed), error_rows=len(selected), mean_training_rows=reduced.tree.n,
            mean_base_training_rows=base_rows, mean_added_training_rows=reduced.tree.n-base_rows,
            excluded_mean_rows=int((~keep).sum()), source_seed_excluded_from_all_mean_rows=True))
        print(f"[ERRORS] seed{seed}: {len(selected)} old-support rows; mean trained on {reduced.tree.n} other-seed rows", flush=True)
    assert all(np.isfinite(e[rows]).all() for e in errors.values())
    return errors, fitting


def cached_errors(bank, groups, mean_bank, mean_groups, seed, condition, directory):
    from scripts.v83_cross_seed_covariance import ALPHA

    path = directory/f"outer_seed{seed}_{condition}.pt"
    if path.is_file():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        assert saved["revision"] == REVISION and saved["condition"] == condition and saved["outer_seed"] == seed
        assert saved["ridge_alpha"] == ALPHA and saved["mean_rows"] == mean_bank.tree.n
        np.testing.assert_array_equal(saved["covariance_groups"], groups)
        np.testing.assert_array_equal(saved["mean_groups"], mean_groups)
        errors, fitting = saved["errors"], saved["inner_fits"]
        assert all(errors[m].shape == bank.residuals[m].shape and np.isfinite(errors[m]).all() for m in v84.MODES)
        print(f"[REUSE] outer seed{seed} {condition} full error bank", flush=True)
    else:
        errors, fitting = aligned_errors(bank, groups, mean_bank, mean_groups)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(dict(revision=REVISION, condition=condition, outer_seed=seed, ridge_alpha=ALPHA,
            mean_rows=mean_bank.tree.n, covariance_groups=groups, mean_groups=mean_groups,
            errors=errors, inner_fits=fitting), path)
    return errors, fitting


def make_density(bank, mean_bank, errors, weight):
    from scripts.v83_cross_seed_covariance import CrossSeedDensityBank, LocalLinearMean

    density = CrossSeedDensityBank(bank, errors, weight)
    density.correction = LocalLinearMean(mean_bank)
    return density


def smoke_rows(bank, groups, inputs):
    normalized = (inputs-bank.features.input_mean)/bank.features.input_scale
    _, indices = bank.tree.query(normalized, k=64)
    representatives = [np.flatnonzero(groups == s)[np.linspace(0, np.sum(groups == s)-1, 8, dtype=int)]
                       for s in np.unique(groups)]
    return np.unique(np.concatenate([indices.ravel(), *representatives]))


def verify_geometry(bank, density, inputs, expected_means, shifted, raw, centers, covariance):
    from scripts.v83_cross_seed_covariance import error_covariance

    normalized = (inputs-bank.features.input_mean)/bank.features.input_scale
    _, indices = bank.tree.query(normalized, k=64)
    actual_covariance = np.stack([covariance[m] for m in v84.MODES], axis=1)
    terms = v84.scores(shifted, actual_covariance)
    maximum_condition, maximum_score_difference = 0., 0.
    for c, mode in enumerate(v84.MODES):
        np.testing.assert_allclose(centers[mode], expected_means[:, c], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(bank.geometry(inputs, mode)[1], raw[:, c], rtol=1e-10, atol=1e-10)
        fitted = np.asarray([error_covariance(e)[0] for e in density.errors[mode][indices]])
        np.testing.assert_allclose(covariance[mode], density.raw_weight*raw[:, c]+(1-density.raw_weight)*fitted,
                                   rtol=1e-10, atol=1e-10)
        physical_covariance = covariance[mode]*bank.target_scale[None, :, None]*bank.target_scale[None, None, :]
        for row in range(len(inputs)):
            maximum_condition = max(maximum_condition, float(np.linalg.cond(physical_covariance[row])))
            # Use triangular solves rather than a spectral inverse in physical units.
            distribution = Covariance.from_cholesky(np.linalg.cholesky(physical_covariance[row]))
            for truth in range(len(v84.MODES)):
                expected = multivariate_normal.logpdf(shifted[row, truth, c]*bank.target_scale,
                                                     cov=distribution)
                physical_score = terms[0][row, truth, c]-.5*len(bank.target_scale)*np.log(2*np.pi)-np.log(bank.target_scale).sum()
                maximum_score_difference = max(maximum_score_difference, abs(float(physical_score-expected)))
                np.testing.assert_allclose(physical_score, expected, rtol=1e-10, atol=1e-8)
    return dict(gaussian_checks=len(inputs)*len(v84.MODES)**2,
        maximum_physical_covariance_condition=maximum_condition,
        maximum_physical_score_difference=maximum_score_difference)


def run(args):
    sys.path.insert(0, str(args.v83_dir))
    from scripts import v83_cross_seed_covariance as v83

    torch.set_num_threads(2)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    source = args.v83_dir/"run_checkpoints/cross-seed-covariance"
    selection = json.loads((source/"nested_selection.json").read_text())
    previous = json.loads((args.v89_dir/"run_results/pilot/reference_prefix.json").read_text())
    assert selection["revision"] == v83.REVISION and previous["revision"] == v89.REVISION and not previous["smoke"]
    bank = torch.load(args.v76_dir/"run_checkpoints/formal/behavior_coverage_references.pt",
                      map_location="cpu", weights_only=False)["references"]["updating_mixed"].bank
    assert bank.tree.n == 23000 and tuple(bank.residuals) == v84.MODES
    groups = v83.groups_from_blocks(selection["reference_blocks"], bank.tree.n)
    weight = selection["selection"]["raw_weight"]
    assert weight == previous["raw_weight"]
    stored = args.v76_dir/"run_checkpoints/formal/traces"
    traces = {}
    for seed in v89.SEEDS:
        with np.load(stored/f"density_extension_seed{seed}_updating_normal.npz") as saved:
            traces[seed] = {k: saved[k] for k in (*v89.KEYS, "oracle_observations")}
    v89.verify_sources(bank, selection["reference_blocks"], traces)
    banks, expanded_groups = v89.expand_banks(bank, groups, traces)
    expected_folds = selection["selection"]["folds"]
    assert len(expected_folds) == len(previous["folds"]) == 6
    if args.smoke:
        expected_folds = [f for f in expected_folds if f["seed"] == v89.SEEDS[0]]
        assert len(expected_folds) == 1
    old_by_seed = {f["seed"]: f for f in previous["folds"]}
    weights = dict(raw=1., cross_seed=0., selected_mix=weight)
    collected = {c: [] for c in (*weights, *v89.CONDITIONS, *CONDITIONS)}
    records, changes = [], []
    checked, gaussians, controls, inner_checks = 0, 0, 0, 0
    maximum_condition, maximum_score_difference = 0., 0.
    for expected in expected_folds:
        seed = expected["seed"]
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
        checked += len(rows)
        conditions, fitting, uncertainty = {}, {}, {}
        for name, w in weights.items():
            terms = v84.scores(offsets, w*raw+(1-w)*cross)
            stats = v84.summarize(*terms)
            v89.prior.verify_control(stats, old_by_seed[seed]["conditions"][name])
            collected[name].append(terms)
            conditions[name] = v89.prior.compact(stats)
            controls += 1
        frozen_covariance = weight*raw+(1-weight)*cross
        for condition, mean_condition in CONDITIONS.items():
            expanded = banks[mean_condition]
            predicted, metadata = v89.predict_heldout(expanded, expanded_groups, seed, inputs, bank.tree.n)
            shifted = v89.coverage_offsets(offsets, baseline, predicted)
            frozen = v84.scores(shifted, frozen_covariance)
            stats = v84.summarize(*frozen)
            v89.prior.verify_control(stats, old_by_seed[seed]["conditions"][mean_condition])
            assert metadata == old_by_seed[seed]["mean_fit"][mean_condition]
            collected[mean_condition].append(frozen)
            conditions[mean_condition] = v89.prior.compact(stats)
            controls += 1
            covariance_bank, covariance_groups, mean_bank, mean_groups = outer_banks(bank, groups, expanded, expanded_groups, seed)
            assert seed not in covariance_groups and seed not in mean_groups
            fitting[condition] = metadata
            query_rows = rows if args.smoke else np.arange(len(inputs))
            query_inputs = inputs[query_rows]
            if args.smoke:
                error_rows = smoke_rows(covariance_bank, covariance_groups, query_inputs)
                errors, inner = aligned_errors(covariance_bank, covariance_groups, mean_bank, mean_groups, error_rows)
            else:
                errors, inner = cached_errors(covariance_bank, covariance_groups, mean_bank, mean_groups,
                                             seed, mean_condition, args.checkpoint_dir)
                error_rows = np.arange(covariance_bank.tree.n)
            assert {f["seed"] for f in inner} == set(np.unique(covariance_groups))
            assert all(f["source_seed_excluded_from_all_mean_rows"] for f in inner)
            inner_checks += len(inner)
            density = make_density(covariance_bank, mean_bank, errors, weight)
            centers, covariance, diagnostic = density.geometry(query_inputs)
            expected_centers = predicted[query_rows]-np.stack(
                [bank.features.features(query_inputs)@bank.coefficients[m] for m in v84.MODES], axis=1)
            check_rows = np.arange(len(rows)) if args.smoke else rows
            geometry_check = verify_geometry(covariance_bank, density, query_inputs[check_rows],
                expected_centers[check_rows], shifted[query_rows][check_rows], raw[query_rows][check_rows],
                {m: c[check_rows] for m, c in centers.items()}, {m: c[check_rows] for m, c in covariance.items()})
            gaussians += geometry_check["gaussian_checks"]
            maximum_condition = max(maximum_condition, geometry_check["maximum_physical_covariance_condition"])
            maximum_score_difference = max(maximum_score_difference, geometry_check["maximum_physical_score_difference"])
            for c, mode in enumerate(v84.MODES):
                np.testing.assert_allclose(centers[mode], expected_centers[:, c], rtol=1e-10, atol=1e-10)
            mixed = np.stack([covariance[m] for m in v84.MODES], axis=1)
            updated = v84.scores(shifted[query_rows], mixed)
            stats = v84.summarize(*updated)
            collected[condition].append(updated)
            conditions[condition] = v89.prior.compact(stats)
            uncertainty[condition] = dict(error_rows_computed=len(error_rows), covariance_support_rows=covariance_bank.tree.n,
                partial_smoke_bank=args.smoke, inner_fits=inner,
                per_mode={m: dict(frozen_trace=float(np.trace(frozen_covariance[query_rows, c], axis1=1, axis2=2).mean()),
                    matched_trace=float(np.trace(mixed[:, c], axis1=1, axis2=2).mean()),
                    mean_logdet_change=float((updated[2][:, c]-frozen[2][query_rows, c]).mean()),
                    neighbor_cross_seed_error_rms=float(diagnostic["cross_seed_error_rms"][:, c].mean()))
                    for c, m in enumerate(v84.MODES)})
            frozen_query_stats = v84.summarize(*(a[query_rows] for a in frozen))
            for mode in v84.MODES:
                before = frozen_query_stats["per_mode"][mode]
                after = stats["per_mode"][mode]
                changes.append(dict(seed=seed, mode=mode, condition=condition, control=mean_condition,
                    classification_nll_delta=after["classification_nll"]-before["classification_nll"],
                    accuracy_delta=after["accuracy"]-before["accuracy"],
                    mean_margin_delta=after["mean_true_minus_best_other"]-before["mean_true_minus_best_other"]))
            print(f"[FOLD] seed{seed} {condition}: {len(query_rows)} rows; NLL={stats['classification_nll']:.8f}, accuracy={stats['accuracy']:.6f}", flush=True)
        records.append(dict(values[3], mean_fit=fitting, uncertainty=uncertainty, conditions=conditions))
        if not args.smoke:
            (args.checkpoint_dir/"progress.json").write_text(json.dumps(dict(revision=REVISION, raw_weight=weight,
                folds=records), allow_nan=False)+"\n")
    aggregate = {c: dict(raw_weight=weights.get(c, weight), **v89.prior.compact(v84.summarize(
        *(np.concatenate([v[i] for v in terms]) for i in range(3))))) for c, terms in collected.items()}
    if not args.smoke:
        for c in (*weights, *v89.CONDITIONS):
            v89.prior.verify_control(aggregate[c], previous["conditions"][c])
    result = dict(revision=REVISION, source_revision=v89.REVISION, smoke=args.smoke, passed=True,
        modes=list(v84.MODES), conditions=aggregate, folds=records, fold_mode_changes_vs_frozen_errors=changes,
        new_conditions=CONDITIONS, raw_weight=weight, mean_fitting_rows=25200, covariance_reference_rows=23000,
        validation_rows_per_fold=512, matched_covariance_queries_per_fold=8 if args.smoke else 512,
        matched_covariance_queries=len(records)*(8 if args.smoke else 512),
        verification=dict(source_base_rows_checked=2200, all_five_source_labels_match=True,
            direct_baseline_mean_queries=checked, all_baseline_means_match=True,
            fold_control_comparisons=controls, all_five_frozen_controls_match_v89=True,
            inner_seed_exclusion_checks=inner_checks, outer_seed_excluded_before_all_inner_fits=True,
            seed_exclusion_includes_added_rows=True, expanded_means_unchanged=True,
            raw_covariance_and_neighborhoods_unchanged=True, mixture_covariances_match_formula=True,
            gaussian_checks=gaussians, physical_gaussian_scores_match=True,
            gaussian_reference="SciPy Covariance.from_cholesky in physical delta-state coordinates",
            maximum_physical_covariance_condition=maximum_condition,
            maximum_physical_score_difference=maximum_score_difference),
        covariance_formula="fixed v83 raw weight * original raw covariance + complementary weight * OAS second moment of expanded-mean whole-seed-held-out errors",
        error_rows="original covariance-support inputs only; added rows train means but do not enter the covariance tree",
        weighting="equal seed / true mode / row", prior="uniform, instantaneous posterior",
        new_error_components_refitted=True, covariance_support_changed=False, conditional_mean_rule_changed=False,
        weight_refitted=False, global_forward_refitted=False, scaling_refitted=False, temporal_rules_changed=False,
        evaluation_paths_loaded=False, normal_null_loaded=False, readout_controls_actor=False,
        environment_steps=0, optimizer_updates=0, additional_counterfactual_steps=0, independent_confirmation=False,
        decision="native-correspondence-only" if args.smoke else "reference-only-matched-uncertainty-pending-analysis",
        limitations=["Previously inspected reference folds and a globally selected frozen v83 weight, not independent confirmation.",
            "Fixed original scaling/forward models and added mean labels are inherited; no extra simulator data.",
            "Inner seed exclusions reduce mean support; error transport to full-bank deployment is not guaranteed.",
            "Smoke scores the matched covariance on eight rows only; its full-fold controls have different sample counts.",
            "Instantaneous reference classification is not recursive trajectory or RL return evidence."])
    name = "native_smoke.json" if args.smoke else "matched_uncertainty.json"
    path = args.result_dir/name
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] {REVISION}: {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("v89-dir", "v83-dir", "v76-dir", "checkpoint-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
