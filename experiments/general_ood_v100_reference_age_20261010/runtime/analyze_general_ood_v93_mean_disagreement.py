"""Source-seed mean displacement covariance with frozen v92 predictions/errors."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import Covariance, multivariate_normal
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_general_ood_v92_error_support as v92

v91, v89, v84 = v92.v91, v92.v89, v92.v84
REVISION = "general_ood_v93_mean_disagreement"
CONDITIONS = {"prefix_disagreement_mix": "prefix_support_mix", "suffix_disagreement_mix": "suffix_support_mix"}


def seed_members(bank, groups, inputs, base_rows):
    from scripts.v83_cross_seed_covariance import LocalLinearMean, subset_bank

    assert len(groups) == bank.tree.n
    members, fitting = [], []
    for seed in np.unique(groups):
        keep = groups != seed
        reduced = subset_bank(bank, keep)
        predicted = LocalLinearMean(reduced).predict(inputs)
        members.append(np.stack([predicted[m] for m in v84.MODES], axis=1))
        original = int(keep[:base_rows].sum())
        fitting.append(dict(seed=int(seed), training_rows=reduced.tree.n,
            base_training_rows=original, added_training_rows=reduced.tree.n-original,
            excluded_rows=int((~keep).sum()), source_seed_excluded_from_original_and_added_rows=True))
        print(f"[MEMBER] without seed{seed}: {len(inputs)} queries, {reduced.tree.n} training rows", flush=True)
    return np.asarray(members), fitting


def cached_members(bank, groups, inputs, base_rows, seed, condition, directory):
    from scripts.v83_cross_seed_covariance import ALPHA

    path = directory/f"outer_seed{seed}_{condition}_q{len(inputs)}.pt"
    if path.is_file():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        assert saved["revision"] == REVISION and saved["condition"] == condition and saved["outer_seed"] == seed
        assert saved["base_rows"] == base_rows and saved["ridge_alpha"] == ALPHA
        np.testing.assert_array_equal(saved["groups"], groups)
        np.testing.assert_array_equal(saved["inputs"], inputs)
        members, fitting = saved["members"], saved["member_fits"]
        print(f"[REUSE] seed{seed} {condition}: {len(inputs)} member queries", flush=True)
    else:
        members, fitting = seed_members(bank, groups, inputs, base_rows)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(dict(revision=REVISION, condition=condition, outer_seed=seed, ridge_alpha=ALPHA,
            base_rows=base_rows, groups=groups, inputs=inputs, members=members, member_fits=fitting), path)
    assert members.shape == (len(np.unique(groups)), len(inputs), len(v84.MODES), len(bank.target_scale))
    assert np.isfinite(members).all()
    return members, fitting


def displacement_second_moment(centers, members):
    assert members.shape[1:] == centers.shape
    displacements = members-centers[None]
    return np.einsum("gnmd,gnme->nmde", displacements, displacements)/len(members)


def verify_members(bank, groups, inputs, base_rows, members, fitting):
    rows = np.linspace(0, len(inputs)-1, 2, dtype=int)
    design = bank.features.features(inputs[rows])
    forward = np.stack([design@bank.coefficients[m] for m in v84.MODES], axis=1)
    assert [f["seed"] for f in fitting] == list(np.unique(groups))
    for member, fit in zip(members, fitting):
        predicted, metadata = v89.predict_heldout(bank, groups, fit["seed"], inputs[rows], base_rows)
        np.testing.assert_allclose(member[rows]+forward, predicted, rtol=1e-10, atol=1e-10)
        for key in ("training_rows", "base_training_rows", "added_training_rows", "excluded_rows"):
            assert fit[key] == metadata[key]
        assert fit["source_seed_excluded_from_original_and_added_rows"]
    return len(rows)*len(fitting)


def verify_geometry(bank, shifted, centers, members, base_covariance, covariance):
    terms = v84.scores(shifted, covariance)
    maximum_condition, maximum_difference = 0., 0.
    for c, mode in enumerate(v84.MODES):
        for row in range(len(centers)):
            vectors = members[:, row, c]-centers[row, c]
            added = sum(np.outer(v, v) for v in vectors)/len(vectors)
            np.testing.assert_allclose(covariance[row, c], base_covariance[row, c]+added, rtol=1e-10, atol=1e-10)
            physical = covariance[row, c]*bank.target_scale[:, None]*bank.target_scale[None, :]
            maximum_condition = max(maximum_condition, float(np.linalg.cond(physical)))
            distribution = Covariance.from_cholesky(np.linalg.cholesky(physical))
            for truth in range(len(v84.MODES)):
                target = shifted[row, truth, c]*bank.target_scale
                reference = multivariate_normal.logpdf(target, cov=distribution)
                score = terms[0][row, truth, c]-.5*len(target)*np.log(2*np.pi)-np.log(bank.target_scale).sum()
                maximum_difference = max(maximum_difference, abs(float(score-reference)))
                np.testing.assert_allclose(score, reference, rtol=1e-10, atol=1e-8)
    return dict(gaussian_checks=len(centers)*len(v84.MODES)**2,
        maximum_physical_covariance_condition=maximum_condition, maximum_physical_score_difference=maximum_difference)


def run(args):
    sys.path.insert(0, str(args.v83_dir))
    from scripts import v83_cross_seed_covariance as v83

    torch.set_num_threads(2)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    source = args.v83_dir/"run_checkpoints/cross-seed-covariance"
    selection = json.loads((source/"nested_selection.json").read_text())
    previous = json.loads((args.v92_dir/"run_results/pilot/error_support.json").read_text())
    assert selection["revision"] == v83.REVISION and previous["revision"] == v92.REVISION
    assert previous["passed"] and not previous["smoke"] and len(previous["folds"]) == 6
    bank = torch.load(args.v76_dir/"run_checkpoints/formal/behavior_coverage_references.pt",
                      map_location="cpu", weights_only=False)["references"]["updating_mixed"].bank
    assert bank.tree.n == 23000 and tuple(bank.residuals) == v84.MODES
    groups = v83.groups_from_blocks(selection["reference_blocks"], bank.tree.n)
    weight = selection["selection"]["raw_weight"]
    assert weight == previous["raw_weight"]
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
        assert len(folds) == 1
    old_by_seed = {f["seed"]: f for f in previous["folds"]}
    weights = dict(raw=1., cross_seed=0., selected_mix=weight)
    collected = {c: [] for c in (*previous["conditions"], *CONDITIONS)}
    records, changes = [], []
    verification = dict(fold_control_comparisons=0, direct_baseline_mean_queries=0,
        direct_member_prediction_checks=0, member_seed_exclusion_checks=0, gaussian_checks=0,
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
        fitting, disagreement, terms_by_name = {}, {}, {}
        for name, w in weights.items():
            terms_by_name[name] = v84.scores(offsets, w*raw+(1-w)*cross)
        frozen_covariance = weight*raw+(1-weight)*cross
        for condition, control in CONDITIONS.items():
            mean_condition = v92.CONDITIONS[control]
            old_condition = next(c for c, m in v91.CONDITIONS.items() if m == mean_condition)
            predicted, metadata = v89.predict_heldout(banks[mean_condition], expanded_groups, seed, inputs, bank.tree.n)
            assert metadata == old_by_seed[seed]["mean_fit"][control]
            shifted = v89.coverage_offsets(offsets, baseline, predicted)
            terms_by_name[mean_condition] = v84.scores(shifted, frozen_covariance)
            raw_bank, raw_groups, mean_bank, mean_groups = v91.outer_banks(
                bank, groups, banks[mean_condition], expanded_groups, seed)
            assert seed not in raw_groups and seed not in mean_groups
            # The previous error intervention is frozen, including its server-side caches.
            error_directory = args.v92_dir/"run_checkpoints/pilot"
            assert (error_directory/f"outer_seed{seed}_{mean_condition}_added.pt").is_file()
            errors, old_errors, _ = v92.load_errors(raw_bank, raw_groups, mean_bank, mean_groups, seed,
                mean_condition, args.v91_dir/"run_checkpoints/pilot", error_directory)
            _, old_covariance, _ = v91.make_density(raw_bank, mean_bank, old_errors, weight).geometry(inputs)
            terms_by_name[old_condition] = v84.scores(shifted, np.stack([old_covariance[m] for m in v84.MODES], axis=1))
            centers, base_covariance, _ = v92.support_geometry(raw_bank, mean_bank, errors, inputs, weight)
            fixed_centers = np.stack([centers[m] for m in v84.MODES], axis=1)
            forward = np.stack([bank.features.features(inputs)@bank.coefficients[m] for m in v84.MODES], axis=1)
            np.testing.assert_allclose(fixed_centers+forward, predicted, rtol=1e-10, atol=1e-10)
            base = np.stack([base_covariance[m] for m in v84.MODES], axis=1)
            terms_by_name[control] = v84.scores(shifted, base)
            query_rows = rows if args.smoke else np.arange(len(inputs))
            query = inputs[query_rows]
            members, member_fits = cached_members(mean_bank, mean_groups, query, raw_bank.tree.n,
                seed, mean_condition, args.checkpoint_dir)
            verification["direct_member_prediction_checks"] += verify_members(
                mean_bank, mean_groups, query, raw_bank.tree.n, members, member_fits)
            verification["member_seed_exclusion_checks"] += len(member_fits)
            assert len(member_fits) == 5 and all(f["seed"] != seed for f in member_fits)
            moment = displacement_second_moment(fixed_centers[query_rows], members)
            expanded_covariance = base[query_rows]+moment
            check_rows = np.arange(len(rows)) if args.smoke else rows
            check = verify_geometry(bank, shifted[query_rows][check_rows], fixed_centers[query_rows][check_rows],
                members[:, check_rows], base[query_rows][check_rows], expanded_covariance[check_rows])
            verification["gaussian_checks"] += check["gaussian_checks"]
            for name in ("maximum_physical_covariance_condition", "maximum_physical_score_difference"):
                verification[name] = max(verification[name], check[name])
            terms_by_name[condition] = v84.scores(shifted[query_rows], expanded_covariance)
            fitting[condition] = metadata
            # Retain the bias term: the second moment is around the fixed outer mean.
            member_average = members.mean(axis=0)
            centered = displacement_second_moment(member_average, members)
            bias = member_average-fixed_centers[query_rows]
            bias_moment = np.einsum("nmd,nme->nmde", bias, bias)
            np.testing.assert_allclose(moment, centered+bias_moment, rtol=1e-10, atol=1e-10)
            disagreement[condition] = dict(member_fits=member_fits,
                per_mode={m: dict(base_trace=float(np.trace(base[query_rows, c], axis1=1, axis2=2).mean()),
                    added_trace=float(np.trace(moment[:, c], axis1=1, axis2=2).mean()),
                    centered_member_trace=float(np.trace(centered[:, c], axis1=1, axis2=2).mean()),
                    member_bias_trace=float(np.trace(bias_moment[:, c], axis1=1, axis2=2).mean()),
                    final_trace=float(np.trace(expanded_covariance[:, c], axis1=1, axis2=2).mean()),
                    mean_logdet_change=float((terms_by_name[condition][2][:, c]-terms_by_name[control][2][query_rows, c]).mean()))
                    for c, m in enumerate(v84.MODES)})
            stats = v84.summarize(*terms_by_name[condition])
            for compared in (control, mean_condition):
                before = v84.summarize(*(a[query_rows] for a in terms_by_name[compared]))
                for mode in v84.MODES:
                    new, old = stats["per_mode"][mode], before["per_mode"][mode]
                    changes.append(dict(seed=seed, mode=mode, condition=condition, control=compared,
                        classification_nll_delta=new["classification_nll"]-old["classification_nll"],
                        accuracy_delta=new["accuracy"]-old["accuracy"]))
            print(f"[FOLD] seed{seed} {condition}: NLL={stats['classification_nll']:.8f}, accuracy={stats['accuracy']:.6f}", flush=True)
        conditions = {}
        for name, terms in terms_by_name.items():
            stats = v84.summarize(*terms)
            if name in previous["conditions"]:
                v89.prior.verify_control(stats, old_by_seed[seed]["conditions"][name])
                verification["fold_control_comparisons"] += 1
            collected[name].append(terms)
            conditions[name] = v89.prior.compact(stats)
        records.append(dict(values[3], mean_fit=fitting, disagreement=disagreement, conditions=conditions))
    aggregate = {c: dict(raw_weight=weights.get(c, weight), **v89.prior.compact(v84.summarize(
        *(np.concatenate([v[i] for v in terms]) for i in range(3))))) for c, terms in collected.items()}
    if not args.smoke:
        for name in previous["conditions"]:
            v89.prior.verify_control(aggregate[name], previous["conditions"][name])
    result = dict(revision=REVISION, source_revision=v92.REVISION, smoke=args.smoke, passed=True,
        modes=list(v84.MODES), raw_weight=weight, conditions=aggregate, folds=records,
        new_conditions=CONDITIONS, fold_mode_changes=changes, verification=dict(verification,
            all_nine_controls_match_v92=True, conditional_means_unchanged=True, all_error_arrays_reused=True,
            raw_covariance_and_weight_unchanged=True, outer_seed_excluded_before_member_fits=True,
            member_exclusion_includes_original_and_added_rows=True, covariance_matches_fixed_center_second_moment=True,
            physical_gaussian_scores_match=True),
        mean_fitting_rows=25200, raw_reference_rows=23000, error_reference_rows=25200,
        members_per_outer_fold=5, disagreement_coefficient=1.,
        covariance_formula="v92 covariance + equal-source-seed second moment of leave-source-seed-out means around the unchanged outer mean",
        disagreement_units="normalized delta-state target coordinates",
        queries_per_new_condition=len(records)*(8 if args.smoke else 512),
        true_mode_labels_per_new_condition=len(records)*(8 if args.smoke else 512)*5,
        weighting="equal seed / true mode / row", prior="uniform, instantaneous posterior",
        error_components_refitted=False, conditional_means_changed=False, query_targets_used_for_covariance=False,
        weight_refitted=False, global_forward_refitted=False, scaling_refitted=False, temporal_rules_changed=False,
        evaluation_paths_loaded=False, normal_null_loaded=False, readout_controls_actor=False,
        environment_steps=0, optimizer_updates=0, additional_counterfactual_steps=0, independent_confirmation=False,
        decision="native-correspondence-only" if args.smoke else "reference-only-mean-disagreement-pending-analysis",
        limitations=["Reused development folds and a globally selected frozen v83 weight, not independent confirmation.",
            "Seed blocks have unequal sizes; deletion sensitivity is not a calibrated predictive variance or unbiased jackknife estimate.",
            "The added second moment can double-count error or worsen mode discrimination; its coefficient is fixed, not tuned.",
            "Smoke scores new conditions on eight inputs, controls on all 512; only paired subset deltas are comparable.",
            "Instantaneous reference classification is not recursive trajectory or RL return evidence."])
    path = args.result_dir/("native_smoke.json" if args.smoke else "mean_disagreement.json")
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] {REVISION}: {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("v92-dir", "v91-dir", "v83-dir", "v76-dir", "checkpoint-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
