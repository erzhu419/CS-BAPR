from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import analyze_general_ood_v92_error_support as worker
from scripts.v83_cross_seed_covariance import LocalLinearMean, error_covariance, subset_bank
from test_general_ood_v68 import trace
from test_general_ood_v78 import banks
from test_general_ood_v89 import sources
from test_general_ood_v91 import expanded


def test_added_errors_exclude_all_own_original_and_added_labels(expanded):
    base, _, additions, groups = expanded
    bank = additions[worker.v89.CONDITIONS[0]]
    expected, fits = worker.added_errors(bank, groups, base.tree.n)
    changed = deepcopy(bank)
    seed = worker.v89.SEEDS[0]
    for mode in worker.v84.MODES:
        changed.residuals[mode][groups == seed] += 1000.
    actual, after = worker.added_errors(changed, groups, base.tree.n)
    rows = groups[base.tree.n:] == seed
    for mode in worker.v84.MODES:
        np.testing.assert_allclose(actual[mode][rows]-1000., expected[mode][rows], rtol=1e-10, atol=1e-10)
    assert fits == after and sum(f["error_rows"] for f in fits) == 2200
    assert all(f["excluded_mean_rows"] == 1180 and f["mean_base_training_rows"] == 160 and
               f["mean_added_training_rows"] == 1100 for f in fits)


def test_added_errors_match_direct_whole_seed_excluded_predictions(expanded):
    base, _, additions, groups = expanded
    for bank in additions.values():
        errors, _ = worker.added_errors(bank, groups, base.tree.n)
        for seed in worker.v89.SEEDS:
            rows = np.flatnonzero((groups == seed) & (np.arange(bank.tree.n) >= base.tree.n))[:4]
            inputs = bank.tree.data[rows]*bank.features.input_scale+bank.features.input_mean
            predicted = LocalLinearMean(subset_bank(bank, groups != seed)).predict(inputs)
            for mode in worker.v84.MODES:
                np.testing.assert_array_equal(errors[mode][rows-base.tree.n], bank.residuals[mode][rows]-predicted[mode])


def test_outer_seed_cannot_enter_added_errors_or_geometry(expanded):
    base, raw_groups, additions, groups = expanded
    seed = worker.v89.SEEDS[0]
    bank = additions[worker.v89.CONDITIONS[0]]
    changed = deepcopy(bank)
    for mode in worker.v84.MODES:
        changed.residuals[mode][groups == seed] += 500.
    results = []
    for mean in (bank, changed):
        raw, rg, reduced, mg = worker.v91.outer_banks(base, raw_groups, mean, groups, seed)
        assert seed not in rg and seed not in mg
        old, _ = worker.v91.aligned_errors(raw, rg, reduced, mg)
        extra, _ = worker.added_errors(reduced, mg, raw.tree.n)
        errors = {m: np.concatenate((old[m], extra[m])) for m in worker.v84.MODES}
        inputs = raw.features.inputs(trace(993, "heavy", 2))
        results.append((errors, worker.support_geometry(raw, reduced, errors, inputs, .24619454443573643)))
    for mode in worker.v84.MODES:
        np.testing.assert_array_equal(results[0][0][mode], results[1][0][mode])
        np.testing.assert_array_equal(results[0][1][0][mode], results[1][1][0][mode])
        np.testing.assert_array_equal(results[0][1][1][mode], results[1][1][1][mode])


def test_error_neighbors_expand_but_raw_component_and_mean_stay_fixed(expanded):
    raw, _, additions, _ = expanded
    weight = .24619454443573643
    for mean in additions.values():
        rows = np.arange(raw.tree.n, raw.tree.n+4)
        inputs = mean.tree.data[rows]*mean.features.input_scale+mean.features.input_mean
        errors = {m: np.full_like(mean.residuals[m], .1) for m in worker.v84.MODES}
        for mode in worker.v84.MODES:
            errors[mode][raw.tree.n:] = 5.
        centers, covariance, diagnostic = worker.support_geometry(raw, mean, errors, inputs, weight)
        direct = LocalLinearMean(mean).predict(inputs)
        _, indices = mean.tree.query((inputs-mean.features.input_mean)/mean.features.input_scale, k=64)
        assert diagnostic["error_added_neighbor_fraction"] > 0
        raw_covariance = np.stack([raw.geometry(inputs, m)[1] for m in worker.v84.MODES], axis=1)
        expected = np.stack([direct[m] for m in worker.v84.MODES], axis=1)
        shifted = np.full((4, 5, 5, len(raw.target_scale)), .1)
        for c, mode in enumerate(worker.v84.MODES):
            fitted = np.asarray([error_covariance(e)[0] for e in errors[mode][indices]])
            np.testing.assert_array_equal(centers[mode], direct[mode])
            np.testing.assert_array_equal(covariance[mode], weight*raw_covariance[:, c]+(1-weight)*fitted)
            old = error_covariance(errors[mode][:64])[0]
            assert not np.allclose(fitted, old)
        result = worker.verify_geometry(raw, mean, errors, inputs, weight, expected, shifted,
                                        raw_covariance, centers, covariance)
        assert result["gaussian_checks"] == 100 and result["maximum_physical_score_difference"] < 1e-8


def test_original_errors_are_reused_and_added_cache_avoids_refitting(expanded, tmp_path, monkeypatch):
    raw, groups, additions, mean_groups = expanded
    mean = additions[worker.v89.CONDITIONS[0]]
    source, target = tmp_path/"v91", tmp_path/"v92"
    old, old_fits = worker.v91.cached_errors(raw, groups, mean, mean_groups, 999, "prefix_mean_mix", source)
    expected, reused, fits = worker.load_errors(raw, groups, mean, mean_groups, 999, "prefix_mean_mix", source, target)
    for mode in worker.v84.MODES:
        np.testing.assert_array_equal(reused[mode], old[mode])
        np.testing.assert_array_equal(expected[mode][:raw.tree.n], old[mode])
    assert fits[:len(old_fits)] == old_fits
    assert sum(f["error_rows"] for f in fits) == mean.tree.n
    assert worker.verify_error_rows(mean, mean_groups, expected, raw.tree.n) == 5
    def no_refit(*args):
        pytest.fail("Existing v91 and added-row errors must be reused")
    monkeypatch.setattr(worker, "added_errors", no_refit)
    monkeypatch.setattr(worker.v91, "aligned_errors", no_refit)
    actual, _, after = worker.load_errors(raw, groups, mean, mean_groups, 999, "prefix_mean_mix", source, target)
    for mode in worker.v84.MODES:
        np.testing.assert_array_equal(actual[mode], expected[mode])
    assert after == fits


def test_missing_v91_cache_does_not_silently_refit_it(expanded, tmp_path, monkeypatch):
    raw, groups, additions, mean_groups = expanded
    def no_refit(*args):
        pytest.fail("A missing source cache must not trigger a different v91 run")
    monkeypatch.setattr(worker.v91, "aligned_errors", no_refit)
    with pytest.raises(FileNotFoundError):
        worker.load_errors(raw, groups, additions[worker.v89.CONDITIONS[0]], mean_groups,
                           999, "prefix_mean_mix", tmp_path/"v91", tmp_path/"v92")


def test_protocol_changes_only_error_support_and_keeps_matched_budget():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root/"experiments/general_ood_v92_error_support_20261009/dispatch_manifest.json").read_text())
    assert manifest["raw_reference_rows"] == 23000 and manifest["error_reference_rows"] == manifest["mean_fitting_rows"] == 25200
    assert manifest["additional_inputs_per_condition"] == 2200 and manifest["additional_labels_per_condition"] == 11000
    assert manifest["raw_weight"] == .24619454443573643 and len(manifest["controls"]) == 7
    assert manifest["reference_folds"] == 6 and manifest["validation_rows_per_fold"] == 512
    assert manifest["outer_seed_excluded_before_inner_fits"] and manifest["inner_seed_exclusion_includes_additions"]
    assert manifest["error_covariance_support_changed"] and manifest["original_error_rows_reused_from_v91"]
    assert not manifest["raw_covariance_component_changed"] and not manifest["conditional_means_changed_vs_v89"]
    assert not manifest["weight_refitted"] and not manifest["evaluation_paths_loaded"] and not manifest["normal_null_loaded"]
    assert not manifest["readout_controls_actor"] and manifest["environment_steps"] == 0
