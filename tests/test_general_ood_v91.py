from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import analyze_general_ood_v91_matched_uncertainty as worker
from scripts.v83_cross_seed_covariance import LocalLinearMean, cross_seed_errors
from test_general_ood_v68 import trace
from test_general_ood_v78 import banks
from test_general_ood_v89 import sources


@pytest.fixture(scope="module")
def expanded(banks, sources):
    bank = banks["updating"]
    groups = np.repeat([135050, *worker.v89.SEEDS], 80)
    additions, mean_groups = worker.v89.expand_banks(bank, groups, sources)
    return bank, groups, additions, mean_groups


def test_without_additions_errors_match_existing_cross_seed_estimator(banks):
    bank, groups = banks["updating"], np.repeat([1, 2, 3], 80)
    errors, fitting = worker.aligned_errors(bank, groups, bank, groups)
    expected = cross_seed_errors(bank, groups)
    for mode in worker.v84.MODES:
        np.testing.assert_array_equal(errors[mode], expected[mode])
    assert all(f["mean_added_training_rows"] == 0 for f in fitting)


def test_inner_exclusion_removes_own_original_and_added_labels(expanded):
    bank, groups, additions, mean_groups = expanded
    seed = worker.v89.SEEDS[0]
    rows = np.flatnonzero(groups == seed)[:8]
    mean_bank = additions[worker.v89.CONDITIONS[0]]
    errors, fitting = worker.aligned_errors(bank, groups, mean_bank, mean_groups, rows)
    changed = deepcopy(mean_bank)
    for mode in worker.v84.MODES:
        changed.residuals[mode][mean_groups == seed] += 1000.
    actual, after = worker.aligned_errors(bank, groups, changed, mean_groups, rows)
    for mode in worker.v84.MODES:
        np.testing.assert_array_equal(actual[mode][rows], errors[mode][rows])
    assert fitting == after and fitting[0]["excluded_mean_rows"] == 1180
    assert fitting[0]["mean_base_training_rows"] == 160 and fitting[0]["mean_added_training_rows"] == 1100


def test_outer_labels_cannot_enter_inner_errors_or_covariance(expanded):
    bank, groups, additions, mean_groups = expanded
    seed = worker.v89.SEEDS[0]
    mean_bank = additions[worker.v89.CONDITIONS[0]]
    original = worker.outer_banks(bank, groups, mean_bank, mean_groups, seed)
    changed_base, changed_mean = deepcopy(bank), deepcopy(mean_bank)
    for mode in worker.v84.MODES:
        changed_base.residuals[mode][groups == seed] += 500.
        changed_mean.residuals[mode][mean_groups == seed] += 500.
    actual = worker.outer_banks(changed_base, groups, changed_mean, mean_groups, seed)
    query = bank.features.inputs(trace(999, "heavy", 3))
    computed = []
    for covariance_bank, covariance_groups, predictor, predictor_groups in [original, actual]:
        errors, fitting = worker.aligned_errors(covariance_bank, covariance_groups, predictor, predictor_groups)
        density = worker.make_density(covariance_bank, predictor, errors, .3)
        computed.append((errors, fitting, density.geometry(query)))
        assert seed not in covariance_groups and seed not in predictor_groups
    for mode in worker.v84.MODES:
        np.testing.assert_array_equal(computed[0][0][mode], computed[1][0][mode])
        np.testing.assert_array_equal(computed[0][2][0][mode], computed[1][2][0][mode])
        np.testing.assert_array_equal(computed[0][2][1][mode], computed[1][2][1][mode])
    assert computed[0][1] == computed[1][1]


def test_both_expanded_means_use_original_support_covariances_and_physical_gaussians(expanded):
    bank, groups, additions, mean_groups = expanded
    query = trace(997, "slippery", 4)
    inputs, design, targets = bank.features.inputs(query), bank.design(query), bank.targets(query)
    for mean_bank in additions.values():
        errors, _ = worker.aligned_errors(bank, groups, mean_bank, mean_groups)
        density = worker.make_density(bank, mean_bank, errors, .24619454443573643)
        centers, covariance, _ = density.geometry(inputs)
        direct = LocalLinearMean(mean_bank).predict(inputs)
        expected = np.stack([direct[m] for m in worker.v84.MODES], axis=1)
        predictions = np.stack([design@bank.coefficients[m]+direct[m] for m in worker.v84.MODES], axis=1)
        shifted = np.broadcast_to(targets[:, None, None, :]-predictions[:, None, :, :], (4, 5, 5, 11))
        raw = np.stack([bank.geometry(inputs, m)[1] for m in worker.v84.MODES], axis=1)
        assert density.bank is bank and density.correction.bank is mean_bank
        verification = worker.verify_geometry(bank, density, inputs, expected, shifted, raw, centers, covariance)
        assert verification["gaussian_checks"] == 100
        assert verification["maximum_physical_score_difference"] < 1e-8


def test_physical_gaussian_check_handles_anisotropic_supported_units(expanded):
    original, _, additions, _ = expanded
    bank = deepcopy(original)
    bank.target_scale *= np.geomspace(1e-3, 10., len(bank.target_scale))
    inputs = bank.features.inputs(trace(995, "heavy", 2))
    mean_bank = additions[worker.v89.CONDITIONS[0]]
    errors = {m: bank.residuals[m].copy() for m in worker.v84.MODES}
    density = worker.make_density(bank, mean_bank, errors, .3)
    centers, covariance, _ = density.geometry(inputs)
    expected = np.stack([centers[m] for m in worker.v84.MODES], axis=1)
    shifted = np.full((2, 5, 5, len(bank.target_scale)), .1)
    raw = np.stack([bank.geometry(inputs, m)[1] for m in worker.v84.MODES], axis=1)
    result = worker.verify_geometry(bank, density, inputs, expected, shifted, raw, centers, covariance)
    assert result["gaussian_checks"] == 50
    assert result["maximum_physical_covariance_condition"] > 1e8
    assert result["maximum_physical_score_difference"] < 1e-8


def test_partial_smoke_computes_every_used_neighbor_and_each_inner_seed(expanded):
    bank, groups, additions, mean_groups = expanded
    predictor = additions[worker.v89.CONDITIONS[0]]
    inputs = bank.features.inputs(trace(998, "heavy", 2))
    rows = worker.smoke_rows(bank, groups, inputs)
    assert set(groups[rows]) == set(groups)
    sparse, fitting = worker.aligned_errors(bank, groups, predictor, mean_groups, rows)
    full, _ = worker.aligned_errors(bank, groups, predictor, mean_groups)
    assert sum(f["error_rows"] for f in fitting) == len(rows)
    sparse_geometry = worker.make_density(bank, predictor, sparse, .3).geometry(inputs)
    full_geometry = worker.make_density(bank, predictor, full, .3).geometry(inputs)
    for mode in worker.v84.MODES:
        np.testing.assert_array_equal(sparse[mode][rows], full[mode][rows])
        np.testing.assert_array_equal(sparse_geometry[1][mode], full_geometry[1][mode])


def test_full_error_cache_reuses_completed_inner_fits(expanded, tmp_path, monkeypatch):
    bank, groups, additions, mean_groups = expanded
    mean_bank = additions[worker.v89.CONDITIONS[0]]
    expected, fitting = worker.cached_errors(bank, groups, mean_bank, mean_groups, 999, "prefix_mean_mix", tmp_path)
    def no_refit(*args):
        pytest.fail("Completed full error bank must be reused")
    monkeypatch.setattr(worker, "aligned_errors", no_refit)
    actual, after = worker.cached_errors(bank, groups, mean_bank, mean_groups, 999, "prefix_mean_mix", tmp_path)
    for mode in worker.v84.MODES:
        np.testing.assert_array_equal(actual[mode], expected[mode])
    assert fitting == after


def test_protocol_is_error_component_only_on_reference_folds():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root/"experiments/general_ood_v91_matched_uncertainty_20261009/dispatch_manifest.json").read_text())
    assert manifest["covariance_reference_rows"] == 23000 and manifest["mean_fitting_rows"] == 25200
    assert manifest["raw_weight"] == .24619454443573643 and manifest["reference_folds"] == 6
    assert manifest["validation_rows_per_fold"] == 512 and manifest["smoke_queries"] == 8
    assert manifest["outer_seed_excluded_before_inner_fits"] and manifest["inner_seed_exclusion_includes_additions"]
    assert not manifest["covariance_support_changed"] and not manifest["weight_refitted"]
    assert not manifest["evaluation_paths_loaded"] and not manifest["normal_null_loaded"]
    assert not manifest["readout_controls_actor"] and manifest["environment_steps"] == 0
