from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import logsumexp

from scripts import analyze_general_ood_v113_discriminative_evidence as worker
from scripts import probe_general_ood_v76_behavior_coverage as trace_source
from scripts.v83_cross_seed_covariance import LocalLinearMean, subset_bank
from test_general_ood_v58 import shared_start
from test_general_ood_v68 import trace
from test_general_ood_v78 import banks
from test_general_ood_v89 import sources
from test_general_ood_v91 import expanded
from test_general_ood_v94 import fitted
from test_general_ood_v98 import expanded_coverage, fitted_coverage
from test_general_ood_v106 import family


def test_paired_features_use_observed_transition_and_same_five_predictions():
    rng = np.random.RandomState(3)
    targets, predicted = rng.normal(size=(8, 5, 11)), rng.normal(size=(8, 5, 11))
    x, y = worker.paired_features(targets, predicted)
    assert x.shape == (40, 110)
    np.testing.assert_array_equal(y, np.tile(np.arange(5), 8))
    for row in range(8):
        for mode in range(5):
            residual = (targets[row, mode]-predicted[row]).reshape(-1)
            np.testing.assert_array_equal(x[5*row+mode], np.r_[residual, residual**2])


def test_seed_weighting_balances_sources_and_all_five_modes():
    groups = np.array([1]*20+[2]*3+[3]*7)
    x = np.random.RandomState(4).normal(size=(150, 110))
    _, metadata = worker.fit_classifier(x, np.tile(np.arange(5), 30), groups)
    assert metadata["weighted_mode_fractions"] == pytest.approx([.2]*5)
    assert list(metadata["weighted_seed_fractions"].values()) == pytest.approx([1/3]*3)
    assert metadata["feature_dimension"] == 110


def test_outer_validation_seeds_excluded_before_training_features_and_scaling(family):
    before = deepcopy(family.mean.residuals)
    validation, model = worker.heldout_validation(family, smoke=True)
    assert validation["validation_source_seeds"] == [135230]
    assert 135230 not in validation["training"]["source_seeds"]
    assert validation["heldout_seeds_excluded_from_classifier_scaler_and_all_local_mean_fits"]
    changed = deepcopy(family)
    for m in worker.MODES:
        changed.mean.residuals[m][changed.mean_groups == 135230] += 10
    other, other_model = worker.heldout_validation(changed, smoke=True)
    np.testing.assert_array_equal(model[0].mean_, other_model[0].mean_)
    np.testing.assert_array_equal(model[1].coef_, other_model[1].coef_)
    assert validation["aggregate"] != other["aggregate"]
    for m in worker.MODES:
        np.testing.assert_array_equal(family.mean.residuals[m], before[m])


def test_heldout_reference_reconstruction_and_real_transition_inference(family):
    bank, groups = family.mean, family.mean_groups
    rows = worker.selected_rows(groups, True)
    targets = worker.reference_targets(bank, rows)
    predicted = np.empty_like(targets)
    for seed in np.unique(groups):
        keep = groups[rows] == seed
        selected = rows[keep]
        errors, _ = worker.v98.heldout_errors(bank, groups, seed, selected)
        predicted[keep] = targets[keep]-np.stack([errors[m] for m in worker.MODES], axis=1)
        inputs = bank.tree.data[selected]*bank.features.input_scale+bank.features.input_mean
        direct = LocalLinearMean(subset_bank(bank, groups != seed)).predict(inputs)
        design = bank.features.features(inputs)
        np.testing.assert_allclose(predicted[keep], np.stack([design@bank.coefficients[m]+direct[m] for m in worker.MODES], axis=1))
    x, y = worker.paired_features(targets, predicted)
    fitted, _ = worker.fit_classifier(x, y, groups[rows])
    query = trace(135252, "compound", 6)
    inputs = family.bank.features.inputs(query)
    correction = family.correction.predict(inputs)
    means = np.stack([family.bank.design(query)@family.bank.coefficients[m]+correction[m] for m in worker.MODES], axis=1)
    observed = family.bank.targets(query)
    scores = worker.scores(fitted, observed, means)
    for values in scores.values():
        np.testing.assert_allclose(logsumexp(values, axis=1), 0., atol=1e-12)
    altered = observed.copy(); altered[3:] += 100
    changed = worker.scores(fitted, altered, means)
    for a in worker.ARMS:
        np.testing.assert_array_equal(scores[a][:3], changed[a][:3])
        assert not np.allclose(scores[a][3:], changed[a][3:])
        assert np.isfinite(changed[a]).all()
        np.testing.assert_allclose(logsumexp(changed[a], axis=1), 0., atol=1e-12)
    stats = worker.statistics(scores["quadratic_logistic"], 4)
    assert stats["classification_nll"] >= 0
    assert 0 <= stats["mean_true_mode_probability"] <= 1


def test_native_population_retains_both_historical_real_only_windows():
    root = Path(__file__).resolve().parents[1]
    previous = json.loads((root/"results/general_ood_v112_matched_errors/formal/matched_errors.json").read_text())
    smoke = [r for r in previous["records"] if (r["seed"], r["mode"], r["updating"], r["window"]) in worker.SMOKE_WINDOWS]
    assert len(smoke) == 6 and sum(r["transitions"] for r in smoke) == 328
    assert len(previous["records"]) == 76 and sum(not r["oracle_available"] for r in previous["records"]) == 2
    for record in smoke:
        cell = worker.trace_cell(record)
        if record["origin"] == "v110":
            assert cell["role"] == "confirmation"
            assert trace_source.trace_path(Path("/native"), cell).name.startswith("confirmation_seed")
