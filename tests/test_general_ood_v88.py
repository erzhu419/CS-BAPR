from copy import deepcopy

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from scripts import analyze_general_ood_v88_mean_bias as worker
from scripts import v83_cross_seed_covariance as v83
from test_general_ood_v68 import trace
from test_general_ood_v78 import banks


def test_bias_is_equal_neighbor_average_in_candidate_mode_order(banks):
    bank = banks['updating']
    inputs = bank.features.inputs(trace(90, 'heavy', 4))
    errors = {m: bank.residuals[m].copy()+c for c, m in enumerate(worker.v84.MODES)}
    before = {m: e.copy() for m, e in errors.items()}
    actual = worker.neighborhood_bias(bank, errors, inputs)
    _, indices = bank.tree.query((inputs-bank.features.input_mean)/bank.features.input_scale, k=64)
    for c, mode in enumerate(worker.v84.MODES):
        np.testing.assert_allclose(actual[:, c], np.asarray([errors[mode][n].mean(axis=0) for n in indices]))
        np.testing.assert_array_equal(errors[mode], before[mode])


def test_bias_correction_adds_error_to_prediction_without_changing_covariance():
    rng = np.random.default_rng(91)
    offsets = rng.normal(size=(3, 5, 5, 2))
    bias = rng.normal(size=(3, 5, 2))
    before = offsets.copy()
    shifted = worker.corrected_offsets(offsets, bias)
    a = rng.normal(size=(3, 5, 2, 2))
    covariance = a@a.transpose(0, 1, 3, 2)+.1*np.eye(2)
    covariance_before = covariance.copy()
    ll, _, logdet = worker.v84.scores(shifted, covariance)
    old_logdet = worker.v84.scores(offsets, covariance)[2]
    np.testing.assert_array_equal(logdet, old_logdet)
    for row in range(3):
        for truth in range(5):
            expected = [multivariate_normal.logpdf(offsets[row, truth, c], mean=bias[row, c], cov=covariance[row, c]) for c in range(5)]
            np.testing.assert_allclose(ll[row, truth]-np.log(2*np.pi), expected)
    np.testing.assert_array_equal(offsets, before)
    np.testing.assert_array_equal(covariance, covariance_before)
    np.testing.assert_array_equal(worker.corrected_offsets(offsets, np.zeros_like(bias)), offsets)


def test_outer_seed_labels_do_not_enter_bias_and_all_inner_fits_exclude_source(banks, monkeypatch):
    bank = banks['updating']
    groups = np.repeat([1, 2, 3], 80)
    inputs = bank.tree.data[:4]*bank.features.input_scale+bank.features.input_mean
    original = v83.LocalLinearMean
    seen = []

    class AuditedMean:
        def __init__(self, reduced):
            self.model = original(reduced)
            self.fit_seeds = set(groups[bank.tree.query(reduced.tree.data, k=1)[1]])

        def predict(self, query):
            normalized = (query-bank.features.input_mean)/bank.features.input_scale
            source_seeds = set(groups[bank.tree.query(normalized, k=1)[1]])
            assert 1 not in self.fit_seeds and not self.fit_seeds.intersection(source_seeds)
            seen.append(source_seeds)
            return self.model.predict(query)

    monkeypatch.setattr(v83, 'LocalLinearMean', AuditedMean)
    bias, metadata = worker.fit_bias(bank, groups, 1, inputs)
    changed = deepcopy(bank)
    for mode in worker.v84.MODES:
        changed.residuals[mode][groups == 1] += 1000
    actual, actual_metadata = worker.fit_bias(changed, groups, 1, inputs)
    np.testing.assert_array_equal(actual, bias)
    assert actual_metadata == metadata and metadata['inner_seeds'] == [2, 3]
    assert seen == [{2}, {3}, {2}, {3}]


def test_query_next_state_and_future_inputs_do_not_enter_bias(banks):
    bank = banks['updating']
    query = trace(92, 'slippery', 6)
    errors = bank.residuals
    before = worker.neighborhood_bias(bank, errors, bank.features.inputs(query))
    changed = deepcopy(query)
    changed['next_states'] += 1000
    changed['states'][3:] += 100
    actual = worker.neighborhood_bias(bank, errors, bank.features.inputs(changed))
    np.testing.assert_array_equal(actual[:3], before[:3])


def test_completed_bias_cache_avoids_expensive_inner_refits(banks, tmp_path, monkeypatch):
    bank = banks['updating']
    groups = np.repeat([1, 2, 3], 80)
    inputs = bank.tree.data[:3]*bank.features.input_scale+bank.features.input_mean
    path = tmp_path/'outer_seed1_bias.pt'
    expected, metadata = worker.freeze_bias(path, bank, groups, 1, inputs, .24619454443573643)

    def unexpected_fit(*args):
        pytest.fail('Completed native smoke bias must be reused by the formal pilot')

    monkeypatch.setattr(worker, 'fit_bias', unexpected_fit)
    actual, actual_metadata = worker.freeze_bias(path, bank, groups, 1, inputs, .24619454443573643)
    np.testing.assert_array_equal(actual, expected)
    assert actual_metadata == metadata
