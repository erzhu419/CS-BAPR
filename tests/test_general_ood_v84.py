import numpy as np
import pytest
from scipy.stats import multivariate_normal

from scripts import analyze_general_ood_v84_reference_discrimination as worker
from scripts.v77_mean_covariance import MODES
from scripts import v83_cross_seed_covariance as v83
from scripts.v78_local_linear_mean import LocalLinearMean
from test_general_ood_v78 import banks


def test_cross_mode_offsets_reconstruct_heldout_predictions(banks):
    bank, groups = banks['updating'], np.repeat([1, 2, 3], 80)
    values = v83.validation_fold(bank, groups, 1, 8)
    inputs, predictions, offsets, cross, raw = worker.reconstruct_fold(bank, groups, values)
    reduced = v83.subset_bank(bank, groups != 1)
    design = bank.features.features(inputs)
    predicted_residuals = LocalLinearMean(reduced).predict(inputs)
    selected = np.flatnonzero(groups == 1)[np.linspace(0, 79, 8, dtype=int)]
    for t, mode in enumerate(MODES):
        targets = design@bank.coefficients[mode]+bank.residuals[mode][selected]
        for c, candidate in enumerate(MODES):
            expected = design@bank.coefficients[candidate]+predicted_residuals[candidate]
            np.testing.assert_allclose(predictions[:, c], expected, rtol=1e-11, atol=1e-11)
            np.testing.assert_allclose(offsets[:, t, c], targets-expected, rtol=1e-11, atol=1e-11)
    assert cross.shape == raw.shape == (8, 5, 11, 11)


def test_candidate_specific_covariance_and_logdet_match_gaussian_density():
    rng = np.random.default_rng(84)
    offsets = rng.normal(size=(3, 5, 5, 2))
    matrices = rng.normal(size=(3, 5, 2, 2))
    covariance = matrices@matrices.transpose(0, 1, 3, 2)+.1*np.eye(2)
    ll, energy, logdet = worker.scores(offsets, covariance)
    for n in range(3):
        for t in range(5):
            for c in range(5):
                expected = multivariate_normal.logpdf(offsets[n, t, c], cov=covariance[n, c])+np.log(2*np.pi)
                assert ll[n, t, c] == pytest.approx(expected, abs=1e-10)
    stats = worker.summarize(ll, energy, logdet)
    for mode in MODES:
        for pair in stats['per_mode'][mode]['pairs'].values():
            assert pair['mean_log_density_margin'] == pytest.approx(pair['mean_energy_advantage']+pair['mean_volume_advantage'])


def test_relative_posterior_is_invariant_to_common_score_offset_and_balances_modes():
    rng = np.random.default_rng(85)
    ll = rng.normal(size=(7, 5, 5))
    energy, logdet = -2*ll, np.zeros((7, 5))
    expected = worker.summarize(ll, energy, logdet)
    actual = worker.summarize(ll+rng.normal(size=(7, 5, 1)), energy, logdet)
    for key in ('accuracy', 'classification_nll', 'mean_true_minus_best_other', 'nonpositive_margin_fraction'):
        assert actual[key] == pytest.approx(expected[key])
        assert expected[key] == pytest.approx(np.mean([s[key] for s in expected['per_mode'].values()]))
    for mode in MODES:
        assert actual['per_mode'][mode]['predicted_mode_fractions'] == expected['per_mode'][mode]['predicted_mode_fractions']


def test_better_true_density_can_have_worse_discrimination():
    ll = np.full((2, 5, 5), -2.)
    ll[:, np.arange(5), np.arange(5)] = 0.
    wrong = ll+2.
    wrong[:, np.arange(5), (np.arange(5)+1)%5] = 3.
    original = worker.summarize(ll, -2*ll, np.zeros((2, 5)))
    changed = worker.summarize(wrong, -2*wrong, np.zeros((2, 5)))
    assert changed['true_density_nll_without_constant'] < original['true_density_nll_without_constant']
    assert original['accuracy'] == 1. and changed['accuracy'] == 0.
    assert changed['classification_nll'] > original['classification_nll']
