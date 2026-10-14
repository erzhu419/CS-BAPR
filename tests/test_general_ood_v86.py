from copy import deepcopy

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from scripts import analyze_general_ood_v86_directional_errors as worker
from scripts import analyze_general_ood_v83_cross_seed_covariance as prior
from scripts.v78_local_linear_mean import LocalLinearMean, correction_terms
from scripts.v83_cross_seed_covariance import cross_seed_errors
from test_general_ood_v68 import trace
from test_general_ood_v78 import banks


def test_margin_components_match_candidate_specific_gaussians():
    rng = np.random.default_rng(86)
    error, delta = rng.normal(size=(9, 3)), rng.normal(size=(9, 3))
    a, b = rng.normal(size=(9, 3, 3)), rng.normal(size=(9, 3, 3))
    true, other = a@a.transpose(0, 2, 1)+.1*np.eye(3), b@b.transpose(0, 2, 1)+.1*np.eye(3)
    reference = rng.normal(size=(9, 64, 3))
    result = worker.pair_diagnostics(error, delta, true, other, reference)
    expected = [multivariate_normal.logpdf(e, cov=t)-multivariate_normal.logpdf(e+d, cov=o)
                for e, d, t, o in zip(error, delta, true, other)]
    assert result['mean_log_density_margin'] == pytest.approx(np.mean(expected))
    assert result['mean_log_density_margin'] == pytest.approx(result['mean_common_covariance_margin']+
        result['mean_covariance_shape_adjustment']+result['mean_covariance_volume_advantage'])


def test_error_toward_wrong_mean_crosses_common_covariance_boundary():
    error, delta = np.array([[-1.5, 0.]]), np.array([[2., 0.]])
    covariance = np.eye(2)[None]
    result = worker.pair_diagnostics(error, delta, covariance, covariance, np.zeros((1, 64, 2)), delta)
    assert result['query_toward_other_mean'] == pytest.approx(1.5)
    assert result['mean_common_covariance_margin'] == pytest.approx(-1.)
    assert result['common_covariance_wrong_fraction'] == 1.
    assert result['reference_common_covariance_wrong_fraction'] == 0.
    assert result['query_exceeds_reference95_fraction'] == 1.
    assert result['mean_predicted_oracle_direction_cosine'] == pytest.approx(1.)


def test_covariance_volume_can_reverse_correct_energy_ranking():
    result = worker.pair_diagnostics(np.array([[.5, 0.]]), np.array([[.2, 0.]]),
        4*np.eye(2)[None], np.eye(2)[None], np.ones((1, 64, 2)))
    assert result['mean_common_covariance_margin'] > 0
    assert result['mean_energy_advantage'] > 0 and result['mean_log_density_margin'] < 0
    assert result['volume_flips_margin_fraction'] == 1.


def test_directional_statistics_are_invariant_to_coordinate_units():
    rng = np.random.default_rng(87)
    error, delta = rng.normal(size=(5, 3)), rng.normal(size=(5, 3))
    true, other = np.repeat(np.eye(3)[None], 5, axis=0), np.repeat((2*np.eye(3))[None], 5, axis=0)
    reference, oracle = rng.normal(size=(5, 64, 3)), rng.normal(size=(5, 3))
    original = worker.pair_diagnostics(error, delta, true, other, reference, oracle)
    scale = np.array([.1, 2., 5.])
    changed = worker.pair_diagnostics(error*scale, delta*scale, true*scale[:, None]*scale[None, :],
        other*scale[:, None]*scale[None, :], reference*scale, oracle*scale)
    for key, value in original.items():
        assert changed[key] == pytest.approx(value) if value is not None else changed[key] is None


def test_equal_predicted_means_have_no_separation_projection():
    result = worker.pair_diagnostics(np.ones((2, 2)), np.zeros((2, 2)),
        np.repeat(np.eye(2)[None], 2, axis=0), np.repeat(np.eye(2)[None], 2, axis=0), np.zeros((2, 64, 2)))
    assert result['nonzero_separation_fraction'] == 0.
    assert result['mean_common_covariance_margin'] == 0.
    assert result['query_toward_other_mean'] is None


def test_wrong_saved_comparison_blocks_interpretation():
    keys = ('mean_true_prediction_error_rms', 'mean_true_minus_best_other_log_density',
            'mean_energy_advantage', 'mean_covariance_volume_advantage', 'raw_mode_accuracy')
    result = {k: 1. for k in keys}
    pair = dict(mean_log_density_margin=1., mean_energy_advantage=2., mean_covariance_volume_advantage=-1., positive_margin_fraction=.5)
    result['pairs'] = {'heavy': pair}
    expected = dict({k: 1. for k in keys}, fixed_competitor={'heavy': deepcopy(pair)})
    worker.verify_source(result, expected)
    expected['fixed_competitor']['heavy']['mean_log_density_margin'] += .1
    with pytest.raises(AssertionError):
        worker.verify_source(result, expected)


def test_full_diagnosis_reproduces_existing_window_attribution(banks):
    bank = banks['updating']
    source_errors = cross_seed_errors(bank, np.repeat([1, 2, 3], 80))
    query = trace(38, 'heavy', 6)
    density = prior.CrossSeedDensityBank(bank, source_errors, .24619454443573643)
    result = worker.diagnose(query, bank, density, source_errors, 'heavy', prior)
    raw = correction_terms(query, banks, LocalLinearMean(bank))['linear_mean_updating_cov']
    mixed = density.terms(query)[0]
    for condition, terms in (('linear_mean_updating_cov', raw), (prior.MIX_CONDITION, mixed)):
        expected = prior.v78.v77.attribution(query, terms, bank, 'heavy', 0, len(query['states']))
        worker.verify_source(result[condition], expected)
