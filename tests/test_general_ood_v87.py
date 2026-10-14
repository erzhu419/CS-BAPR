from copy import deepcopy

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from scripts import analyze_general_ood_v87_shared_covariance as worker
from scripts.v78_local_linear_mean import LocalLinearMean
from scripts.v83_cross_seed_covariance import subset_bank, validation_fold
from test_general_ood_v78 import banks


def covariance():
    rng = np.random.default_rng(87)
    a = rng.normal(size=(4, 5, 3, 3))
    return a@a.transpose(0, 1, 3, 2)+.1*np.eye(3)


def test_pool_is_per_input_equal_mode_mean_and_preserves_components():
    mixed = covariance()
    before = mixed.copy()
    pooled = worker.shared_covariance(mixed)
    for row in range(4):
        for candidate in range(5):
            np.testing.assert_allclose(pooled[row, candidate], mixed[row].mean(axis=0))
    assert not np.allclose(pooled[0], pooled[1])
    assert np.all(np.linalg.eigvalsh(pooled) > 0.)
    np.testing.assert_array_equal(mixed, before)
    np.testing.assert_allclose(worker.shared_covariance(mixed[:, ::-1]), pooled)


def test_scores_remain_normalized_gaussians_and_pair_volume_cancels():
    rng = np.random.default_rng(88)
    offsets = rng.normal(size=(4, 5, 5, 3))
    pooled = worker.shared_covariance(covariance())
    ll, energy, logdet = worker.v84.scores(offsets, pooled)
    constant = -1.5*np.log(2*np.pi)
    for row in range(4):
        for truth in range(5):
            expected = [multivariate_normal.logpdf(offsets[row, truth, c], cov=pooled[row, c]) for c in range(5)]
            np.testing.assert_allclose(ll[row, truth]+constant, expected)
    np.testing.assert_array_equal(logdet, np.repeat(logdet[:, :1], 5, axis=1))
    stats = worker.v84.summarize(ll, energy, logdet)
    for values in stats['per_mode'].values():
        for pair in values['pairs'].values():
            assert pair['mean_volume_advantage'] == 0.
            assert pair['mean_log_density_margin'] == pytest.approx(pair['mean_energy_advantage'])


def test_covariance_sharing_preserves_cached_local_linear_means(banks):
    bank = banks['updating']
    groups = np.repeat([1, 2, 3], 80)
    values = validation_fold(bank, groups, 1, 8)
    inputs, predictions, offsets, cross, raw = worker.v84.reconstruct_fold(bank, groups, values)
    correction = LocalLinearMean(subset_bank(bank, groups != 1)).predict(inputs)
    for c, mode in enumerate(worker.v84.MODES):
        np.testing.assert_allclose(predictions[:, c], bank.features.features(inputs)@bank.coefficients[mode]+correction[mode])
    before = offsets.copy()
    worker.v84.scores(offsets, worker.shared_covariance(.24619454443573643*raw+.7538054555642636*cross))
    np.testing.assert_array_equal(offsets, before)


def test_changed_control_statistics_block_interpretation():
    offsets = np.random.default_rng(89).normal(size=(4, 5, 5, 3))
    stats = worker.v84.summarize(*worker.v84.scores(offsets, covariance()))
    expected = deepcopy(stats)
    worker.verify_control(stats, expected)
    expected['per_mode']['slippery']['classification_nll'] += .1
    with pytest.raises(AssertionError):
        worker.verify_control(stats, expected)
