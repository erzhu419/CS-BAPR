from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial import cKDTree

from scripts import analyze_general_ood_v79_linear_support as worker
from scripts.v78_local_linear_mean import LocalLinearMean


def bank():
    rng = np.random.RandomState(29)
    x, y = rng.normal(size=(96, 14)), rng.normal(size=(96, 55))
    return SimpleNamespace(features=SimpleNamespace(input_mean=np.zeros(14), input_scale=np.ones(14)),
        tree=cKDTree(x), residuals=dict(zip(worker.MODES, np.split(y, 5, axis=1))))


def test_prediction_weights_reproduce_v78_and_do_not_modify_reference():
    reference = bank()
    original = deepcopy(reference.residuals)
    inputs = np.random.RandomState(30).normal(size=(3, 14))
    result = worker.local_prediction(reference, inputs)
    expected = LocalLinearMean(reference).predict(inputs)
    for mode in worker.MODES:
        np.testing.assert_allclose(result['linear'][mode], expected[mode], rtol=1e-11, atol=1e-11)
        np.testing.assert_array_equal(reference.residuals[mode], original[mode])
    changed = inputs.copy()
    changed[1:] += 100
    prefix = worker.local_prediction(reference, changed)
    for key in result['support']:
        np.testing.assert_array_equal(result['support'][key][:1], prefix['support'][key][:1])


def test_reference_loo_errors_match_explicit_refits():
    reference = bank()
    query = np.zeros((1, 14))
    result = worker.local_prediction(reference, query)
    indices = reference.tree.query(query, k=64)[1][0]
    x = reference.tree.data[indices]
    y = np.concatenate([reference.residuals[m][indices] for m in worker.MODES], axis=1)
    errors = []
    for i in range(64):
        keep = np.arange(64) != i
        xx, yy = x[keep], y[keep]
        xc, yc = xx-xx.mean(axis=0), yy-yy.mean(axis=0)
        slope = np.linalg.solve(xc.T@xc+.064*np.eye(14), xc.T@yc)
        errors.append(y[i]-(yy.mean(axis=0)+(x[i]-xx.mean(axis=0))@slope))
    expected = np.sqrt(np.mean(np.asarray(errors).reshape(64, 5, 11)**2, axis=(0, 2)))
    np.testing.assert_allclose(result['loo_rms'][0], expected, rtol=1e-11, atol=1e-11)


def test_energy_gain_decomposes_alignment_and_penalty_per_coordinate():
    rng = np.random.RandomState(31)
    error, correction = rng.normal(size=(8, 11)), rng.normal(size=(8, 11))
    matrix = rng.normal(size=(8, 11, 11))
    covariance = matrix@matrix.transpose(0, 2, 1)+.1*np.eye(11)
    result = worker.energy_change(error, correction, covariance)
    np.testing.assert_allclose(result['coordinate_gain'].sum(axis=1), result['gain'], rtol=1e-11, atol=1e-10)
    ideal = worker.energy_change(error, error, covariance)
    assert np.all(ideal['gain'] > 0)
    excessive = worker.energy_change(error, 3*error, covariance)
    assert np.all(excessive['gain'] < 0)


def test_wrong_source_window_statistics_block_attribution():
    values = {k: 1. for k in ('mean_true_prediction_error_rms', 'mean_true_minus_best_other_log_density',
        'mean_energy_advantage', 'mean_covariance_volume_advantage', 'raw_mode_accuracy')}
    result = {'conditions': {'linear_mean_frozen_cov': deepcopy(values)}}
    previous = {'attribution': {'linear_mean_frozen_cov': deepcopy(values)}}
    worker.verify_source(result, previous)
    previous['attribution']['linear_mean_frozen_cov']['mean_true_prediction_error_rms'] += .1
    with pytest.raises(AssertionError):
        worker.verify_source(result, previous)
