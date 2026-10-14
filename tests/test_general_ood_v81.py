from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import multivariate_normal
from sklearn.covariance import oas
from sklearn.linear_model import Ridge

from scripts import analyze_general_ood_v81_prediction_error_covariance as worker
from scripts.v77_mean_covariance import MODES
from scripts.v78_local_linear_mean import LocalLinearMean
from scripts.v81_prediction_error_covariance import ALPHA, PredictionErrorDensityBank, error_covariance, fit_reference
from test_general_ood_v68 import trace
from test_general_ood_v78 import banks


def test_reference_loo_errors_match_explicit_refits_with_fixed_alpha():
    rng = np.random.RandomState(34)
    x, y, query = rng.normal(size=(64, 14)), rng.normal(size=(64, 55)), rng.normal(size=14)
    mean, errors, leverage = fit_reference(x, y, query)
    expected = []
    for i in range(64):
        keep = np.arange(64) != i
        model = Ridge(alpha=ALPHA, fit_intercept=True, solver='cholesky').fit(x[keep], y[keep])
        expected.append(y[i]-model.predict(x[i:i+1])[0])
    np.testing.assert_allclose(errors, expected, rtol=1e-11, atol=1e-11)
    model = Ridge(alpha=ALPHA, fit_intercept=True, solver='cholesky').fit(x-query, y)
    np.testing.assert_array_equal(mean, model.intercept_)
    assert leverage >= 1/64


def test_error_second_moment_keeps_bias_and_uses_existing_relative_ridge():
    errors = np.random.RandomState(35).normal(size=(64, 11))+2
    expected, shrinkage = oas(errors, assume_centered=True)
    covariance, actual_shrinkage = error_covariance(errors)
    np.testing.assert_allclose(covariance, expected+.001*np.trace(expected)/11*np.eye(11), rtol=1e-12, atol=1e-12)
    assert actual_shrinkage == shrinkage
    assert np.linalg.eigvalsh(covariance).min() > 0
    centered, _ = oas(errors)
    assert np.trace(covariance) > np.trace(centered)


def test_means_stay_v78_and_densities_match_physical_gaussians(banks):
    query = trace(10, 'heavy', 6)
    bank = banks['updating']
    density = PredictionErrorDensityBank(bank)
    inputs = bank.features.inputs(query)
    means, covariances, _ = density.geometry(inputs)
    expected = LocalLinearMean(bank).predict(inputs)
    terms, _ = density.terms(query)
    np.testing.assert_allclose(terms['likelihoods'], density.log_likelihoods(query), rtol=1e-11, atol=1e-10)
    for i, mode in enumerate(MODES):
        np.testing.assert_array_equal(means[mode], expected[mode])
        error = (bank.targets(query)-bank.design(query)@bank.coefficients[mode]-means[mode])*bank.target_scale
        for row in range(len(inputs)):
            covariance = covariances[mode][row]*bank.target_scale[:, None]*bank.target_scale[None, :]
            value = multivariate_normal.logpdf(error[row], cov=covariance)
            assert terms['likelihoods'][row, i] == pytest.approx(value, abs=1e-8)


def test_covariance_has_no_query_target_or_future_dependence_and_preserves_bank(banks):
    bank = banks['updating']
    density = PredictionErrorDensityBank(bank)
    residuals, coefficients = deepcopy(bank.residuals), deepcopy(bank.coefficients)
    query = trace(10, 'slippery', 8)
    means, covariances, diagnostic = density.geometry(bank.features.inputs(query))
    changed = deepcopy(query)
    changed['next_states'] += 1000
    same_means, same_covariances, same_diagnostic = density.geometry(bank.features.inputs(changed))
    for mode in MODES:
        np.testing.assert_array_equal(means[mode], same_means[mode])
        np.testing.assert_array_equal(covariances[mode], same_covariances[mode])
    for key in diagnostic:
        np.testing.assert_array_equal(diagnostic[key], same_diagnostic[key])
    changed['states'][4:] += 100
    prefix_means, prefix_covariances, prefix_diagnostic = density.geometry(bank.features.inputs(changed))
    for mode in MODES:
        np.testing.assert_array_equal(means[mode][:4], prefix_means[mode][:4])
        np.testing.assert_array_equal(covariances[mode][:4], prefix_covariances[mode][:4])
        np.testing.assert_array_equal(bank.residuals[mode], residuals[mode])
        np.testing.assert_array_equal(bank.coefficients[mode], coefficients[mode])
    for key in diagnostic:
        np.testing.assert_array_equal(diagnostic[key][:4], prefix_diagnostic[key][:4])


def test_changed_mean_statistics_block_covariance_only_claim():
    attribution = {'last64': {'mean_true_prediction_error_rms': .1}}
    previous = {'attribution': {'linear_mean_updating_cov': deepcopy(attribution)}}
    worker.verify_mean_statistics(attribution, previous)
    attribution['last64']['mean_true_prediction_error_rms'] = .2
    with pytest.raises(AssertionError):
        worker.verify_mean_statistics(attribution, previous)


def test_resume_reuses_completed_prefix_and_rejects_mean_protocol_changes(tmp_path):
    cell = dict(group='v76_fresh_updating', seed=135180, mode='normal', updating=True)
    record = worker.v78.replay_cell(cell)
    saved = dict(revision=worker.REVISION, spec=worker.v76.SPEC, ridge_alpha=ALPHA, records=[record])
    progress = tmp_path/'progress.json'
    progress.write_text(json.dumps(saved))
    args = SimpleNamespace(resume_from=progress)
    assert worker.evaluate(args, [(cell, Path('/unused'))], None, None, {}) == [record]
    with pytest.raises(AssertionError):
        worker.evaluate(args, [(dict(cell, seed=135181), Path('/unused'))], None, None, {})
    saved['ridge_alpha'] *= 10
    progress.write_text(json.dumps(saved))
    with pytest.raises(AssertionError):
        worker.evaluate(args, [(cell, Path('/unused'))], None, None, {})


def test_protocol_is_covariance_only_without_actor_feedback():
    root = Path(__file__).resolve().parents[1]
    m = json.loads((root/'experiments/general_ood_v81_prediction_error_covariance_20261009/dispatch_manifest.json').read_text())
    assert m['ridge_alpha'] == ALPHA and m['neighbors'] == 64 and m['evaluation_paths'] == 65
    assert m['conditional_covariances_changed'] and not m['conditional_mean_rule_changed']
    assert m['error_second_moment_assume_centered'] and m['covariance_ridge_relative'] == .001
    assert not m['query_residual_used_for_geometry'] and not m['query_leverage_inflation']
    assert not m['readout_controls_actor'] and len(m['controls_reused_from_v78']) == 5
    assert m['environment_steps'] == m['optimizer_updates'] == m['resource_request']['vram_mb'] == 0
