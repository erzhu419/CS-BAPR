from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from scripts import analyze_general_ood_v80_loo_ridge_mean as worker
from scripts.v78_local_linear_mean import CONTROLS, correction_terms
from scripts.v77_mean_covariance import MODES, factorial_terms
from scripts.v80_loo_ridge_mean import ALPHAS, CONDITIONS, LooRidgeDensityBank, LooRidgeMean, fit_neighborhood
from test_general_ood_v68 import trace
from test_general_ood_v78 import banks


def test_loo_scores_and_selected_prediction_match_explicit_sklearn_refits():
    rng = np.random.RandomState(32)
    x, y, query = rng.normal(size=(64, 14)), rng.normal(size=(64, 55)), rng.normal(size=14)
    prediction, details = fit_neighborhood(x, y, query)
    scores = []
    for alpha in ALPHAS:
        errors = []
        for i in range(64):
            keep = np.arange(64) != i
            model = Ridge(alpha=alpha, fit_intercept=True, solver='cholesky').fit(x[keep], y[keep])
            errors.append(y[i]-model.predict(x[i:i+1])[0])
        scores.append(np.mean(np.asarray(errors)**2))
    np.testing.assert_allclose(details['loo_mse'], scores, rtol=1e-11, atol=1e-11)
    assert details['selected_alpha'] == ALPHAS[int(np.argmin(scores))]
    expected = Ridge(alpha=details['selected_alpha'], fit_intercept=True, solver='cholesky').fit(x, y).predict(query[None])[0]
    np.testing.assert_allclose(prediction, expected, rtol=1e-11, atol=1e-11)


def test_constant_reference_ties_choose_strongest_and_preserve_intercept():
    rng = np.random.RandomState(33)
    x = rng.normal(size=(64, 14))
    prediction, details = fit_neighborhood(x, np.full((64, 55), 3.25), rng.normal(size=14))
    assert details['selected_alpha'] == max(ALPHAS)
    np.testing.assert_array_equal(details['loo_mse'], np.zeros(len(ALPHAS)))
    np.testing.assert_array_equal(prediction, np.full(55, 3.25))


def test_selection_uses_only_current_inputs_and_preserves_reference(banks):
    bank = banks['updating']
    correction = LooRidgeMean(bank)
    query = trace(9, 'slippery', 8)
    old = deepcopy(bank.residuals)
    expected, details = correction.predict_details(bank.features.inputs(query))
    changed = deepcopy(query)
    changed['next_states'] += 1000
    actual, unchanged_details = correction.predict_details(bank.features.inputs(changed))
    for key in details:
        np.testing.assert_array_equal(details[key], unchanged_details[key])
    for mode in MODES:
        np.testing.assert_array_equal(expected[mode], actual[mode])
        np.testing.assert_array_equal(old[mode], bank.residuals[mode])
    changed['states'][4:] += 100
    prefix, prefix_details = correction.predict_details(bank.features.inputs(changed))
    for key in details:
        np.testing.assert_array_equal(details[key][:4], prefix_details[key][:4])
    for mode in MODES:
        np.testing.assert_array_equal(expected[mode][:4], prefix[mode][:4])


def test_selected_batch_density_matches_readout_and_preserves_controls(banks):
    query = trace(9, 'heavy', 6)
    correction = LooRidgeMean(banks['updating'])
    centers, _ = correction.predict_details(banks['updating'].features.inputs(query))
    terms = correction_terms(query, banks, worker.SavedMeans(centers))
    baseline = factorial_terms(query, banks)
    for condition, source in CONTROLS.items():
        np.testing.assert_allclose(terms[condition]['likelihoods'], baseline[source]['likelihoods'], rtol=1e-11, atol=1e-10)
    for source in CONDITIONS.values():
        density = LooRidgeDensityBank(banks, correction, source)
        np.testing.assert_allclose(terms['linear_mean_'+source+'_cov']['likelihoods'], density.log_likelihoods(query), rtol=1e-11, atol=1e-10)
        assert density.metadata()['shared_strength_across_modes']
        assert not density.metadata()['conditional_covariance_changed']


def test_resume_reuses_completed_prefix_and_rejects_changed_protocol(tmp_path):
    cell = dict(group='v76_fresh_updating', seed=135180, mode='normal', updating=True)
    record = worker.v78.replay_cell(cell)
    saved = dict(revision=worker.REVISION, spec=worker.v76.SPEC, alphas=ALPHAS, records=[record])
    progress = tmp_path/'progress.json'
    progress.write_text(json.dumps(saved))
    args = SimpleNamespace(resume_from=progress)
    assert worker.evaluate(args, [(cell, Path('/unused'))], {}, {}, None, {}) == [record]
    with pytest.raises(AssertionError):
        worker.evaluate(args, [(dict(cell, seed=135181), Path('/unused'))], {}, {}, None, {})
    saved['alphas'] = [1., 2.]
    progress.write_text(json.dumps(saved))
    with pytest.raises(AssertionError):
        worker.evaluate(args, [(cell, Path('/unused'))], {}, {}, None, {})


def test_protocol_changes_only_reference_selected_means():
    root = Path(__file__).resolve().parents[1]
    m = json.loads((root/'experiments/general_ood_v80_loo_ridge_mean_20261009/dispatch_manifest.json').read_text())
    assert m['new_conditions'] == CONDITIONS and tuple(m['ridge_alpha_candidates']) == ALPHAS
    assert m['evaluation_paths'] == 65 and m['neighbors'] == 64
    assert m['shared_strength_across_modes'] and not m['intercept_penalized']
    assert not m['selection_uses_query_next_state'] and not m['selection_uses_true_mode']
    assert not m['conditional_covariances_changed'] and not m['readout_controls_actor']
    assert m['environment_steps'] == m['optimizer_updates'] == m['resource_request']['vram_mb'] == 0
