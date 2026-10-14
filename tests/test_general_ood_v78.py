from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial import cKDTree
from scipy.stats import multivariate_normal
import torch

from scripts import analyze_general_ood_v78_local_linear_mean as worker
from scripts.v68_mode_likelihood import ModeLikelihoodBank
from scripts.v72_conditional_likelihood import ConditionalDensityBank
from scripts.v76_behavior_coverage import BehaviorCoverageBank
from scripts.v77_mean_covariance import MODES, factorial_terms
from scripts.v78_local_linear_mean import CONDITIONS, CONTROLS, LocalLinearDensityBank, LocalLinearMean, correction_terms
from test_general_ood_v68 import trace


@pytest.fixture(scope='module')
def banks():
    torch.set_num_threads(1)
    fitting, calibration = [{m: trace(s, m, 160) for m in MODES} for s in (3, 4)]
    base = ConditionalDensityBank.fit_density(ModeLikelihoodBank.fit(fitting, calibration), calibration)
    extra = {m: trace(5, m, 80) for m in MODES}
    return {'frozen': base, 'updating': BehaviorCoverageBank.extend(base, extra, 'updating_mixed')}


def test_local_affine_ridge_matches_mean_loss_solution_and_unpenalized_intercept():
    rng = np.random.RandomState(19)
    inputs, query = rng.normal(size=(128, 14)), rng.normal(size=(4, 14))
    slopes, intercept = rng.normal(scale=.1, size=(14, 55)), rng.normal(size=55)
    targets = inputs@slopes+intercept
    bank = SimpleNamespace(features=SimpleNamespace(input_mean=np.zeros(14), input_scale=np.ones(14)),
        tree=cKDTree(inputs), residuals=dict(zip(MODES, np.split(targets, 5, axis=1))))
    expected = []
    for q, indices in zip(query, bank.tree.query(query, k=64)[1]):
        x, y = inputs[indices], targets[indices]
        xc, yc = x-x.mean(axis=0), y-y.mean(axis=0)
        fitted = np.linalg.solve(xc.T@xc+.064*np.eye(14), xc.T@yc)
        expected.append(y.mean(axis=0)+(q-x.mean(axis=0))@fitted)
    actual = np.concatenate(list(LocalLinearMean(bank).predict(query).values()), axis=1)
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)
    assert np.sqrt(np.mean((actual-(query@slopes+intercept))**2)) < .005
    constant_bank = deepcopy(bank)
    constant_bank.residuals = {m: np.full((128, 11), 3.25) for m in MODES}
    for values in LocalLinearMean(constant_bank).predict(query).values():
        np.testing.assert_allclose(values, 3.25, rtol=0, atol=1e-14)


def test_local_mean_has_no_query_target_or_future_dependence_and_preserves_bank(banks):
    bank = banks['updating']
    query = trace(8, 'slippery', 12)
    correction = LocalLinearMean(bank)
    residuals = {m: bank.residuals[m].copy() for m in MODES}
    coefficients = {m: bank.coefficients[m].copy() for m in MODES}
    expected = correction.predict(bank.features.inputs(query))
    changed = deepcopy(query)
    changed['next_states'] += 1000
    changed['states'][6:] += 100
    actual = correction.predict(bank.features.inputs(changed))
    for mode in MODES:
        np.testing.assert_array_equal(actual[mode][:6], expected[mode][:6])
        np.testing.assert_array_equal(bank.residuals[mode], residuals[mode])
        np.testing.assert_array_equal(bank.coefficients[mode], coefficients[mode])


def test_control_densities_and_linear_physical_gaussians(banks):
    query = trace(8, 'heavy', 8)
    correction = LocalLinearMean(banks['updating'])
    terms, original = correction_terms(query, banks, correction), factorial_terms(query, banks)
    for condition, source in CONTROLS.items():
        np.testing.assert_allclose(terms[condition]['likelihoods'], original[source]['likelihoods'], rtol=1e-12, atol=1e-10)
    for condition, (mean_source, covariance_source) in CONDITIONS.items():
        if mean_source != 'updating_linear':
            continue
        density = LocalLinearDensityBank(banks, correction, covariance_source)
        np.testing.assert_allclose(terms[condition]['likelihoods'], density.log_likelihoods(query), rtol=1e-12, atol=1e-10)
        bank = banks['updating']
        centers = correction.predict(bank.features.inputs(query))
        for i, mode in enumerate(MODES):
            covariance = banks[covariance_source].geometry(bank.features.inputs(query), mode)[1]
            errors = (bank.targets(query)-bank.design(query)@bank.coefficients[mode]-centers[mode])*bank.target_scale
            for row in range(len(errors)):
                physical_cov = covariance[row]*bank.target_scale[:, None]*bank.target_scale[None, :]
                expected = multivariate_normal.logpdf(errors[row], cov=physical_cov)
                assert terms[condition]['likelihoods'][row, i] == pytest.approx(expected, abs=1e-8)


def test_control_statistics_reject_mismatched_risk():
    values = dict(mean_coefficient=.01, mode_accuracy=.9, mean_true_minus_best_other_score=2.,
        instantaneous_mode_accuracy=.8, instantaneous_mean_true_minus_best_other_log_density=1.,
        predicted_mode_fractions={'normal': .9, 'heavy': .1})
    observers = {c: {'full': deepcopy(values)} for c in CONTROLS}
    previous = {'observers': {s: {'full': deepcopy(values)} for s in CONTROLS.values()}}
    worker.verify_controls(observers, previous)
    observers['constant_mean_frozen_cov']['full']['mean_coefficient'] = .2
    with pytest.raises(AssertionError):
        worker.verify_controls(observers, previous)


def test_protocol_is_single_mean_intervention_without_new_training():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root/'experiments/general_ood_v78_local_linear_mean_20261009/dispatch_manifest.json').read_text())
    assert manifest['conditions'] == {k: list(v) for k, v in CONDITIONS.items()}
    assert manifest['evaluation_paths'] == 65 and manifest['neighbors'] == 64
    assert manifest['ridge_alpha'] == 64*manifest['ridge_mean_loss_coefficient'] == .064
    assert not manifest['intercept_penalized'] and not manifest['conditional_covariances_changed']
    assert manifest['environment_steps'] == manifest['optimizer_updates'] == 0
    assert not manifest['readout_controls_actor'] and manifest['resource_request']['vram_mb'] == 0


def test_scalar_resume_reuses_completed_prefix(tmp_path):
    cell = dict(group='v76_fresh_updating', seed=135180, mode='normal', updating=True)
    record = worker.replay_cell(cell)
    progress = tmp_path/'progress.json'
    progress.write_text(json.dumps(dict(revision=worker.REVISION, spec=worker.v76.SPEC, records=[record])))
    args = SimpleNamespace(resume_from=progress)
    assert worker.evaluate(args, [(cell, Path('/unused'))], {}, {}, None, {}) == [record]
    with pytest.raises(AssertionError):
        worker.evaluate(args, [(dict(cell, seed=135181), Path('/unused'))], {}, {}, None, {})


def test_shared_summary_uses_supplied_five_conditions():
    stats = dict(mean_coefficient=.01, mean_true_minus_best_other_score=1., mode_accuracy=.8)
    rows = [dict(group='test', mode=m, observers={c: {w: deepcopy(stats) for w in worker.v77.WINDOWS} for c in CONDITIONS},
        change_minus_nochange={c: {w: .1 for w in worker.v76.v68.BASE.CHANGE_WINDOWS} for c in CONDITIONS}) for m in ('normal', 'heavy')]
    result = worker.v77.summarize(rows, CONDITIONS)['test']
    assert set(result) == set(CONDITIONS)
    assert all(r['screen_pass'] and r['changed_pairs'] == 1 for r in result.values())
