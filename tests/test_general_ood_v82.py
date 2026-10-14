from copy import copy, deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial import cKDTree
from scipy.stats import multivariate_normal

from scripts import analyze_general_ood_v82_reference_covariance_mix as worker
from scripts.v77_mean_covariance import MODES
from scripts.v78_local_linear_mean import LocalLinearMean, correction_terms
from scripts.v81_prediction_error_covariance import ALPHA, PredictionErrorDensityBank
from scripts.v82_reference_covariance_mix import ReferenceMixedDensityBank, fit_selection, validation_fold
from test_general_ood_v68 import trace
from test_general_ood_v78 import banks


def test_mixture_formula_means_endpoints_and_physical_density(banks):
    bank, query = banks['updating'], trace(36, 'heavy', 5)
    inputs = bank.features.inputs(query)
    original = LocalLinearMean(bank)
    means, loo, _ = PredictionErrorDensityBank(bank).geometry(inputs)
    density = ReferenceMixedDensityBank(bank, .3)
    mixed_means, covariances, _ = density.geometry(inputs)
    terms, _ = density.terms(query)
    expected_means = original.predict(inputs)
    for i, mode in enumerate(MODES):
        np.testing.assert_array_equal(mixed_means[mode], expected_means[mode])
        raw = bank.geometry(inputs, mode)[1]
        np.testing.assert_allclose(covariances[mode], .3*raw+.7*loo[mode], rtol=1e-12, atol=1e-12)
        errors = (bank.targets(query)-bank.design(query)@bank.coefficients[mode]-means[mode])*bank.target_scale
        for row, error in enumerate(errors):
            physical = covariances[mode][row]*bank.target_scale[:, None]*bank.target_scale[None, :]
            assert terms['likelihoods'][row, i] == pytest.approx(multivariate_normal.logpdf(error, cov=physical), abs=1e-8)
    np.testing.assert_allclose(ReferenceMixedDensityBank(bank, 0.).log_likelihoods(query),
        PredictionErrorDensityBank(bank).log_likelihoods(query), rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(ReferenceMixedDensityBank(bank, 1.).log_likelihoods(query),
        correction_terms(query, banks, original)['linear_mean_updating_cov']['likelihoods'], rtol=1e-12, atol=1e-9)


def test_fold_excludes_entire_seed_from_means_and_both_covariances(banks):
    bank = banks['updating']
    groups = np.repeat([1, 2, 3], 80)
    before = deepcopy(bank.residuals)
    errors, loo, raw, fold = validation_fold(bank, groups, 2, 8)
    keep = groups != 2
    reduced = copy(bank)
    reduced.tree = cKDTree(bank.tree.data[keep])
    reduced.residuals = {m: bank.residuals[m][keep] for m in MODES}
    selected = np.flatnonzero(groups == 2)[np.linspace(0, 79, 8, dtype=int)]
    query = bank.tree.data[selected]*bank.features.input_scale+bank.features.input_mean
    expected_means, expected_loo, _ = PredictionErrorDensityBank(reduced).geometry(query)
    np.testing.assert_allclose(errors, np.concatenate([bank.residuals[m][selected]-expected_means[m] for m in MODES]))
    np.testing.assert_array_equal(loo, np.concatenate(list(expected_loo.values())))
    np.testing.assert_array_equal(raw, np.concatenate([reduced.geometry(query, m)[1] for m in MODES]))
    assert fold['excluded_rows'] == 80 and fold['training_rows'] == 160 and fold['validation_labels'] == 40
    for mode in MODES:
        np.testing.assert_array_equal(bank.residuals[mode], before[mode])


def test_weight_is_reference_only_and_equally_weights_seeds_and_modes(banks):
    bank, groups = banks['updating'], np.repeat([1, 2, 3], 80)
    selection = fit_selection(bank, groups, rows_per_seed=8)
    assert selection['validation_queries'] == 24 and selection['validation_labels'] == 120
    assert all(f['validation_rows'] == 8 and f['validation_labels'] == 40 for f in selection['folds'])
    assert set(selection['per_mode']) == set(MODES)
    assert 0 <= selection['raw_weight'] <= 1
    assert selection['selected_nll'] <= min(selection['raw_only_nll'], selection['loo_only_nll'])+1e-10
    assert selection['selected_nll'] == pytest.approx(np.mean([f['selected_nll'] for f in selection['folds']]))
    assert not selection['selection_uses_evaluation_paths'] and not selection['selection_uses_normal_null']
    assert selection['selected_nll'] == pytest.approx(np.mean([v['selected_nll'] for v in selection['per_mode'].values()]))


def test_resume_requires_same_frozen_weight_and_reuses_prefix(tmp_path):
    cell = dict(group='v76_fresh_updating', seed=135180, mode='normal', updating=True)
    record = worker.v78.replay_cell(cell)
    saved = dict(revision=worker.REVISION, spec=worker.v76.SPEC, ridge_alpha=ALPHA, raw_weight=.3, records=[record])
    path = tmp_path/'progress.json'
    path.write_text(json.dumps(saved))
    args, density = SimpleNamespace(resume_from=path), SimpleNamespace(raw_weight=.3)
    assert worker.evaluate(args, [(cell, Path('/unused'))], None, density, {}) == [record]
    density.raw_weight = .4
    with pytest.raises(AssertionError):
        worker.evaluate(args, [(cell, Path('/unused'))], None, density, {})


def test_protocol_is_covariance_only_and_reference_selection_is_predeclared():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root/'experiments/general_ood_v82_reference_covariance_mix_20261009/dispatch_manifest.json').read_text())
    assert manifest['validation_rows_per_seed'] == 512 and len(manifest['validation_seeds']) == 6
    assert manifest['reference_labels_per_mode'] == 23000 and manifest['evaluation_paths'] == 65
    assert manifest['ridge_alpha'] == ALPHA and manifest['neighbors'] == 64
    assert len(manifest['controls_reused']) == 6
    assert not manifest['conditional_mean_rule_changed'] and manifest['conditional_covariances_changed']
    assert not manifest['selection_uses_evaluation_paths'] and not manifest['readout_controls_actor']
    assert manifest['environment_steps'] == manifest['optimizer_updates'] == manifest['resource_request']['vram_mb'] == 0
