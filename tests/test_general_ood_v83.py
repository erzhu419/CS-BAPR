from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from scripts import analyze_general_ood_v83_cross_seed_covariance as worker
from scripts import v83_cross_seed_covariance as readout
from scripts.v77_mean_covariance import MODES
from scripts.v78_local_linear_mean import LocalLinearMean, correction_terms
from scripts.v81_prediction_error_covariance import ALPHA, error_covariance
from test_general_ood_v68 import trace
from test_general_ood_v78 import banks


def test_reference_blocks_keep_coupled_streams_in_one_seed():
    blocks = [dict(seed=1, bank_row_range=[0, 80]), dict(seed=1, bank_row_range=[80, 160]),
              dict(seed=2, bank_row_range=[160, 240])]
    groups = readout.groups_from_blocks(blocks, 240)
    np.testing.assert_array_equal(groups, np.repeat([1, 2], [160, 80]))
    blocks[1]['bank_row_range'][0] += 1
    with pytest.raises(AssertionError):
        readout.groups_from_blocks(blocks, 240)


def test_each_error_is_predicted_without_its_entire_seed(banks):
    bank, groups = banks['updating'], np.repeat([1, 2, 3], 80)
    errors = readout.cross_seed_errors(bank, groups)
    for seed in (1, 2, 3):
        heldout = groups == seed
        reduced = readout.subset_bank(bank, ~heldout)
        inputs = bank.tree.data[heldout]*bank.features.input_scale+bank.features.input_mean
        expected = LocalLinearMean(reduced).predict(inputs)
        for mode in MODES:
            np.testing.assert_array_equal(errors[mode][heldout], bank.residuals[mode][heldout]-expected[mode])
    changed = deepcopy(bank)
    for mode in MODES:
        changed.residuals[mode][groups == 1] += 10
    actual = readout.cross_seed_errors(changed, groups)
    for mode in MODES:
        np.testing.assert_allclose(actual[mode][groups == 1]-errors[mode][groups == 1], 10., rtol=0, atol=1e-12)


def test_outer_labels_cannot_enter_any_inner_fit_or_covariance(banks):
    bank, groups = banks['updating'], np.repeat([1, 2, 3], 80)
    original = readout.validation_fold(bank, groups, 1, 8)
    changed = deepcopy(bank)
    for mode in MODES:
        changed.residuals[mode][groups == 1] += 20
    actual = readout.validation_fold(changed, groups, 1, 8)
    np.testing.assert_allclose(actual[0]-original[0], 20., rtol=0, atol=1e-12)
    np.testing.assert_array_equal(actual[1], original[1])
    np.testing.assert_array_equal(actual[2], original[2])
    assert actual[3]['inner_seeds'] == [2, 3]
    assert actual[3]['outer_seed_excluded_from_inner_fits'] and actual[3]['inner_seed_excluded_from_its_error_fit']


def test_mixture_uses_aligned_errors_and_preserves_mean_and_physical_units(banks):
    bank, query = banks['updating'], trace(37, 'heavy', 5)
    errors = readout.cross_seed_errors(bank, np.repeat([1, 2, 3], 80))
    density = readout.CrossSeedDensityBank(bank, errors, .3)
    inputs = bank.features.inputs(query)
    means, covariance, _ = density.geometry(inputs)
    expected = LocalLinearMean(bank).predict(inputs)
    terms, _ = density.terms(query)
    _, indices = bank.tree.query((inputs-bank.features.input_mean)/bank.features.input_scale, k=64)
    for i, mode in enumerate(MODES):
        np.testing.assert_array_equal(means[mode], expected[mode])
        cross = np.asarray([error_covariance(e)[0] for e in errors[mode][indices]])
        raw = bank.geometry(inputs, mode)[1]
        np.testing.assert_allclose(covariance[mode], .3*raw+.7*cross, rtol=1e-12, atol=1e-12)
        offset = (bank.targets(query)-bank.design(query)@bank.coefficients[mode]-means[mode])*bank.target_scale
        for row in range(len(inputs)):
            physical = covariance[mode][row]*bank.target_scale[:, None]*bank.target_scale[None, :]
            assert terms['likelihoods'][row, i] == pytest.approx(multivariate_normal.logpdf(offset[row], cov=physical), abs=1e-8)
    np.testing.assert_allclose(readout.CrossSeedDensityBank(bank, errors, 1.).log_likelihoods(query),
        correction_terms(query, banks, LocalLinearMean(bank))['linear_mean_updating_cov']['likelihoods'], rtol=1e-12, atol=1e-9)


def test_query_targets_and_future_states_do_not_change_geometry(banks):
    bank, query = banks['updating'], trace(38, 'slippery', 6)
    errors = {m: bank.residuals[m].copy() for m in MODES}
    density = readout.CrossSeedDensityBank(bank, errors, .3)
    means, covariances, _ = density.geometry(bank.features.inputs(query))
    changed = deepcopy(query)
    changed['next_states'] += 100
    changed['states'][3:] += 100
    actual_means, actual_covariances, _ = density.geometry(bank.features.inputs(changed))
    for mode in MODES:
        np.testing.assert_array_equal(means[mode][:3], actual_means[mode][:3])
        np.testing.assert_array_equal(covariances[mode][:3], actual_covariances[mode][:3])


def test_nested_selection_balances_seeds_modes_and_resumes_fold_cache(banks, tmp_path, monkeypatch):
    bank, groups = banks['updating'], np.repeat([1, 2, 3], 80)
    selected = readout.fit_selection(bank, groups, tmp_path, rows_per_seed=8)
    assert selected['validation_queries'] == 24 and selected['validation_labels'] == 120
    assert selected['selected_nll'] <= min(selected['raw_only_nll'], selected['cross_seed_only_nll'])+1e-10
    assert selected['selected_nll'] == pytest.approx(np.mean([f['selected_nll'] for f in selected['folds']]))
    assert selected['selected_nll'] == pytest.approx(np.mean([f['selected_nll'] for f in selected['per_mode'].values()]))
    assert not selected['selection_uses_evaluation_paths'] and not selected['selection_uses_normal_null']
    def no_refitting(*args):
        pytest.fail('Completed validation folds must be reused')
    monkeypatch.setattr(readout, 'validation_fold', no_refitting)
    assert readout.fit_selection(bank, groups, tmp_path, rows_per_seed=8) == selected


def test_scalar_resume_requires_matching_conditions_and_frozen_weights(tmp_path):
    cell = dict(group='v76_fresh_updating', seed=135180, mode='normal', updating=True)
    record = worker.v78.replay_cell(cell)
    weights = {readout.ERROR_CONDITION: 0., readout.MIX_CONDITION: .3}
    saved = dict(revision=worker.REVISION, spec=worker.v76.SPEC, ridge_alpha=ALPHA, raw_weights=weights, records=[record])
    path = tmp_path/'progress.json'
    path.write_text(json.dumps(saved))
    args = SimpleNamespace(resume_from=path)
    densities = {c: SimpleNamespace(raw_weight=w) for c, w in weights.items()}
    assert worker.evaluate(args, [(cell, Path('/unused'))], {}, densities, {}) == [record]
    densities[readout.MIX_CONDITION].raw_weight = .4
    with pytest.raises(AssertionError):
        worker.evaluate(args, [(cell, Path('/unused'))], {}, densities, {})


def test_protocol_is_nested_reference_only_covariance_intervention():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root/'experiments/general_ood_v83_cross_seed_covariance_20261009/dispatch_manifest.json').read_text())
    assert manifest['reference_labels_per_mode'] == manifest['error_bank_rows'] == 23000
    assert manifest['nested_cross_fitting'] and manifest['error_source_seed_excluded_from_mean_fit']
    assert manifest['outer_seed_excluded_from_inner_fits'] and manifest['validation_rows_per_seed'] == 512
    assert manifest['ridge_alpha'] == ALPHA and manifest['neighbors'] == 64 and manifest['evaluation_paths'] == 65
    assert not manifest['conditional_mean_rule_changed'] and manifest['conditional_covariances_changed']
    assert not manifest['selection_uses_evaluation_paths'] and not manifest['readout_controls_actor']
    assert manifest['environment_steps'] == manifest['optimizer_updates'] == manifest['resource_request']['vram_mb'] == 0
