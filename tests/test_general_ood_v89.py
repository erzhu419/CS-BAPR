from copy import deepcopy

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from scripts import analyze_general_ood_v89_reference_prefix as worker
from scripts.v74_density_coverage import DensityCoverageBank
from test_general_ood_v68 import trace
from test_general_ood_v78 import banks


@pytest.fixture(scope='module')
def sources():
    output = {}
    for seed in worker.SEEDS:
        t = trace(seed, 'normal', worker.STEPS)
        t['oracle_observations'] = np.stack([trace(seed, m, worker.STEPS)['next_states'] for m in worker.v84.MODES], axis=1)
        output[seed] = t
    return output


def test_matched_samples_are_unique_unused_and_same_count():
    prefix, suffix = (worker.source_indices(c) for c in worker.CONDITIONS)
    old = np.arange(worker.SWITCH, worker.STEPS, 2)
    for indices in (prefix, suffix):
        assert len(indices) == len(np.unique(indices)) == 1100
        assert not len(np.intersect1d(indices, old))
    assert prefix.min() == 0 and prefix.max() < worker.SWITCH
    assert suffix.min() == worker.SWITCH+1 and suffix.max() == worker.STEPS-1
    assert not len(np.intersect1d(prefix, suffix)) and np.all(suffix % 2 == 1)


def test_extensions_preserve_inputs_and_all_counterfactual_labels(sources):
    for condition in worker.CONDITIONS:
        extra = worker.extensions(sources, condition)
        indices = worker.source_indices(condition)
        assert tuple(extra) == worker.v84.MODES
        for c, mode in enumerate(worker.v84.MODES):
            np.testing.assert_array_equal(extra[mode]['states'], np.concatenate([t['states'][indices] for t in sources.values()]))
            np.testing.assert_array_equal(extra[mode]['next_states'], np.concatenate([t['oracle_observations'][indices, c] for t in sources.values()]))
            assert len(extra[mode]['states']) == 2200


def test_expansion_keeps_original_forward_scaling_rows_and_seed_identity(banks, sources):
    bank = banks['updating']
    groups = np.repeat([135050, *worker.SEEDS], 80)
    expanded, expanded_groups = worker.expand_banks(bank, groups, sources)
    np.testing.assert_array_equal(expanded_groups, np.concatenate([groups, np.repeat(worker.SEEDS, 1100)]))
    for b in expanded.values():
        assert b.tree.n == bank.tree.n+2200
        np.testing.assert_array_equal(b.tree.data[:bank.tree.n], bank.tree.data)
        np.testing.assert_array_equal(b.features.input_mean, bank.features.input_mean)
        np.testing.assert_array_equal(b.features.input_scale, bank.features.input_scale)
        np.testing.assert_array_equal(b.target_scale, bank.target_scale)
        for mode in worker.v84.MODES:
            np.testing.assert_array_equal(b.coefficients[mode], bank.coefficients[mode])
            np.testing.assert_array_equal(b.residuals[mode][:bank.tree.n], bank.residuals[mode])


def test_heldout_seed_exclusion_removes_original_and_added_targets(banks, sources):
    bank = banks['updating']
    groups = np.repeat([135050, *worker.SEEDS], 80)
    expanded, extended_groups = worker.expand_banks(bank, groups, sources)
    inputs = bank.features.inputs(trace(96, 'heavy', 4))
    for b in expanded.values():
        expected, metadata = worker.predict_heldout(b, extended_groups, worker.SEEDS[0], inputs, bank.tree.n)
        changed = deepcopy(b)
        for mode in worker.v84.MODES:
            changed.residuals[mode][extended_groups == worker.SEEDS[0]] += 1000
        actual, after = worker.predict_heldout(changed, extended_groups, worker.SEEDS[0], inputs, bank.tree.n)
        np.testing.assert_array_equal(actual, expected)
        assert metadata == after and metadata['excluded_rows'] == 1180
        assert metadata['base_training_rows'] == 160 and metadata['added_training_rows'] == 1100


def test_source_verification_rejects_wrong_reference_labels(banks, sources):
    old_indices = np.arange(worker.SWITCH, worker.STEPS, 2)
    old = {m: {k: np.concatenate([t['oracle_observations'][old_indices, c] if k == 'next_states'
        else t[k][old_indices] for t in sources.values()]) for k in worker.KEYS} for c, m in enumerate(worker.v84.MODES)}
    base = banks['updating']
    bank = DensityCoverageBank.extend(base, old, 'original_even_suffix')
    blocks = [dict(origin='v76', seed=s, source_mode='normal', updating=True, source_steps=[3800, 6000], stride=2,
        bank_row_range=[base.tree.n+i*1100, base.tree.n+(i+1)*1100]) for i, s in enumerate(worker.SEEDS)]
    worker.verify_sources(bank, blocks, sources)
    changed = deepcopy(sources)
    changed[worker.SEEDS[0]]['oracle_observations'][old_indices, 2] += .1
    with pytest.raises(AssertionError):
        worker.verify_sources(bank, blocks, changed)


def test_new_means_use_frozen_covariances_and_correct_candidate_axis():
    rng = np.random.default_rng(97)
    targets, baseline, expanded = (rng.normal(size=(3, 5, 2)) for _ in range(3))
    offsets = targets[:, :, None, :]-baseline[:, None, :, :]
    a = rng.normal(size=(3, 5, 2, 2))
    covariance = a@a.transpose(0, 1, 3, 2)+.1*np.eye(2)
    before = covariance.copy()
    shifted = worker.coverage_offsets(offsets, baseline, expanded)
    ll, _, logdet = worker.v84.scores(shifted, covariance)
    np.testing.assert_allclose(shifted, targets[:, :, None, :]-expanded[:, None, :, :])
    np.testing.assert_array_equal(logdet, worker.v84.scores(offsets, covariance)[2])
    for row in range(3):
        for truth in range(5):
            expected = [multivariate_normal.logpdf(targets[row, truth], mean=expanded[row, c], cov=covariance[row, c]) for c in range(5)]
            np.testing.assert_allclose(ll[row, truth]-np.log(2*np.pi), expected)
    np.testing.assert_array_equal(covariance, before)
