from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import analyze_general_ood_v98_regime_replay as worker
from scripts import analyze_general_ood_v83_cross_seed_covariance as source
from scripts.v73_recursive_mode_readout import RecursiveModeReference, observe_likelihoods
from scripts.v83_cross_seed_covariance import LocalLinearMean, subset_bank
from test_general_ood_v58 import shared_start
from test_general_ood_v68 import trace
from test_general_ood_v78 import banks
from test_general_ood_v89 import sources
from test_general_ood_v91 import expanded
from test_general_ood_v94 import fitted


@pytest.fixture(scope="module")
def expanded_coverage(fitted):
    original = fitted[0]
    data = {m: {k: np.concatenate([trace(s, m, 64)[k] for s in worker.v97.SEEDS])
                for k in worker.v97.KEYS} for m in worker.v94.MODES}
    groups = np.repeat(worker.v97.SEEDS, 64)
    raw, mean, labels = worker.expand(original, data, groups, "old_recipe")
    return original, data, raw, mean, labels


def test_expansion_keeps_raw_prefix_and_original_suffix_and_all_five_targets(expanded_coverage):
    original, payload, raw, mean, groups = expanded_coverage
    assert raw.tree.n == original.bank.tree.n+128
    assert mean.tree.n == original.mean.tree.n+128
    assert len(np.unique(groups)) == len(np.unique(original.groups))+2
    np.testing.assert_array_equal(mean.tree.data[:raw.tree.n], raw.tree.data)
    np.testing.assert_array_equal(mean.tree.data[raw.tree.n:], original.mean.tree.data[original.bank.tree.n:])
    for m in worker.v94.MODES:
        prediction = raw.design(payload[m])@raw.coefficients[m]
        np.testing.assert_allclose(raw.residuals[m][-128:], raw.targets(payload[m])-prediction)
        np.testing.assert_array_equal(mean.residuals[m][:raw.tree.n], raw.residuals[m])
        np.testing.assert_array_equal(raw.coefficients[m], original.bank.coefficients[m])
    np.testing.assert_array_equal(groups[original.bank.tree.n:raw.tree.n], np.repeat(worker.v97.SEEDS, 64))


@pytest.fixture(scope="module")
def fitted_coverage(expanded_coverage, tmp_path_factory):
    original, _, raw, mean, groups = expanded_coverage
    args = SimpleNamespace(recipe="old_recipe", checkpoint_dir=tmp_path_factory.mktemp("v98-errors"))
    errors, fits = worker.freeze_errors(args, mean, groups, source)
    family = worker.CoverageDensity(raw, mean, groups, errors, original.raw_weight)
    family.recipe = args.recipe
    return args, family, fits


def test_whole_seed_errors_exclude_original_and_added_rows_and_reuse_cache(fitted_coverage, monkeypatch):
    args, family, fits = fitted_coverage
    for fit in fits:
        seed = fit["seed"]
        rows = np.flatnonzero(family.groups == seed)
        reduced = subset_bank(family.mean, family.groups != seed)
        assert fit["excluded_mean_rows"] == len(rows)
        inputs = family.mean.tree.data[rows]*family.mean.features.input_scale+family.mean.features.input_mean
        expected = LocalLinearMean(reduced).predict(inputs)
        for m in worker.v94.MODES:
            np.testing.assert_allclose(family.errors[m][rows], family.mean.residuals[m][rows]-expected[m])
    monkeypatch.setattr(worker, "heldout_errors", lambda *a: pytest.fail("completed error folds recomputed"))
    actual, actual_fits = worker.freeze_errors(args, family.mean, family.groups, source)
    assert actual_fits == fits
    for m in worker.v94.MODES:
        np.testing.assert_array_equal(actual[m], family.errors[m])


def test_covariance_controls_physical_scores_and_truth_free_geometry(fitted_coverage):
    _, family, _ = fitted_coverage
    result = worker.gaussian_check(family, source)
    assert result["physical_gaussian_checks"] == 75
    assert result["direct_member_checks"] == 2*len(family.members)
    query = trace(135219, "compound", 3)
    changed = deepcopy(query)
    changed["next_states"] += 999
    fixed, members, cov, _ = family.geometry(family.bank.features.inputs(query))
    other, other_members, other_cov, _ = family.geometry(family.bank.features.inputs(changed))
    np.testing.assert_array_equal(fixed, other)
    np.testing.assert_array_equal(members, other_members)
    for c in worker.v94.CONDITIONS:
        np.testing.assert_array_equal(cov[c], other_cov[c])
    metadata = family.metadata(worker.v94.CONDITIONS[1])
    assert metadata["recipe"] == "old_recipe" and metadata["reference_revision"] == worker.REVISION
    assert metadata["members"] == len(family.members) and metadata["full_error_bank_recomputed"]


def test_replay_preserves_full_history_controls_normal_pairs_and_completed_resume(fitted_coverage, shared_start, tmp_path, monkeypatch):
    _, family, _ = fitted_coverage
    args = SimpleNamespace(recipe="old_recipe", checkpoint_dir=tmp_path, resume_from=None)
    monkeypatch.setitem(source.v76.SPEC, "steps", 8)
    monkeypatch.setitem(source.v76.SPEC, "switch_step", 4)
    monkeypatch.setitem(source.v76.v73.SPEC, "switch_step", 4)
    windows = dict(first800=(0, 4), full=(0, 8), last1000=(4, 8), first_post50=(4, 6), last64=(6, 8))
    monkeypatch.setattr(source.v76.v73.v68.v66.v65.v64, "window_ranges", lambda n, s: windows)
    monkeypatch.setattr(source.v78.v77, "WINDOWS", tuple(windows))
    monkeypatch.setattr(source.v78.v77, "windows", lambda: (("first_post50", 4, 6), ("last64", 6, 8)))
    population, queries = [], []
    for i, mode in enumerate(("normal", "compound")):
        cell = dict(seed=135220, mode=mode, updating=True, group="v95_development_updating")
        path = tmp_path/f"{mode}.npz"
        data = trace(135220+i, mode, 8)
        np.savez_compressed(path, **data)
        previous = dict(observers={}, attribution={}, change_minus_nochange={})
        for c in worker.v94.CONDITIONS:
            previous["observers"][c] = {w: dict(mean_coefficient=.01, mode_accuracy=.5,
                mean_true_minus_best_other_score=1.) for w in windows}
            previous["attribution"][c] = {w: dict(control_sentinel=123) for w in windows}
            previous["change_minus_nochange"][c] = {w: 0. for w in ("first_post50", "last64")}
        population.append((cell, path, previous));queries.append(data)
    monkeypatch.setattr(worker, "evaluation_inputs", lambda *a: population)
    refs = {}
    for c in worker.v94.CONDITIONS:
        ref = RecursiveModeReference()
        ref.bank = SimpleNamespace(metadata=lambda c=c: family.metadata(c))
        ref.phase, ref.normal_null_calibration_transitions = deepcopy(shared_start[4]), 12000
        refs[c] = ref
    actual = worker.replay(args, family, refs, source)
    assert actual["evaluation_paths"] == 2 and actual["evaluated_transitions"] == 16
    for record, query, (_, _, previous) in zip(actual["records"], queries, population):
        terms = family.terms(query)[0]
        for c, control in zip(worker.v94.CONDITIONS, worker.CONTROLS):
            series = observe_likelihoods(terms[c]["likelihoods"], query["dones"], refs[c], "recursive")
            expected = source.v76.v73.window_statistics(query, terms[c]["likelihoods"], series, record["mode"])
            assert record["observers"][c] == expected
            for field in ("observers", "attribution", "change_minus_nochange"):
                assert record[field][control] == previous[field][c]
    normal, changed = actual["records"]
    for c in worker.v94.CONDITIONS:
        assert changed["change_minus_nochange"][c]["first_post50"] == pytest.approx(
            changed["observers"][c]["first_post50"]["mean_coefficient"]-normal["observers"][c]["first_post50"]["mean_coefficient"])
    monkeypatch.setattr(source.v76.v68, "load_trace", lambda *a: pytest.fail("completed replay repeated"))
    args.resume_from = tmp_path/"progress.json"
    assert worker.replay(args, family, refs, source) == actual
    json.dumps(actual, allow_nan=False)
