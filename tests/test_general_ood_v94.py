from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from scipy.stats import Covariance, multivariate_normal

from scripts import analyze_general_ood_v94_suffix_replay as worker
from scripts import analyze_general_ood_v83_cross_seed_covariance as source
from scripts.v83_cross_seed_covariance import LocalLinearMean
from test_general_ood_v68 import trace
from test_general_ood_v78 import banks
from test_general_ood_v89 import sources
from test_general_ood_v91 import expanded


@pytest.fixture(scope="module")
def fitted(expanded, tmp_path_factory):
    raw, groups, additions, mean_groups = expanded
    mean = additions["suffix_mean_mix"]
    directory = tmp_path_factory.mktemp("v94-errors")
    errors, fits = worker.full_errors(raw, groups, mean, mean_groups, directory)
    family = worker.PairedSuffixDensity(raw, mean, mean_groups, errors, .24619454443573643)
    return family, groups, additions, directory, fits


def test_full_error_bank_matches_source_excluded_predictions_and_reuses_cache(fitted, monkeypatch):
    family, groups, _, directory, fits = fitted
    assert sum(f["error_rows"] for f in fits) == family.mean.tree.n
    assert worker.v92.verify_error_rows(family.mean, family.groups, family.errors, family.bank.tree.n) == 5
    def no_refit(*args):
        pytest.fail("Completed full-bank errors must be reused")
    monkeypatch.setattr(worker.v91, "aligned_errors", no_refit)
    monkeypatch.setattr(worker.v92, "added_errors", no_refit)
    errors, actual_fits = worker.full_errors(family.bank, groups, family.mean, family.groups, directory)
    for mode in worker.MODES:
        np.testing.assert_array_equal(errors[mode], family.errors[mode])
    assert fits == actual_fits


def test_trace_control_matches_trace_but_preserves_base_shape():
    base = np.broadcast_to(np.diag([1., 2., 4.]), (4, 5, 3, 3))
    vector = np.array([2., 1., -1.])
    moment = np.broadcast_to(np.outer(vector, vector), base.shape)
    actual = worker.covariance_variants(base, moment)
    np.testing.assert_array_equal(actual[worker.CONDITIONS[0]], base)
    np.testing.assert_array_equal(actual[worker.CONDITIONS[1]], base+moment)
    np.testing.assert_allclose(np.trace(actual[worker.CONDITIONS[1]], axis1=-2, axis2=-1),
                               np.trace(actual[worker.CONDITIONS[2]], axis1=-2, axis2=-1))
    np.testing.assert_allclose(actual[worker.CONDITIONS[2]], base*(1+6/7))
    assert not np.allclose(actual[worker.CONDITIONS[1]], actual[worker.CONDITIONS[2]])
    zero = worker.covariance_variants(base, np.zeros_like(base))
    for covariance in zero.values():
        np.testing.assert_array_equal(covariance, base)


def test_all_conditions_keep_full_suffix_mean_and_physical_scores(fitted):
    family = fitted[0]
    query = trace(987, "slippery", 3)
    inputs = family.bank.features.inputs(query)
    fixed, members, covariance, _ = family.geometry(inputs)
    direct = LocalLinearMean(family.mean).predict(inputs)
    np.testing.assert_array_equal(fixed, np.stack([direct[m] for m in worker.MODES], axis=1))
    assert worker.v93.verify_members(family.mean, family.groups, inputs, family.bank.tree.n,
                                     members, family.member_fits) == 6
    terms = worker.terms_for_covariances(query, family.bank, fixed, covariance)
    for condition in worker.CONDITIONS:
        np.testing.assert_array_equal(terms[condition]["mean_error_rms"], terms[worker.CONDITIONS[0]]["mean_error_rms"])
        for c, mode in enumerate(worker.MODES):
            errors = (family.bank.targets(query)-family.bank.design(query)@family.bank.coefficients[mode]-fixed[:, c])*family.bank.target_scale
            for row in range(3):
                physical = covariance[condition][row, c]*family.bank.target_scale[:, None]*family.bank.target_scale[None, :]
                expected = multivariate_normal.logpdf(errors[row], cov=Covariance.from_cholesky(np.linalg.cholesky(physical)))
                assert terms[condition]["likelihoods"][row, c] == pytest.approx(expected, abs=1e-8)


def test_members_exclude_all_own_labels(fitted):
    family = fitted[0]
    changed = deepcopy(family.mean)
    seed = worker.v89.SEEDS[0]
    for mode in worker.MODES:
        changed.residuals[mode][family.groups == seed] += 1000.
    other = worker.PairedSuffixDensity(family.bank, changed, family.groups, family.errors, family.raw_weight)
    index = list(np.unique(family.groups)).index(seed)
    inputs = family.bank.features.inputs(trace(986, "heavy", 3))
    before, after = family.members[index].predict(inputs), other.members[index].predict(inputs)
    for mode in worker.MODES:
        np.testing.assert_array_equal(before[mode], after[mode])
    assert other.member_fits == family.member_fits and family.member_fits[index]["excluded_rows"] == 1180


def test_geometry_is_query_label_free_and_prefix_causal(fitted):
    family = fitted[0]
    original = trace(985, "normal", 5)
    changed = deepcopy(original)
    changed["next_states"] += 999.
    inputs, altered = family.bank.features.inputs(original), family.bank.features.inputs(changed)
    np.testing.assert_array_equal(inputs, altered)
    fixed, members, covariance, _ = family.geometry(inputs)
    other_fixed, other_members, other_covariance, _ = family.geometry(altered)
    short_fixed, short_members, short_covariance, _ = family.geometry(inputs[:2])
    np.testing.assert_array_equal(fixed, other_fixed)
    np.testing.assert_array_equal(members, other_members)
    np.testing.assert_array_equal(fixed[:2], short_fixed)
    np.testing.assert_array_equal(members[:, :2], short_members)
    for c in worker.CONDITIONS:
        np.testing.assert_array_equal(covariance[c], other_covariance[c])
        np.testing.assert_array_equal(covariance[c][:2], short_covariance[c])


def test_paired_term_cache_avoids_recomputing_member_geometry(fitted, tmp_path, monkeypatch):
    family = fitted[0]
    query = trace(984, "heavy", 3)
    path = tmp_path/"paired.pt"
    expected, diagnostic = worker.cached_terms(path, family, query, ("test", 984))
    def no_geometry(*args):
        pytest.fail("Completed paired terms must be reused")
    monkeypatch.setattr(family, "terms", no_geometry)
    actual, other = worker.cached_terms(path, family, query, ("test", 984))
    for c in worker.CONDITIONS:
        for key in expected[c]:
            np.testing.assert_array_equal(actual[c][key], expected[c][key])
    assert diagnostic == other


def test_all_phases_use_same_nulls_and_unchanged_calibration_api(tmp_path, monkeypatch):
    import scripts.v73_recursive_mode_readout as readout

    nulls = [dict(states=np.zeros((6000, 11))) for _ in range(2)]
    loaded, geometry, calibrated = [], [], []
    def load(path):
        loaded.append(path)
        return nulls[len(loaded)-1]
    def terms(trace):
        geometry.append(trace)
        return {c: {k: np.full((6000, 5), i) for k in ("energy", "logdet", "likelihoods", "mean_error_rms")}
                for i, c in enumerate(worker.CONDITIONS)}, {}
    family = SimpleNamespace(raw_weight=.24619454443573643, terms=terms, metadata=lambda c: dict(condition=c))
    def calibrate(cls, bank, population):
        assert population[0] is nulls[0] and population[1] is nulls[1]
        c = bank.metadata()["condition"]
        for t in population:
            np.testing.assert_array_equal(bank.log_likelihoods(t), worker.CONDITIONS.index(c))
        calibrated.append(c)
        return SimpleNamespace(phase=dict(condition=c), normal_null_calibration_transitions=12000)
    monkeypatch.setattr(source.v76.v68, "load_trace", load)
    monkeypatch.setattr(readout.RecursiveModeReference, "calibrate", classmethod(calibrate))
    args = SimpleNamespace(checkpoint_dir=tmp_path)
    first = worker.freeze_references(args, family, source, ["null0", "null1"])
    second = worker.freeze_references(args, family, source, ["null0", "null1"])
    assert len(loaded) == len(geometry) == 2 and calibrated == list(worker.CONDITIONS)
    assert set(first) == set(second) == set(worker.CONDITIONS)
    assert all(r.normal_null_calibration_transitions == 12000 for r in second.values())


def test_resume_keeps_complete_path_prefix_and_all_three_conditions(tmp_path):
    cell = dict(group="v76_fresh_updating", seed=135180, mode="normal", updating=True)
    record = source.v78.replay_cell(cell)
    path = tmp_path/"progress.json"
    saved = dict(revision=worker.REVISION, spec=source.v76.SPEC, raw_weight=.24619454443573643,
                 conditions=list(worker.CONDITIONS), records=[record])
    path.write_text(json.dumps(saved))
    args, family = SimpleNamespace(resume_from=path), SimpleNamespace(raw_weight=saved["raw_weight"])
    assert worker.evaluate(args, [(cell, None)], {}, family, {}, source) == [record]
    family.raw_weight = .5
    with pytest.raises(AssertionError):
        worker.evaluate(args, [(cell, None)], {}, family, {}, source)


def test_native_smoke_reproduces_old_controls_and_all_three_physical_densities(fitted):
    family, _, additions, _, _ = fitted
    original_errors = family.bank.residuals
    old = worker.v90.make_densities(family.bank, additions, original_errors, family.raw_weight)
    old[source.MIX_CONDITION] = source.CrossSeedDensityBank(family.bank, original_errors, family.raw_weight)
    cells = [dict(group=g, seed=135180 if g == "v76_fresh_updating" else 1, mode=m, updating=True)
             for g, m in (("v76_fresh_updating", "compound"), ("v61_development_replay", "slippery"), ("v61_development_replay", "heavy"))]
    trajectories = [trace(101+i, c["mode"], 128) for i, c in enumerate(cells)]
    windows = lambda: (("first_post50", 0, 50), ("last64", 64, 128))
    scores = {}
    for cell, trajectory in zip(cells, trajectories):
        details = {}
        for c, density in old.items():
            terms = density.terms(trajectory)[0]
            details[c] = {w: source.v78.v77.attribution(trajectory, terms, family.bank, cell["mode"], a, b)
                          for w, a, b in windows()}
        scores[source.v78.v77.cell_key(source.v78.replay_cell(cell))] = dict(attribution=details)
    with patch.object(source.v78.v77, "windows", windows), patch.object(source.v76.v68, "load_trace", side_effect=trajectories):
        result = worker.native_smoke([(c, None) for c in cells], family, additions, original_errors, scores, source)
    assert result["passed"] and result["checked_transitions"] == 342
    assert result["gaussian_checks"] == 450 and result["old_attribution_comparisons"] == 18
    assert result["direct_member_prediction_checks"] == 36


def test_protocol_preserves_population_and_has_no_new_rl_data():
    root = Path(__file__).resolve().parents[1]
    m = json.loads((root/"experiments/general_ood_v94_suffix_replay_20261009/dispatch_manifest.json").read_text())
    assert m["evaluation_paths"] == 65 and m["changed_windows"] == 96
    assert m["normal_null_transitions_per_condition"] == 12000 and m["members"] == 6
    assert m["conditions"] == list(worker.CONDITIONS) and m["controls_reused"] == list(worker.CONTROLS)
    assert m["trace_matched_rescaling_control"] and m["paired_geometry_shared"]
    assert m["error_seed_exclusion_includes_additions"] and m["full_reference_errors_fitted_before_queries"]
    assert not m["conditional_means_changed_vs_v89"] and not m["query_targets_used_for_geometry"]
    assert not m["evaluation_paths_used_for_reference_fit"] and not m["temporal_rules_changed"]
    assert not m["readout_controls_actor"] and not m["weight_refitted"]
    assert m["environment_steps"] == m["optimizer_updates"] == m["additional_counterfactual_steps"] == 0
