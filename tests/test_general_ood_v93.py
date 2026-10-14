from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import analyze_general_ood_v93_mean_disagreement as worker
from scripts.v83_cross_seed_covariance import LocalLinearMean
from test_general_ood_v68 import trace
from test_general_ood_v78 import banks
from test_general_ood_v89 import sources
from test_general_ood_v91 import expanded


def test_second_moment_uses_fixed_mean_equal_seed_weight_and_retains_bias():
    rng = np.random.default_rng(93)
    centers = rng.normal(size=(4, 5, 3))
    members = rng.normal(size=(3, 4, 5, 3))+2.
    actual = worker.displacement_second_moment(centers, members)
    direct = np.array([[sum(np.outer(v, v) for v in members[:, row, mode]-centers[row, mode])/3
                        for mode in range(5)] for row in range(4)])
    np.testing.assert_allclose(actual, direct, rtol=1e-12, atol=1e-12)
    average = members.mean(axis=0)
    centered = worker.displacement_second_moment(average, members)
    bias = average-centers
    np.testing.assert_allclose(actual, centered+np.einsum("nmd,nme->nmde", bias, bias), rtol=1e-12, atol=1e-12)
    assert not np.allclose(actual, centered)
    assert np.linalg.eigvalsh(actual).min() > -1e-12
    np.testing.assert_allclose(worker.displacement_second_moment(centers, members[::-1]), actual)


def test_identical_members_add_zero_and_reproduce_frozen_scores():
    centers = np.ones((4, 5, 3))
    members = np.repeat(centers[None], 5, axis=0)
    moment = worker.displacement_second_moment(centers, members)
    np.testing.assert_array_equal(moment, 0.)
    covariance = np.broadcast_to(np.eye(3), (4, 5, 3, 3))
    shifted = np.full((4, 5, 5, 3), .2)
    before, after = worker.v84.scores(shifted, covariance), worker.v84.scores(shifted, covariance+moment)
    for a, b in zip(before, after):
        np.testing.assert_array_equal(a, b)


def test_each_member_excludes_own_original_and_added_targets(expanded):
    base, _, additions, groups = expanded
    bank = additions[worker.v89.CONDITIONS[0]]
    inputs = bank.features.inputs(trace(992, "heavy", 3))
    expected, fits = worker.seed_members(bank, groups, inputs, base.tree.n)
    changed = deepcopy(bank)
    seed = worker.v89.SEEDS[0]
    for mode in worker.v84.MODES:
        changed.residuals[mode][groups == seed] += 1000.
    actual, after = worker.seed_members(changed, groups, inputs, base.tree.n)
    index = list(np.unique(groups)).index(seed)
    np.testing.assert_array_equal(actual[index], expected[index])
    assert fits == after and fits[index]["excluded_rows"] == 1180
    assert fits[index]["base_training_rows"] == 160 and fits[index]["added_training_rows"] == 1100
    assert worker.verify_members(bank, groups, inputs, base.tree.n, expected, fits) == 6


def test_outer_labels_cannot_enter_any_member_or_disagreement(expanded):
    base, raw_groups, additions, groups = expanded
    bank = additions[worker.v89.CONDITIONS[0]]
    seed = worker.v89.SEEDS[0]
    changed = deepcopy(bank)
    for mode in worker.v84.MODES:
        changed.residuals[mode][groups == seed] += 500.
    inputs = bank.features.inputs(trace(991, "slippery", 2))
    results = []
    for mean in (bank, changed):
        raw, rg, reduced, mg = worker.v91.outer_banks(base, raw_groups, mean, groups, seed)
        assert seed not in rg and seed not in mg
        fixed = np.stack(list(LocalLinearMean(reduced).predict(inputs).values()), axis=1)
        members, fits = worker.seed_members(reduced, mg, inputs, raw.tree.n)
        results.append((fixed, members, fits, worker.displacement_second_moment(fixed, members)))
    for index in (0, 1, 3):
        np.testing.assert_array_equal(results[0][index], results[1][index])
    assert results[0][2] == results[1][2]


def test_member_geometry_does_not_use_query_next_state_or_true_mode(expanded):
    base, _, additions, groups = expanded
    bank = additions[worker.v89.CONDITIONS[0]]
    original = trace(990, "normal", 3)
    changed = deepcopy(original)
    changed["next_states"] += 999.
    changed["oracle_observations"] = np.full((3, 5, 11), -999.)
    inputs = bank.features.inputs(original)
    altered = bank.features.inputs(changed)
    np.testing.assert_array_equal(inputs, altered)
    before, fits = worker.seed_members(bank, groups, inputs, base.tree.n)
    after, other = worker.seed_members(bank, groups, altered, base.tree.n)
    np.testing.assert_array_equal(before, after)
    assert fits == other


def test_added_covariance_keeps_mean_and_matches_physical_gaussian(expanded):
    base, _, additions, groups = expanded
    bank = additions[worker.v89.CONDITIONS[0]]
    inputs = bank.features.inputs(trace(989, "compound", 3))
    fixed = np.stack(list(LocalLinearMean(bank).predict(inputs).values()), axis=1)
    members, _ = worker.seed_members(bank, groups, inputs, base.tree.n)
    unchanged = fixed.copy()
    covariance = np.stack([base.geometry(inputs, m)[1] for m in worker.v84.MODES], axis=1)
    updated = covariance+worker.displacement_second_moment(fixed, members)
    physical_bank = deepcopy(bank)
    physical_bank.target_scale *= np.geomspace(1e-3, 10., len(bank.target_scale))
    shifted = np.full((3, 5, 5, len(bank.target_scale)), .1)
    check = worker.verify_geometry(physical_bank, shifted, fixed, members, covariance, updated)
    np.testing.assert_array_equal(fixed, unchanged)
    assert check["gaussian_checks"] == 75 and check["maximum_physical_covariance_condition"] > 1e8
    assert check["maximum_physical_score_difference"] < 1e-8


def test_member_cache_reuses_exact_query_population(expanded, tmp_path, monkeypatch):
    base, _, additions, groups = expanded
    bank = additions[worker.v89.CONDITIONS[0]]
    inputs = bank.features.inputs(trace(988, "heavy", 3))
    expected, fits = worker.cached_members(bank, groups, inputs, base.tree.n, 999, "prefix_mean_mix", tmp_path)
    def no_refit(*args):
        pytest.fail("Completed member queries must be reused")
    monkeypatch.setattr(worker, "seed_members", no_refit)
    actual, after = worker.cached_members(bank, groups, inputs, base.tree.n, 999, "prefix_mean_mix", tmp_path)
    np.testing.assert_array_equal(actual, expected)
    assert fits == after
    with pytest.raises(AssertionError):
        worker.cached_members(bank, groups, inputs+.1, base.tree.n, 999, "prefix_mean_mix", tmp_path)


def test_protocol_is_one_fixed_additive_component_without_new_data():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root/"experiments/general_ood_v93_mean_disagreement_20261009/dispatch_manifest.json").read_text())
    assert manifest["disagreement_coefficient"] == 1. and manifest["members_per_outer_fold"] == 5
    assert manifest["disagreement_center"] == "unchanged outer mean, not ensemble average"
    assert len(manifest["controls"]) == 9 and manifest["reference_folds"] == 6
    assert manifest["validation_rows_per_fold"] == 512 and manifest["mean_fitting_rows"] == 25200
    assert manifest["outer_seed_excluded_before_member_fits"] and manifest["member_seed_exclusion_includes_additions"]
    assert manifest["v91_and_v92_error_arrays_reused"] and manifest["disagreement_component_added"]
    assert not manifest["conditional_means_changed"] and not manifest["error_covariance_component_changed"]
    assert not manifest["raw_covariance_component_changed"] and manifest["raw_weight"] == .24619454443573643
    assert not manifest["weight_refitted"] and not manifest["query_targets_used_for_covariance"]
    assert not manifest["readout_controls_actor"] and not manifest["evaluation_paths_loaded"]
    assert manifest["environment_steps"] == manifest["optimizer_updates"] == manifest["additional_counterfactual_steps"] == 0
