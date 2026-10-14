from copy import deepcopy
import json

import numpy as np
import pytest

from scripts import probe_general_ood_v43 as probe
from scripts import summarize_general_ood_v43 as summary
from tests.test_general_ood_v42 import synthetic_trace


def reference(seed):
    trace = synthetic_trace()
    rng = np.random.default_rng(seed)
    trace["states"] = rng.normal(size=(1536, 2))
    trace["executed_actions"] = rng.normal(size=(1536, 1))
    x, a = trace["states"], trace["executed_actions"][:, 0]
    trace["next_states"] = x + np.column_stack((.3 * x[:, 0] + .4 * a ** 2, -.2 * x[:, 1] + .7 * a)) + .05 * rng.normal(size=x.shape)
    trace["dones"][900] = trace["terminated"][900] = True
    return trace


def test_episode_age_is_past_only_and_independent_rollouts_start_at_zero():
    dones = np.array([False, False, True, False, False, False])
    np.testing.assert_array_equal(probe.episode_steps(dones), [0, 1, 2, 0, 1, 2])
    changed = dones.copy()
    changed[4:] = True
    np.testing.assert_array_equal(probe.episode_steps(changed)[:5], probe.episode_steps(dones)[:5])
    traces = [reference(1), reference(2)]
    combined = probe.concatenate_references(traces)
    assert combined["episode_steps"][1535] == 634
    assert combined["episode_steps"][1536] == 0
    np.testing.assert_array_equal(combined["next_states"][900], traces[0]["next_states"][900])


def test_centered_residual_detects_bias_shift_that_reduces_uncentered_error():
    ages = np.arange(1024)
    errors = np.tile([[1.9, 1.9], [2.1, 2.1]], (512, 1))
    raw = probe.ResidualCalibration("global-rms", errors, ages, np.ones(2))
    centered = probe.ResidualCalibration("global-centered", errors, ages, np.ones(2))
    shifted = np.full((1, 2), 1.5)
    raw_score, _ = raw.scores(shifted, np.array([0]))
    centered_score, threshold = centered.scores(shifted, np.array([0]))
    assert raw_score[0] < 1.
    assert centered_score[0] > 20 * threshold[0]


def test_episode_centered_calibration_uses_all_four_fixed_age_groups():
    rng = np.random.default_rng(1)
    ages = np.arange(1536)
    groups = np.minimum(ages // 256, 3)
    errors = (groups + 1)[:, None] + rng.normal(scale=.1, size=(1536, 2))
    calibration = probe.ResidualCalibration("episode-centered", errors, ages, np.ones(2))
    assert calibration.counts == [256, 256, 256, 768]
    scores, thresholds = calibration.scores(errors, ages)
    for group in range(4):
        np.testing.assert_allclose(scores[groups == group].mean(), 1.)
        assert (thresholds[groups == group] == calibration.thresholds[group]).all()


def test_calibration_cannot_refit_predictor_and_test_never_updates_either():
    fitting = probe.concatenate_references([reference(1), reference(2)])
    calibration = probe.concatenate_references([reference(3), reference(4)])
    model = probe.ReferenceDynamics(2, True).fit_references(fitting, calibration)
    changed_calibration = deepcopy(calibration)
    changed_calibration["next_states"] += .2
    other = probe.ReferenceDynamics(2, True).fit_references(fitting, changed_calibration)
    np.testing.assert_array_equal(model.coefficients, other.coefficients)
    np.testing.assert_array_equal(model.input_scale, other.input_scale)
    metadata, weights = model.metadata(), model.coefficients.copy()
    control = reference(5)
    changed = deepcopy(control)
    changed["next_states"][1024:] += 2.
    baseline, _ = probe.observe_trace(control, {"quadratic-state-action": model})
    actual, _ = probe.observe_trace(changed, {"quadratic-state-action": model})
    for channel in probe.CALIBRATORS:
        assert baseline["quadratic-state-action"][channel]["validation256"] == actual["quadratic-state-action"][channel]["validation256"]
        assert actual["quadratic-state-action"][channel]["first_post16"]["mean_score"] > 100 * baseline["quadratic-state-action"][channel]["first_post16"]["mean_score"]
    assert model.metadata() == metadata
    np.testing.assert_array_equal(model.coefficients, weights)


def test_missing_policy_matrix_is_pending(tmp_path):
    manifest = {"source_policies": ["fixed-minus2", "relative-adaptive"], "revision": probe.REVISION}
    result = summary.summarize(manifest, tmp_path, "formal")
    assert not result["matrix_complete"] and result["decision"] == "pending-missing-policies"


def test_operational_and_formal_reference_seeds_are_disjoint_from_tests():
    for stage in ("preflight", "formal"):
        fitting, calibration, tests = probe.stage_seeds(stage)
        combined = fitting + calibration + tests
        assert len(set(combined)) == len(combined)
        assert len({s % 10000 for s in combined}) == len(combined)


def test_summary_rejects_non_normal_reference_refit_and_false_pairing():
    manifest = json.loads((summary.STUDY / "dispatch_manifest.json").read_text())
    data = json.loads((summary.DESTINATION / "preflight" / f"{probe.REVISION}_fixed-minus2.json").read_text())
    changed = deepcopy(data)
    changed["references"][0]["mode"] = "heavy"
    with pytest.raises(AssertionError):
        summary.validate_run(changed, manifest, "preflight")
    changed = deepcopy(data)
    changed["predictors_updated_after_calibration"] = True
    with pytest.raises(AssertionError):
        summary.validate_run(changed, manifest, "preflight")
    changed = deepcopy(data)
    changed["records"][1]["change_minus_nochange"]["quadratic-state-action"]["episode-centered"]["validation256"]["mean_score"] = 1.
    with pytest.raises(AssertionError):
        summary.validate_run(changed, manifest, "preflight")


def test_summary_retains_coverage_comparison_and_all_calibrations():
    manifest = json.loads((summary.STUDY / "dispatch_manifest.json").read_text())
    result = summary.summarize(manifest, summary.DESTINATION, "preflight")
    assert result["matrix_complete"] and result["optimizer_updates"] == result["new_training_seeds"] == 0
    for method in result["methods"].values():
        assert method["reference_rollouts"] == 4
        assert "v42_fraction_above_calibration95" in method["v42_coverage_comparison"]["quadratic-state-action"]["first_post256"]
        assert set(method["null"]["quadratic-state-action"]) == set(probe.CALIBRATORS)
        assert method["profiles"]["high_grav"]["predictors"]["quadratic-state-action"]["episode-centered"]["first_post16"]["mean_score"]["pairs"] == 2
