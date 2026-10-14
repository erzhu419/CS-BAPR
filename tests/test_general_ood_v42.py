from copy import deepcopy
import json

import numpy as np
import pytest

from scripts import probe_general_ood_v42 as probe
from scripts import summarize_general_ood_v42 as summary


def synthetic_trace():
    rng = np.random.default_rng(42)
    states, actions = rng.normal(size=(1536, 2)), rng.normal(size=(1536, 1))
    delta = np.column_stack((.3 * states[:, 0] + .4 * actions[:, 0] ** 2,
                             -.2 * states[:, 1] + .7 * actions[:, 0]))
    delta += .05 * rng.normal(size=delta.shape)
    return {"states": states, "next_states": states + delta, "executed_actions": actions,
            "dones": np.zeros(1536, dtype=bool), "terminated": np.zeros(1536, dtype=bool),
            "truncated": np.zeros(1536, dtype=bool)}


def test_fit_and_calibration_do_not_read_validation_or_future():
    original = synthetic_trace()
    changed = deepcopy(original)
    for key in ("states", "next_states", "executed_actions"):
        changed[key][768:] += 50.
    for spec in probe.PREDICTORS.values():
        baseline = probe.DynamicsResidual(*spec).fit(original)
        future_changed = probe.DynamicsResidual(*spec).fit(changed)
        assert baseline.metadata() == future_changed.metadata()
        np.testing.assert_array_equal(baseline.coefficients, future_changed.coefficients)
        np.testing.assert_array_equal(baseline.scores(original)[:768], future_changed.scores(changed)[:768])
        np.testing.assert_allclose(baseline.scores(original)[512:768].mean(), 1.)


def test_quadratic_features_and_executed_action_control():
    trace = synthetic_trace()
    action = probe.DynamicsResidual(2, True).fit(trace)
    state = probe.DynamicsResidual(2, False).fit(trace)
    linear = probe.DynamicsResidual(1, True).fit(trace)
    assert action.metadata()["feature_count"] == 10
    assert state.metadata()["feature_count"] == 6
    assert linear.metadata()["feature_count"] == 4
    assert action.calibration_mse_in_target_units < state.calibration_mse_in_target_units / 10
    assert action.calibration_mse_in_target_units < linear.calibration_mse_in_target_units / 10


def test_residual_phases_keep_null_control_and_separate_end_causes():
    control = synthetic_trace()
    model = probe.DynamicsResidual(2, True).fit(control)
    original_weights = model.coefficients.copy()
    changed = deepcopy(control)
    changed["next_states"][1024:] += 2.
    changed["terminated"][1200] = changed["dones"][1200] = True
    changed["truncated"][-1] = changed["dones"][-1] = True
    baseline, observed = probe.phase_statistics(control, model), probe.phase_statistics(changed, model)
    assert baseline["validation256"] == observed["validation256"]
    assert observed["first_post16"]["mean_score"] > 100 * baseline["first_post16"]["mean_score"]
    assert observed["first_post256"]["terminated"] == 1
    assert observed["first_post256"]["truncated"] == 0
    assert observed["first_post256"]["nonterminal_transitions"] == 255
    assert observed["second_post256"]["truncated"] == 1
    np.testing.assert_array_equal(model.coefficients, original_weights)


def test_missing_policy_results_are_pending_not_completion(tmp_path):
    manifest = {"source_policies": ["fixed-minus2", "relative-adaptive"], "revision": probe.REVISION}
    result = summary.summarize(manifest, tmp_path, "formal")
    assert not result["matrix_complete"]
    assert result["decision"] == "pending-missing-policies"


def test_summary_rejects_action_coupling_future_refit_and_false_contrasts():
    manifest = json.loads((summary.STUDY / "dispatch_manifest.json").read_text())
    data = json.loads((summary.DESTINATION / "preflight" / f"{probe.REVISION}_fixed-minus2.json").read_text())
    for key in ("auxiliary_signal_controls_actions", "predictors_updated_after_calibration"):
        changed = deepcopy(data)
        changed[key] = True
        with pytest.raises(AssertionError):
            summary.validate_run(changed, manifest, "preflight")
    changed = deepcopy(data)
    changed["records"][1]["change_minus_nochange"]["quadratic-state-action"]["first_post256"]["mean_score"] += 1.
    with pytest.raises(AssertionError):
        summary.validate_run(changed, manifest, "preflight")


def test_summary_retains_null_forecasts_early_response_and_end_causes():
    manifest = json.loads((summary.STUDY / "dispatch_manifest.json").read_text())
    result = summary.summarize(manifest, summary.DESTINATION, "preflight")
    assert result["matrix_complete"] and result["optimizer_updates"] == result["new_training_seeds"] == 0
    for policy in manifest["source_policies"]:
        method = result["methods"][policy]
        assert "validation256" in method["null"]["quadratic-state-action"]
        assert method["profiles"]["high_grav"]["predictors"]["quadratic-state-action"]["first_post16"]["first_transition_score"]["pairs"] == 2
        assert "actual_truncated" in method["profiles"]["high_grav"]["endings"]["second_post256"]
