from copy import deepcopy

import numpy as np
import pytest
from scipy.optimize import brentq

from csbapr.envs.lqr_env import B
from csbapr.envs.nonlinear_load_env import local_linear_gain
from scripts.diagnose_nonlinear_load_v30 import rollout
from scripts.diagnose_nonlinear_load_v32 import FittedInverseReference
from scripts.summarize_nonlinear_load_v32 import decide


def reference(linear=(2.5, 2.5), cubic=(2.5, 2.5)):
    return FittedInverseReference({"gain": local_linear_gain().tolist(), "load_matrix": B.tolist(),
                                  "linear_action_matrix": (B @ np.diag(linear)).tolist(),
                                  "cubic_action_matrix": (B @ np.diag(cubic)).tolist()})


@pytest.mark.parametrize("linear,cubic", (((2.5, 2.5), (2.5, 2.5)), ((1.5, 2.), (3., 1.))))
def test_fitted_cubic_inverse_matches_scalar_solver_and_support(linear, cubic):
    ref = reference(linear, cubic)
    force = np.linspace(-1, 1, 201)[:, None] * ref.cap
    action = ref.inverse(force)
    np.testing.assert_allclose(ref.linear * action + ref.cubic * action ** 3, force, rtol=1e-12, atol=1e-12)
    for i in (0, 12, 100, 189, 200):
        for j in range(2):
            expected = brentq(lambda u: linear[j] * u + cubic[j] * u ** 3 - force[i, j], -1, 1)
            assert np.isclose(action[i, j], expected, rtol=1e-10, atol=1e-12)
    assert np.max(np.abs(action)) <= 1 + 1e-15
    assert np.max(np.abs(ref(np.full((3, 6), 100.)))) <= 1 + 1e-15


def test_origin_and_full_action_jacobian_preserve_local_feedback():
    ref = reference()
    np.testing.assert_allclose(ref.jacobian(np.zeros((1, 6)))[0], -ref.gain, rtol=1e-12, atol=1e-12)
    states = np.random.default_rng(31).normal(size=(12, 6))
    states[:, 4:] *= 3.
    jacobian = ref.jacobian(states)
    for column in range(6):
        shift = np.zeros(6)
        shift[column] = 1e-5
        finite = (ref(states + shift) - ref(states - shift)) / 2e-5
        np.testing.assert_allclose(jacobian[:, :, column], finite, rtol=1e-6, atol=1e-9)
    np.testing.assert_array_equal(ref(np.zeros((1, 6))), np.zeros((1, 2)))


def test_reference_consumes_fitted_coefficients_not_oracle_actuator():
    states = np.asarray([[.2, -.3, .1, .2, 3., -2.]])
    original, changed = reference(), reference((2., 1.5), (3., 2.))
    assert not np.allclose(original(states), changed(states))
    np.testing.assert_allclose(changed.jacobian(np.zeros((1, 6)))[0], -changed.gain, atol=1e-12)


def test_custom_control_keeps_v30_costs_and_load_paths():
    default = rollout("sustained", "zero", .2, 50, [63000, 63001])
    custom = rollout("sustained", "custom", .2, 50, [63000, 63001],
                     action_fn=lambda states: np.zeros((len(states), 2)))
    assert default == custom
    fitted = rollout("sustained", "inverse", .2, 50, [63000, 63001], action_fn=reference())
    for key in ("force_rms", "maximum_load_components", "force_change_counts"):
        assert fitted[key] == default[key]
    assert fitted["mean_cost"] < default["mean_cost"]


@pytest.mark.parametrize("failure", (None, "id", "ood"))
def test_diagnostic_gate_preserves_individual_seed_failures(failure):
    runs = [{"radius_results": {str(radius): {str(load): {"inverse_minus_local": {
        "mean_total_cost_change": -.1}} for load in (1, 50)} for radius in (.2, .5, 1.)}} for _ in (21, 22)]
    if failure:
        runs[-1]["radius_results"]["0.2"]["1" if failure == "id" else "50"]["inverse_minus_local"]["mean_total_cost_change"] = 0.
    decision = decide(runs)
    assert decision["reference_intervention_admitted"] == (failure is None)
    assert not decision["learner_replication_admitted"] and decision["historical_hold_unchanged"]
