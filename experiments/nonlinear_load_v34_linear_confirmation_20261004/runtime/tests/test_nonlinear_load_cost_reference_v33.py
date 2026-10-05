from copy import deepcopy

import numpy as np
import pytest

from csbapr.envs.lqr_env import A, B, R, ACTION_SCALE
from csbapr.envs.nonlinear_load_env import local_linear_gain
from csbapr.envs.sustained_load_env import SustainedLoadEnv
from scripts.diagnose_nonlinear_load_v33 import CostReference, COMPARATORS
from scripts.summarize_nonlinear_load_v33 import decide


def reference():
    fit = {"dynamics": A.tolist(), "load_matrix": B.tolist(),
           "linear_action_matrix": (2.5 * B).tolist(), "cubic_action_matrix": (2.5 * B).tolist()}
    return CostReference(fit, SustainedLoadEnv.specification())


def test_objective_matches_command_cost_and_full_next_value():
    ref = reference()
    states = np.random.default_rng(33).normal(size=(15, 6))
    actions = np.random.default_rng(34).uniform(-1, 1, (15, 2))
    baseline = states @ ref.dynamics.T
    following = baseline + actions @ ref.linear.T + actions ** 3 @ ref.cubic.T
    truth = (np.einsum("bi,ij,bj->b", following, ref.value, following)
             - np.einsum("bi,ij,bj->b", baseline, ref.value, baseline)
             + np.einsum("bi,ij,bj->b", actions, R * ACTION_SCALE ** 2, actions))
    np.testing.assert_allclose(ref.objective(states, actions).sum(1), truth, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(ref.gain[:, :4], local_linear_gain()[:, :4], atol=1e-10)


def test_global_solver_including_nonconvex_states_and_endpoints():
    ref = reference()
    states = np.random.default_rng(35).normal(size=(30, 6))
    states[:, 4:] *= 10
    actions = ref(states)
    assert np.max(np.abs(actions)) <= 1 and np.any(np.abs(actions) == 1)
    for state, action in zip(states, actions):
        grid = np.repeat(np.linspace(-1, 1, 10001)[:, None], 2, axis=1)
        grid_cost = ref.objective(state[None], grid).min(axis=0)
        assert np.all(ref.objective(state[None], action[None])[0] <= grid_cost + 1e-10)
    assert ref.action_components_solved == 60 and ref.maximum_surrogate_excess <= 1e-9


def test_implicit_jacobian_and_matched_origin():
    ref = reference()
    np.testing.assert_allclose(ref.solve(np.zeros((1, 6))), 0., atol=1e-14)
    np.testing.assert_allclose(ref.jacobian(np.zeros((1, 6)))[0], -ref.gain, atol=1e-12)
    states = np.random.default_rng(36).normal(size=(8, 6)) * .05
    for column in range(6):
        shift = np.zeros(6)
        shift[column] = 1e-5
        finite = (ref.solve(states + shift) - ref.solve(states - shift)) / 2e-5
        np.testing.assert_allclose(ref.jacobian(states)[:, :, column], finite, rtol=1e-6, atol=1e-8)
    saturated = np.asarray([[.5, .5, .5, .5, 4., 4.]])
    np.testing.assert_array_equal(ref.jacobian(saturated), 0.)


def test_candidate_uses_id_cubic_dynamics_not_oracle_actuator():
    ref = reference()
    fit = {"dynamics": A.tolist(), "load_matrix": B.tolist(),
           "linear_action_matrix": (2.5 * B).tolist(), "cubic_action_matrix": (B @ np.diag([1., 4.])).tolist()}
    changed = CostReference(fit, SustainedLoadEnv.specification())
    states = np.asarray([[.04, -.06, .02, .04, .06, -.04]])
    assert np.max(np.abs(ref.solve(states))) < 1
    assert not np.allclose(changed.solve(states), ref.solve(states))
    np.testing.assert_array_equal(changed.gain, ref.gain)


@pytest.mark.parametrize("failure", (None, *COMPARATORS))
def test_gate_preserves_each_seed_and_both_comparators(failure):
    run = {"radius_results": {str(radius): {str(load): {"nonlinear_minus_baselines": {
        name: {"mean_total_cost_change": -.1} for name in COMPARATORS}}
        for load in (1, 50)} for radius in (.2, .5, 1.)}}
    runs = [deepcopy(run), deepcopy(run)]
    if failure:
        runs[1]["radius_results"]["0.2"]["1"]["nonlinear_minus_baselines"][failure]["mean_total_cost_change"] = 0.
    result = decide(runs)
    assert result["reference_intervention_admitted"] == (failure is None)
    assert not result["learner_replication_admitted"] and result["historical_hold_unchanged"]
