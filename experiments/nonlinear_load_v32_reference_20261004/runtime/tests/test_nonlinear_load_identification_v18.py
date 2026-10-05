import copy

import numpy as np

from csbapr.envs.lqr_env import A, B
from scripts.diagnose_nonlinear_load_v18 import (
    RADII, SEVERITIES, decide, fit_models, local_gain, physical_metrics,
)
from scripts.preflight_nonlinear_load_v9 import local_linear_gain


def synthetic_transitions():
    rng = np.random.default_rng(18)
    x = np.hstack((rng.normal(0, .2, (1000, 4)), rng.uniform(-.05, .05, (1000, 2))))
    u = rng.uniform(-1, 1, (1000, 2))
    next_x = x.copy()
    next_x[:, :4] = x[:, :4] @ A.T + (2.5 * (u + u ** 3) + x[:, 4:]) @ B.T
    next_x[:, 4:] = rng.uniform(-.05, .05, (1000, 2))
    states = [np.stack((s, sn)) for s, sn in zip(x, next_x)]
    actions = [action[None] for action in u]
    return states, actions, x, u, next_x - x


def test_cubic_fit_recovers_local_feedback_without_changing_data():
    states, actions, x, u, y = synthetic_transitions()
    original_states = [s.copy() for s in states]
    original_actions = [a.copy() for a in actions]
    models = fit_models(states, actions)
    np.testing.assert_allclose(local_gain(models['cubic']), local_linear_gain(), rtol=0, atol=1e-7)
    assert physical_metrics(models['cubic'], x, u, y)['physical_rmse'] < 1e-10
    assert physical_metrics(models['linear'], x, u, y)['physical_rmse'] > 1e-3
    for old, current in zip(original_states + original_actions, states + actions):
        np.testing.assert_array_equal(old, current)


def test_physical_validation_does_not_score_stochastic_load_innovations():
    states, actions, x, u, y = synthetic_transitions()
    model = fit_models(states, actions)['cubic']
    expected = physical_metrics(model, x, u, y)
    y[:, 4:] = 1000
    assert physical_metrics(model, x, u, y) == expected


def test_decision_preserves_individual_seed_and_block_failures():
    run = {'validation': {str(s): {'linear': {'physical_rmse': .1}, 'cubic': {'physical_rmse': .001}}
                          for s in SEVERITIES},
           'fits': {'cubic': {'parameter_errors': {'load_gain_max_abs_error': .0001}}},
           'radius_results': {str(r): {str(s): {'cubic_minus_linear': {'mean_total_cost': -.1}}
                                      for s in SEVERITIES} for r in RADII}}
    assert decide([run, copy.deepcopy(run)])['ready_for_matched_actor_experiment']
    bad = copy.deepcopy(run)
    bad['radius_results']['0.2']['50']['cubic_minus_linear']['mean_total_cost'] = .01
    assert not decide([run, bad])['direct_feedback_improved_all_blocks']
    bad = copy.deepcopy(run)
    bad['validation']['50']['cubic']['physical_rmse'] = .2
    assert not decide([run, bad])['physical_validation_improved_each_seed_and_load']
    bad = copy.deepcopy(run)
    bad['fits']['cubic']['parameter_errors']['load_gain_max_abs_error'] = .002
    assert not decide([run, bad])['local_load_gain_recovered_each_seed']
