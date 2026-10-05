import copy

import numpy as np
import pytest

from csbapr.envs.lqr_env import A, B, Q, R, ACTION_SCALE, get_optimal_gain
from csbapr.envs.nonlinear_load_env import NonlinearLoadEnv, actuator_force, inverse_force
from csbapr.envs.sustained_load_env import SustainedLoadEnv
from scripts.diagnose_nonlinear_load_v30 import CONTROLLERS, PROFILES, REVISION, control_actions, qualify_radius, rollout
from scripts.summarize_nonlinear_load_v30 import validate


@pytest.mark.parametrize("severity", (1,50))
def test_active_bounded_load_keeps_original_physics_and_latent_stream(severity):
    env, old = SustainedLoadEnv(severity, 411), NonlinearLoadEnv(severity, 411)
    observation, _ = env.reset()
    other, _ = old.reset()
    np.testing.assert_array_equal(observation[:4], other[:4])
    action = np.array([.2,-.8])
    for step in range(200):
        physical, applied = env.state.copy(), env.disturbance.copy()
        assert np.min(np.abs(applied)) >= .55 * .08 * severity and np.max(np.abs(applied)) <= .08 * severity
        observation, reward, done, truncated, info = env.step(action)
        old.step(-action)
        np.testing.assert_allclose(env.state, A @ physical + B @ (actuator_force(action) + applied))
        assert -reward == pytest.approx(physical @ Q @ physical + (ACTION_SCALE * action) @ R @ (ACTION_SCALE * action))
        np.testing.assert_array_equal(info['applied_disturbance'], applied)
        np.testing.assert_array_equal(observation[4:], env.disturbance.astype(np.float32))
        assert done == (step == 199) and not truncated
    assert env._rng.bit_generator.state == old._rng.bit_generator.state


def test_force_paths_are_action_independent_and_severity_is_only_a_scale():
    small, large, other = [SustainedLoadEnv(s, 73) for s in (1,50,50)]
    for env in (small,large,other):
        env.reset()
    for _ in range(200):
        np.testing.assert_allclose(large.disturbance, 50 * small.disturbance)
        np.testing.assert_array_equal(large.disturbance, other.disturbance)
        small.step([0.,0.]); large.step([1.,1.]); other.step([-1.,-1.])


def test_current_force_cancellation_is_feasible_and_legacy_inverse_unchanged():
    observations = np.array([[0.,0.,0.,0.,4.,-4.],[.1,-.2,.3,-.4,2.,-3.]])
    exact = control_actions('force_cancel_feedback', observations)
    np.testing.assert_allclose(actuator_force(exact[0]) + observations[0,4:], 0, atol=1e-10)
    assert np.max(np.abs(exact)) <= 1
    legacy = control_actions('known_inverse', observations)
    expected = np.stack([inverse_force(5 * np.tanh(-get_optimal_gain() @ o[:4] - o[4:] / 5)) for o in observations])
    np.testing.assert_allclose(legacy, expected, atol=1e-12)


def test_window_accounting_pairing_and_validator_rejects_changed_evidence():
    seeds = [61000,61001]
    results = {p:{str(s):{m:rollout(p,m,.2,s,seeds) for m in CONTROLLERS} for s in (1,50)} for p in PROFILES}
    report = {'revision':REVISION,'radius':.2,'preflight_only':True,
              'environment_settings':{p:cls.specification() for p,cls in PROFILES.items()},
              'evaluation_seeds':seeds,'controllers':list(CONTROLLERS),'severities':[1,50],'window_steps':50,
              'controller_episodes':40,'evaluation_interactions':8000,'rl_training_runs':0,'optimizer_updates':0,
              'fitting_interactions':0,'historical_hold_unchanged':True,'reference_is_optimal':False,
              'runtime':{'device':'cpu','numpy':'2.2.6','scipy':'1.15.3'},
              'results':results,'qualification':qualify_radius(results,.2)}
    validate(report,.2,True)
    altered = copy.deepcopy(report)
    altered['results']['sustained']['50']['zero']['force_rms'][0] += .01
    with pytest.raises(AssertionError):
        validate(altered,.2,True)
    altered = copy.deepcopy(report)
    altered['results']['sustained']['50']['force_cancel_feedback']['mean_cost'] += 1
    with pytest.raises(AssertionError):
        validate(altered,.2,True)


def test_qualification_rejects_an_initial_only_or_uncontrollable_task():
    baseline = {'mean_cost':1.,'maximum_physical_state_norms':[.5],
                'saturated_component_fractions':[0.], 'late_cost_share':.7,
                'mean_state_costs_by_window':[.1,.01,.01,.01]}
    results = {'sustained':{str(s):{'force_cancel_feedback':copy.deepcopy(baseline),
                                  'zero':{'mean_cost':20.},
                                  'no_load_feedback':{'mean_state_costs_by_window':[.1,1.,1.,1.]}}
                           for s in (1,50)}}
    assert qualify_radius(results,.2)['qualified']
    results['sustained']['50']['force_cancel_feedback']['late_cost_share'] = .1
    assert not qualify_radius(results,.2)['qualified']
    results['sustained']['50']['force_cancel_feedback']['late_cost_share'] = .7
    results['sustained']['50']['force_cancel_feedback']['maximum_physical_state_norms'] = [3.]
    assert not qualify_radius(results,.2)['qualified']
