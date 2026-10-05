import numpy as np
import pytest

from csbapr.envs.lqr_env import A, B, Q, R, ACTION_SCALE, get_optimal_gain
from csbapr.envs.nonlinear_load_env import NonlinearLoadEnv, actuator_force
from csbapr.envs.random_load_env import RandomLoadEnv, SEVERITIES
from scripts.preflight_nonlinear_load_v9 import inverse_force


def test_actuator_preserves_equilibrium_and_authority_but_is_nonlinear():
    actions = np.linspace(-1, 1, 101)
    force = actuator_force(actions)
    assert force[0] == -5 and force[50] == 0 and force[-1] == 5
    assert np.all(np.diff(force) > 0)
    np.testing.assert_allclose(force, -actuator_force(-actions))
    assert actuator_force(0.5) == 1.5625
    assert actuator_force(0.5) != 0.5 * actuator_force(1)


@pytest.mark.parametrize("severity", SEVERITIES)
def test_nonlinear_physics_command_cost_and_paired_force_stream(severity):
    nonlinear, linear = NonlinearLoadEnv(severity, 451), RandomLoadEnv(severity, 451)
    observation, _ = nonlinear.reset()
    other, _ = linear.reset()
    np.testing.assert_array_equal(observation, other)
    action = np.array([0.37, -0.82])
    for step in range(200):
        state, force = nonlinear.state.copy(), nonlinear.disturbance.copy()
        observation, reward, done, truncated, info = nonlinear.step(action)
        other, _, _, _, _ = linear.step(-action)
        np.testing.assert_allclose(nonlinear.state, A @ state + B @ (actuator_force(action) + force))
        assert -reward == pytest.approx(state @ Q @ state + (ACTION_SCALE * action) @ R @ (ACTION_SCALE * action))
        np.testing.assert_array_equal(observation[4:], other[4:])
        np.testing.assert_array_equal(info["applied_disturbance"], force)
        assert done == (step == 199) and not truncated


def test_inverse_reproduces_the_linear_reference_physics_with_command_cost():
    np.testing.assert_allclose(actuator_force(inverse_force(np.array([-5., -2.5, 0., 2.5, 5.]))),
                               [-5., -2.5, 0., 2.5, 5.], atol=1e-10)
    nonlinear, linear = NonlinearLoadEnv(50, 801), RandomLoadEnv(50, 801)
    observation, _ = linear.reset()
    nonlinear.reset()
    for _ in range(200):
        action = np.tanh(-get_optimal_gain() @ observation[:4] - observation[4:] / ACTION_SCALE)
        observation, linear_reward, _, _, _ = linear.step(action)
        _, nonlinear_reward, _, _, _ = nonlinear.step(inverse_force(ACTION_SCALE * action))
        np.testing.assert_allclose(nonlinear.state, linear.state, atol=1e-8)
        assert -nonlinear_reward >= -linear_reward - 1e-8


def test_zero_state_and_zero_applied_load_remain_a_physical_equilibrium():
    env = NonlinearLoadEnv(50, 1)
    env.reset()
    env.state[:] = 0
    env.disturbance[:] = 0
    _, reward, _, _, info = env.step(np.zeros(2))
    np.testing.assert_array_equal(env.state, np.zeros(4))
    np.testing.assert_array_equal(info["actuator_force"], np.zeros(2))
    assert reward == 0
