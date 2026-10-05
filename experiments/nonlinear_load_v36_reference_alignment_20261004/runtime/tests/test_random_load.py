from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from csbapr.agent import CSBAPRAgent
from csbapr.envs.lqr_env import A, B, ACTION_SCALE, get_optimal_gain
from csbapr.envs import random_load_env as load
from csbapr.sindy.load_feedback import fitted_load_reference, known_load_reference
from scripts import test_lqr_convergence as runner

METHODS = (
    "csbapr-no-target-shift", "bapr-no-target-shift", "resac-no-target-shift",
    "csbapr-no-sindy-no-target-shift", "csbapr-no-nau-no-target-shift",
)


@pytest.mark.parametrize("severity", load.SEVERITIES)
def test_force_stream_is_observed_bounded_and_action_independent(severity):
    left, right = load.RandomLoadEnv(severity, 41), load.RandomLoadEnv(severity, 41)
    obs, _ = left.reset()
    other, _ = right.reset()
    np.testing.assert_array_equal(obs, other)
    assert np.linalg.norm(obs[:4]) <= load.INITIAL_AMPLITUDE
    for step in range(200):
        physical, force = left.state.copy(), left.disturbance.copy()
        next_obs, _, done, truncated, info = left.step(np.ones(2))
        other, _, _, _, _ = right.step(-np.ones(2))
        np.testing.assert_allclose(left.state, A @ physical + B @ (ACTION_SCALE + force))
        np.testing.assert_array_equal(info["applied_disturbance"], force)
        np.testing.assert_array_equal(next_obs[4:], other[4:])
        assert np.max(np.abs(next_obs[4:])) <= left.force_cap + 1e-7
        assert done == (step == 199) and not truncated


def test_fitted_reference_uses_physics_not_unpredictable_force_rows():
    powers = np.vstack((np.zeros((1, 8), dtype=int), np.eye(8, dtype=int)))
    coefficients = np.zeros((6, 9))
    coefficients[:4, 1:5] = A - np.eye(4)
    coefficients[:4, 5:7] = B
    coefficients[:4, 7:] = B * ACTION_SCALE
    coefficients[4:, 0] = 0.1  # Stochastic force innovations need not have zero fitted intercept.
    model = SimpleNamespace(n_state=6, n_control=2, poly_degree=1, discrete_time=True,
                            coeffs=coefficients, library=SimpleNamespace(powers_=powers))
    fitted = fitted_load_reference(model)
    np.testing.assert_allclose(fitted.gain.numpy(), known_load_reference().gain.numpy(), atol=1e-6)
    assert fitted.gain.shape == (2, 6)
    np.testing.assert_allclose(fitted.gain[:, :4], get_optimal_gain(), atol=1e-6)


@pytest.mark.parametrize("method", METHODS)
def test_load_presets_keep_v5_coefficients(method):
    original = asdict(runner.make_config(method, "cpu"))
    actual = asdict(runner.make_config(method, "cpu", "random_load"))
    assert actual == {**original, "state_dim": 6, "env_name": "RandomLoad-v0"}


@pytest.mark.parametrize("method", METHODS)
def test_random_load_resume_reproduces_next_episode(method, tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(runner, "EP_LEN", 6)
    monkeypatch.setattr(load, "EP_LEN", 6)
    runner.seed_training(17)
    config = runner.make_config(method, "cpu", "random_load")
    config.hidden_dim, config.num_critics, config.batch_size = 8, 2, 4
    config.buffer_size, config.warmup_steps = 64, 2
    agent = CSBAPRAgent(6, 2, config)
    if method in runner.SYMBOLIC_LQR_METHODS:
        agent.pi_sym_torch = known_load_reference()
    env = load.RandomLoadEnv(50, 17)
    env.action_space.seed(17)
    protocol = {"method": method, "config": asdict(config)}
    first, steps = runner.run_episode(env, agent, train=True)
    path = str(tmp_path / "last.pt")
    runner.save_training_checkpoint(agent, env, path, next_ep=1, env_steps=steps,
                                    history=[first], protocol=protocol)
    expected, expected_steps = runner.run_episode(env, agent, train=True, env_steps=steps)
    runner.seed_training(999)
    resumed = CSBAPRAgent(6, 2, config)
    resumed_env = load.RandomLoadEnv(50, 999)
    resumed_env.action_space.seed(999)
    _, restored_steps, _ = runner.restore_training_checkpoint(resumed, resumed_env, path, protocol)
    actual, actual_steps = runner.run_episode(resumed_env, resumed, train=True, env_steps=restored_steps)
    assert actual == expected and actual_steps == expected_steps
    assert resumed.training_steps == agent.training_steps
    for name in ("actor", "critic", "target_critic"):
        for actual, expected in zip(getattr(resumed, name).parameters(), getattr(agent, name).parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    np.testing.assert_array_equal(resumed_env.disturbance, env.disturbance)
    if method in runner.SYMBOLIC_LQR_METHODS:
        torch.testing.assert_close(resumed.pi_sym_torch.gain, agent.pi_sym_torch.gain, rtol=0, atol=0)
