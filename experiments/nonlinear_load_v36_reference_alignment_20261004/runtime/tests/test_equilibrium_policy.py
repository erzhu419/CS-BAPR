from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from csbapr.agent import CSBAPRAgent
from csbapr.envs import random_load_env as load
from csbapr.losses.jacobian import deterministic_action, compute_jacobian_loss
from csbapr.networks.policy import EquilibriumGaussianPolicy, GaussianPolicy
from csbapr.sindy.load_feedback import known_load_reference
from scripts import test_lqr_convergence as runner

METHODS = (
    "csbapr-no-nau-no-target-shift", "csbapr-no-nau-no-sindy-no-target-shift",
    "csbapr-centered-mlp-no-target-shift", "csbapr-centered-mlp-no-sindy-no-target-shift",
)


def test_centered_forward_keeps_std_and_backpropagates_through_origin():
    ordinary = GaussianPolicy(6, 2, 8, activation="tanh")
    centered = EquilibriumGaussianPolicy(6, 2, 8, activation="tanh")
    centered.load_state_dict(ordinary.state_dict())
    states = torch.randn(5, 6)
    mean, std = ordinary(states)
    expected = mean - ordinary(torch.zeros(1, 6))[0]
    actual, actual_std = centered(states)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_std, std)
    expected_gradients = torch.autograd.grad(expected.square().mean(), tuple(ordinary.net.parameters()), retain_graph=True)
    actual_gradients = torch.autograd.grad(actual.square().mean(), tuple(centered.net.parameters()), retain_graph=True)
    for left, right in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(left, right)
    loss = compute_jacobian_loss(centered, known_load_reference(), states)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in centered.parameters() if p.grad is not None)
    assert np.isfinite(centered.compute_L_eff())


def test_zero_action_remains_zero_after_optimizer_update():
    actor = EquilibriumGaussianPolicy(6, 2, 8, activation="tanh")
    optimizer = torch.optim.Adam(actor.parameters(), lr=0.01)
    for _ in range(3):
        optimizer.zero_grad()
        action, log_prob, *_ = actor.evaluate(torch.randn(7, 6))
        (action.square().mean() + 0.1 * log_prob.mean()).backward()
        optimizer.step()
        torch.testing.assert_close(deterministic_action(actor, torch.zeros(4, 6)), torch.zeros(4, 2), rtol=0, atol=1e-6)


def test_factorial_configs_change_only_centering_and_symbolic_weights():
    configs = [asdict(runner.make_config(m, "cpu", "random_load")) for m in METHODS]
    base = configs[0]
    assert configs[1] == {**base, "weight_sym": 0.0, "jac_weight": 0.0}
    assert configs[2] == {**base, "actor_type": "centered-tanh-mlp"}
    assert configs[3] == {**configs[1], "actor_type": "centered-tanh-mlp"}
    assert all(not c["use_nau_actor"] and c["nau_reg_weight"] == c["weight_reg"] == c["action_anchor_weight"] == 0 for c in configs)
    assert [m in runner.SYMBOLIC_LQR_METHODS for m in METHODS] == [True, False, True, False]


@pytest.mark.parametrize("method", (
    "csbapr-no-target-shift", "bapr-no-target-shift", "resac-no-target-shift",
    "csbapr-no-sindy-no-target-shift", "csbapr-no-nau-no-target-shift",
))
def test_v6_presets_remain_unchanged(method):
    root = Path(__file__).resolve().parents[1]
    path = root / f"results/random_load_v6/{method}_seed0/random_load_v6_{method}_seed0_evaluation.json"
    saved = json.loads(path.read_text())["config"]
    current = runner.make_config(method, "cuda", "random_load")
    current.max_episodes = 500
    assert asdict(current) == saved


@pytest.mark.parametrize("method", METHODS)
def test_factorial_checkpoint_restores_next_update(method, tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(runner, "EP_LEN", 6)
    monkeypatch.setattr(load, "EP_LEN", 6)
    runner.seed_training(23)
    config = runner.make_config(method, "cpu", "random_load")
    config.hidden_dim, config.num_critics, config.batch_size = 8, 2, 4
    config.buffer_size, config.warmup_steps = 64, 2
    agent = CSBAPRAgent(6, 2, config)
    if method in runner.SYMBOLIC_LQR_METHODS:
        agent.pi_sym_torch = known_load_reference()
    env = load.RandomLoadEnv(50, 23)
    env.action_space.seed(23)
    protocol = {"method": method, "config": asdict(config)}
    first, steps = runner.run_episode(env, agent, train=True)
    path = str(tmp_path / "last.pt")
    runner.save_training_checkpoint(agent, env, path, next_ep=1, env_steps=steps, history=[first], protocol=protocol)
    expected, expected_steps = runner.run_episode(env, agent, train=True, env_steps=steps)
    resumed = CSBAPRAgent(6, 2, config)
    resumed_env = load.RandomLoadEnv(50, 999)
    _, restored_steps, _ = runner.restore_training_checkpoint(resumed, resumed_env, path, protocol)
    actual, actual_steps = runner.run_episode(resumed_env, resumed, train=True, env_steps=restored_steps)
    assert actual == expected and actual_steps == expected_steps
    for name in ("actor", "critic", "target_critic"):
        for actual_parameter, expected_parameter in zip(getattr(resumed, name).parameters(), getattr(agent, name).parameters()):
            torch.testing.assert_close(actual_parameter, expected_parameter, rtol=0, atol=0)
