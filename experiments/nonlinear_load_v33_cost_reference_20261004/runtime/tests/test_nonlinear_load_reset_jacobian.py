from dataclasses import asdict
import random

import numpy as np
import pytest
import torch

from csbapr.losses.jacobian import compute_jacobian_loss
from scripts import test_lqr_convergence as training
from scripts.train_nonlinear_load_v17 import CANDIDATE, ResetBalancedJacobianAgent, make_agent, train_episode
from scripts.summarize_nonlinear_load_v17 import summarize
from csbapr.envs import nonlinear_load_env as nonlinear


def small_config():
    config = training.make_config(CANDIDATE, "cpu", "nonlinear_load")
    config.hidden_dim, config.num_critics, config.batch_size = 8, 2, 4
    config.buffer_size, config.warmup_steps = 64, 2
    return config


def test_reset_mixture_and_global_rng_isolation():
    training.seed_training(11)
    agent = ResetBalancedJacobianAgent(6, 2, small_config(), 11, episode_length=6)
    for index in range(15):
        state = np.full(6, index, dtype=np.float32)
        agent.replay_buffer.push(state, np.zeros(2), 0., state, float(index % 6 == 5))
    ordinary = torch.full((256, 6), -1.)
    numpy_rng, python_rng, torch_rng = np.random.get_state(), random.getstate(), torch.get_rng_state()
    mixed = agent._jacobian_loss_states(ordinary)
    torch.testing.assert_close(mixed[:128], ordinary[:128], rtol=0, atol=0)
    assert set(mixed[128:, 0].tolist()) == {0., 6., 12.}
    assert torch.equal(torch.get_rng_state(), torch_rng) and random.getstate() == python_rng
    current_rng = np.random.get_state()
    assert current_rng[0] == numpy_rng[0] and current_rng[2:] == numpy_rng[2:]
    np.testing.assert_array_equal(current_rng[1], numpy_rng[1])
    assert agent.jacobian_batches == 1 and agent.jacobian_reset_samples == agent.jacobian_replay_samples == 128
    assert sum(agent.jacobian_radius_samples.values()) == 128


def test_control_hook_preserves_native_loss_and_gradient():
    training.seed_training(11)
    agent = make_agent(small_config(), "replay", 11)
    agent.pi_sym_torch = training.known_reference("nonlinear_load")
    states = torch.randn(8, 6) / 4
    assert agent._jacobian_loss_states(states) is states
    expected = compute_jacobian_loss(agent.actor, agent.pi_sym_torch, states)
    actual = compute_jacobian_loss(agent.actor, agent.pi_sym_torch, agent._jacobian_loss_states(states))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    parameters = tuple(agent.actor.parameters())
    left = torch.autograd.grad(expected, parameters, allow_unused=True)
    right = torch.autograd.grad(actual, parameters, allow_unused=True)
    for a, b in zip(left, right):
        if a is None:
            assert b is None
        else:
            torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("sampling", ["replay", "reset-balanced"])
def test_resume_restores_sampling_and_next_training_episode(sampling, tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(training, "EP_LEN", 6)
    monkeypatch.setattr(nonlinear, "EP_LEN", 6)
    config = small_config()
    training.seed_training(11)

    def agent_for(seed):
        return (ResetBalancedJacobianAgent(6, 2, config, seed, episode_length=6)
                if sampling == "reset-balanced" else make_agent(config, sampling, seed))

    agent = agent_for(11)
    agent.pi_sym_torch = training.known_reference("nonlinear_load")
    env = training.make_env("nonlinear_load", 1, 11)
    env.action_space.seed(11)
    protocol = {"sampling": sampling, "config": asdict(config), "reset_cycle": [.2, .5, 1.]}
    history, steps = [], 0
    for episode in range(3):
        reward, steps = train_episode(env, agent, "balanced", episode, steps)
        history.append(reward)
    path = str(tmp_path / "last.pt")
    training.save_training_checkpoint(agent, env, path, next_ep=3, env_steps=steps, history=history, protocol=protocol)
    expected, expected_steps = train_episode(env, agent, "balanced", 3, steps)
    training.seed_training(999)
    resumed, restored_env = agent_for(999), training.make_env("nonlinear_load", 1, 999)
    restored_env.action_space.seed(999)
    episode, restored_steps, restored_history = training.restore_training_checkpoint(resumed, restored_env, path, protocol)
    actual, actual_steps = train_episode(restored_env, resumed, "balanced", episode, restored_steps)
    assert (episode, restored_history, actual, actual_steps) == (3, history, expected, expected_steps)
    assert resumed.training_steps == agent.training_steps and restored_env.amplitude == env.amplitude == .2
    for name in ("actor", "critic", "target_critic"):
        for a, b in zip(getattr(resumed, name).parameters(), getattr(agent, name).parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    np.testing.assert_array_equal(restored_env.disturbance, env.disturbance)
    if sampling == "reset-balanced":
        assert resumed.sampling_state() == agent.sampling_state()
    with pytest.raises(ValueError, match="exact protocol"):
        training.restore_training_checkpoint(resumed, restored_env, path, {**protocol, "sampling": "different"})


def test_first_sac_critic_update_is_identical_before_actor_intervention():
    config = small_config()

    def updated(sampling):
        training.seed_training(11)
        agent = make_agent(config, sampling, 11)
        agent.pi_sym_torch = training.known_reference("nonlinear_load")
        for index in range(8):
            state = np.full(6, index / 20., dtype=np.float32)
            agent.replay_buffer.push(state, np.full(2, .1), -.2, state / 2, 0.)
        agent.update()
        return agent

    control, intervention = updated("replay"), updated("reset-balanced")
    for a, b in zip(control.critic.parameters(), intervention.critic.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert control.alpha == intervention.alpha
    np.testing.assert_array_equal(control.belief_tracker.belief, intervention.belief_tracker.belief)


@pytest.mark.parametrize("failure", [None, "total", "early", "small", "fitted"])
def test_frozen_sampling_gates_keep_single_seed_failure(failure):
    runs = {(arm, seed): {"radius_results": {
        str(radius): {str(load): {"controllers": {"actor": {"mean_cost": 1. if arm == "reset-balanced" else 2.,
                                                  "mean_early_state_cost": 1. if arm == "reset-balanced" else 2.}},
                                 "paired_excess_costs": {"fitted": {"mean_total_cost": -.1}}}
                      for load in (1, 50)} for radius in (.2, .5, 1.)}}
            for arm in ("replay", "reset-balanced") for seed in (11, 12)}
    block = runs[("reset-balanced", 12)]["radius_results"]
    if failure in ("total", "early"):
        block["1.0"]["50"]["controllers"]["actor"]["mean_cost" if failure == "total" else "mean_early_state_cost"] = 2.
    elif failure == "small":
        block["0.2"]["50"]["controllers"]["actor"]["mean_cost"] = 2.01
    elif failure == "fitted":
        block["0.5"]["1"]["paired_excess_costs"]["fitted"]["mean_total_cost"] = 0.
    decision = summarize(runs, (11, 12))["decision"]
    assert decision["ready_for_learner_replication"] == (failure is None)
    assert decision["historical_v13_v15_hold_unchanged"]
