from dataclasses import asdict

import numpy as np
import pytest
import torch

from csbapr.envs import nonlinear_load_env as nonlinear
from csbapr.losses.jacobian import compute_jacobian_loss
from scripts import test_lqr_convergence as training
from scripts.train_nonlinear_load_v17 import CANDIDATE, ResetBalancedJacobianAgent, train_episode
from scripts.train_nonlinear_load_v19 import ReferenceAgent, identify_reference
from scripts.summarize_nonlinear_load_v19 import summarize


def small_config():
    config = training.make_config(CANDIDATE, "cpu", "nonlinear_load")
    config.hidden_dim, config.num_critics, config.batch_size = 8, 2, 4
    config.buffer_size, config.warmup_steps = 64, 2
    return config


def identification_env(seed):
    env = training.make_env("nonlinear_load", 1, seed)
    env.action_space.seed(seed)
    return env


@pytest.mark.parametrize("seed", [13, 14])
def test_linear_arm_matches_native_identification_and_reference_gradient(seed):
    torch.set_num_threads(1)
    config = small_config()
    training.seed_training(seed)
    native = ResetBalancedJacobianAgent(6, 2, config, seed)
    native.sindy_preidentify(identification_env(seed), reference_builder=lambda m: training.fitted_reference(m, "nonlinear_load"))
    training.seed_training(seed)
    control = ReferenceAgent(config, seed)
    identify_reference(control, identification_env(seed), "linear")
    assert control._sindy_report == native._sindy_report
    torch.testing.assert_close(control.pi_sym_torch.gain, native.pi_sym_torch.gain, rtol=0, atol=0)
    torch.testing.assert_close(control.f_sym_torch.coeffs, native.f_sym_torch.coeffs, rtol=0, atol=0)
    states = torch.randn(8, 6) / 4
    left = compute_jacobian_loss(control.actor, control.pi_sym_torch, states)
    right = compute_jacobian_loss(native.actor, native.pi_sym_torch, states)
    torch.testing.assert_close(left, right, rtol=0, atol=0)
    for a, b in zip(torch.autograd.grad(left, tuple(control.actor.parameters()), allow_unused=True),
                    torch.autograd.grad(right, tuple(native.actor.parameters()), allow_unused=True)):
        if a is None:
            assert b is None
        else:
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_identification_is_matched_and_initial_critic_alpha_update_is_unchanged():
    config = small_config()
    agents = []
    for reference in ("linear", "cubic-local"):
        training.seed_training(13)
        agent = ReferenceAgent(config, 13, episode_length=6)
        identify_reference(agent, identification_env(13), reference)
        for index in range(8):
            state = np.full(6, index / 20., dtype=np.float32)
            agent.replay_buffer.push(state, np.full(2, .1), -.2, state / 2, 0.)
        training.seed_training(13)
        agent.update()
        agents.append(agent)
    control, modified = agents
    for key in ("gains", "fits", "validation"):
        assert control.reference_identification[key] == modified.reference_identification[key]
    for a, b in zip(control.critic.parameters(), modified.critic.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert control.alpha == modified.alpha
    np.testing.assert_array_equal(control.belief_tracker.belief, modified.belief_tracker.belief)
    assert control.sampling_state() == modified.sampling_state()
    assert any(not torch.equal(a, b) for a, b in zip(control.actor.parameters(), modified.actor.parameters()))


@pytest.mark.parametrize("reference", ["linear", "cubic-local"])
def test_resume_preserves_reference_metadata_and_next_episode(reference, tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(training, "EP_LEN", 6)
    monkeypatch.setattr(nonlinear, "EP_LEN", 6)
    config = small_config()
    config.sindy_n_explore_episodes, config.max_steps_per_episode = 50, 6
    training.seed_training(13)
    agent = ReferenceAgent(config, 13, episode_length=6)
    identify_reference(agent, identification_env(13), reference)
    env = identification_env(13)
    protocol = {"reference": reference, "config": asdict(config), "reset_cycle": [.2, .5, 1.]}
    history, steps = [], 0
    for episode in range(3):
        reward, steps = train_episode(env, agent, "balanced", episode, steps)
        history.append(reward)
    path = str(tmp_path / "last.pt")
    training.save_training_checkpoint(agent, env, path, next_ep=3, env_steps=steps, history=history, protocol=protocol)
    expected, expected_steps = train_episode(env, agent, "balanced", 3, steps)
    training.seed_training(999)
    resumed, resumed_env = ReferenceAgent(config, 999, episode_length=6), identification_env(999)
    episode, restored_steps, restored_history = training.restore_training_checkpoint(resumed, resumed_env, path, protocol)
    actual, actual_steps = train_episode(resumed_env, resumed, "balanced", episode, restored_steps)
    assert (episode, restored_history, actual, actual_steps) == (3, history, expected, expected_steps)
    assert resumed.reference_identification == agent.reference_identification
    torch.testing.assert_close(resumed.pi_sym_torch.gain, agent.pi_sym_torch.gain, rtol=0, atol=0)
    assert resumed.sampling_state() == agent.sampling_state()
    for name in ("actor", "critic", "target_critic"):
        for a, b in zip(getattr(resumed, name).parameters(), getattr(agent, name).parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert resumed.alpha == agent.alpha and resumed.training_steps == agent.training_steps
    np.testing.assert_array_equal(resumed_env.disturbance, env.disturbance)
    with pytest.raises(ValueError, match="exact protocol"):
        training.restore_training_checkpoint(resumed, resumed_env, path, {**protocol, "reference": "other"})


@pytest.mark.parametrize("failure", [None, "total", "early", "small", "fitted"])
def test_reference_gates_keep_individual_seed_failures(failure):
    runs = {(arm, seed): {"radius_results": {
        str(radius): {str(load): {"controllers": {"actor": {"mean_cost": 1. if arm == "cubic-local" else 2.,
                                                  "mean_early_state_cost": 1. if arm == "cubic-local" else 2.}},
                                 "paired_excess_costs": {"fitted": {"mean_total_cost": -.1}}}
                      for load in (1, 50)} for radius in (.2, .5, 1.)}}
            for arm in ("linear", "cubic-local") for seed in (13, 14)}
    block = runs[("cubic-local", 14)]["radius_results"]
    if failure in ("total", "early"):
        block["1.0"]["50"]["controllers"]["actor"]["mean_cost" if failure == "total" else "mean_early_state_cost"] = 2.
    elif failure == "small":
        block["0.2"]["50"]["controllers"]["actor"]["mean_cost"] = 2.01
    elif failure == "fitted":
        block["0.5"]["1"]["paired_excess_costs"]["fitted"]["mean_total_cost"] = 0.
    decision = summarize(runs, (13, 14))["decision"]
    assert decision["ready_for_learner_replication"] == (failure is None)
    assert decision["historical_v13_v15_v17_hold_unchanged"]
