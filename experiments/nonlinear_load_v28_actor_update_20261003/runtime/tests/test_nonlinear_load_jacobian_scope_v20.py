from dataclasses import asdict

import numpy as np
import pytest
import torch

from csbapr.envs import nonlinear_load_env as nonlinear
from csbapr.losses.jacobian import compute_jacobian_loss
from scripts import test_lqr_convergence as training
from scripts.train_nonlinear_load_v17 import CANDIDATE, train_episode
from scripts.train_nonlinear_load_v19 import ReferenceAgent, identify_reference
from scripts.train_nonlinear_load_v20 import INPUT_COLUMNS, ScopedReferenceAgent
from scripts.summarize_nonlinear_load_v20 import summarize


def small_config():
    config = training.make_config(CANDIDATE, "cpu", "nonlinear_load")
    config.hidden_dim, config.num_critics, config.batch_size = 8, 2, 4
    config.buffer_size, config.warmup_steps = 64, 2
    return config


def identification_env(seed):
    env = training.make_env("nonlinear_load", 1, seed)
    env.action_space.seed(seed)
    return env


def test_retained_column_sum_and_gradient_do_not_rescale_or_merge_action_rows():
    actor = torch.nn.Linear(6, 2, bias=False)
    reference = torch.nn.Linear(6, 2, bias=False)
    with torch.no_grad():
        actor.weight.copy_(torch.arange(1, 13, dtype=torch.float32).reshape(2, 6))
        reference.weight.zero_()
    states = torch.randn(4, 6)
    full = compute_jacobian_loss(actor, reference, states)
    explicit_full = compute_jacobian_loss(actor, reference, states, INPUT_COLUMNS["full"])
    load = compute_jacobian_loss(actor, reference, states, INPUT_COLUMNS["load-only"])
    torch.testing.assert_close(full, actor.weight.square().sum(), rtol=0, atol=0)
    torch.testing.assert_close(explicit_full, full, rtol=0, atol=0)
    torch.testing.assert_close(load, actor.weight[:, 4:].square().sum(), rtol=0, atol=0)
    expected_gradient = torch.zeros_like(actor.weight)
    expected_gradient[:, 4:] = 2 * actor.weight[:, 4:]
    torch.testing.assert_close(torch.autograd.grad(load, actor.weight)[0], expected_gradient, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(full, actor.weight)[0],
                               torch.autograd.grad(explicit_full, actor.weight)[0], rtol=0, atol=0)
    assert reference.weight.grad is None


@pytest.mark.parametrize("seed", [15, 16])
def test_full_updates_match_v19_and_scope_changes_only_the_actor_intervention(seed):
    torch.set_num_threads(1)
    config = small_config()
    agents, metrics = [], []
    for scope in (None, "full", "load-only"):
        training.seed_training(seed)
        agent = (ReferenceAgent(config, seed, episode_length=6) if scope is None
                 else ScopedReferenceAgent(config, seed, scope, episode_length=6))
        identify_reference(agent, identification_env(seed), "cubic-local")
        for index in range(8):
            state = np.full(6, index / 20., dtype=np.float32)
            agent.replay_buffer.push(state, np.full(2, .1), -.2, state / 2, 0.)
        training.seed_training(seed)
        metrics.append(agent.update())
        agents.append(agent)
    native, full, load = agents
    assert metrics[0] == metrics[1]
    assert native.reference_identification == full.reference_identification == load.reference_identification
    assert native.sampling_state() == full.sampling_state() == load.sampling_state()
    for name in ("actor", "critic", "target_critic"):
        for a, b in zip(getattr(native, name).parameters(), getattr(full, name).parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    for name in ("critic", "target_critic"):
        for a, b in zip(getattr(full, name).parameters(), getattr(load, name).parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert native.alpha == full.alpha == load.alpha
    np.testing.assert_array_equal(full.belief_tracker.belief, load.belief_tracker.belief)
    assert any(not torch.equal(a, b) for a, b in zip(full.actor.parameters(), load.actor.parameters()))


@pytest.mark.parametrize("scope", ["full", "load-only"])
def test_resume_preserves_scope_and_actual_next_training_episode(scope, tmp_path, monkeypatch):
    torch.set_num_threads(1)
    config = small_config()
    training.seed_training(15)
    agent = ScopedReferenceAgent(config, 15, scope, episode_length=6)
    identify_reference(agent, identification_env(15), "cubic-local")
    monkeypatch.setattr(training, "EP_LEN", 6)
    monkeypatch.setattr(nonlinear, "EP_LEN", 6)
    config.max_steps_per_episode = 6
    env = identification_env(15)
    protocol = {"jacobian_scope": scope, "reference": "cubic-local", "config": asdict(config), "reset_cycle": [.2, .5, 1.]}
    history, steps = [], 0
    for episode in range(3):
        reward, steps = train_episode(env, agent, "balanced", episode, steps)
        history.append(reward)
    path = str(tmp_path / "last.pt")
    training.save_training_checkpoint(agent, env, path, next_ep=3, env_steps=steps, history=history, protocol=protocol)
    expected, expected_steps = train_episode(env, agent, "balanced", 3, steps)
    training.seed_training(999)
    resumed = ScopedReferenceAgent(config, 999, scope, episode_length=6)
    resumed_env = identification_env(999)
    episode, restored_steps, restored_history = training.restore_training_checkpoint(resumed, resumed_env, path, protocol)
    actual, actual_steps = train_episode(resumed_env, resumed, "balanced", episode, restored_steps)
    assert (episode, restored_history, actual, actual_steps) == (3, history, expected, expected_steps)
    assert resumed.jacobian_scope == scope
    assert resumed.reference_identification == agent.reference_identification
    torch.testing.assert_close(resumed.pi_sym_torch.gain, agent.pi_sym_torch.gain, rtol=0, atol=0)
    assert resumed.sampling_state() == agent.sampling_state()
    for name in ("actor", "critic", "target_critic"):
        for a, b in zip(getattr(resumed, name).parameters(), getattr(agent, name).parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert resumed.alpha == agent.alpha and resumed.training_steps == agent.training_steps
    np.testing.assert_array_equal(resumed_env.disturbance, env.disturbance)
    wrong_scope = "load-only" if scope == "full" else "full"
    wrong = ScopedReferenceAgent(config, 15, wrong_scope, episode_length=6)
    with pytest.raises(ValueError, match="Jacobian scope"):
        wrong.load(path)
    with pytest.raises(ValueError, match="exact protocol"):
        training.restore_training_checkpoint(resumed, resumed_env, path, {**protocol, "jacobian_scope": wrong_scope})


@pytest.mark.parametrize("failure", [None, "total", "early", "small", "fitted"])
def test_scope_gates_keep_individual_seed_and_radius_failures(failure):
    runs = {(scope, seed): {"radius_results": {
        str(radius): {str(load): {"controllers": {"actor": {"mean_cost": 1. if scope == "load-only" else 2.,
                                                  "mean_early_state_cost": 1. if scope == "load-only" else 2.}},
                                 "paired_excess_costs": {"fitted": {"mean_total_cost": -.1}}}
                      for load in (1, 50)} for radius in (.2, .5, 1.)}}
            for scope in ("full", "load-only") for seed in (15, 16)}
    block = runs[("load-only", 16)]["radius_results"]
    if failure in ("total", "early"):
        block["1.0"]["50"]["controllers"]["actor"]["mean_cost" if failure == "total" else "mean_early_state_cost"] = 2.
    elif failure == "small":
        block["0.2"]["50"]["controllers"]["actor"]["mean_cost"] = 2.01
    elif failure == "fitted":
        block["0.5"]["1"]["paired_excess_costs"]["fitted"]["mean_total_cost"] = 0.
    decision = summarize(runs, (15, 16))["decision"]
    assert decision["ready_for_learner_replication"] == (failure is None)
    assert decision["historical_v13_v15_v17_v19_hold_unchanged"]
