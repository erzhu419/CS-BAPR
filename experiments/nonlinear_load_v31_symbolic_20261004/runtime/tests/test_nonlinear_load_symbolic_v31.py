from copy import deepcopy
from dataclasses import asdict
import random

import numpy as np
import pytest
import torch

from scripts import test_lqr_convergence as training
from scripts.diagnose_nonlinear_load_v14 import rollout
from scripts.train_nonlinear_load_v31 import SymbolicConstraintAgent, make_config, make_env, train_episode
from scripts.summarize_nonlinear_load_v31 import summarize
from test_nonlinear_load_critic_replay_v25 import make_agent as original_agent, parameters_equal


def make_agent(mode, critic_mode="early"):
    original = original_agent(critic_mode)
    training.seed_training(19)
    config = deepcopy(original.config)
    config.jac_weight = .01 if mode == "on" else 0.
    agent = SymbolicConstraintAgent(config, 19, mode, critic_mode)
    agent.pi_sym_torch = deepcopy(original.pi_sym_torch)
    agent.reference_identification = deepcopy(original.reference_identification)
    agent.replay_buffer.buffer = deepcopy(original.replay_buffer.buffer)
    agent.replay_buffer.position = original.replay_buffer.position
    return agent


@pytest.mark.parametrize("mode", ("off", "on"))
def test_loss_branch_matches_native_v25(mode):
    torch.set_num_threads(1)
    native, arm = original_agent("early"), make_agent(mode)
    native.config.jac_weight = arm.config.jac_weight
    training.seed_training(23)
    expected = native.update()
    training.seed_training(23)
    assert arm.update() == expected
    parameters_equal(native, arm)
    assert native.critic_sampler_state() == arm.critic_sampler_state()
    if mode == "on":
        assert native.sampling_state() == arm.sampling_state()
    else:
        assert native.sampling_state()["batches"] == 0 and arm.sampling_state()["batches"] == 1


def test_only_jacobian_weight_changes_the_first_update(monkeypatch):
    torch.set_num_threads(1)
    import csbapr.agent as module
    native_q = module.compute_q_loss_csbapr
    critic_batches, actor_batches, jacobian_batches, metrics, snapshots, agents = [], [], [], [], [], []

    def capture_q(*args, **kwargs):
        critic_batches.append(tuple(v.detach().clone() for v in args[4:11]))
        return native_q(*args, **kwargs)

    monkeypatch.setattr(module, "compute_q_loss_csbapr", capture_q)
    for mode in ("off", "on"):
        agent = make_agent(mode)
        native_actor, native_jacobian = agent._actor_loss_batch, agent._jacobian_loss_states

        def capture_actor(*args):
            result = native_actor(*args)
            actor_batches.append(tuple(v.detach().clone() for v in result))
            return result

        def capture_jacobian(states):
            result = native_jacobian(states)
            jacobian_batches.append(result.detach().clone())
            return result

        agent._actor_loss_batch, agent._jacobian_loss_states = capture_actor, capture_jacobian
        training.seed_training(23)
        metrics.append(agent.update())
        snapshots.append((np.random.get_state(), random.getstate(), torch.get_rng_state()))
        agents.append(agent)
    for batches in (critic_batches, actor_batches, jacobian_batches):
        left, right = batches
        if isinstance(left, tuple):
            for a, b in zip(left, right):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        else:
            torch.testing.assert_close(left, right, rtol=0, atol=0)
    off, on = agents
    assert off.alpha == on.alpha and off._reward_ema_var == on._reward_ema_var
    assert metrics[0]["surprise"] == metrics[1]["surprise"] and metrics[0]["q_loss"] == metrics[1]["q_loss"]
    np.testing.assert_array_equal(off.belief_tracker.belief, on.belief_tracker.belief)
    assert off.sampling_state() == on.sampling_state()
    assert off.critic_sampler_state() == on.critic_sampler_state() and off.actor_sampler_state() == on.actor_sampler_state()
    for name in ("critic", "target_critic"):
        for a, b in zip(getattr(off, name).parameters(), getattr(on, name).parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert any(not torch.equal(a, b) for a, b in zip(off.actor.parameters(), on.actor.parameters()))
    np.testing.assert_array_equal(snapshots[0][0][1], snapshots[1][0][1])
    assert snapshots[0][0][0] == snapshots[1][0][0] and snapshots[0][0][2:] == snapshots[1][0][2:]
    assert snapshots[0][1] == snapshots[1][1] and torch.equal(snapshots[0][2], snapshots[1][2])


@pytest.mark.parametrize("mode", ("off", "on"))
def test_full_prefix_fork_and_next_environment_episode_resume(mode, tmp_path):
    torch.set_num_threads(1)
    training.seed_training(21)
    config = make_config("off", "cpu")
    config.hidden_dim, config.num_critics = 8, 2
    config.batch_size, config.buffer_size = 8, 10000
    prefix_agent = SymbolicConstraintAgent(config, 21, "off", "uniform")
    prefix_agent.pi_sym_torch = training.known_reference("nonlinear_load")
    prefix_agent.reference_identification = {"selected_reference": "test-fixed"}
    env = make_env(1, 21)
    env.action_space.seed(21)
    history, steps = [], 0
    for episode in range(5):
        reward, steps = train_episode(env, prefix_agent, "balanced", episode, steps)
        history.append(reward)
        prefix_agent.record_temperature(episode)
    assert steps == 1000 and prefix_agent.training_steps == 1
    prefix_protocol = {"phase": "prefix", "mode": "off", "config": asdict(config)}
    prefix = str(tmp_path / "shared_prefix.pt")
    training.save_training_checkpoint(prefix_agent, env, prefix, next_ep=5,
                                      env_steps=steps, history=history, protocol=prefix_protocol)
    training.seed_training(999)
    arm = SymbolicConstraintAgent(deepcopy(config), 21, "off", "uniform")
    start, restored_steps, restored_history = training.restore_training_checkpoint(arm, env, prefix, prefix_protocol)
    assert (start, restored_steps, restored_history) == (5, 1000, history)
    parameters_equal(prefix_agent, arm)
    assert arm.sampling_state() == prefix_agent.sampling_state()
    assert arm.critic_sampler_state() == prefix_agent.critic_sampler_state()
    assert arm.actor_sampler_state() == prefix_agent.actor_sampler_state()
    arm.critic_sampling = "early"
    arm.set_constraint(mode)
    reward, steps = train_episode(env, arm, "balanced", 5, steps)
    history.append(reward)
    arm.record_temperature(5)
    protocol = {"phase": "arm", "mode": mode, "config": asdict(arm.config)}
    last = str(tmp_path / "last.pt")
    training.save_training_checkpoint(arm, env, last, next_ep=6, env_steps=steps, history=history, protocol=protocol)
    expected, expected_steps = train_episode(env, arm, "balanced", 6, steps)
    training.seed_training(999)
    resumed, resumed_env = SymbolicConstraintAgent(deepcopy(arm.config), 21, mode), make_env(1, 999)
    episode, restored_steps, restored_history = training.restore_training_checkpoint(resumed, resumed_env, last, protocol)
    actual, actual_steps = train_episode(resumed_env, resumed, "balanced", episode, restored_steps)
    assert (episode, restored_history, actual, actual_steps) == (6, history, expected, expected_steps)
    parameters_equal(arm, resumed)
    assert arm.alpha == resumed.alpha and arm.training_steps == resumed.training_steps
    assert arm.sampling_state() == resumed.sampling_state()
    assert arm.critic_sampler_state() == resumed.critic_sampler_state() and arm.actor_sampler_state() == resumed.actor_sampler_state()
    np.testing.assert_array_equal(env.disturbance, resumed_env.disturbance)
    with pytest.raises(ValueError, match="exact protocol"):
        training.restore_training_checkpoint(resumed, resumed_env, last, {**protocol, "mode": "wrong"})
    with pytest.raises(ValueError, match="symbolic constraint"):
        make_agent("on" if mode == "off" else "off").load(last)


def test_evaluation_uses_explicit_sustained_factory_without_changing_default():
    actor = make_agent("on").actor
    calls = []

    def factory(severity, seed, radius):
        calls.append((severity, seed, radius))
        env = make_env(severity, seed, radius)
        assert env.storm_probability == 1. and env.force_cap == 4.
        return env

    old, _ = rollout(actor, {}, "actor", .2, 50, [62000])
    sustained, _ = rollout(actor, {}, "actor", .2, 50, [62000], env_factory=factory)
    assert calls == [(50, 62000, .2)]
    assert old["returns"] != sustained["returns"]
    repeated, _ = rollout(actor, {}, "actor", .2, 50, [62000])
    assert repeated == old


@pytest.mark.parametrize("failure", (None, "total", "early", "small", "fitted"))
def test_gates_keep_one_seed_failure_and_hold(failure):
    runs = {(mode, seed): {"radius_results": {
        str(radius): {str(load): {"controllers": {"actor": {"mean_cost": 1. if mode == "on" else 2.,
                                                  "mean_early_state_cost": 1. if mode == "on" else 2.}},
                                 "paired_excess_costs": {"fitted": {"mean_total_cost": -.1}}}
                      for load in (1, 50)} for radius in (.2, .5, 1.)}}
            for mode in ("off", "on") for seed in (21, 22)}
    block = runs["on", 22]["radius_results"]
    if failure in ("total", "early"):
        block["1.0"]["50"]["controllers"]["actor"]["mean_cost" if failure == "total" else "mean_early_state_cost"] = 2.
    elif failure == "small":
        block["0.2"]["50"]["controllers"]["actor"]["mean_cost"] = 2.01
    elif failure == "fitted":
        block["0.5"]["1"]["paired_excess_costs"]["fitted"]["mean_total_cost"] = 0.
    decision = summarize(runs, (21, 22))["decision"]
    assert decision["ready_for_learner_replication"] == (failure is None)
    assert decision["historical_hold_unchanged"]
