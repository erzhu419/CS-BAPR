from copy import deepcopy
import random

import numpy as np
import pytest
import torch

from scripts import test_lqr_convergence as training
from scripts.train_nonlinear_load_v29 import ActorReplayAgent, policy_sample
from test_nonlinear_load_critic_replay_v25 import make_agent as original_agent, parameters_equal


def make_agent(mode):
    original = original_agent("early")
    training.seed_training(19)
    agent = ActorReplayAgent(original.config, 19, mode)
    agent.pi_sym_torch = deepcopy(original.pi_sym_torch)
    agent.reference_identification = deepcopy(original.reference_identification)
    agent.replay_buffer.buffer = deepcopy(original.replay_buffer.buffer)
    agent.replay_buffer.position = original.replay_buffer.position
    return agent


def test_uniform_actor_control_is_bitwise_v25_early():
    torch.set_num_threads(1)
    native, control = original_agent("early"), make_agent("uniform")
    training.seed_training(23)
    expected = native.update()
    training.seed_training(23)
    assert control.update() == expected
    parameters_equal(native, control)
    assert native.sampling_state() == control.sampling_state()
    assert native.critic_sampler_state() == control.critic_sampler_state()


@pytest.mark.parametrize("size", (401, 425, 600))
def test_actor_reset_early_selection_handles_native_partial_episodes(size):
    agent = make_agent("early")
    while agent.replay_buffer.size < size:
        agent.replay_buffer.push(np.zeros(6), np.zeros(2), 0., np.zeros(6), 0.)
    agent.replay_buffer.position = size
    before = deepcopy(agent.critic_rng.bit_generator.state)
    reset, early = agent.replacement_indices(10000, rng=agent.actor_rng)
    assert (reset % 200 == 0).all() and (reset < size).all()
    assert ((early % 200 >= 1) & (early % 200 < 50) & (early < size)).all()
    assert set(reset // 200) == set(range((size + 199) // 200))
    assert agent.critic_rng.bit_generator.state == before


def test_only_actor_base_loss_coverage_changes_with_matched_noise(monkeypatch):
    torch.set_num_threads(1)
    import csbapr.agent as module
    original = module.compute_q_loss_csbapr
    critic_batches, actor_batches, jacobian, snapshots, agents, metrics = [], [], [], [], [], []
    def capture_critic(*args, **kwargs):
        critic_batches.append(tuple(value.detach().clone() for value in args[4:11]))
        return original(*args, **kwargs)
    monkeypatch.setattr(module, "compute_q_loss_csbapr", capture_critic)
    for mode in ("uniform", "early"):
        agent = make_agent(mode)
        actor_before = deepcopy(agent.actor)
        native_actor = agent._actor_loss_batch
        native_jacobian = agent._jacobian_loss_states
        def capture_actor(states, actions, log_probs, noise):
            result = native_actor(states, actions, log_probs, noise)
            actor_batches.append(tuple(v.detach().clone() for v in result))
            for current, old in zip(result, (states, actions, log_probs)):
                torch.testing.assert_close(current[:4], old[:4], rtol=0, atol=0)
            if mode == "early":
                expected_action, expected_log = policy_sample(actor_before, result[0][4:], noise[4:])
                torch.testing.assert_close(result[1][4:], expected_action, rtol=0, atol=0)
                torch.testing.assert_close(result[2][4:], expected_log, rtol=0, atol=0)
            return result
        def capture_jacobian(states):
            result = native_jacobian(states)
            jacobian.append(result.detach().clone())
            return result
        agent._actor_loss_batch = capture_actor
        agent._jacobian_loss_states = capture_jacobian
        training.seed_training(23)
        metrics.append(agent.update())
        snapshots.append((np.random.get_state(), random.getstate(), torch.get_rng_state()))
        agents.append(agent)
    for left, right in zip(*critic_batches):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    torch.testing.assert_close(jacobian[0], jacobian[1], rtol=0, atol=0)
    assert not torch.equal(actor_batches[0][0][4:], actor_batches[1][0][4:])
    a, b = agents
    assert a.alpha == b.alpha and a._reward_ema_var == b._reward_ema_var
    assert metrics[0]["surprise"] == metrics[1]["surprise"] and metrics[0]["q_loss"] == metrics[1]["q_loss"]
    np.testing.assert_array_equal(a.belief_tracker.belief, b.belief_tracker.belief)
    assert a.sampling_state() == b.sampling_state() and a.critic_sampler_state() == b.critic_sampler_state()
    for name in ("critic", "target_critic"):
        for left, right in zip(getattr(a, name).parameters(), getattr(b, name).parameters()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert any(not torch.equal(left, right) for left, right in zip(a.actor.parameters(), b.actor.parameters()))
    np.testing.assert_array_equal(snapshots[0][0][1], snapshots[1][0][1])
    assert snapshots[0][0][0] == snapshots[1][0][0] and snapshots[0][0][2:] == snapshots[1][0][2:]
    assert snapshots[0][1] == snapshots[1][1] and torch.equal(snapshots[0][2], snapshots[1][2])
    assert b.actor_counts == {"uniform": 4, "reset": 2, "early": 2}


def test_active_actor_sampler_and_adam_resume_exactly(tmp_path):
    torch.set_num_threads(1)
    agent = make_agent("early")
    agent.update()
    path = str(tmp_path / "last.pt")
    agent.save(path, extra_state={"episode": 2})
    metrics = agent.update()
    resumed = make_agent("early")
    resumed.load(path)
    assert resumed.update() == metrics
    parameters_equal(agent, resumed)
    assert resumed.actor_sampler_state() == agent.actor_sampler_state()
    assert resumed.critic_sampler_state() == agent.critic_sampler_state()
    assert resumed.sampling_state() == agent.sampling_state()
    with pytest.raises(ValueError, match="actor sampling"):
        make_agent("uniform").load(path)
