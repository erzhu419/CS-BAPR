from copy import deepcopy
import random

import numpy as np
import pytest
import torch

from scripts import test_lqr_convergence as training
from scripts.train_nonlinear_load_v22 import EntropyFloorAgent, make_config
from scripts.train_nonlinear_load_v25 import CriticReplayAgent


def make_agent(mode=None):
    training.seed_training(19)
    config = make_config("floor", "cpu")
    config.hidden_dim, config.num_critics, config.batch_size = 8, 2, 8
    config.buffer_size = 10000
    agent = EntropyFloorAgent(config, 19) if mode is None else CriticReplayAgent(config, 19, mode)
    agent.pi_sym_torch = training.known_reference("nonlinear_load")
    agent.reference_identification = {"selected_reference": "test-fixed"}
    for index in range(401):
        state = np.full(6, index / 1000, dtype=np.float32)
        agent.replay_buffer.push(state, np.full(2, index / 10000, dtype=np.float32),
                                 -(index + 1) / 1000, state + .0001, float(index % 200 == 199))
    return agent


def parameters_equal(a, b):
    for name in ("actor", "critic", "target_critic"):
        for left, right in zip(getattr(a, name).parameters(), getattr(b, name).parameters()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_uniform_control_is_bitwise_native():
    torch.set_num_threads(1)
    native, control = make_agent(), make_agent("uniform")
    training.seed_training(23)
    metrics = native.update()
    training.seed_training(23)
    assert metrics == control.update()
    parameters_equal(native, control)
    assert native.sampling_state() == control.sampling_state()
    assert control.critic_counts == {"uniform": 8, "reset": 0, "early": 0}


@pytest.mark.parametrize("size", [401, 425, 600])
def test_replacement_indices_include_only_existing_reset_and_early_transitions(size):
    agent = make_agent("early")
    while agent.replay_buffer.size < size:
        agent.replay_buffer.push(np.zeros(6), np.zeros(2), 0., np.zeros(6), 0.)
    agent.replay_buffer.buffer = agent.replay_buffer.buffer[:size]
    agent.replay_buffer.position = size
    reset, early = agent.replacement_indices(10000)
    assert (reset % 200 == 0).all() and (reset < size).all()
    assert ((early % 200 >= 1) & (early % 200 <= 49) & (early < size)).all()
    assert set(reset // 200) == set(range((size + 199) // 200))


def test_only_critic_inputs_change_and_global_rng_streams_stay_native(monkeypatch):
    torch.set_num_threads(1)
    captured, jacobian = [], []
    import csbapr.agent as module
    original = module.compute_q_loss_csbapr

    def capture(*args, **kwargs):
        captured.append(tuple(value.detach().clone() for value in args[4:11]))
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "compute_q_loss_csbapr", capture)
    snapshots, agents, metrics = [], [], []
    for mode in ("uniform", "early"):
        agent = make_agent(mode)
        native_sampler = agent._jacobian_loss_states

        def capture_jacobian(states):
            jacobian.append(states.detach().clone())
            return native_sampler(states)

        agent._jacobian_loss_states = capture_jacobian
        training.seed_training(23)
        metrics.append(agent.update())
        snapshots.append((np.random.get_state(), random.getstate(), torch.get_rng_state()))
        agents.append(agent)
    left, right = agents
    assert left.alpha == right.alpha and left._reward_ema_var == right._reward_ema_var
    assert metrics[0]["surprise"] == metrics[1]["surprise"]
    np.testing.assert_array_equal(left.belief_tracker.belief, right.belief_tracker.belief)
    assert left.sampling_state() == right.sampling_state()
    torch.testing.assert_close(jacobian[0], jacobian[1], rtol=0, atol=0)
    assert snapshots[0][0][0] == snapshots[1][0][0] and snapshots[0][0][2:] == snapshots[1][0][2:]
    np.testing.assert_array_equal(snapshots[0][0][1], snapshots[1][0][1])
    assert snapshots[0][1] == snapshots[1][1] and torch.equal(snapshots[0][2], snapshots[1][2])
    assert not torch.equal(captured[0][0][4:], captured[1][0][4:])
    for original_values, changed_values in zip(*captured):
        torch.testing.assert_close(original_values[:4], changed_values[:4], rtol=0, atol=0)
    states, actions, rewards, next_states, dones = [v[4:] for v in captured[1][:5]]
    torch.testing.assert_close(actions, states[:, :2] / 10)
    torch.testing.assert_close(next_states, states + .0001)
    torch.testing.assert_close(rewards[:, 0], -(states[:, 0] + .001) / right._reward_ema_var ** .5)
    assert torch.count_nonzero(dones) == 0
    assert right.critic_counts == {"uniform": 4, "reset": 2, "early": 2}


def test_active_sampler_and_next_update_resume_exactly(tmp_path):
    torch.set_num_threads(1)
    agent = make_agent("early")
    agent.update()
    path = str(tmp_path / "last.pt")
    agent.save(path, extra_state={"episode": 2})
    metrics = agent.update()
    expected = deepcopy(agent.critic_sampler_state())
    resumed = make_agent("early")
    resumed.load(path)
    assert resumed.update() == metrics
    assert resumed.critic_sampler_state() == expected
    assert resumed.sampling_state() == agent.sampling_state()
    parameters_equal(agent, resumed)
    with pytest.raises(ValueError, match="critic sampling"):
        make_agent("uniform").load(path)
