from dataclasses import asdict

import numpy as np
import pytest
import torch

from csbapr.agent import CSBAPRAgent
from csbapr.losses.jacobian import compute_action_anchor_loss, compute_jacobian_loss
from scripts.test_lqr_convergence import make_config


def test_anchor_corrects_offset_without_reference_gradients():
    actor = torch.nn.Linear(4, 2)
    reference = torch.nn.Linear(4, 2)
    with torch.no_grad():
        actor.weight.zero_()
        actor.bias.fill_(0.6)
        reference.weight.zero_()
        reference.bias.zero_()
    states = torch.zeros(8, 4)
    assert compute_jacobian_loss(actor, reference, states).item() == 0
    before = compute_action_anchor_loss(actor, reference, states)
    optimizer = torch.optim.SGD(actor.parameters(), lr=0.2)
    before.backward()
    assert all(p.grad is None for p in reference.parameters())
    optimizer.step()
    assert compute_action_anchor_loss(actor, reference, states) < before


def test_anchor_preset_changes_only_the_anchor_weight():
    control = asdict(make_config("csbapr", "cpu"))
    anchored = asdict(make_config("csbapr-action-anchor", "cpu"))
    assert anchored.pop("action_anchor_weight") == 1.0
    assert control.pop("action_anchor_weight") == 0.0
    assert anchored == control


def test_missing_reference_fails_before_any_update():
    config = make_config("csbapr-action-anchor", "cpu")
    config.batch_size = 2
    agent = CSBAPRAgent(4, 2, config)
    for _ in range(2):
        agent.replay_buffer.push(np.zeros(4), np.zeros(2), 0., np.zeros(4), 0.)
    with pytest.raises(ValueError, match="fitted reference"):
        agent.update()
    assert agent.training_steps == 0
    assert not agent.actor_optimizer.state and not agent.critic_optimizer.state
