import math

import numpy as np
import pytest
import torch

from csbapr.agent import CSBAPRAgent
from csbapr.config import CSBAPRConfig
from csbapr.sindy.feedback import LQRFeedbackReference
from scripts.diagnose_lqr_feedback_v2 import alignment_metrics, gradient_profile


def test_action_offset_survives_perfect_jacobian_match():
    actor = torch.nn.Linear(4, 2)
    reference = torch.nn.Linear(4, 2)
    with torch.no_grad():
        actor.weight.zero_()
        actor.bias.fill_(0.6)
        reference.weight.zero_()
        reference.bias.zero_()
    reference.requires_grad_(False)
    result = alignment_metrics(actor, reference, np.zeros((5, 4)))
    assert result["jacobian_squared_frobenius_mean"] == 0
    assert result["action_component_rmse"] == pytest.approx(0.6)
    assert result["oddness_component_rmse"] == pytest.approx(1.2)


def test_unit_ball_coverage_uses_training_reset_support():
    actor = torch.nn.Linear(4, 2)
    reference = torch.nn.Linear(4, 2)
    reference.requires_grad_(False)
    states = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]], dtype=np.float32)
    result = alignment_metrics(actor, reference, states)
    assert result["unit_ball_state_fraction"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("anchor_weight", [0.0, 1.0])
def test_gradient_probe_does_not_change_training_state(anchor_weight):
    torch.set_num_threads(1)
    config = CSBAPRConfig(hidden_dim=8, batch_size=8, actor_type="smooth-nau", device="cpu")
    config.action_anchor_weight = anchor_weight
    agent = CSBAPRAgent(4, 2, config)
    agent.pi_sym_torch = LQRFeedbackReference(np.array([[1, 0, 0.5, 0], [0, 1, 0, 0.5]]))
    agent._reward_ema_var = 2.0
    agent.training_steps = 200
    rng = np.random.default_rng(3)
    for _ in range(32):
        state = rng.normal(size=4).astype(np.float32)
        agent.replay_buffer.push(state, np.zeros(2), -float(state @ state), state * 0.9, 0.)
    parameters = list(agent.actor.parameters()) + list(agent.critic.parameters()) + list(agent.target_critic.parameters())
    before = [p.detach().clone() for p in parameters]
    belief = agent.belief_tracker.belief.copy()
    profile = gradient_profile(agent, agent.pi_sym_torch, n_batches=2)
    for batch in profile:
        assert "jacobian" in batch["actor_terms"]
        assert ("action_anchor" in batch["actor_terms"]) == (anchor_weight > 0)
        assert batch["action_anchor_weight"] == anchor_weight
        if anchor_weight > 0:
            assert batch["actor_terms"]["action_anchor"]["weighted_loss"] == pytest.approx(
                anchor_weight * batch["diagnostic_action_mse"])
        assert all(math.isfinite(batch[key]) for key in ("critic_total_loss", "critic_gradient_l2", "actor_total_gradient_l2"))
    for initial, current in zip(before, parameters):
        torch.testing.assert_close(initial, current, rtol=0, atol=0)
        assert current.grad is None
    np.testing.assert_array_equal(belief, agent.belief_tracker.belief)
    assert agent.training_steps == 200 and agent._reward_ema_var == 2.0
    assert not agent.actor_optimizer.state and not agent.critic_optimizer.state
