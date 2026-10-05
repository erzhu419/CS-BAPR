import numpy as np
import torch

from csbapr.losses.jacobian import compute_jacobian_loss
from csbapr.networks.critic import EnsembleQNet
from csbapr.networks.policy import EquilibriumGaussianPolicy
from csbapr.sindy.feedback import LQRFeedbackReference
from scripts.diagnose_nonlinear_load_v16 import flat_gradient, gradient_terms, relation, support


def test_native_actor_loss_and_split_gradient():
    torch.manual_seed(37)
    actor = EquilibriumGaussianPolicy(6, 2, 8, activation="tanh")
    critic = EnsembleQNet(6, 2, 8, 3).requires_grad_(False)
    reference = LQRFeedbackReference(np.arange(12).reshape(2, 6) / 20.)
    states = torch.randn(16, 6) / 4
    before = [p.detach().clone() for p in actor.parameters()]
    losses, gradients = gradient_terms(actor, critic, reference, states, .03, -3.2, .01, 100)
    torch.manual_seed(100)
    action, log_prob, *_ = actor.evaluate(states)
    values = critic(states, action)
    native_jac = .01 * compute_jacobian_loss(actor, reference, states)
    native_sac = -(values.mean(0) - 3.2 * values.std(0) - .03 * log_prob).mean()
    np.testing.assert_allclose(losses["jac_physical"] + losses["jac_load"], native_jac.item(), rtol=1e-5)
    torch.testing.assert_close(gradients["total"], flat_gradient(native_sac + native_jac, tuple(actor.parameters())),
                               rtol=1e-5, atol=1e-6)
    assert all(torch.equal(old, p) and p.grad is None for old, p in zip(before, actor.parameters()))
    assert all(p.grad is None for p in critic.parameters())


def test_direction_sign_and_zero_gradient():
    assert relation(torch.tensor([1.]), torch.tensor([-2.])) == {
        "dot": -2., "cosine": -1., "right_loss_derivative_under_left_descent": 2.}
    assert relation(torch.zeros(1), torch.ones(1))["cosine"] is None


def test_support_bin_counts_conserve_states():
    states = np.zeros((4, 6))
    states[:, 0] = [.1, .3, .7, 1.2]
    summary = support(states)
    assert summary["count"] == 4 and summary["physical_norm_bin_counts"] == [1, 1, 1, 1]
