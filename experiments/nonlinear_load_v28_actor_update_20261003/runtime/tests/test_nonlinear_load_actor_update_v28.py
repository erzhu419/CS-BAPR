from copy import deepcopy

import numpy as np
import pytest
import torch

from csbapr.sindy.feedback import LQRFeedbackReference
from scripts import diagnose_nonlinear_load_v28 as diagnostic


def models():
    torch.manual_seed(5)
    actor = diagnostic.EquilibriumGaussianPolicy(6, 2, 8, activation="tanh")
    critic = diagnostic.EnsembleQNet(6, 2, 8, 3).requires_grad_(False)
    reference = LQRFeedbackReference(np.array([[1, 0, .5, 0, .4, 0], [0, 1, 0, .5, 0, .4]]))
    return actor, critic, reference


def test_sampling_matches_native_streams_and_changes_only_actor_coverage():
    states = np.arange(600000, dtype=np.float32).reshape(100000, 6) / 1000000
    native = np.random.RandomState(19)
    jac = np.random.default_rng(100019)
    numpy_state, jac_state = native.get_state(), deepcopy(jac.bit_generator.state)
    banks = diagnostic.sampled_batches(states, numpy_state, jac_state, 19, count=2)
    for bank in banks:
        uniform = native.randint(0, 100000, size=256)
        jacobian = np.concatenate((uniform[:128], jac.integers(0, 500, size=128) * 200))
        np.testing.assert_array_equal(bank["uniform"]["indices"], uniform)
        np.testing.assert_array_equal(bank["jacobian"]["indices"], jacobian)
        early = np.asarray(bank["early"]["indices"])
        np.testing.assert_array_equal(early[:128], uniform[:128])
        assert (early[128:192] % 200 == 0).all()
        assert ((early[192:] % 200 >= 1) & (early[192:] % 200 < 50)).all()
        for name, values in bank.items():
            np.testing.assert_array_equal(values["states"], states[values["indices"]])


def test_loss_components_equal_native_equation_and_full_gradient():
    actor, critic, reference = models()
    states, jac = torch.randn(8, 6), torch.randn(8, 6)
    noise = torch.randn(8, 2)
    terms = diagnostic.loss_terms(actor, critic, reference, states, jac, .01, -2.4, .01, noise)
    action, log_prob = diagnostic.policy_sample(actor, states, noise)
    q = critic(states, action)
    expected = -(q.mean(0) - 2.4 * q.std(0) - .01 * log_prob).mean()
    expected += .01 * diagnostic.compute_jacobian_loss(actor, reference, jac)
    torch.testing.assert_close(sum(terms.values()), expected)
    parameters = tuple(actor.parameters())
    summed = sum(diagnostic.flat_gradient(value, parameters) for value in terms.values())
    torch.testing.assert_close(summed, diagnostic.flat_gradient(expected, parameters), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("zero", (False, True))
def test_shadow_adam_matches_native_moments_weight_decay_and_clipping(zero):
    actor, _, _ = models()
    optimizer = torch.optim.Adam(actor.parameters(), lr=.0003, weight_decay=.0001)
    for _ in range(3):
        optimizer.zero_grad()
        sum(p.square().sum() for p in actor.parameters()).backward()
        optimizer.step()
    before = deepcopy(actor.state_dict())
    parameters = tuple(actor.parameters())
    gradient = torch.zeros(sum(p.numel() for p in parameters)) if zero else torch.randn(sum(p.numel() for p in parameters)) * 2
    encoded = diagnostic.json_optimizer(optimizer.state_dict())
    restored = diagnostic.tensor_optimizer(encoded)
    delta, norm, scale = diagnostic.adam_delta(actor, restored, gradient, 1.)
    shadow = deepcopy(actor)
    native = torch.optim.Adam(shadow.parameters())
    native.load_state_dict(deepcopy(optimizer.state_dict()))
    offset = 0
    for p in shadow.parameters():
        p.grad = gradient[offset:offset + p.numel()].reshape_as(p).clone()
        offset += p.numel()
    expected_norm = torch.nn.utils.clip_grad_norm_(shadow.parameters(), 1.)
    native.step()
    expected = torch.cat([(new.detach() - old.detach()).flatten() for new, old in zip(shadow.parameters(), parameters)])
    torch.testing.assert_close(delta, expected, rtol=0, atol=0)
    assert norm == float(expected_norm) and scale <= 1
    if zero:
        assert delta.norm() > 0
    for key, value in actor.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_full_parameter_perturbation_changes_std_and_paired_return_summary():
    actor, _, _ = models()
    delta = torch.ones(sum(p.numel() for p in actor.parameters())) * .001
    plus, minus = [diagnostic.displaced_actor(actor, delta, sign) for sign in (1, -1)]
    states = torch.randn(2, 6)
    assert not torch.equal(plus(states)[1], minus(states)[1])
    for p, m, original in zip(plus.parameters(), minus.parameters(), actor.parameters()):
        torch.testing.assert_close((p + m) / 2, original)
    def values(multiplier):
        return {"continuations": {"actor": {k: np.ones((2, 3)) * multiplier for k in diagnostic.COSTS + diagnostic.RETURNS}},
                "critic_values": {"actor": {k: np.ones((2, 3)) * multiplier for k in diagnostic.SCORES}}}
    raw = {f"{arm}_{sign}_{h}": values(2 + direction * h)
           for arm in diagnostic.ARMS for h in diagnostic.PARAMETER_SCALES
           for sign, direction in (("plus", 1), ("minus", -1))}
    result = diagnostic.compact_continuations(raw, values(2), diagnostic.PARAMETER_SCALES)
    for arm in diagnostic.ARMS:
        for h in diagnostic.PARAMETER_SCALES:
            block = result[arm][str(h)]
            assert block["continuations"]["mean_derivatives"]["soft_actor_return"] == 1
            assert block["plus_step_discounted_cost_change"]["mean"] == h
