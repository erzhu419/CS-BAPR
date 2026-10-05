from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from csbapr.losses.q_loss import compute_q_loss_csbapr
from scripts.test_lqr_convergence import make_config


@pytest.mark.parametrize("method,anchor,shift", [
    ("csbapr", 0.0, 0.01),
    ("csbapr-action-anchor", 1.0, 0.01),
    ("csbapr-no-target-shift", 0.0, 0.0),
    ("csbapr-action-anchor-no-target-shift", 1.0, 0.0),
])
def test_factorial_presets_change_only_the_two_coefficients(method, anchor, shift):
    original = asdict(make_config("csbapr", "cpu"))
    actual = asdict(make_config(method, "cpu"))
    expected = {**original, "action_anchor_weight": anchor, "weight_reg": shift}
    assert actual == expected


@pytest.mark.parametrize("base_method", ["csbapr-no-sindy", "csbapr-no-nau", "bapr", "resac"])
def test_fair_baseline_presets_change_only_target_shift(base_method):
    original = asdict(make_config(base_method, "cpu"))
    actual = asdict(make_config(base_method + "-no-target-shift", "cpu"))
    assert actual == {**original, "weight_reg": 0.0}


def test_zero_coefficient_removes_the_positive_target_shift():
    class ZeroEnsemble(torch.nn.Module):
        num_critics = 2

        def forward(self, state, action):
            return torch.zeros(self.num_critics, len(state))

    actor = torch.nn.Linear(4, 2, bias=False)
    reference = torch.nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        actor.weight.zero_()
        reference.weight.zero_()
    reference.requires_grad_(False)
    state = torch.zeros(2, 4)
    action = torch.zeros(2, 2)
    reward = torch.zeros(2, 1)
    done = torch.tensor([[0.0], [1.0]])
    belief = SimpleNamespace(belief=[0.5, 0.5], max_H=2)

    def loss(weight_reg):
        return compute_q_loss_csbapr(
            ZeroEnsemble(), ZeroEnsemble(), actor, reference,
            state, action, reward, state, done, action, torch.zeros(2),
            torch.tensor([2.0, 4.0]), belief, alpha=0.0, gamma=0.5,
            weight_reg=weight_reg, weight_sym=0.01, beta_ood=0.0,
            penalty_decay_rate=0.1, batch_size=2,
        )[0].item()

    # The terminal transition has no bootstrap shift; the other has 0.01/0.02.
    assert loss(0.01) == pytest.approx((0.01 ** 2 + 0.02 ** 2) / 4)
    assert loss(0.0) == 0.0
