from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from csbapr.networks.policy import EquilibriumGaussianPolicy
from scripts.diagnose_nonlinear_load_v9 import MODES, control_action, control_derivatives
from scripts.diagnose_nonlinear_load_v14 import paired_excess_costs, rollout
from scripts.preflight_nonlinear_load_v9 import local_linear_gain
from summarize_nonlinear_load_v14 import validate_costs


def actor_and_gains():
    torch.manual_seed(42)
    actor = EquilibriumGaussianPolicy(6, 2, hidden_dim=8, activation="tanh").eval()
    actor.requires_grad_(False)
    gain = torch.tensor(local_linear_gain(), dtype=torch.float32)
    return actor, {"fitted": gain, "known_local": gain}


def test_diagnostic_action_and_output_force_jacobians_match_finite_differences():
    from csbapr.envs.nonlinear_load_env import actuator_force

    actor, gains = actor_and_gains()
    observation = np.array([[0.3, -0.2, 0.1, -0.1, 0.4, -0.3]], dtype=np.float32)
    step = 1e-3
    for mode in MODES:
        actions, jacobian, force_jacobian = control_derivatives(actor, gains, mode, observation)
        np.testing.assert_allclose(actions, control_action(actor, gains, mode, observation), atol=1e-7)
        numerical, force_numerical = [], []
        for index in range(6):
            offset = np.zeros_like(observation)
            offset[:, index] = step
            plus = control_action(actor, gains, mode, observation + offset)
            minus = control_action(actor, gains, mode, observation - offset)
            numerical.append((plus - minus) / (2 * step))
            force_numerical.append((actuator_force(plus) - actuator_force(minus)) / (2 * step))
        np.testing.assert_allclose(jacobian, np.stack(numerical, axis=2), rtol=5e-3, atol=1e-4)
        np.testing.assert_allclose(force_jacobian, np.stack(force_numerical, axis=2), rtol=5e-3, atol=3e-4)


def test_diagnostic_cost_split_and_paired_excess_conserve_episode_return():
    actor, gains = actor_and_gains()
    seeds = [60000, 60002]
    actual, samples = rollout(actor, gains, "actor", 1.0, 50, seeds)
    reference, _ = rollout(actor, gains, "fitted", 1.0, 50, seeds)
    assert samples.shape == (20, 2, 6)
    for result in (actual, reference):
        for kind in ("state", "command"):
            np.testing.assert_allclose(result[f"{kind}_costs"],
                                       np.asarray(result[f"early_{kind}_costs"]) + result[f"late_{kind}_costs"],
                                       rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(-np.asarray(result["returns"]),
                                   np.asarray(result["state_costs"]) + result["command_costs"], rtol=1e-12)
    paired = paired_excess_costs(actual, reference, seeds)
    assert [p["evaluation_seed"] for p in paired["per_path"]] == seeds
    np.testing.assert_allclose([p["total_cost"] for p in paired["per_path"]],
                               np.asarray(reference["returns"]) - actual["returns"])
    assert all(not p.requires_grad and p.grad is None for p in actor.parameters())
    actual["maximum_absolute_reproduction_error"] = 0.0
    validate_costs(actual, actual, seeds, "test-input")
    changed = deepcopy(actual)
    changed["early_state_costs"][-1] += 1.0
    with pytest.raises(AssertionError):
        validate_costs(changed, actual, seeds, "test-input")
