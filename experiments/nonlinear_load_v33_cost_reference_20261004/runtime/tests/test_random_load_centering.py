import numpy as np
import torch

from csbapr.networks.policy import GaussianPolicy
from csbapr.sindy.load_feedback import known_load_reference
from scripts.diagnose_random_load_v6 import alignment, policy_action, rollout


def test_centering_subtracts_pre_tanh_mean_and_preserves_bounds():
    actor = GaussianPolicy(6, 2, 8, activation="tanh")
    with torch.no_grad():
        actor.mean_linear.bias.copy_(torch.tensor([1.0, -0.5]))
    origin = torch.zeros(1, 6)
    origin_mean = actor(origin)[0].detach()
    states = torch.randn(7, 6)
    expected = torch.tanh(actor(states)[0] - origin_mean)
    torch.testing.assert_close(policy_action(actor, states, origin_mean), expected)
    torch.testing.assert_close(policy_action(actor, origin, origin_mean), torch.zeros(1, 2), rtol=0, atol=0)
    assert policy_action(actor, states, origin_mean).abs().max() <= 1
    assert not torch.allclose(expected, policy_action(actor, states) - policy_action(actor, origin))


def test_jacobian_partition_covers_all_six_input_columns():
    from csbapr.losses.jacobian import action_jacobian

    actor = GaussianPolicy(6, 2, 8, activation="tanh")
    observations = np.random.default_rng(5).normal(size=(16, 6)).astype(np.float32)
    reference = known_load_reference()
    metrics = alignment(actor, reference, observations, None)
    states = torch.from_numpy(observations).requires_grad_(True)
    difference = action_jacobian(policy_action(actor, states), states, False) - action_jacobian(reference(states), states, False)
    combined = metrics["physical_input_jacobian_squared_frobenius_mean"] + metrics["force_input_jacobian_squared_frobenius_mean"]
    np.testing.assert_allclose(combined, difference.square().sum((1, 2)).mean().item(), rtol=1e-6)


def test_rollout_keeps_seed_pairing_and_reference_cost():
    from csbapr.envs.random_load_env import EP_LEN, RandomLoadEnv

    actor = GaussianPolicy(6, 2, 8, activation="tanh")
    reference = known_load_reference()
    seeds = [10000, 10001]
    result = rollout(actor, reference, 50, seeds, "reference", torch.zeros(1, 2))
    returns = []
    for seed in seeds:
        env = RandomLoadEnv(50, seed)
        obs, _ = env.reset()
        total = 0
        for _ in range(EP_LEN):
            with torch.no_grad():
                action = reference(torch.from_numpy(obs).unsqueeze(0))[0].numpy()
            obs, reward, _, _, _ = env.step(action)
            total += reward
        returns.append(total)
    assert result["evaluation_seeds"] == seeds
    np.testing.assert_allclose(result["returns"], returns, rtol=1e-6)
