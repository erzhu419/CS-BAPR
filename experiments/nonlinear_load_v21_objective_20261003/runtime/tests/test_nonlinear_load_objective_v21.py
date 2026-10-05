import numpy as np
import pytest
import torch
from torch.distributions import Normal

from csbapr.envs.lqr_env import ACTION_SCALE, Q, R
from csbapr.losses.jacobian import deterministic_action
from csbapr.networks.critic import EnsembleQNet
from csbapr.networks.policy import EquilibriumGaussianPolicy
from scripts import diagnose_nonlinear_load_v21 as diagnostic


class ConstantActor(torch.nn.Module):
    action_range = 1.

    def __init__(self):
        super().__init__()
        self.mean = torch.nn.Parameter(torch.full((2,), float(np.arctanh(.25))))

    def forward(self, states):
        return self.mean.expand(len(states), -1), torch.zeros((len(states), 2))


def test_only_first_action_changes_and_discount_uses_step_zero(monkeypatch):
    environments = []

    class RecordingEnv:
        def __init__(self, seed):
            self.seed = seed
            self.state = np.array([.1, .1, 0., 0.])
            self.actions, self.t = [], 0
            environments.append(self)

        def reset(self):
            return np.r_[self.state, 0., 0.].astype(np.float32), {}

        def step(self, action):
            self.actions.append(action.copy())
            command = ACTION_SCALE * action.astype(float)
            cost = float(self.state @ Q @ self.state + command @ R @ command)
            self.t += 1
            load = np.array([self.seed + self.t, self.seed - self.t], dtype=float)
            return np.r_[self.state, 0., 0.].astype(np.float32), -cost, self.t == 5, False, {"applied_disturbance": load}

    monkeypatch.setattr(diagnostic, "EP_LEN", 5)
    monkeypatch.setattr(diagnostic, "make_radius_env", lambda severity, seed, radius: RecordingEnv(seed))
    gain = torch.tensor([[2., 0., 0., 0., 0., 0.], [0., 2., 0., 0., 0., 0.]])
    _, totals = diagnostic.paired_rollout(ConstantActor(), gain, .2, 1, [60000, 60001], gamma=.7)
    for i in range(2):
        actor, swapped = environments[i], environments[i + 2]
        np.testing.assert_allclose(actor.actions, .25, rtol=1e-6)
        np.testing.assert_allclose(swapped.actions[0], -np.tanh(.2), rtol=1e-6)
        np.testing.assert_array_equal(swapped.actions[1:], actor.actions[1:])
        for mode, env in (("actor", actor), ("fitted-first", swapped)):
            costs = [.02 + (ACTION_SCALE * a.astype(float)) @ R @ (ACTION_SCALE * a.astype(float)) for a in env.actions]
            assert totals[mode]["total_cost"][i] == pytest.approx(sum(costs))
            assert totals[mode]["discounted_cost"][i] == pytest.approx(sum(.7 ** t * cost for t, cost in enumerate(costs)))


def test_native_risk_and_pointwise_entropy_scores_are_read_only():
    torch.manual_seed(21)
    actor = EquilibriumGaussianPolicy(6, 2, 8, activation="tanh")
    critic = EnsembleQNet(6, 2, 8, 3)
    gain = torch.tensor([[1., 0., .5, 0., .4, 0.], [0., 1., 0., .5, 0., .4]])
    states = torch.randn(4, 6) / 5
    before = [p.detach().clone() for net in (actor, critic) for p in net.parameters()]
    alpha, beta = .03, -3.1
    result = diagnostic.initial_scores(actor, critic, gain, states.numpy(), alpha, beta)
    with torch.no_grad():
        mean, log_std = actor(states)
        for mode in diagnostic.MODES:
            action = deterministic_action(actor, states) if mode == "actor" else torch.tanh(-states @ gain.T)
            q = critic(states, action)
            score = q.mean(0) + beta * q.std(0)
            latent = torch.atanh(action.clamp(-1 + 1e-6, 1 - 1e-6))
            log_prob = (Normal(mean, log_std.exp()).log_prob(latent) - torch.log(1 - action.square() + 1e-6)).sum(-1)
            np.testing.assert_allclose(result[mode]["risk_score"], score.numpy(), rtol=1e-6)
            np.testing.assert_allclose(result[mode]["log_probability"], log_prob.numpy(), rtol=1e-6)
            np.testing.assert_allclose(result[mode]["soft_surrogate_score"], (score - alpha * log_prob).numpy(), rtol=1e-6)
    assert all(torch.equal(old, p) and p.grad is None for old, p in zip(before, [p for net in (actor, critic) for p in net.parameters()]))


def test_identical_actions_reproduce_identical_real_environment_paths():
    class LinearActor(torch.nn.Module):
        def __init__(self, gain):
            super().__init__()
            self.gain = torch.nn.Parameter(gain)

        def forward(self, states):
            mean = -states @ self.gain.T
            return mean, torch.zeros_like(mean)

    gain = torch.tensor([[1., 0., .5, 0., .4, 0.], [0., 1., 0., .5, 0., .4]])
    _, totals = diagnostic.paired_rollout(LinearActor(gain), gain, .5, 50, [60000, 60001])
    for name in totals["actor"]:
        np.testing.assert_array_equal(totals["actor"][name], totals["fitted-first"][name])


def test_ordering_does_not_mix_cost_and_score_signs_or_hide_discount_reversals():
    a = np.array([1., 1., 1.])
    totals = {"actor": {"total_cost": a, "discounted_cost": a},
              "fitted-first": {"total_cost": a + [-.3, .1, -.1], "discounted_cost": a + [.3, -.1, .1]}}
    scores = {"actor": {k: a for k in ("q_mean", "risk_score", "soft_surrogate_score")},
              "fitted-first": {k: a + [-.2, .2, .2] for k in ("q_mean", "risk_score", "soft_surrogate_score")}}
    result = diagnostic.summarize_block(totals, scores)
    assert result["discount_changes_path_ordering_count"] == 3
    assert result["discount_changes_mean_ordering"]
    ordering = result["ordering"]["total_cost"]["q_mean"]
    assert ordering["actor_preferred_despite_lower_fitted_first_cost_paths"] == 1
    assert ordering["fitted_first_preferred_despite_higher_cost_paths"] == 1


def test_full_diagnostic_reproduces_saved_actor_and_preserves_parameters():
    torch.set_num_threads(1)
    torch.manual_seed(15)
    actor = EquilibriumGaussianPolicy(6, 2, 8, activation="tanh")
    critic = EnsembleQNet(6, 2, 8, 3)
    gain = torch.tensor([[1., 0., .5, 0., .4, 0.], [0., 1., 0., .5, 0., .4]])
    _, totals = diagnostic.paired_rollout(actor, gain, 1., 50, [60000, 60001])
    payload = {"source_protocol": {"config": {"hidden_dim": 8, "num_critics": 3, "actor_type": "centered-tanh-mlp",
                                              "gamma": .99, "policy_q_std_cap": 0., "use_context": False, "penalty_scale": 2.}},
               "actor_parameters": {k: v.tolist() for k, v in actor.state_dict().items()},
               "critic_parameters": {k: v.tolist() for k, v in critic.state_dict().items()},
               "fitted_gain": gain.tolist(), "alpha": .03, "weighted_lambda": .5, "reward_std": 1.2,
               "source_actor_results": {"1.0": {"50": {"evaluation_seeds": [60000, 60001],
                    "returns": (-totals["actor"]["total_cost"]).tolist(), "state_costs": totals["actor"]["state_cost"].tolist(),
                    "command_costs": totals["actor"]["command_cost"].tolist()}}}}
    output = diagnostic.diagnose(payload, "cpu", 2, (1.,), (50,))
    assert output["parameters_unchanged"] and output["optimizer_updates"] == output["fitting_interactions"] == 0
    assert output["controller_episodes"] == 4
    assert output["risk_beta"] == -3.
    assert all(error == 0 for error in output["radius_results"]["1.0"]["50"]["reproduction_max_abs_errors"].values())
