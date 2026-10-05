import numpy as np
import torch

from csbapr.envs.lqr_env import ACTION_SCALE, Q, R
from csbapr.networks.critic import EnsembleQNet
from csbapr.networks.policy import EquilibriumGaussianPolicy, GaussianPolicy
from scripts import diagnose_nonlinear_load_v23 as diagnostic


def test_reparameterized_components_match_native_output_mean_gradients():
    class MeanPolicy(GaussianPolicy):
        def __init__(self):
            super().__init__(6, 2, 8)
            self.local_mean = torch.nn.Parameter(torch.tensor([[.2, -.1], [-.3, .4]]))

        def forward(self, states):
            return self.local_mean, torch.full_like(self.local_mean, -.7)

    torch.manual_seed(23)
    actor, critic = MeanPolicy(), EnsembleQNet(6, 2, 8, 3)
    states = torch.randn(2, 6) / 5
    gain = torch.zeros(2, 6)
    direction = -actor.local_mean.detach() / actor.local_mean.detach().norm(dim=-1, keepdim=True)
    alpha, beta, seeds = .01, -2.7, (92000, 92001)
    result = diagnostic.directional_scores(actor, critic, gain, states.numpy(), alpha, beta, seeds)
    for index, seed in enumerate(seeds):
        torch.manual_seed(seed)
        action, log_prob, _, _, _ = actor.evaluate(states)
        q = critic(states, action)
        components = {"q_mean": q.mean(0), "risk_penalty": beta * q.std(0), "entropy_bonus": -alpha * log_prob}
        for name, value in components.items():
            native = torch.autograd.grad(value.sum(), actor.local_mean, retain_graph=True)[0]
            np.testing.assert_allclose(result["expected"][name][index], (native * direction).sum(-1).numpy(), rtol=1e-6, atol=1e-8)
    assert all(p.grad is None for net in (actor, critic) for p in net.parameters())


def test_only_first_action_changes_and_finite_differences_match_known_cost(monkeypatch):
    envs = []

    class ConstantActor(torch.nn.Module):
        action_range = 1.

        def __init__(self):
            super().__init__()
            self.mean = torch.nn.Parameter(torch.full((2,), .2))

        def forward(self, states):
            return self.mean.expand(len(states), -1), torch.zeros((len(states), 2))

    class RecordingEnv:
        def __init__(self, seed):
            self.seed, self.t = seed, 0
            self.state = np.array([.1, .1, 0., 0.])
            self.actions = []
            envs.append(self)

        def reset(self):
            return np.r_[self.state, 0., 0.].astype(np.float32), {}

        def step(self, action):
            command = ACTION_SCALE * action.astype(float)
            cost = float(self.state @ Q @ self.state + command @ R @ command)
            self.actions.append(action.copy())
            if self.t == 0:
                self.state[:2] = action
            self.t += 1
            return np.r_[self.state, 0., 0.].astype(np.float32), -cost, self.t == 5, False, {"applied_disturbance": np.array([self.seed, self.t])}

    monkeypatch.setattr(diagnostic, "EP_LEN", 5)
    monkeypatch.setattr(diagnostic, "make_radius_env", lambda severity, seed, radius: RecordingEnv(seed))
    actors = {mode: ConstantActor() for mode in diagnostic.SOURCES}
    gains = {mode: torch.zeros(2, 6) for mode in actors}
    _, totals = diagnostic.paired_rollout(actors, gains, 1., 50, [60000])
    action = np.tanh(np.array([.2, .2]))
    action_derivative = (1 - action ** 2) * (-np.ones(2) / np.sqrt(2))
    projection = action @ action_derivative
    for model in totals.values():
        for h in diagnostic.STEPS:
            derivative = {k: (np.array(model[f"plus_{h}"][k]) - model[f"minus_{h}"][k]) / (2 * h)
                          for k in model["actor"]}
            np.testing.assert_allclose(derivative["total_cost"], 8.5 * projection, rtol=1e-4)
            np.testing.assert_allclose(derivative["discounted_cost"], (.5 + 2 * sum(.99 ** t for t in range(1, 5))) * projection, rtol=1e-4)
    for env in envs:
        np.testing.assert_allclose(env.actions[1:], np.tile(action, (4, 1)), rtol=1e-6)


def test_read_only_real_rollout_reproduces_both_source_actors():
    torch.set_num_threads(1)
    payloads, actors, gains = {}, {}, {}
    for seed, mode in enumerate(diagnostic.SOURCES):
        torch.manual_seed(seed)
        actors[mode] = EquilibriumGaussianPolicy(6, 2, 8, activation="tanh")
        gains[mode] = torch.tensor([[1., 0., .5, 0., .4, 0.], [0., 1., 0., .5, 0., .4]])
        critic = EnsembleQNet(6, 2, 8, 3)
        payloads[mode] = {"source_protocol": {"config": {"hidden_dim": 8, "num_critics": 3, "penalty_scale": 2.}},
                          "actor_parameters": {k: v.tolist() for k, v in actors[mode].state_dict().items()},
                          "critic_parameters": {k: v.tolist() for k, v in critic.state_dict().items()},
                          "fitted_gain": gains[mode].tolist(), "alpha": .01, "weighted_lambda": .2}
    seeds = [60000, 60001]
    _, totals = diagnostic.paired_rollout(actors, gains, 1., 50, seeds)
    for mode, payload in payloads.items():
        baseline = totals[mode]["actor"]
        payload["source_actor_results"] = {"1.0": {"50": {"evaluation_seeds": seeds,
            "returns": (-np.array(baseline["total_cost"])).tolist(), "state_costs": baseline["state_cost"],
            "command_costs": baseline["command_cost"]}}}
    output = diagnostic.diagnose(payloads, "cpu", 2, (1.,), (50,), diagnostic.NOISE_SEEDS[:2])
    assert output["parameters_unchanged"] and output["optimizer_updates"] == output["fitting_interactions"] == 0
    assert output["controller_episodes"] == 20
    for model in output["radius_results"]["1.0"]["50"]["models"].values():
        assert all(v == 0 for v in model["reproduction_max_abs_errors"].values())
        assert model["summary"] == diagnostic.summarize_block(model["continuations"], model["scores"])


def test_zero_direction_and_cost_score_signs():
    actor, critic = EquilibriumGaussianPolicy(6, 2, 8, activation="tanh"), EnsembleQNet(6, 2, 8, 3)
    with torch.no_grad():
        for p in actor.parameters():
            p.zero_()
    scores = diagnostic.directional_scores(actor, critic, torch.zeros(2, 6), np.zeros((2, 6)), .01, -2.5, diagnostic.NOISE_SEEDS[:2])
    assert not np.asarray(scores["latent_direction"]).any()
    assert all(not np.asarray(v).any() for mode in ("deterministic", "expected") for v in scores[mode].values())
    costs = {branch: {"total_cost": [2 + shift, 2 - shift], "discounted_cost": [2 + shift, 2 - shift]}
             for branch, shift in diagnostic.BRANCHES.items()}
    for mode in ("deterministic", "expected"):
        for name in scores[mode]:
            scores[mode][name] = [[1., -1.]]
    summary = diagnostic.summarize_block(costs, scores)
    ordering = summary["ordering"]["0.01"]["total_cost"]["expected"]["q_mean"]
    assert ordering["toward_fitted_lower_cost_paths"] == 1
    assert ordering["score_opposes_lower_cost_direction_paths"] == ordering["score_favors_higher_cost_direction_paths"] == 1
