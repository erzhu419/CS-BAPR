import json

import numpy as np
import torch

from csbapr.networks.critic import EnsembleQNet
from csbapr.networks.policy import EquilibriumGaussianPolicy, GaussianPolicy
from scripts import diagnose_nonlinear_load_v24 as diagnostic
from scripts.summarize_nonlinear_load_v24 import validate


def test_sampling_and_shifted_density_match_native_evaluate():
    class MeanPolicy(GaussianPolicy):
        def __init__(self):
            super().__init__(6, 2, 8)
            self.local_mean = torch.nn.Parameter(torch.tensor([[.2, -.1], [-.3, .4]]))
            self.shift = 0.

        def forward(self, states):
            return self.local_mean + self.shift, torch.full_like(self.local_mean, -.7)

    actor = MeanPolicy()
    states = torch.zeros(2, 6)
    noise = torch.randn((2, 2), generator=torch.Generator().manual_seed(92000))
    shift = torch.tensor([[.01, -.02], [-.01, .02]])
    for delta in (0., shift):
        actual_action, actual_density = diagnostic.policy_sample(actor, states, noise, delta)
        actor.shift = delta
        torch.manual_seed(92000)
        native_action, native_density, native_noise, _, _ = actor.evaluate(states)
        torch.testing.assert_close(actual_action, native_action, rtol=0, atol=0)
        torch.testing.assert_close(actual_density, native_density, rtol=0, atol=0)
        torch.testing.assert_close(noise, native_noise, rtol=0, atol=0)
        actor.shift = 0.


def test_common_noise_first_step_and_soft_return_timing(monkeypatch):
    recorded = []

    class ConstantActor(torch.nn.Module):
        action_range = 1.

        def __init__(self):
            super().__init__()
            self.mean = torch.nn.Parameter(torch.full((2,), .2))

        def forward(self, states):
            return self.mean.expand(len(states), -1), torch.full((len(states), 2), float(np.log(.3)))

    class LinearCritic(torch.nn.Module):
        def forward(self, states, actions):
            value = actions.sum(-1)
            return torch.stack((value - .1, value, value + .1))

    class RecordingEnv:
        def __init__(self, seed):
            self.seed, self.t = seed, 0
            self.state = np.array([.1, .1, 0., 0.])
            self.actions = []
            recorded.append(self)

        def reset(self):
            return np.r_[self.state, 0., 0.].astype(np.float32), {}

        def step(self, action):
            command = diagnostic.ACTION_SCALE * action.astype(float)
            cost = float(self.state @ diagnostic.Q @ self.state + command @ diagnostic.R @ command)
            self.actions.append(action.copy())
            if self.t == 0:
                self.state[:2] = action
            self.t += 1
            return np.r_[self.state, 0., 0.].astype(np.float32), -cost, self.t == 5, False, {"applied_disturbance": np.array([self.seed, self.t])}

    monkeypatch.setattr(diagnostic, "EP_LEN", 5)
    monkeypatch.setattr(diagnostic, "make_radius_env", lambda severity, seed, radius: RecordingEnv(seed))
    actors = {mode: ConstantActor() for mode in diagnostic.SOURCES}
    critics = {mode: LinearCritic() for mode in actors}
    gains = {mode: torch.zeros(2, 6) for mode in actors}
    terminals = {mode: {"alpha": .1, "beta": -2.5, "reward_std": 2.} for mode in actors}
    seeds, noises = [60000, 60001], diagnostic.NOISE_SEEDS[:2]
    _, raw = diagnostic.stochastic_rollout(actors, critics, gains, terminals, 1., 50, seeds, noises)
    per_draw = len(actors) * len(diagnostic.BRANCHES) * len(seeds)
    for draw, seed in enumerate(noises):
        generator = torch.Generator().manual_seed(seed)
        z = np.stack([torch.randn((2, 2), generator=generator).numpy().astype(float) for _ in range(5)])
        for index, env in enumerate(recorded[draw * per_draw:(draw + 1) * per_draw]):
            mode = diagnostic.SOURCES[index // (len(diagnostic.BRANCHES) * len(seeds))]
            branch = tuple(diagnostic.BRANCHES)[(index // len(seeds)) % len(diagnostic.BRANCHES)]
            path = index % len(seeds)
            action = np.asarray(env.actions).astype(float)
            oracle = np.tanh(.2 + .3 * z[:, path])
            oracle[0] = np.tanh(.2 - diagnostic.BRANCHES[branch] / np.sqrt(2) + .3 * z[0, path])
            np.testing.assert_allclose(action, oracle, rtol=1e-6, atol=1e-7)
            effort = .25 * np.square(action).sum(1)
            state = np.r_[.02, np.repeat(np.square(action[0]).sum(), 4)]
            discounts = .99 ** np.arange(5)
            density = (-.5 * z[:, path] ** 2 - np.log(.3) - .5 * np.log(2 * np.pi)
                       - np.log(1 - action ** 2 + 1e-6)).sum(1)
            costs = raw[mode]["continuations"][branch]
            expected_reward = -np.dot(discounts, state + effort) / 2
            expected_future = np.dot(discounts[1:], -.1 * density[1:])
            np.testing.assert_allclose(costs["normalized_reward_return"][draw, path], expected_reward, rtol=1e-6)
            np.testing.assert_allclose(costs["future_entropy_bonus"][draw, path], expected_future, rtol=1e-6, atol=1e-7)
            np.testing.assert_allclose(costs["soft_q_return"][draw, path], expected_reward + expected_future, rtol=1e-6, atol=1e-7)
            np.testing.assert_allclose(costs["soft_actor_return"][draw, path], expected_reward + expected_future - .1 * density[0], rtol=1e-6, atol=1e-7)


def test_real_rollout_read_only_raw_retention_and_json_validation(tmp_path):
    torch.set_num_threads(1)
    payloads = {}
    for seed, mode in enumerate(diagnostic.SOURCES):
        torch.manual_seed(seed)
        actor, critic = EquilibriumGaussianPolicy(6, 2, 8, activation="tanh"), EnsembleQNet(6, 2, 8, 3)
        payloads[mode] = {"source_protocol": {"config": {"hidden_dim": 8, "num_critics": 3, "penalty_scale": 2.}},
                          "actor_parameters": {k: v.tolist() for k, v in actor.state_dict().items()},
                          "critic_parameters": {k: v.tolist() for k, v in critic.state_dict().items()},
                          "fitted_gain": [[1., 0., .5, 0., .4, 0.], [0., 1., 0., .5, 0., .4]],
                          "alpha": .01, "weighted_lambda": .2, "reward_std": 2.}
    initial = np.stack([diagnostic.make_radius_env(50, seed, 1.).reset()[0] for seed in (60000, 60001)])
    source = {"radius_results": {"1.0": {"50": {"initial_observations": initial.tolist()}}}}
    output = diagnostic.diagnose(payloads, source, "cpu", 2, (1.,), (50,), diagnostic.NOISE_SEEDS[:2], tmp_path)
    assert output["parameters_unchanged"] and output["optimizer_updates"] == output["fitting_interactions"] == 0
    assert output["controller_episodes"] == 40 and len(output["raw_files"]) == 1
    raw = json.loads((tmp_path / 'radius1.0_load50.json').read_text())
    assert np.asarray(raw["models"]["floor"]["continuations"]["actor"]["total_cost"]).shape == (2, 2)
    versions = {"torch": torch.__version__, "numpy": np.__version__, "scipy": diagnostic.scipy.__version__,
                "pysindy": diagnostic.pysindy.__version__, "sklearn": diagnostic.sklearn.__version__}
    manifest = {"revision": diagnostic.REVISION, "input_revision": diagnostic.INPUT_REVISION,
                "source_tasks": {m: {"id": m} for m in diagnostic.SOURCES}, "configs": {}, "environment_settings": {},
                "runtime_requirement": {**versions, "python_minor_versions": ['.'.join(diagnostic.sys.version.split()[0].split('.')[:2])]}}
    data = {**output, "revision": diagnostic.REVISION, "input_revision": diagnostic.INPUT_REVISION,
            "source_ids": {m: m for m in diagnostic.SOURCES}, "training_seed": 18, "checkpoint_selection": "last",
            "preflight_only": True, "configs": {}, "environment_settings": {}, "gamma": .99,
            "finite_difference_steps": list(diagnostic.STEPS), "runtime": {**versions, "python": diagnostic.sys.version, "device": "cuda"}}
    validate(json.loads(json.dumps(data)), manifest, preflight=True)


def test_noise_average_direction_and_score_signs():
    delta = np.array([[1., -1.], [3., -3.]])
    raw = {"continuations": {branch: {key: np.full((2, 2), 2.) + shift * delta for key in diagnostic.COSTS + diagnostic.RETURNS}
                             for branch, shift in diagnostic.BRANCHES.items()},
           "critic_values": {branch: {key: np.full((2, 2), 3.) - shift * delta for key in diagnostic.SCORES}
                             for branch, shift in diagnostic.BRANCHES.items()}}
    scores = {"expected": {key: (-delta).tolist() for key in diagnostic.SCORES if key != "unpenalized_soft_score"},
              "latent_distance_to_fitted": [1., 1.]}
    compact = diagnostic.compact_model(raw, scores)
    summary = compact['summary']
    ordering = summary['ordering']['0.005']['q_mean_vs_soft_q_return']
    assert ordering == {"improving_return_paths": 1, "prediction_opposes_improving_paths": 1,
                        "prediction_favors_worsening_paths": 1, "sign_disagreements": 2}
    assert all(v == 0 for v in summary['finite_difference_step_sign_disagreements'].values())
    np.testing.assert_allclose(compact['per_draw_mean_derivatives']['0.005']['continuations']['soft_q_return'], [0., 0.])
    np.testing.assert_allclose(compact['analytic_score_derivatives']['q_mean'], [-2., 2.])
