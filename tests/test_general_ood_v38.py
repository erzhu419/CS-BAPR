from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import run_general_ood_v38 as runner
from scripts.v38_legacy_belief import BeliefTracker as LegacyBeliefTracker


class FakeSpace:
    shape = (1,)
    low, high = np.array([-1.0]), np.array([1.0])

    def seed(self, seed):
        self.seed_value = seed


class FakeEnv:
    def __init__(self, episode_length=5):
        self.action_space = FakeSpace()
        self.observation_space = SimpleNamespace(shape=(3,))
        self.unwrapped = SimpleNamespace(m=1.0, g=10.0, l=1.0)
        self.episode_length = episode_length

    def reset(self, seed):
        self.age = 0
        return np.zeros(3), {}

    def step(self, action):
        self.age += 1
        return np.zeros(3), 1.0, self.age == self.episode_length, False, {}


class FakeAgent:
    config = SimpleNamespace(surprise_window=8)

    def __init__(self):
        self.dones = []

    def reset_context(self):
        pass

    def select_action(self, state):
        return np.zeros(1)

    def _current_context_np(self):
        return np.zeros(2)

    def store_transition(self, state, action, reward, next_state, done):
        self.dones.append(done)

    def update(self, recent_rollout):
        return {key: 0.0 for key in runner.METRICS}


def test_exact_budget_cutoff_does_not_create_terminal():
    agent = FakeAgent()
    result = runner.train(agent, FakeEnv(), "Pendulum-v1", seed=0, budget=7)
    assert result["interactions"] == result["optimizer_update_steps"] == 7
    assert agent.dones == [0., 0., 0., 0., 1., 0., 0.]
    assert not result["last_segment_terminated"]
    assert result["recent_completed_episode_returns"] == [5.0]


def test_mode_schedule_is_independent_of_episode_termination(monkeypatch):
    monkeypatch.setattr(runner, "TRAIN_SWITCH_STEPS", 2)
    runs = [runner.train(FakeAgent(), FakeEnv(length), "Pendulum-v1", 1, 7)
            for length in (2, 5)]
    assert runs[0]["mode_events"] == runs[1]["mode_events"]
    assert [event["step"] for event in runs[0]["mode_events"]] == [0, 2, 4, 6]


def test_variants_have_matched_nonbelief_configuration():
    configs = {v: asdict(runner.make_config("Hopper-v4", v, "cpu")) for v in runner.VARIANTS}
    assert configs["legacy-belief"] == configs["corrected-belief"]
    assert configs["fixed-penalty"] == {**configs["corrected-belief"], "penalty_scale": 0.0}
    assert configs["corrected-belief"]["use_context"]
    assert not configs["corrected-belief"]["use_nau_actor"]
    assert configs["corrected-belief"]["weight_sym"] == configs["corrected-belief"]["jac_weight"] == 0


def test_recorded_dimensions_match_hopper_spaces():
    env = SimpleNamespace(observation_space=SimpleNamespace(shape=(11,)),
                          action_space=SimpleNamespace(shape=(3,)))
    agent = runner.make_agent("Hopper-v4", "corrected-belief", "cpu", env)
    assert agent.config.state_dim == agent.state_dim == 11
    assert agent.config.action_dim == agent.action_dim == 3


@pytest.mark.parametrize("variant", runner.VARIANTS)
def test_checkpoint_keeps_selected_belief_and_update_diagnostics(variant, tmp_path, monkeypatch):
    original = runner.make_config

    def small_config(environment, variant, device):
        config = original(environment, variant, device)
        config.hidden_dim, config.num_critics = 8, 2
        config.batch_size, config.buffer_size = 4, 32
        config.track_L_eff = False
        return config

    monkeypatch.setattr(runner, "make_config", small_config)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    env = FakeEnv()
    agent = runner.make_agent("Pendulum-v1", variant, "cpu", env)
    for i in range(4):
        agent.store_transition(np.full(3, i, dtype=np.float32), np.zeros(1),
                               float(i), np.full(3, i + 1, dtype=np.float32), 0.)
    metrics = agent.update()
    assert metrics["belief_reset_probability"] == agent.belief_tracker.belief[0]
    assert metrics["belief_effective_window"] == agent.belief_tracker.effective_window
    checkpoint = str(tmp_path / "final.pt")
    agent.save(checkpoint, extra_state={"stage": "training-test"})
    restored = runner.make_agent("Pendulum-v1", variant, "cpu", env)
    restored.load(checkpoint)
    assert isinstance(restored.belief_tracker, LegacyBeliefTracker) == (variant == "legacy-belief")
    np.testing.assert_array_equal(restored.belief_tracker.belief, agent.belief_tracker.belief)
    assert restored.training_steps == agent.training_steps


def test_inference_snapshot_preserves_context_without_replay(tmp_path):
    env = FakeEnv()
    agent = runner.make_agent("Pendulum-v1", "corrected-belief", "cpu", env)
    agent.observe_context(np.zeros(3), np.zeros(1), 1., np.ones(3))
    agent.observe_context(np.ones(3), np.zeros(1), 2., np.full(3, 3.))
    checkpoint = str(tmp_path / "inference.pt")
    agent.save(checkpoint)
    stored = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert stored["training_state"] is None
    restored = runner.make_agent("Pendulum-v1", "corrected-belief", "cpu", env)
    restored.load(checkpoint)
    assert restored.context_tracker._count == 2
    for key in ("_mean", "_m2", "_ema"):
        np.testing.assert_array_equal(getattr(restored.context_tracker, key),
                                      getattr(agent.context_tracker, key))
    restored.reset_context()
    assert restored.context_tracker._count == 2
    np.testing.assert_array_equal(restored.context_tracker._mean, agent.context_tracker._mean)


def test_real_pendulum_evaluation_uses_wrapper_config_and_isolated_context():
    agent = FakeAgent()
    agent.config = SimpleNamespace(max_steps_per_episode=3)
    agent.actor = SimpleNamespace(eval=lambda: None, requires_grad_=lambda value: None)
    agent.training_steps = 13
    agent.context_count = 7
    agent.select_action = lambda state, deterministic=False: np.zeros(1)
    agent.snapshot_context = lambda: agent.context_count
    agent.restore_context = lambda snapshot: setattr(agent, "context_count", snapshot)
    agent.observe_context = lambda *args, **kwargs: setattr(agent, "context_count", agent.context_count + 1)
    modes, fixed = runner.evaluate(agent, "Pendulum-v1", primary_count=2, fixed_count=1)
    assert set(modes) == {"id", "markov_50x"}
    assert set(fixed) == {"low_load", "rush_load", "mega_event", "recovery_aftershock"}
    assert all(len(result["rewards"]) == 2 for result in modes.values())
    assert all(len(result["rewards"]) == 1 for result in fixed.values())
    assert agent.context_count == 7 and agent.training_steps == 13
