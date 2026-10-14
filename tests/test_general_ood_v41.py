from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import torch

from csbapr.belief.surprise import RelativeSurpriseComputer
from scripts import probe_general_ood_v41 as probe
from scripts import run_general_ood_v38 as v38
from scripts import run_general_ood_v40 as v40
from tests.test_general_ood_v38 import FakeEnv


def config():
    return SimpleNamespace(surprise_ema=.1, max_run_length=20, hazard_rate=.02, base_variance=1.)


def test_observer_cadences_and_historical_reference_are_causal():
    rewards = np.tile([.9, 1.1], 768).astype(np.float32)
    changed = rewards.copy()
    changed[1024:] += 1.
    q = np.ones(1536, dtype=np.float32)
    norms = torch.ones(2)
    baseline = probe.observe_trace(rewards, q, norms, norms, config())
    result = probe.observe_trace(changed, q, norms, norms, config())
    assert [b["observations"] for b in result["step-overlap"]] == [1] + [256] * 5
    assert [b["observations"] for b in result["chunk256"]] == [1] * 6
    assert [b["observations"] for b in result["previous-chunk-reference"]] == [0] + [1] * 5
    for name in probe.OBSERVERS:
        assert baseline[name][:4] == result[name][:4]
        assert all(np.isfinite(v) for b in result[name] if b["means"] for v in b["means"].values())
    reference = result["previous-chunk-reference"]
    assert reference[3]["means"]["surprise_raw"] == 0.
    np.testing.assert_allclose(reference[4]["means"]["surprise_reward_zscore"], 10., rtol=1e-6)
    assert reference[5]["means"]["surprise_reward_zscore"] == 0.
    assert probe.paired_contrasts(baseline, baseline)["step-overlap"]["first_post256"]["surprise"] == 0.


def test_step_observer_reuses_existing_relative_surprise_values():
    rewards = torch.linspace(.5, 2., 1536)
    q = torch.linspace(1., 3., 1536)
    current, target = torch.tensor([1.2, 2.4]), torch.tensor([1., 2.])
    observed = probe.observe_trace(rewards, q, current, target, config())
    detector = RelativeSurpriseComputer(.1)
    values = []
    for end in range(1281, 1537):
        # Advance through the complete prefix before the final bin.
        if end == 1281:
            for prefix_end in range(256, end):
                detector.compute(rewards[prefix_end - 256:prefix_end], q[prefix_end - 256:prefix_end].mean().item(), current, target)
        values.append(detector.compute(rewards[end - 256:end], q[end - 256:end].mean().item(), current, target))
    np.testing.assert_allclose(observed["step-overlap"][-1]["means"]["surprise"], np.mean(values), rtol=1e-12)


def test_collection_pairs_prefix_restores_context_and_never_uses_optimizer(monkeypatch):
    class ModeEnv(FakeEnv):
        def __init__(self):
            super().__init__(episode_length=23)
            self.unwrapped = SimpleNamespace(model=SimpleNamespace(
                body_mass=np.ones(2), geom_friction=np.ones((2, 3)),
                opt=SimpleNamespace(gravity=np.array([0., 0., -9.81]))))

        def step(self, action):
            self.age += 1
            mass = self.unwrapped.model.body_mass.mean()
            return np.full(3, self.age * mass), float(mass), self.age == 23, False, {}

    original = v38.make_config

    def small_config(environment, preset, device):
        c = original(environment, preset, device)
        c.hidden_dim, c.num_critics = 8, 2
        c.buffer_size = 32
        return c

    monkeypatch.setattr(v38, "make_config", small_config)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    agent = v40.make_agent("Pendulum-v1", "fixed-minus2", "cpu", ModeEnv())
    snapshot, original_config = agent.snapshot_context(), asdict(agent.config)
    weights = [p.detach().clone() for m in (agent.actor, agent.critic, agent.target_critic) for p in m.parameters()]
    agent.update = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("optimizer invoked"))
    agent.store_transition = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("replay insertion invoked"))
    normal = probe.collect_trajectory(agent, ModeEnv(), 3, "normal", snapshot, budget=128, switch_step=64)
    changed = probe.collect_trajectory(agent, ModeEnv(), 3, "heavy", snapshot, budget=128, switch_step=64)
    assert probe.prefix_equal(normal, changed, 64)
    assert not np.array_equal(normal["rewards"][64:], changed["rewards"][64:])
    assert len(changed["q_spread"]) == len(changed["rewards"]) == 128
    np.testing.assert_array_equal(normal["next_states"][22], np.full(3, 23.))
    np.testing.assert_array_equal(normal["states"][23], np.zeros(3))
    assert normal["terminated"][22] and not normal["truncated"].any()
    np.testing.assert_array_equal(normal["dones"], normal["terminated"] | normal["truncated"])
    np.testing.assert_array_equal(normal["executed_actions"], normal["actions"])
    assert agent.context_tracker._count == 128 and snapshot._count == 0
    assert agent.training_steps == agent.replay_buffer.size == 0
    assert asdict(agent.config) == original_config
    now = [p for m in (agent.actor, agent.critic, agent.target_critic) for p in m.parameters()]
    assert all(torch.equal(a, b) for a, b in zip(weights, now))
