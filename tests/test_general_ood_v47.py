from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import probe_general_ood_v41 as v41
from scripts import probe_general_ood_v46 as v46
from scripts import probe_general_ood_v47 as probe
from scripts import run_general_ood_v38 as v38
from scripts import run_general_ood_v40 as v40
from scripts.summarize_general_ood_v47 import validate_run
from tests.test_general_ood_v38 import FakeEnv
from tests.test_general_ood_v45 import fixture


def test_collector_applies_both_switches_before_transition_and_preserves_old_path(monkeypatch):
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
        c.hidden_dim, c.num_critics, c.buffer_size = 8, 2, 32
        return c

    monkeypatch.setattr(v38, "make_config", small_config)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    agent = v40.make_agent("Pendulum-v1", "fixed-minus2", "cpu", ModeEnv())
    snapshot = agent.snapshot_context()
    weights = [p.detach().clone() for m in (agent.actor, agent.critic, agent.target_critic) for p in m.parameters()]
    agent.update = lambda *a, **k: (_ for _ in ()).throw(AssertionError("optimizer invoked"))
    agent.store_transition = lambda *a, **k: (_ for _ in ()).throw(AssertionError("replay insertion invoked"))
    old = v41.collect_trajectory(agent, ModeEnv(), 3, "heavy", snapshot, budget=128, switch_step=32)
    stay = probe.collect_trajectory(agent, ModeEnv(), 3, {32: "heavy"}, snapshot, budget=128)
    changed = probe.collect_trajectory(agent, ModeEnv(), 3, {32: "heavy", 64: "normal"}, snapshot, budget=128)
    assert all(np.array_equal(old[k], stay[k]) for k in old)
    assert v41.prefix_equal(stay, changed, 64)
    assert all(np.array_equal(stay[k][64], changed[k][64]) for k in ("states", "executed_actions", "contexts"))
    assert changed["rewards"][31] == changed["rewards"][64] == 1.
    assert changed["rewards"][32] == changed["rewards"][63] == stay["rewards"][64] > 1.
    np.testing.assert_array_equal(changed["next_states"][22], np.full(3, 23.))
    np.testing.assert_array_equal(changed["states"][23], np.zeros(3))
    assert changed["terminated"][22] and not changed["truncated"].any()
    assert agent.training_steps == agent.replay_buffer.size == 0
    assert snapshot._count == 0 and agent.context_tracker._count == 128
    now = [p for m in (agent.actor, agent.critic, agent.target_critic) for p in m.parameters()]
    assert all(torch.equal(a, b) for a, b in zip(weights, now))


def test_observer_preserves_v46_and_does_not_leak_future_or_reset_at_episode_end():
    model, trace = fixture()
    for key, values in trace.items():
        trace[key] = np.concatenate((values, values[:256]))
    trace["errors"][1024:] = 20.
    trace["dones"][1250] = trace["terminated"][1250] = True
    shortened = {k: v[:1536] for k, v in trace.items()}
    old_arms, _, _ = v46.observe(shortened, model)
    arms, histories, signals, _ = probe.observe(trace, model)
    for arm in probe.ARM_CALIBRATIONS:
        for phase in v46.PHASES:
            assert arms[arm]["phases"][phase] == old_arms[arm]["phases"][phase]
    changed = deepcopy(trace)
    changed["errors"][1280:] = .01
    new_arms, new_histories, new_signals, _ = probe.observe(changed, model)
    for arm in histories:
        np.testing.assert_array_equal(histories[arm][:1280], new_histories[arm][:1280])
        assert new_arms[arm]["phases"]["prefix1280"] == arms[arm]["phases"]["prefix1280"]
    for channel in signals:
        np.testing.assert_array_equal(signals[channel][:1280], new_signals[channel][:1280])
        assert not np.array_equal(signals[channel][1280:], new_signals[channel][1280:])


def test_worker_results_reject_incomplete_pairs_or_changed_source():
    root = Path(__file__).resolve().parents[1]
    path = root / "results/general_ood_v47_multiple_switch/preflight/general_ood_v47_multiple_switch_fixed-minus2.json"
    if not path.exists():
        pytest.skip("worker preflight not collected")
    data = json.loads(path.read_text())
    manifest = json.loads((root / "experiments/general_ood_v47_multiple_switch_20261006/dispatch_manifest.json").read_text())
    validate_run(data, manifest, "preflight", root)
    altered = deepcopy(data)
    altered["records"].pop()
    with pytest.raises(AssertionError, match="incomplete-or-duplicate-branches"):
        validate_run(altered, manifest, "preflight", root)
    altered = deepcopy(data)
    altered["records"][1]["arms"]["global-robust"]["phases"]["prefix1280"]["penalty"]["mean"] += .01
    with pytest.raises(AssertionError, match="unpaired-belief-history"):
        validate_run(altered, manifest, "preflight", root)
