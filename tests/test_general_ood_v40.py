from dataclasses import asdict

import numpy as np
import pytest
import torch

from scripts import run_general_ood_v38 as v38
from scripts import run_general_ood_v40 as v40
from tests.test_general_ood_v38 import FakeAgent, FakeEnv


@pytest.mark.parametrize("variant", v40.VARIANTS)
def test_matched_config_and_constant_beta_after_identical_warmup(variant, monkeypatch):
    original = v38.make_config

    def small_config(environment, preset, device):
        config = original(environment, preset, device)
        config.hidden_dim, config.num_critics = 8, 2
        config.batch_size, config.buffer_size = 4, 32
        config.track_L_eff, config.bapr_warmup_iters = False, 2
        return config

    monkeypatch.setattr(v38, "make_config", small_config)
    torch.set_num_threads(1)
    agent = v40.make_agent("Pendulum-v1", variant, "cpu", FakeEnv())
    for i in range(4):
        agent.store_transition(np.full(3, i, dtype=np.float32), np.zeros(1),
                               float(i), np.full(3, i + 1, dtype=np.float32), 0.)
    metrics = [agent.update() for _ in range(3)]
    assert [m["effective_beta"] for m in metrics[:2]] == [-2., -2.]
    assert all(np.isfinite(m[key]) for m in metrics for key in v40.METRICS)
    if variant != "relative-adaptive":
        assert metrics[-1]["effective_beta"] == (-2. if variant == "fixed-minus2" else -2.3)


def test_configs_differ_only_in_predeclared_penalty_scale():
    configs = {v: asdict(v40.make_agent("Pendulum-v1", v, "cpu", FakeEnv()).config)
               for v in v40.VARIANTS}
    for variant, scale in (("fixed-minus2", 0.), ("fixed-minus2p3", .3)):
        assert configs[variant] == {**configs["relative-adaptive"], "penalty_scale": scale}


def test_switch_bins_exclude_initialization_and_keep_noop_ticks_and_update_peaks():
    collector = v40.SwitchDiagnostics()
    collector.on_mode(0, "normal", "normal")
    for step in range(1007):
        if step in (300, 900):
            collector.on_mode(step, "normal" if step == 300 else "rush", "rush")
        metrics = {key: float(step) for key in v40.SWITCH_METRICS} if step % 2 else None
        collector.observe(step, float(step), step % 137 == 0, metrics)
    events = collector.finish()["events"]
    assert [event["step"] for event in events] == [300, 900]
    assert [event["actual_change"] for event in events] == [True, False]
    first = events[0]["bins"]
    assert [(b["relative_start"], b["relative_end"]) for b in first] == [
        (offset, offset + 50) for offset in range(-250, 500, 50)]
    assert all(b["interactions"] == 50 and b["updates"] == 25 for b in first)
    post = first[5]
    assert post["means"]["surprise"] == 325.
    assert post["extrema"]["surprise"] == {"min": 301., "max": 349.}
    assert events[-1]["bins"][-1]["relative_end"] == 107
    assert events[-1]["bins"][-1]["interactions"] == 7


def test_observation_preserves_training_random_draws_and_update_sequence(monkeypatch):
    class DiagnosticAgent(FakeAgent):
        def __init__(self):
            super().__init__()
            self.actions = []

        def select_action(self, state):
            action = np.random.uniform(-1, 1, 1)
            self.actions.append(float(action[0]))
            return action

        def update(self, recent_rollout):
            value = float(np.random.uniform())
            return {key: value for key in v40.METRICS}

    monkeypatch.setattr(v38, "TRAIN_SWITCH_STEPS", 500)
    outputs, actions = [], []
    for collector in (None, v40.SwitchDiagnostics()):
        np.random.seed(17)
        agent = DiagnosticAgent()
        output = v38.train(agent, FakeEnv(137), "Pendulum-v1", 1, 1107,
                           metric_keys=v40.METRICS, diagnostic_interval=100,
                           switch_diagnostics=collector)
        output.pop("seconds")
        if collector is not None:
            events = output.pop("switch_diagnostics")["events"]
            assert [event["step"] for event in events] == [500, 1000]
        outputs.append(output)
        actions.append(agent.actions)
    assert outputs[0] == outputs[1]
    assert actions[0] == actions[1]
