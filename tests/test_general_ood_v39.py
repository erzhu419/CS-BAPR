from dataclasses import asdict
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from csbapr.belief.surprise import RelativeSurpriseComputer, SurpriseComputer
from csbapr.belief.tracker import BeliefTracker
from scripts import run_general_ood_v38 as v38
from scripts import run_general_ood_v39 as v39
from tests.test_general_ood_v38 import FakeAgent, FakeEnv


def test_raw_instrumentation_preserves_frozen_v38_values():
    path = Path(__file__).resolve().parents[1] / "experiments/general_ood_v38_belief_20261005/runtime/csbapr/belief/surprise.py"
    spec = importlib.util.spec_from_file_location("frozen_v38_surprise", path)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    old, new = legacy.SurpriseComputer(ema_alpha=0.1), SurpriseComputer(ema_alpha=0.1)
    for reward, disagreement, current, target in ((0, 1, 100, 100), (1, 2, 150, 100), (1, .5, 160, 140), (-2, 4, 200, 150)):
        inputs = (torch.full((8, 1), float(reward)), torch.tensor(float(disagreement)),
                  torch.tensor([float(current)]), torch.tensor([float(target)]))
        assert new.compute(*inputs) == old.compute(*inputs)
        components = new.last_components
        assert sum(components[k] for k in ("surprise_reward_dominant", "surprise_q_std_dominant", "surprise_reg_dominant")) == 1
    new.reset()
    assert new.last_components == {} and new.prev_q_std is None


def test_relative_drift_is_invariant_to_norm_scale():
    values = []
    for scale in (1., 100.):
        detector = RelativeSurpriseComputer(ema_alpha=1.)
        values.append(detector.compute(torch.zeros(8), torch.tensor(1.),
                                      torch.tensor([120., 240.]) * scale,
                                      torch.tensor([100., 200.]) * scale))
        assert detector.last_components["surprise_reg_dominant"] == 1
    np.testing.assert_allclose(values, [0.1, 0.1])


def test_relative_mode_preserves_one_sided_q_std_signal():
    detector = RelativeSurpriseComputer(ema_alpha=1.)
    for q_std, expected in ((2., 0.), (4., 1.), (1., 0.)):
        assert detector.compute(torch.zeros(8), torch.tensor(q_std)) == expected
        assert detector.last_components["surprise_q_std_spike"] == expected


def test_norm_scale_alone_does_not_force_relative_filter_to_reset():
    raw, relative = SurpriseComputer(ema_alpha=1.), RelativeSurpriseComputer(ema_alpha=1.)
    trackers = [BeliefTracker(20, .02, 1.) for _ in range(2)]
    for _ in range(40):
        for detector, tracker in zip((raw, relative), trackers):
            surprise = detector.compute(torch.zeros(8), torch.tensor(1.),
                                        torch.tensor([12000.]), torch.tensor([10000.]))
            tracker.update(surprise)
    assert trackers[0].belief[0] > .99
    assert trackers[1].effective_window > 15


@pytest.mark.parametrize("variant", v39.VARIANTS)
def test_matched_agent_metrics_and_constant_coefficients(variant, monkeypatch):
    original = v38.make_config

    def small_config(environment, preset, device):
        config = original(environment, preset, device)
        config.hidden_dim, config.num_critics = 8, 2
        config.batch_size, config.buffer_size = 4, 32
        config.track_L_eff, config.bapr_warmup_iters = False, 0
        return config

    monkeypatch.setattr(v38, "make_config", small_config)
    torch.set_num_threads(1)
    agent = v39.make_agent("Pendulum-v1", variant, "cpu", FakeEnv())
    for i in range(4):
        agent.store_transition(np.full(3, i, dtype=np.float32), np.zeros(1),
                               float(i), np.full(3, i + 1, dtype=np.float32), 0.)
    metrics = agent.update()
    assert all(np.isfinite(metrics[k]) for k in v39.METRICS)
    if variant in ("fixed-minus2", "fixed-minus3"):
        assert metrics["effective_beta"] == (-2. if variant == "fixed-minus2" else -3.)
    assert isinstance(agent.surprise_computer, RelativeSurpriseComputer) == (variant != "raw-adaptive")


def test_four_arm_configs_differ_only_in_minus2_penalty():
    configs = {v: asdict(v39.make_agent("Pendulum-v1", v, "cpu", FakeEnv()).config) for v in v39.VARIANTS}
    assert configs["raw-adaptive"] == configs["relative-adaptive"] == configs["fixed-minus3"]
    assert configs["fixed-minus2"] == {**configs["relative-adaptive"], "penalty_scale": 0.0}


def test_short_diagnostic_blocks_keep_all_interactions():
    class DiagnosticAgent(FakeAgent):
        def update(self, recent_rollout):
            return {key: 0.0 for key in v39.METRICS}

    result = v38.train(DiagnosticAgent(), FakeEnv(), "Pendulum-v1", 1, 7,
                      metric_keys=v39.METRICS, diagnostic_interval=2)
    assert [b["step"] for b in result["diagnostic_blocks"]] == [2, 4, 6, 7]
    assert sum(b["updates"] for b in result["diagnostic_blocks"]) == 7
    assert not result["last_segment_terminated"]
