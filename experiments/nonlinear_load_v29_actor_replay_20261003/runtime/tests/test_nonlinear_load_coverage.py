from collections import Counter
from dataclasses import asdict

import numpy as np
import pytest
import torch

from csbapr.agent import CSBAPRAgent
from csbapr.envs import nonlinear_load_env as nonlinear
from scripts import test_lqr_convergence as training
from scripts import train_nonlinear_load_v15 as coverage
from scripts.summarize_nonlinear_load_v15 import summarize


def test_balanced_absolute_episode_schedule_and_budget():
    assert [coverage.reset_radius("balanced", e) for e in range(6)] == [0.2, 0.5, 1.0] * 2
    assert Counter(coverage.reset_radius("balanced", e) for e in range(500)) == {0.2: 167, 0.5: 167, 1.0: 166}
    assert all(coverage.reset_radius("narrow", e) == 0.2 for e in range(500))


def test_reset_coverage_keeps_disturbance_draws_paired():
    narrow = training.make_env("nonlinear_load", 1, 9)
    balanced = training.make_env("nonlinear_load", 1, 9)
    for episode in range(6):
        balanced.amplitude = coverage.reset_radius("balanced", episode)
        x, _ = narrow.reset()
        y, _ = balanced.reset()
        np.testing.assert_allclose(x[:4] / 0.2, y[:4] / balanced.amplitude, atol=3e-8)
        np.testing.assert_array_equal(x[4:], y[4:])
        for _ in range(200):
            narrow.step(np.zeros(2))
            balanced.step(np.zeros(2))
            np.testing.assert_array_equal(narrow.disturbance, balanced.disturbance)


@pytest.mark.parametrize("method", coverage.METHODS)
def test_resume_crosses_reset_cycle_without_changing_next_update(method, tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(training, "EP_LEN", 6)
    monkeypatch.setattr(nonlinear, "EP_LEN", 6)
    training.seed_training(9)
    config = training.make_config(method, "cpu", "nonlinear_load")
    config.hidden_dim, config.num_critics, config.batch_size = 8, 2, 4
    config.buffer_size, config.warmup_steps = 64, 2
    agent = CSBAPRAgent(6, 2, config)
    if method == coverage.CANDIDATE:
        agent.pi_sym_torch = training.known_reference("nonlinear_load")
    env = training.make_env("nonlinear_load", 1, 9)
    env.action_space.seed(9)
    protocol = {"method": method, "config": asdict(config), "condition": "balanced",
                "reset_cycle": [0.2, 0.5, 1.0], "reset_index": "absolute_episode"}
    history, steps = [], 0
    for episode in range(3):
        reward, steps = coverage.train_episode(env, agent, "balanced", episode, steps)
        history.append(reward)
    assert env.amplitude == 1.0
    path = str(tmp_path / "last.pt")
    training.save_training_checkpoint(agent, env, path, next_ep=3, env_steps=steps,
                                      history=history, protocol=protocol)
    expected, expected_steps = coverage.train_episode(env, agent, "balanced", 3, steps)
    training.seed_training(999)
    resumed = CSBAPRAgent(6, 2, config)
    resumed_env = training.make_env("nonlinear_load", 1, 999)
    resumed_env.action_space.seed(999)
    next_ep, restored_steps, restored_history = training.restore_training_checkpoint(resumed, resumed_env, path, protocol)
    actual, actual_steps = coverage.train_episode(resumed_env, resumed, "balanced", next_ep, restored_steps)
    assert next_ep == 3 and restored_history == history
    assert actual == expected and actual_steps == expected_steps
    assert resumed_env.amplitude == env.amplitude == 0.2
    assert resumed.training_steps == agent.training_steps
    for name in ("actor", "critic", "target_critic"):
        for actual, expected in zip(getattr(resumed, name).parameters(), getattr(agent, name).parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    np.testing.assert_array_equal(resumed_env.disturbance, env.disturbance)
    with pytest.raises(ValueError, match="exact protocol"):
        training.restore_training_checkpoint(resumed, resumed_env, path,
                                             {**protocol, "condition": "narrow", "reset_cycle": [0.2]})


def passing_runs():
    runs = {}
    for condition, method in [("balanced", m) for m in coverage.METHODS] + [("narrow", coverage.CANDIDATE)]:
        for seed in (9, 10):
            cost = 1.0 if condition == "balanced" and method == coverage.CANDIDATE else 2.0
            runs[(condition, method, seed)] = {"radius_results": {
                str(r): {str(a): {"controllers": {"actor": {"mean_cost": cost, "mean_early_state_cost": cost}},
                                 "paired_excess_costs": {"fitted": {"mean_total_cost": -0.1}}}
                         for a in coverage.SEVERITIES} for r in coverage.RADII}}
    return runs


@pytest.mark.parametrize("failure", [None, "comparison", "fitted", "early", "total"])
def test_decision_keeps_each_frozen_failure_on_hold(failure):
    runs = passing_runs()
    candidate = runs[("balanced", coverage.CANDIDATE, 10)]["radius_results"]["1.0"]["50"]
    if failure == "comparison":
        runs[("balanced", coverage.METHODS[1], 10)]["radius_results"]["0.5"]["1"]["controllers"]["actor"]["mean_cost"] = 0.5
    elif failure == "fitted":
        candidate["paired_excess_costs"]["fitted"]["mean_total_cost"] = 0.0
    elif failure in ("early", "total"):
        candidate["controllers"]["actor"]["mean_early_state_cost" if failure == "early" else "mean_cost"] = 2.0
    decision = summarize(runs, [9, 10])["decision"]
    assert decision["full_confirmation"] == (failure is None)
    assert decision["historical_v13_hold_unchanged"]
