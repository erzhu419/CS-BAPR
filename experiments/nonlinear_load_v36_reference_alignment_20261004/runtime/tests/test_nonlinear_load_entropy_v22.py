from dataclasses import asdict

import numpy as np
import pytest
import torch

from csbapr.envs import nonlinear_load_env as nonlinear
from scripts import test_lqr_convergence as training
from scripts.train_nonlinear_load_v19 import identify_reference
from scripts.train_nonlinear_load_v20 import ScopedReferenceAgent
from scripts.train_nonlinear_load_v22 import EntropyFloorAgent, make_config, train_episode
from scripts.summarize_nonlinear_load_v22 import compare_pre_floor_prefix, summarize, validate_temperature


def small_config(mode):
    config = make_config(mode, "cpu")
    config.hidden_dim, config.num_critics, config.batch_size = 8, 2, 4
    config.buffer_size, config.warmup_steps = 64, 2
    return config


def env_for(seed):
    env = training.make_env("nonlinear_load", 1, seed)
    env.action_space.seed(seed)
    return env


def test_config_intervention_changes_only_entropy_floor_switch():
    a, b = [asdict(make_config(mode, "cpu")) for mode in ("floor", "no-floor")]
    assert {k for k in a if a[k] != b[k]} == {"enable_entropy_floor"}
    assert a["enable_entropy_floor"] and not b["enable_entropy_floor"]


@pytest.mark.parametrize("seed", [17, 18])
@pytest.mark.parametrize("raw_below_floor", [False, True])
def test_native_floor_updates_and_effective_intervention(seed, raw_below_floor):
    torch.set_num_threads(1)
    agents, metrics = [], []
    for mode in (None, "floor", "no-floor"):
        training.seed_training(seed)
        config = small_config(mode or "floor")
        agent = (ScopedReferenceAgent(config, seed, "full", episode_length=6) if mode is None
                 else EntropyFloorAgent(config, seed, episode_length=6))
        identify_reference(agent, env_for(seed), "cubic-local")
        for index in range(8):
            state = np.full(6, index / 20., dtype=np.float32)
            agent.replay_buffer.push(state, np.full(2, .1), -.2, state / 2, 0.)
        if raw_below_floor:
            with torch.no_grad():
                agent.log_alpha.fill_(-7.)
        training.seed_training(seed)
        metrics.append(agent.update())
        agents.append(agent)
    native, floor, no_floor = agents
    assert metrics[0] == metrics[1]
    assert native.reference_identification == floor.reference_identification == no_floor.reference_identification
    assert native.sampling_state() == floor.sampling_state() == no_floor.sampling_state()
    for name in ("actor", "critic", "target_critic"):
        for a, b in zip(getattr(native, name).parameters(), getattr(floor, name).parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert native.alpha == floor.alpha
    assert floor.log_alpha.item() == no_floor.log_alpha.item()
    assert floor.below_nominal_floor_updates == no_floor.below_nominal_floor_updates == int(raw_below_floor)
    assert floor.floor_binding_updates == int(raw_below_floor) and no_floor.floor_binding_updates == 0
    if raw_below_floor:
        assert floor.alpha == .01 and 0 < no_floor.alpha < .01
        assert any(not torch.equal(a, b) for a, b in zip(floor.actor.parameters(), no_floor.actor.parameters()))
    else:
        assert metrics[1] == metrics[2]
        for name in ("actor", "critic", "target_critic"):
            for a, b in zip(getattr(floor, name).parameters(), getattr(no_floor, name).parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["floor", "no-floor"])
def test_resume_restores_temperature_trace_and_next_episode(mode, tmp_path, monkeypatch):
    torch.set_num_threads(1)
    config = small_config(mode)
    training.seed_training(17)
    agent = EntropyFloorAgent(config, 17, episode_length=6)
    identify_reference(agent, env_for(17), "cubic-local")
    with torch.no_grad():
        agent.log_alpha.fill_(-7.)
    monkeypatch.setattr(training, "EP_LEN", 6)
    monkeypatch.setattr(nonlinear, "EP_LEN", 6)
    config.max_steps_per_episode = 6
    env = env_for(17)
    protocol = {"entropy_mode": mode, "config": asdict(config), "reset_cycle": [.2, .5, 1.]}
    history, steps = [], 0
    for episode in range(3):
        reward, steps = train_episode(env, agent, "balanced", episode, steps)
        history.append(reward)
        agent.record_temperature(episode)
    path = str(tmp_path / "last.pt")
    training.save_training_checkpoint(agent, env, path, next_ep=3, env_steps=steps, history=history, protocol=protocol)
    expected, expected_steps = train_episode(env, agent, "balanced", 3, steps)
    agent.record_temperature(3)
    training.seed_training(999)
    resumed, resumed_env = EntropyFloorAgent(config, 999, episode_length=6), env_for(999)
    episode, restored_steps, restored_history = training.restore_training_checkpoint(resumed, resumed_env, path, protocol)
    actual, actual_steps = train_episode(resumed_env, resumed, "balanced", episode, restored_steps)
    resumed.record_temperature(episode)
    assert (episode, restored_history, actual, actual_steps) == (3, history, expected, expected_steps)
    assert resumed.temperature_trace == agent.temperature_trace
    assert resumed.floor_binding_updates == agent.floor_binding_updates
    assert resumed.sampling_state() == agent.sampling_state()
    assert resumed.reference_identification == agent.reference_identification
    for name in ("actor", "critic", "target_critic"):
        for a, b in zip(getattr(resumed, name).parameters(), getattr(agent, name).parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    wrong = EntropyFloorAgent(small_config("no-floor" if mode == "floor" else "floor"), 17, episode_length=6)
    with pytest.raises(ValueError, match="entropy-floor"):
        wrong.load(path)


@pytest.mark.parametrize("failure", [None, "total", "early", "small", "fitted"])
def test_frozen_gates_retain_each_seed_and_load(failure):
    runs = {(mode, seed): {"radius_results": {
        str(radius): {str(load): {"controllers": {"actor": {"mean_cost": 1. if mode == "no-floor" else 2.,
                                                  "mean_early_state_cost": 1. if mode == "no-floor" else 2.}},
                                 "paired_excess_costs": {"fitted": {"mean_total_cost": -.1}}}
                      for load in (1, 50)} for radius in (.2, .5, 1.)}}
            for mode in ("floor", "no-floor") for seed in (17, 18)}
    block = runs[("no-floor", 18)]["radius_results"]
    if failure in ("total", "early"):
        block["1.0"]["50"]["controllers"]["actor"]["mean_cost" if failure == "total" else "mean_early_state_cost"] = 2.
    elif failure == "small":
        block["0.2"]["50"]["controllers"]["actor"]["mean_cost"] = 2.01
    elif failure == "fitted":
        block["0.5"]["1"]["paired_excess_costs"]["fitted"]["mean_total_cost"] = 0.
    decision = summarize(runs, (17, 18))["decision"]
    assert decision["ready_for_learner_replication"] == (failure is None)
    assert decision["historical_hold_unchanged"]


@pytest.mark.parametrize("mode", ["floor", "no-floor"])
def test_temperature_summary_validates_switch_and_binding_counts(mode):
    enabled = mode == "floor"
    trace = [{"episode": e + 1, "training_updates": max(0, (e + 1) * 200 - 999),
              "alpha": .2 if e < 4 else (.01 if enabled else .001),
              "raw_alpha": 1. if e < 4 else .001, "reward_std": 1.,
              "below_nominal_floor_updates": max(0, (e + 1) * 200 - 999),
              "floor_binding_updates": max(0, (e + 1) * 200 - 999) if enabled else 0}
             for e in range(6)]
    data = {"temperature_trace": trace, "training_episodes": 6, "training_updates": 201,
            "entropy_mode": mode, "config": asdict(make_config(mode, "cpu")),
            "below_nominal_floor_updates": 201, "floor_binding_updates": 201 if enabled else 0}
    validate_temperature(data)
    trace[-1]["alpha"] = .02
    with pytest.raises(AssertionError):
        validate_temperature(data)


@pytest.mark.parametrize("diverge", [False, True])
def test_prefix_comparison_detects_pre_treatment_divergence_without_using_gpu_labels(diverge):
    runs = {}
    for mode in ("floor", "no-floor"):
        runs[mode, 17] = {"training_returns": [-1., -2., -3.], "runtime": {"gpu": "same-gpu"},
                         "temperature_trace": [{"episode": e + 1, "alpha": .5, "raw_alpha": .5,
                                                "reward_std": 1., "below_nominal_floor_updates": int(e == 2)}
                                               for e in range(3)]}
    if diverge:
        runs["no-floor", 17]["training_returns"][1] += .25
    result = compare_pre_floor_prefix(runs, (17,))["17"]
    assert result["episodes_before_either_floor_crossing"] == 2
    assert result["recorded_episode_traces_match"] == (not diverge)
    assert result["differences"]["training_return"]["first_different_episode"] == (2 if diverge else None)
    assert result["differences"]["training_return"]["max_abs_difference"] == (.25 if diverge else 0.)
