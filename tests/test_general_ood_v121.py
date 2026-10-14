from copy import deepcopy
import random

import gymnasium as gym
import numpy as np
import pytest
import torch

from csbapr.agent import CSBAPRAgent
from scripts import run_general_ood_v121_locomotion_source as worker
from scripts import summarize_general_ood_v121 as merger


def make_pair(seed):
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = gym.make("Hopper-v4", max_episode_steps=3)
    config = worker.v39.v38.make_config("Hopper-v4", "fixed-penalty", "cpu")
    config.hidden_dim, config.batch_size = 16, 8
    return env, CSBAPRAgent(11, 3, config)


def test_fixed_protocol_and_unmodified_hopper():
    spec = worker.stage_spec()
    assert worker.SEEDS == (0, 1, 2) and spec["steps"] == 100000
    assert spec["random_action_warmup_steps"] == 0
    env, agent = make_pair(0)
    assert agent.config.penalty_scale == 0. and agent.config.use_context
    assert not agent.config.use_nau_actor and agent.pi_sym_torch is None
    assert (env.unwrapped._healthy_reward, env.unwrapped._ctrl_cost_weight,
            env.unwrapped._forward_reward_weight) == (1., .001, 1.)
    env.close()


def test_evaluation_restores_context_rng_replay_and_updates(monkeypatch):
    env, agent = make_pair(2)
    state, _ = env.reset(seed=2)
    agent.observe_context(state, np.zeros(3), 1., state, False)
    snapshot = deepcopy(vars(agent.context_tracker))
    rng = (torch.get_rng_state(), np.random.get_state(), random.getstate())
    select = agent.select_action
    def consuming_action(*args, **kwargs):
        torch.rand(1)
        np.random.rand()
        random.random()
        return select(*args, **kwargs)
    monkeypatch.setattr(agent, "select_action", consuming_action)
    result = worker.evaluate(agent, 150000, 2, 4)
    assert result["environment_step_calls"] > 0 and len(result["episodes"]) == 2
    assert agent.replay_buffer.size == agent.training_steps == 0
    assert torch.equal(torch.get_rng_state(), rng[0])
    np.testing.assert_array_equal(np.random.get_state()[1], rng[1][1])
    assert np.random.get_state()[2:] == rng[1][2:] and random.getstate() == rng[2]
    for key, value in snapshot.items():
        np.testing.assert_equal(vars(agent.context_tracker)[key], value)
    env.close()


def test_exact_resume_after_reset_updates_and_periodic_evaluation(tmp_path, monkeypatch):
    spec = dict(worker.stage_spec(True), steps=24, checkpoint_interval=4, diagnostic_interval=4,
                evaluation_interval=8, evaluation_max_steps=4)
    env, direct = make_pair(1)
    physics = (env.unwrapped.model.body_mass.copy(), env.unwrapped.model.dof_damping.copy(),
               env.unwrapped.model.opt.gravity.copy())
    expected = worker.train(direct, env, 1, spec, tmp_path/"direct/latest.pt")
    for value, other in zip(physics, (env.unwrapped.model.body_mass, env.unwrapped.model.dof_damping,
                                     env.unwrapped.model.opt.gravity)):
        np.testing.assert_array_equal(value, other)
    env.close()
    save = worker.save_progress
    def interrupt(agent, env, loop, spec, seed, path):
        save(agent, env, loop, spec, seed, path)
        if loop["step"] == 16:
            raise InterruptedError("after periodic evaluation and multiple episode resets")
    monkeypatch.setattr(worker, "save_progress", interrupt)
    env, partial = make_pair(1)
    checkpoint = tmp_path/"resume/latest.pt"
    with pytest.raises(InterruptedError):
        worker.train(partial, env, 1, spec, checkpoint)
    env.close()
    monkeypatch.setattr(worker, "save_progress", save)
    env, resumed = make_pair(99)
    actual = worker.train(resumed, env, 1, spec, checkpoint, checkpoint)
    assert expected == actual
    assert actual["optimizer_updates"] == 17
    assert all(r["metrics"].get("effective_beta", -2.) == -2. for r in actual["diagnostic_blocks"])
    for network in ("actor", "critic", "target_critic"):
        for key, value in getattr(direct, network).state_dict().items():
            assert torch.equal(value, getattr(resumed, network).state_dict()[key])
    for value, other in zip(physics, (env.unwrapped.model.body_mass, env.unwrapped.model.dof_damping,
                                     env.unwrapped.model.opt.gravity)):
        np.testing.assert_array_equal(value, other)
    env.close()


def test_summary_uses_training_seeds_and_retains_bad_source():
    records = []
    for seed, velocity in enumerate((.04, .5, 1.)):
        records.append(dict(revision=worker.REVISION, smoke=False, training_seed=seed,
            trained_from_scratch=True, training_mode="normal-only", config={}, spec=worker.stage_spec(),
            checkpoint_selection="final-not-best", checkpoint="native/final.pt", runtime={},
            training=dict(interactions=100000, optimizer_updates=99745, periodic_evaluations=[]),
            final_evaluation=dict(mean_episode_velocity=velocity, mean_episode_length=1000.,
                mean_return=1000.*(1.+velocity), physical_terminations=0,
                episodes=[dict(fraction_abs_velocity_below_point1=.9 if seed == 0 else .1)])))
    summary = merger.summarize(records)
    assert summary["independent_training_seeds"] == 3
    assert summary["aggregates"]["mean_forward_velocity"]["minimum"] == .04
    assert len(summary["per_training_seed"]) == 3
    with pytest.raises(AssertionError):
        merger.summarize(records[1:])
