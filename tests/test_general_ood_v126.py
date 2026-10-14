from copy import deepcopy

import gymnasium as gym
import numpy as np
import pytest
import torch

from scripts import analyze_general_ood_v126_noise_sensitivity as worker
from scripts import summarize_general_ood_v126 as merger


def make_agent():
    env = gym.make("Hopper-v4")
    agent = worker.v125.make_agent(env, "cpu", -2.)
    state, _ = env.reset(seed=126)
    action = np.zeros(3, dtype=np.float32)
    next_state, reward, term, trunc, _ = env.step(action)
    agent.store_transition(state, action, reward, next_state, term or trunc)
    env.close()
    return agent


def test_noise_draws_match_legacy_and_preserve_global_rng():
    before = torch.get_rng_state().clone()
    expected = worker.noise_schedule(12400, 16, 3)
    assert torch.equal(before, torch.get_rng_state())
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(12400)
        legacy = torch.cat([torch.randn_like(torch.zeros(1, 3)) for _ in range(16)])
    assert torch.equal(expected, legacy)
    assert torch.equal(expected[:8], worker.noise_schedule(12400, 8, 3))
    assert not torch.equal(expected, worker.noise_schedule(12600, 16, 3))


def test_scaled_evaluation_matches_native_policy_and_stays_frozen():
    agent = make_agent()
    context = deepcopy(vars(agent.context_tracker))
    before = deepcopy(agent.actor.state_dict())
    rng = torch.get_rng_state().clone()
    legacy = worker.v124.evaluate_episode(agent, 150000, 12400, 16)
    result = worker.evaluate_episode(agent, 150000, 1., 12400, 16, agent)
    for key in ("length", "episode_return", "mean_velocity", "mean_policy_std", "mean_action_noise_l2"):
        assert result[key] == pytest.approx(legacy[key])
    assert result == worker.evaluate_episode(agent, 150000, 1., 12400, 16, agent)
    assert result["same_input_parity"]["max_action_abs_difference"] == 0.
    zero = worker.evaluate_episode(agent, 150000, 0., None, 16)
    assert zero == worker.evaluate_episode(agent, 150000, 0., 12600, 16) | {"noise_seed": None}
    deterministic = worker.v124.evaluate_episode(agent, 150000, None, 16)
    assert zero["episode_return"] == deterministic["episode_return"]
    assert agent.training_steps == 0 and agent.replay_buffer.size == 1
    for key, value in context.items():
        np.testing.assert_array_equal(value, vars(agent.context_tracker)[key])
    for key, value in before.items():
        assert torch.equal(value, agent.actor.state_dict()[key])
    assert torch.equal(rng, torch.get_rng_state())
    assert all(p.grad is None for p in agent.actor.parameters())
    state = torch.zeros(1, agent.config.state_dim+agent.context_dim)
    eps = torch.ones(1, 3)
    action, mean, std = worker.action_from_noise(agent.actor, state, eps, .5)
    torch.testing.assert_close(action, torch.tanh(mean+.5*std))


def fixture_rows():
    rows = []
    for seed, device in ((0, "cpu"), (1, "cpu"), (2, "cpu"), (2, "cuda")):
        spec = worker.stage_spec(seed, device)
        models = []
        for age in spec["ages"]:
            episodes = [dict(reset_seed=reset, noise_scale=scale, noise_seed=noise, noise_group=group,
                length=1000 if scale == 0 else 110, episode_return=200., mean_velocity=1.,
                mean_policy_std=.2, mean_action_noise_l2=scale*.2, mean_context_l2=.1,
                physical_termination=scale > 0, time_limit_truncation=scale == 0,
                same_input_parity=dict(max_mean_abs_difference=1e-6, max_std_abs_difference=1e-6,
                    max_action_abs_difference=1e-6) if device == "cuda" else None)
                for reset, scale, noise, group in worker.cells(spec)]
            models.append(dict(age=age, checkpoint=f"seed{seed}_{age}", config={}, alpha=.07,
                initial_context_normalizer={}, evaluations=episodes))
        rows.append(dict(revision=worker.REVISION, spec=spec, runtime={}, training_seed=seed,
            evaluation_device=device, models=models, optimizer_updates=0, new_training_interactions=0,
            source_readout_refitted=False, checkpoint_selection="final-not-best", reward_or_physics_changes=[],
            evaluation_interactions=sum(e["length"] for m in models for e in m["evaluations"])))
    return rows


def test_complete_summary_separates_noise_groups_and_devices():
    rows = fixture_rows()
    result = merger.summarize(rows)
    assert result["evaluation_episodes"] == 760 and result["independent_training_seeds"] == 3
    assert len(result["records"]) == 8 and len(result["records"][0]["groups"]) == 7
    assert result["new_training_interactions"] == result["optimizer_updates"] == 0
    rows[-1]["models"][0]["initial_context_normalizer"] = {"count": 1}
    with pytest.raises(AssertionError):
        merger.summarize(rows)
    with pytest.raises(AssertionError):
        merger.summarize(rows[:-1])
