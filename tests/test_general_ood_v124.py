from copy import deepcopy
import random

import gymnasium as gym
import numpy as np
import torch

from scripts import analyze_general_ood_v124_source_stability as worker
from scripts import summarize_general_ood_v124 as merger


def make_agent():
    env = gym.make("Hopper-v4")
    agent = worker.v122.make_agent(env, "cpu")
    state, _ = env.reset(seed=124)
    for _ in range(16):
        action = np.zeros(3, dtype=np.float32)
        next_state, reward, term, trunc, _ = env.step(action)
        agent.store_transition(state, action, reward, next_state, term or trunc)
        state = next_state
    env.close()
    return agent


def test_evaluation_and_gradients_preserve_models_replay_context_rng():
    agent = make_agent()
    before = {name: deepcopy(getattr(agent, name).state_dict()) for name in ("actor", "critic", "target_critic")}
    replay = deepcopy(agent.replay_buffer.buffer)
    context = deepcopy(vars(agent.context_tracker))
    numpy_rng, python_rng, torch_rng = np.random.get_state(), random.getstate(), torch.get_rng_state().clone()
    deterministic = worker.evaluate_episode(agent, 150000, None, 8)
    sampled = worker.evaluate_episode(agent, 150000, 12400, 8)
    assert sampled == worker.evaluate_episode(agent, 150000, 12400, 8)
    assert deterministic["mean_action_noise_l2"] == 0 and sampled["mean_action_noise_l2"] > 0
    assert sampled["length"] == 8 and sampled["diagnostic_cutoff"] and len(sampled["tail"]) == 5
    pools = worker.replay_pools(agent, 8)
    row = worker.gradient_probe(agent, pools["full"], 12400)
    assert row == worker.gradient_probe(agent, pools["full"], 12400)
    assert row["optimizer_updates"] == 0 and row["gradient_norms"]["risk"] > 0
    for name, saved in before.items():
        for key, value in saved.items():
            assert torch.equal(value, getattr(agent, name).state_dict()[key])
    for a, b in zip(replay, agent.replay_buffer.buffer):
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)
    for key, value in context.items():
        np.testing.assert_array_equal(value, vars(agent.context_tracker)[key])
    for a, b in zip(numpy_rng, np.random.get_state()):
        np.testing.assert_array_equal(a, b)
    assert python_rng == random.getstate() and torch.equal(torch_rng, torch.get_rng_state())
    assert agent.training_steps == 0 and all(p.grad is None for p in agent.actor.parameters())


def test_physical_conditions_match_actual_hopper_health():
    env = gym.make("Hopper-v4")
    env.reset(seed=124)
    raw = env.unwrapped
    assert worker.physical_state(raw)["unhealthy_conditions"] == []
    qpos, qvel = raw.data.qpos.copy(), raw.data.qvel.copy()
    qpos[1], qpos[2], qvel[0] = .5, .5, 150.
    raw.set_state(qpos, qvel)
    assert worker.physical_state(raw)["unhealthy_conditions"] == ["height", "angle", "state"]
    env.close()


def test_summary_keeps_noise_repeats_separate_from_training_seeds():
    rows = []
    for seed in (0, 1, 2):
        spec = dict(worker.stage_spec(), training_seed=seed)
        models = []
        for age in spec["ages"]:
            evaluations = [dict(reset_seed=reset, noise_seed=noise,
                action_mode="deterministic" if noise is None else "sampled", length=100 if noise is None else 200,
                episode_return=100., mean_velocity=1., physical_termination=True, time_limit_truncation=False,
                final_physical_state=dict(unhealthy_conditions=["angle"]))
                for reset in spec["reset_seeds"] for noise in (None, *spec["noise_seeds"])]
            models.append(dict(age=age, evaluations=evaluations, gradients={}))
        rows.append(dict(revision=worker.REVISION, training_seed=seed, spec=spec, models=models,
                         evaluation_device="cpu", optimizer_updates=0, evaluation_interactions=7000))
    result = merger.summarize(rows)
    assert result["evaluation_episodes"] == 120 and result["independent_training_seeds"] == 3
    assert result["evaluation_interactions"] == 21000 and result["new_training_interactions"] == 0
    assert result["records"][0]["paired_by_reset"][0]["difference"] == 100
