from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch

from scripts import analyze_general_ood_v120_mechanism as worker
from scripts.v52_agent import ContinuationAgent
from test_general_ood_v58 import shared_start


def test_reward_decomposition_matches_native_hopper_velocity():
    env = gym.make("Hopper-v4")
    env.reset(seed=120)
    settings = worker.reward_settings(env)
    for i in range(20):
        action = np.asarray([.02, -.01, .03])*i
        _, reward, term, trunc, info = env.step(action)
        cost, forward, velocity = worker.decompose(np.asarray([reward]), np.asarray([action]), settings)
        np.testing.assert_allclose(velocity[0], info["x_velocity"], rtol=1e-9, atol=1e-9)
        assert abs(float(forward[0])+1.-float(cost[0])-reward) < 1e-12
        if term or trunc:
            env.reset(seed=120+i)
    _, _, velocity = worker.decompose(np.asarray([1.03]), np.ones((1, 3)), settings)
    np.testing.assert_allclose(velocity, [.033])
    env.close()


def test_gradient_probe_is_repeatable_and_does_not_update_models(shared_start):
    checkpoint, shared, *_ = shared_start
    env = gym.make("Hopper-v4")
    agent = ContinuationAgent(env, "fixed-minus2", "cpu")
    agent.start_shared(checkpoint, shared)
    batch = shared["replay"].buffer[:16]
    pool = tuple(np.stack([r[k] for r in batch]) for k in (0, 1, 5))
    before = {name: deepcopy(getattr(agent, name).state_dict()) for name in ("actor", "critic")}
    row = worker.gradient_probe(agent, pool, 12000)
    assert row == worker.gradient_probe(agent, pool, 12000)
    assert row["samples"] == 16 and row["optimizer_steps"] == 0
    assert row["gradient_norms"]["q_std_unit"] > 0 and row["mean_q_std"] > 0
    assert agent.training_steps == 99745
    for name, state in before.items():
        for key, value in state.items():
            assert torch.equal(value, getattr(agent, name).state_dict()[key])
    assert all(p.grad is None for p in agent.actor.parameters())
    env.close()


def test_source_evaluation_preserves_context_and_training_state(shared_start):
    checkpoint, shared, *_ = shared_start
    env = gym.make("Hopper-v4")
    agent = ContinuationAgent(env, "fixed-minus2", "cpu")
    agent.start_shared(checkpoint, shared)
    before = deepcopy(vars(agent.context_tracker))
    replay_size = agent.replay_buffer.size
    row = worker.evaluate_source(agent, "normal", 140000, 2, 8)
    assert row["environment_step_calls"] <= 16 and len(row["episodes"]) == 2
    assert row["reward_formula_verified_against_info"]
    assert agent.training_steps == 99745 and agent.replay_buffer.size == replay_size
    for key, value in before.items():
        np.testing.assert_array_equal(value, vars(agent.context_tracker)[key])
    env.close()


def test_probe_pools_use_real_saved_contexts_and_correct_trace_indices():
    states = np.arange(6000*11).reshape(6000, 11).astype(float)
    actions = np.zeros((6000, 3))
    contexts = np.ones(28)
    rows = [(states[i], actions[i], 1., states[i], False, contexts, contexts) for i in range(6000)]
    saved = dict(training_steps=105745, training_state=dict(replay=dict(buffer=[rows[0]]*6144+rows)))
    pools = worker.probe_pools(saved, dict(states=states, actions=actions), True)
    assert tuple(pools) == ("prechange", "post_compound")
    for pool in pools.values():
        assert pool[0].shape == (16, 11) and pool[2].shape == (16, 28)
        np.testing.assert_array_equal(pool[2], np.ones((16, 28)))
