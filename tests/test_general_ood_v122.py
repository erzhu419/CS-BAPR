from copy import deepcopy

import numpy as np
import torch

from scripts import probe_general_ood_v122_target_scale as probe
from scripts import run_general_ood_v122_target_l1 as worker
from scripts import summarize_general_ood_v122 as merger
from test_general_ood_v121 import make_pair


def test_target_terms_match_actual_loss_and_zero_uplift_does_not_mutate_agent():
    env, agent = make_pair(1)
    state, _ = env.reset(seed=1)
    for _ in range(12):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        agent.store_transition(state, action, reward, next_state, float(terminated or truncated))
        state = next_state
        if terminated or truncated:
            state, _ = env.reset()
            agent.reset_context()
    agent._reward_ema_var = .25
    before = deepcopy(agent.actor.state_dict()), deepcopy(agent.critic.state_dict()), torch.get_rng_state()
    original = probe.decompose(agent, np.arange(12), 12200)
    assert original["mean_discounted_l1_uplift"] > 0.
    agent.config.weight_reg = 0.
    removed = probe.decompose(agent, np.arange(12), 12200)
    assert removed["mean_discounted_l1_uplift"] == 0.
    np.testing.assert_allclose(original["mean_target"]-removed["mean_target"],
                               original["mean_discounted_l1_uplift"], rtol=1e-6)
    assert agent.training_steps == 0 and agent._reward_ema_var == .25
    assert torch.equal(torch.get_rng_state(), before[2])
    for model, expected in zip((agent.actor, agent.critic), before[:2]):
        for key, value in expected.items():
            assert torch.equal(value, model.state_dict()[key])
    env.close()


def test_pilot_changes_only_target_l1_coefficient_and_reuses_training_schedule():
    env = probe.gym.make("Hopper-v4")
    torch.manual_seed(2)
    original = worker.v121.v39.make_agent("Hopper-v4", "fixed-minus2", "cpu", env)
    torch.manual_seed(2)
    pilot = worker.make_agent(env, "cpu")
    assert {k: (vars(original.config)[k], vars(pilot.config)[k]) for k in vars(original.config)
            if vars(original.config)[k] != vars(pilot.config)[k]} == {"weight_reg": (.01, 0.)}
    for model in ("actor", "critic", "target_critic"):
        for key, value in getattr(original, model).state_dict().items():
            assert torch.equal(value, getattr(pilot, model).state_dict()[key])
    assert all(worker.stage_spec()[k] == value for k, value in worker.v121.stage_spec().items())
    env.close()


def test_summary_pairs_training_seeds_and_keeps_negative_effects():
    controls, pilots = [], []
    for seed in (0, 1, 2):
        control = dict(revision=worker.v121.REVISION, smoke=False, training_seed=seed,
            config=dict(weight_reg=.01), runtime={}, spec=worker.v121.stage_spec(),
            training=dict(interactions=100000, optimizer_updates=99745, periodic_evaluations=[]),
            checkpoint_selection="final-not-best", final_evaluation=dict(mean_episode_velocity=.1,
                mean_episode_length=1000., mean_return=1100., physical_terminations=0, episodes=[dict(seed=150000)]))
        pilot = deepcopy(control)
        pilot.update(revision=worker.REVISION, spec=worker.stage_spec(), config=dict(weight_reg=0.), final_target_scale={})
        pilot["final_evaluation"]["mean_episode_velocity"] += seed-1.
        controls.append(control)
        pilots.append(pilot)
    result = merger.summarize(controls, pilots)
    assert result["independent_training_seeds"] == 3
    assert result["per_training_seed"][0]["pilot_minus_control"]["mean_episode_velocity"] == -1.
    assert abs(result["mean_pilot_minus_control"]["mean_episode_velocity"]) < 1e-12
