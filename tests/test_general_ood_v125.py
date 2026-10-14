from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pytest
import torch

from scripts import run_general_ood_v125_actor_beta as worker
from scripts import summarize_general_ood_v125 as merger
from test_general_ood_v121 import make_pair


def source_pair(seed):
    env, agent = make_pair(seed)
    agent.config.weight_reg = 0.
    return env, agent


def beta_pair(seed, beta):
    env, ordinary = source_pair(seed)
    agent = worker.ActorBetaAgent(11, 3, deepcopy(ordinary.config), beta)
    return env, agent


def test_protocol_and_factory_only_change_actor_beta():
    for arm, beta in worker.ARMS.items():
        spec = worker.stage_spec(arm)
        assert spec["steps"] == 350000 and spec["source_steps"] == 300000 and spec["effective_beta"] == beta
        assert worker.stage_spec(arm, True)["steps"] == 300512
    env, ordinary = source_pair(1)
    config = ordinary.config
    config.penalty_scale = 2.
    assert ordinary._effective_policy_beta(.25) == -2.5
    fixed = worker.make_agent(env, "cpu", 0.)
    old = worker.v122.make_agent(env, "cpu")
    assert asdict(fixed.config) == asdict(old.config)
    assert type(fixed.surprise_computer) is type(old.surprise_computer)
    assert fixed._effective_policy_beta(.25) == 0. and old._effective_policy_beta(.25) == -2.
    env.close()


def test_first_update_changes_actor_only_and_control_matches_original(tmp_path):
    env, source = source_pair(1)
    spec = dict(worker.v123.stage_spec(), steps=12, checkpoint_interval=4,
                diagnostic_interval=4, evaluation_interval=4, evaluation_max_steps=4, evaluation_episodes=1)
    checkpoint = tmp_path/"source.pt"
    worker.v121.train(source, env, 1, spec, checkpoint)
    env.close()
    updated = []
    for beta in (None, -2., 0.):
        env, agent = source_pair(99) if beta is None else beta_pair(99, beta)
        agent.load(str(checkpoint))
        before = deepcopy(agent.actor.state_dict())
        metrics = agent.update()
        assert any(not torch.equal(before[k], v) for k, v in agent.actor.state_dict().items())
        updated.append((agent, metrics))
        env.close()
    ordinary, control, zero = [r[0] for r in updated]
    assert updated[0][1] == updated[1][1]
    assert updated[1][1]["effective_beta"] == -2. and updated[2][1]["effective_beta"] == 0.
    np.testing.assert_allclose(updated[1][1]["policy_loss"]-updated[2][1]["policy_loss"],
                               2.*updated[1][1]["q_std_policy"], rtol=1e-4, atol=1e-5)
    for network in ("actor", "critic", "target_critic"):
        for key, value in getattr(ordinary, network).state_dict().items():
            assert torch.equal(value, getattr(control, network).state_dict()[key])
    for network in ("critic", "target_critic"):
        for key, value in getattr(control, network).state_dict().items():
            assert torch.equal(value, getattr(zero, network).state_dict()[key])
    assert control.alpha == zero.alpha and torch.equal(control.log_alpha, zero.log_alpha)
    assert any(not torch.equal(v, zero.actor.state_dict()[k]) for k, v in control.actor.state_dict().items())


@pytest.mark.parametrize("beta", (-2., 0.))
@pytest.mark.parametrize("cutoff", (11, 12))
def test_fork_and_interruption_preserve_full_training_state(tmp_path, monkeypatch, beta, cutoff):
    source_spec = dict(worker.v123.stage_spec(), steps=cutoff, checkpoint_interval=4,
                      diagnostic_interval=4, evaluation_interval=4, evaluation_episodes=1, evaluation_max_steps=4)
    monkeypatch.setattr(worker.v123, "stage_spec", lambda: source_spec)
    arm = next(k for k, v in worker.ARMS.items() if v == beta)
    spec = dict(worker.stage_spec(arm), steps=24, source_steps=cutoff, checkpoint_interval=4,
                diagnostic_interval=4, evaluation_interval=4, evaluation_episodes=1, evaluation_max_steps=4)
    env, source = source_pair(1)
    source_path = tmp_path/"source/latest.pt"
    worker.v121.train(source, env, 1, source_spec, source_path)
    source_actor = deepcopy(source.actor.state_dict())
    env.close()
    env, direct = beta_pair(99, beta)
    expected = worker.continue_training(direct, env, 1, spec, source_path, tmp_path/"direct/latest.pt")
    env.close()
    save = worker.v121.save_progress
    def interrupt(agent, env, loop, spec, seed, path):
        save(agent, env, loop, spec, seed, path)
        if loop["step"] == 20:
            raise InterruptedError("saved full fork checkpoint")
    monkeypatch.setattr(worker.v121, "save_progress", interrupt)
    env, partial = beta_pair(98, beta)
    path = tmp_path/"resume/latest.pt"
    with pytest.raises(InterruptedError):
        worker.continue_training(partial, env, 1, spec, source_path, path)
    env.close()
    monkeypatch.setattr(worker.v121, "save_progress", save)
    env, resumed = beta_pair(97, beta)
    actual = worker.continue_training(resumed, env, 1, spec, source_path, path, path)
    assert actual == expected and actual["optimizer_updates"] == 17
    assert actual["diagnostic_blocks"][-1]["metrics"]["effective_beta"] == beta
    for network in ("actor", "critic", "target_critic"):
        for key, value in getattr(direct, network).state_dict().items():
            assert torch.equal(value, getattr(resumed, network).state_dict()[key])
    for before, after in zip(direct.replay_buffer.buffer, resumed.replay_buffer.buffer):
        for a, b in zip(before, after):
            np.testing.assert_array_equal(a, b)
    saved = torch.load(source_path, map_location="cpu", weights_only=False)
    for key, value in source_actor.items():
        assert torch.equal(value, saved["actor"][key])
    wrong_spec = dict(spec, arm="beta-zero" if beta == -2. else "beta-minus2", effective_beta=0. if beta == -2. else -2.)
    with pytest.raises(AssertionError):
        worker.continue_training(resumed, env, 1, wrong_spec, source_path, path, path)
    env.close()


def test_final_evaluation_keeps_modes_separate_and_restores_context():
    env, agent = beta_pair(1, 0.)
    state, _ = env.reset(seed=1)
    agent.observe_context(state, np.zeros(3), 1., state, False)
    context = deepcopy(vars(agent.context_tracker))
    result = worker.evaluate_final(agent, True)
    assert set(result["modes"]) == {"deterministic", "sampled"} and result["optimizer_updates"] == 0
    for key, value in context.items():
        np.testing.assert_array_equal(value, vars(agent.context_tracker)[key])
    assert agent.training_steps == agent.replay_buffer.size == 0
    env.close()


def test_merger_counts_paired_paths_not_independent_noise_seeds():
    rows = []
    for seed in (0, 1, 2):
        for arm, beta in worker.ARMS.items():
            length = 200.+([-10., 20., 0.][seed] if beta == 0. else 0.)
            modes = {mode: dict(mean_length=length, mean_return=length, mean_velocity=1., falls=len(noises)*5,
                episodes=[dict(reset_seed=reset, noise_seed=noise) for reset in range(150000, 150005) for noise in noises])
                for mode, noises in (("deterministic", (None,)), ("sampled", worker.v124.NOISE_SEEDS))}
            rows.append(dict(revision=worker.REVISION, training_seed=seed, arm=arm, effective_beta=beta, smoke=False,
                spec=worker.stage_spec(arm), config={}, runtime={}, source_checkpoint=f"source/seed{seed}",
                source_revision=worker.v123.REVISION, source_interactions=300000,
                actual_new_training_interactions=50000, actual_new_optimizer_updates=50000,
                training=dict(interactions=350000, optimizer_updates=349745,
                              diagnostic_blocks=[dict(metrics=dict(effective_beta=beta))]),
                checkpoint_selection="final-not-best", periodic_evaluation_selects_checkpoint=False,
                source_readout_refitted=False, final_target_scale=dict(mean_discounted_l1_uplift=0.),
                final_evaluation=dict(protocol=worker.v124.stage_spec(), device="cpu", modes=modes, optimizer_updates=0)))
    result = merger.summarize(rows)
    assert result["actual_new_training_interactions"] == 300000 and result["independent_training_seeds"] == 3
    assert result["reused_unique_source_interactions"] == 900000
    assert result["per_training_seed"][0]["beta_zero_minus_control"]["deterministic"]["mean_length"] == -10.
    with pytest.raises(AssertionError):
        merger.summarize(rows[:-1])
