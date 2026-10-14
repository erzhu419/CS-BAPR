from copy import deepcopy
from dataclasses import asdict
import random

import numpy as np
import pytest
import torch

from scripts import run_general_ood_v128_mean_retention as worker
from scripts import summarize_general_ood_v128 as merger
from test_general_ood_v125 import source_pair


def retention_pair(seed, coefficient):
    env, ordinary = source_pair(seed)
    agent = worker.MeanRetentionAgent(11, 3, deepcopy(ordinary.config), coefficient)
    agent.set_reference(ordinary.actor.state_dict())
    return env, agent


def test_protocol_and_mean_only_gradient():
    env, agent = retention_pair(1, 1.)
    original = deepcopy(agent.reference_actor.state_dict())
    with torch.no_grad():
        agent.actor.mean_linear.bias.add_(.1)
    agent.alpha = torch.tensor(.2, requires_grad=True)
    states = torch.ones(4, 11+agent.context_dim)
    mean, _ = agent.actor(states)
    with torch.no_grad():
        ref_mean, ref_log_std = agent.reference_actor(states)
    expected = .2 * .5 * ((mean-ref_mean)/ref_log_std.exp()).square().sum(-1).mean()
    loss = agent._actor_auxiliary_loss(states)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert agent.alpha.grad is None
    assert agent.actor.mean_linear.bias.grad.abs().sum() > 0
    assert agent.actor.net[0].weight.grad.abs().sum() > 0
    assert agent.actor.log_std_linear.weight.grad is None
    assert agent.actor.log_std_linear.bias.grad is None
    assert all(p.grad is None and not p.requires_grad for p in agent.reference_actor.parameters())
    assert all(torch.equal(v, original[k]) for k, v in agent.reference_actor.state_dict().items())
    assert agent.retention_metrics["mean_retention_loss"] > 0.
    for arm in worker.ARMS:
        spec = worker.stage_spec(arm)
        assert spec["source_steps"] == 300000 and spec["steps"] == 350000
        assert spec["effective_beta"] == -2. and spec["mean_retention_coefficient"] == worker.ARMS[arm]
        assert worker.stage_spec(arm, True)["steps"] == 300512
        new = worker.make_agent(env, "cpu", arm)
        baseline = worker.v125.make_agent(env, "cpu", -2.)
        assert asdict(new.config) == asdict(baseline.config)
    assert sum(len(v) for v in worker.NOISE_GROUPS.values()) == 9
    assert worker.evaluation_spec(True)["max_steps"] == 1000
    env.close()


def test_control_is_exactly_original_and_penalty_changes_actor_only(tmp_path):
    env, source = source_pair(1)
    spec = dict(worker.v123.stage_spec(), steps=12, checkpoint_interval=4,
        diagnostic_interval=4, evaluation_interval=4, evaluation_max_steps=4, evaluation_episodes=1)
    checkpoint = tmp_path/"source.pt"
    worker.v121.train(source, env, 1, spec, checkpoint)
    reference = deepcopy(source.actor.state_dict())
    env.close()
    updated = []
    for coefficient in (None, 0., 1.):
        env, agent = source_pair(99) if coefficient is None else retention_pair(99, coefficient)
        agent.load(str(checkpoint))
        if coefficient is not None:
            agent.set_reference(reference)
        with torch.no_grad():
            agent.actor.mean_linear.bias.add_(.1)
        metrics = agent.update()
        updated.append((agent, metrics, torch.get_rng_state().clone()))
        env.close()
    baseline, control, treatment = [r[0] for r in updated]
    assert updated[0][1] == {k: v for k, v in updated[1][1].items() if k not in worker.METRICS}
    assert updated[1][1]["mean_retention_loss"] == 0.
    assert updated[2][1]["mean_retention_loss"] > 0.
    for network in ("actor", "critic", "target_critic"):
        assert all(torch.equal(v, getattr(control, network).state_dict()[k])
                   for k, v in getattr(baseline, network).state_dict().items())
    for network in ("critic", "target_critic"):
        assert all(torch.equal(v, getattr(treatment, network).state_dict()[k])
                   for k, v in getattr(control, network).state_dict().items())
    assert control.alpha == treatment.alpha
    assert all(torch.equal(updated[0][2], r[2]) for r in updated)
    assert any(not torch.equal(v, treatment.actor.state_dict()[k]) for k, v in control.actor.state_dict().items())


@pytest.mark.parametrize("coefficient", (0., 1.))
def test_full_continuation_resume_preserves_reference(tmp_path, monkeypatch, coefficient):
    source_spec = dict(worker.v123.stage_spec(), steps=12, checkpoint_interval=4,
        diagnostic_interval=4, evaluation_interval=4, evaluation_episodes=1, evaluation_max_steps=4)
    monkeypatch.setattr(worker.v123, "stage_spec", lambda: source_spec)
    env, source = source_pair(1)
    source_path = tmp_path/"source/latest.pt"
    worker.v121.train(source, env, 1, source_spec, source_path)
    reference = deepcopy(source.actor.state_dict())
    env.close()
    monkeypatch.setattr(worker.v121, "METRICS", (*worker.v121.METRICS, *worker.METRICS))
    arm = next(k for k, v in worker.ARMS.items() if v == coefficient)
    spec = dict(worker.stage_spec(arm), steps=24, source_steps=12, checkpoint_interval=4,
        diagnostic_interval=4, evaluation_interval=4, evaluation_episodes=1, evaluation_max_steps=4)
    def make(seed):
        env, agent = retention_pair(seed, coefficient)
        agent.set_reference(reference)
        return env, agent
    env, direct = make(99)
    expected = worker.v125.continue_training(direct, env, 1, spec, source_path, tmp_path/"direct/latest.pt")
    env.close()
    save = worker.v121.save_progress
    def interrupt(agent, env, loop, spec, seed, path):
        save(agent, env, loop, spec, seed, path)
        if loop["step"] == 20:
            raise InterruptedError("saved mean-retention continuation")
    monkeypatch.setattr(worker.v121, "save_progress", interrupt)
    env, partial = make(98)
    path = tmp_path/"resume/latest.pt"
    with pytest.raises(InterruptedError):
        worker.v125.continue_training(partial, env, 1, spec, source_path, path)
    env.close()
    monkeypatch.setattr(worker.v121, "save_progress", save)
    env, resumed = make(97)
    actual = worker.v125.continue_training(resumed, env, 1, spec, source_path, path, path)
    assert actual == expected
    for network in ("actor", "critic", "target_critic", "reference_actor"):
        assert all(torch.equal(v, getattr(resumed, network).state_dict()[k])
                   for k, v in getattr(direct, network).state_dict().items())
    assert all(torch.equal(v, reference[k]) for k, v in resumed.reference_actor.state_dict().items())
    assert all(p.grad is None for p in resumed.reference_actor.parameters())
    for before, after in zip(direct.replay_buffer.buffer, resumed.replay_buffer.buffer):
        for a, b in zip(before, after):
            np.testing.assert_array_equal(a, b)
    with pytest.raises(AssertionError):
        wrong = dict(spec, mean_retention_coefficient=1.-coefficient)
        worker.v125.continue_training(resumed, env, 1, wrong, source_path, path, path)
    env.close()


def test_final_evaluation_full_limit_groups_and_state_restoration(monkeypatch):
    env, agent = retention_pair(2, 1.)
    state, _ = env.reset(seed=2)
    agent.observe_context(state, np.zeros(3), 1., state, False)
    context = deepcopy(vars(agent.context_tracker))
    rng = (torch.get_rng_state(), np.random.get_state(), random.getstate())
    evaluate = worker.v126.evaluate_episode
    limits = []
    def record(*args):
        limits.append(args[4])
        return evaluate(*args)
    monkeypatch.setattr(worker.v126, "evaluate_episode", record)
    result = worker.evaluate_final(agent, True)
    assert limits == [1000]*4
    assert set(result["groups"]) == {"deterministic", *worker.NOISE_GROUPS}
    assert result["optimizer_updates"] == 0 and agent.training_steps == agent.replay_buffer.size == 0
    for group, row in result["groups"].items():
        episode = row["episodes"][0]
        assert not episode["diagnostic_cutoff"]
        assert episode["noise_seed"] == (None if group == "deterministic" else worker.NOISE_GROUPS[group][0])
    assert torch.equal(torch.get_rng_state(), rng[0])
    np.testing.assert_array_equal(np.random.get_state()[1], rng[1][1])
    assert np.random.get_state()[2:] == rng[1][2:] and random.getstate() == rng[2]
    for key, value in context.items():
        np.testing.assert_equal(value, vars(agent.context_tracker)[key])
    env.close()


def formal_rows():
    rows = []
    protocol = worker.evaluation_spec()
    for seed in (0, 1, 2):
        for arm, coefficient in worker.ARMS.items():
            length = 200+int(coefficient)*([-10, 20, 100][seed])
            groups = {}
            for group, noises in {"deterministic": [None], **worker.NOISE_GROUPS}.items():
                episodes = [dict(reset_seed=reset, noise_seed=noise, noise_scale=0. if noise is None else 1.,
                    length=length, episode_return=float(length), mean_velocity=1., physical_termination=True,
                    time_limit_truncation=False, diagnostic_cutoff=False)
                    for reset in protocol["reset_seeds"] for noise in noises]
                groups[group] = dict(episodes=episodes, mean_length=float(length), mean_return=float(length),
                    mean_velocity=1., falls=len(episodes), time_limits=0)
            rows.append(dict(revision=worker.REVISION, training_seed=seed, arm=arm, smoke=False,
                spec=worker.stage_spec(arm), mean_retention_coefficient=coefficient, effective_beta=-2.,
                source_revision=worker.v123.REVISION, source_interactions=300000,
                source_checkpoint=f"source/seed{seed}/latest.pt", reference_checkpoint=f"source/seed{seed}/final.pt",
                config=dict(weight_reg=0.), runtime={}, reward_settings={}, reward_or_physics_changes=[],
                actual_new_training_interactions=50000, actual_new_optimizer_updates=50000,
                training=dict(interactions=350000, optimizer_updates=349745, diagnostic_blocks=[dict(metrics=dict(
                    effective_beta=-2., mean_shift_kl=.1, mean_retention_loss=coefficient*.02, reference_action_mse=.01))]),
                final_target_scale=dict(mean_discounted_l1_uplift=0.), reference_frozen=True,
                current_std_directly_regularized=False, initial_normalizer_swapped=False,
                checkpoint_selection="final-not-best", periodic_evaluation_selects_checkpoint=False,
                source_readout_refitted=False, final_evaluation=dict(protocol=protocol, device="cpu", groups=groups,
                    optimizer_updates=0, evaluation_interactions=sum(len(g["episodes"])*length for g in groups.values()))))
    return rows


def test_merger_retains_seeds_groups_and_rejects_protocol_mismatch():
    rows = formal_rows()
    summary = merger.summarize(rows)
    assert summary["formal_paths"] == 6 and summary["independent_training_seeds"] == 3
    assert summary["evaluation_episodes"] == 300 and summary["actual_new_training_interactions"] == 300000
    assert summary["per_training_seed"][0]["retention_minus_control"]["new_heldout"]["mean_length"] == -10.
    with pytest.raises(AssertionError):
        merger.summarize(rows[:-1])
    wrong = deepcopy(rows)
    wrong[1]["final_evaluation"]["groups"]["new_heldout"]["episodes"][0]["noise_scale"] = .5
    with pytest.raises(AssertionError):
        merger.summarize(wrong)
    wrong = deepcopy(rows)
    wrong[1]["reference_checkpoint"] = "source/seed2/final.pt"
    with pytest.raises(AssertionError):
        merger.summarize(wrong)
