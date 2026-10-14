from copy import deepcopy

import gymnasium as gym
import numpy as np
import pytest
import torch

from scripts import analyze_general_ood_v127_component_attribution as worker
from scripts import summarize_general_ood_v127 as merger


def make_sources():
    env = gym.make("Hopper-v4")
    agents = {}
    for age in worker.AGES:
        agent = worker.v126.v125.make_agent(env, "cpu", -2.)
        state, _ = env.reset(seed=age)
        action = np.zeros(3, dtype=np.float32)
        next_state, reward, term, trunc, _ = env.step(action)
        agent.store_transition(state, action, reward, next_state, term or trunc)
        agents[age] = agent
    env.close()
    return agents, {age: agent.actor for age, agent in agents.items()}


def test_crossed_functions_use_complete_actors_on_identical_inputs():
    agents, actors = make_sources()
    state = torch.randn(4, agents[300000].config.state_dim+agents[300000].context_dim)
    mean, log_std = worker.CrossedPolicy(actors[300000], actors[350000])(state)
    assert torch.equal(mean, actors[300000](state)[0])
    assert torch.equal(log_std, actors[350000](state)[1])
    actor = actors[300000]
    for x, y in zip(worker.CrossedPolicy(actor, actor)(state), actor(state)):
        assert torch.equal(x, y)


def test_evaluation_preserves_both_sources_and_reproduces_coherent_baselines():
    agents, actors = make_sources()
    snapshots = {age: dict(actor=deepcopy(a.actor.state_dict()), context=deepcopy(vars(a.context_tracker)),
                          updates=a.training_steps, replay=a.replay_buffer.size) for age, a in agents.items()}
    rng = torch.get_rng_state().clone()
    for case in worker.cases():
        cell = dict(case, reset_seed=150000, noise_seed=None if case["mode"] == "deterministic" else 12400,
                    noise_group="deterministic" if case["mode"] == "deterministic" else "legacy")
        result = worker.evaluate_cell(agents, actors, cell, 16)
        assert result == worker.evaluate_cell(agents, actors, cell, 16)
        if worker.coherent(cell):
            reference = worker.v126.evaluate_episode(agents[cell["mean_age"]], 150000,
                result["noise_scale"], cell["noise_seed"], 16)
            worker.verify_baseline(result, reference)
            assert result["tail"] == reference["tail"]
    for age, a in agents.items():
        assert a.actor is actors[age]
        assert (a.training_steps, a.replay_buffer.size) == (snapshots[age]["updates"], snapshots[age]["replay"])
        for k, v in snapshots[age]["context"].items():
            np.testing.assert_array_equal(v, vars(a.context_tracker)[k])
        for k, v in snapshots[age]["actor"].items():
            assert torch.equal(v, a.actor.state_dict()[k])
        assert all(p.grad is None for p in a.actor.parameters())
    assert torch.equal(rng, torch.get_rng_state())


def test_partial_resume_keeps_cell_identity_and_does_not_repeat_finished_work(monkeypatch):
    spec = worker.stage_spec(2, smoke=True)
    expected = worker.cells(spec)
    assert len(expected) == 20 and len(worker.cells(worker.stage_spec(2))) == 260
    progress = dict(evaluations=[dict(cell, baseline_match=True if worker.coherent(cell) else None)
                                for cell in expected[:7]])
    calls = []
    def evaluate(agents, actors, cell, max_steps):
        calls.append(cell)
        return dict(cell, length=1, noise_scale=0. if cell["mode"] == "deterministic" else 1.)
    monkeypatch.setattr(worker, "evaluate_cell", evaluate)
    monkeypatch.setattr(worker, "verify_baseline", lambda a, b: None)
    baseline = dict(models=[dict(age=age, evaluations=[dict(reset_seed=150000, noise_seed=noise,
        noise_scale=0. if noise is None else 1.) for noise in (None, 12400, 12600)]) for age in worker.AGES])
    worker.collect({}, {}, baseline, spec, progress, lambda: None)
    assert calls == expected[7:] and len(progress["evaluations"]) == 20
    progress["evaluations"][0]["normalizer_age"] = 350000
    with pytest.raises(AssertionError):
        worker.collect({}, {}, baseline, spec, progress, lambda: None)


def fixture_rows():
    rows = []
    for seed in (0, 1, 2):
        spec = worker.stage_spec(seed)
        es = [dict(cell, length=100, episode_return=200., mean_velocity=1., mean_policy_std=.2,
            mean_action_noise_l2=.2, mean_context_l2=.3, physical_termination=True, time_limit_truncation=False,
            baseline_match=True if worker.coherent(cell) else None) for cell in worker.cells(spec)]
        rows.append(dict(revision=worker.REVISION, spec=spec, training_seed=seed, evaluation_device="cpu",
            sources=[dict(age=age, config={}) for age in worker.AGES], evaluations=es,
            optimizer_updates=0, new_training_interactions=0, source_readout_refitted=False,
            checkpoint_selection="final-not-best", reward_or_physics_changes=[], coherent_baseline_matches=70,
            evaluation_interactions=sum(e["length"] for e in es)))
    return rows


def test_summary_keeps_full_factorial_and_rejects_missing_or_failed_controls():
    rows = fixture_rows()
    result = merger.summarize(rows)
    assert result["evaluation_episodes"] == 780 and result["coherent_baseline_matches"] == 210
    assert len(result["records"]) == 60 and len(result["single_component_contrasts"]) == 18
    assert result["independent_training_seeds"] == 3
    rows[0]["evaluations"][0]["baseline_match"] = False
    with pytest.raises(AssertionError):
        merger.summarize(rows)
    with pytest.raises(AssertionError):
        merger.summarize(rows[:2])
    bad = dict(rows[1]["evaluations"][0], episode_return=0., diagnostic_cutoff=False,
        final_physical_state={"unhealthy_conditions": []}, mean_context_l2=.3,
        mean_action_noise_l2=.2, mean_policy_std=.2)
    with pytest.raises(AssertionError):
        worker.verify_baseline(dict(bad, length=101), bad)
