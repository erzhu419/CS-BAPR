from copy import deepcopy
from dataclasses import asdict
import json
from types import SimpleNamespace
import random

import gymnasium as gym
import numpy as np
import pytest
import torch

from scripts import analyze_general_ood_v129_heldout_robustness as worker
from scripts import summarize_general_ood_v129 as merger


def small_agent(env, device, beta):
    config = worker.v128.v121.v39.v38.make_config("Hopper-v4", "fixed-penalty", device)
    config.state_dim, config.action_dim = 11, 3
    config.hidden_dim, config.batch_size, config.weight_reg = 16, 8, 0.
    return worker.v128.v125.ActorBetaAgent(11, 3, config, beta)


def make_sources():
    torch.set_num_threads(1)
    env = gym.make("Hopper-v4")
    agents = {}
    for index, model in enumerate(worker.MODELS):
        agent = small_agent(env, "cpu", -2.)
        state, _ = env.reset(seed=index)
        action = np.zeros(3, dtype=np.float32)
        next_state, reward, term, trunc, _ = env.step(action)
        agent.store_transition(state, action, reward, next_state, term or trunc)
        agent.actor.eval().requires_grad_(False)
        agents[model] = agent
    env.close()
    return agents


def test_predeclared_models_noise_and_baselines_first():
    spec = worker.stage_spec(2)
    expected = worker.cells(spec)
    assert len(expected) == 315 and len(worker.cells(worker.stage_spec(2, True))) == 6
    assert spec["noise_seeds"] == list(range(13000, 13020)) and spec["max_steps"] == 1000
    old_noises = {n for ns in worker.v128.NOISE_GROUPS.values() for n in ns}
    assert not old_noises.intersection(spec["noise_seeds"])
    assert all(c["mode"] == "deterministic" for c in expected[:15])
    assert all(c["mode"] == "sampled" for c in expected[15:])
    for model in worker.MODELS:
        assert sum(c["model"] == model for c in expected) == 105


def test_compact_evaluation_matches_existing_sampler_and_preserves_sources():
    agents = make_sources()
    snapshots = {m: dict(actor=deepcopy(a.actor.state_dict()), context=deepcopy(vars(a.context_tracker)),
        updates=a.training_steps, replay=a.replay_buffer.size, samples=a._n_train_samples) for m, a in agents.items()}
    rng = (torch.get_rng_state().clone(), np.random.get_state(), random.getstate())
    for model in worker.MODELS:
        for noise in (None, 13000):
            cell = dict(model=model, mode="deterministic" if noise is None else "sampled",
                reset_seed=150000, noise_seed=noise)
            result = worker.evaluate_cell(agents, cell, 16)
            assert result == worker.evaluate_cell(agents, cell, 16)
            baseline = worker.v126.evaluate_episode(agents[model], 150000, 0. if noise is None else 1., noise, 16)
            worker.v127.verify_baseline(result, baseline)
            assert "tail" not in result and "same_input_parity" not in result
    for model, agent in agents.items():
        assert (agent.training_steps, agent.replay_buffer.size, agent._n_train_samples) == (
            snapshots[model]["updates"], snapshots[model]["replay"], snapshots[model]["samples"])
        for key, value in snapshots[model]["context"].items():
            np.testing.assert_array_equal(value, vars(agent.context_tracker)[key])
        assert all(torch.equal(v, snapshots[model]["actor"][k]) for k, v in agent.actor.state_dict().items())
        assert all(p.grad is None for p in agent.actor.parameters())
    assert torch.equal(torch.get_rng_state(), rng[0])
    np.testing.assert_array_equal(np.random.get_state()[1], rng[1][1])
    assert np.random.get_state()[2:] == rng[1][2:] and random.getstate() == rng[2]


def episode(cell, length=100):
    return dict(cell, length=length, noise_scale=0. if cell["mode"] == "deterministic" else 1.,
        episode_return=float(2*length), mean_velocity=1., mean_policy_std=.2,
        mean_action_noise_l2=.1, mean_context_l2=.3, physical_termination=True,
        time_limit_truncation=False, diagnostic_cutoff=False,
        final_physical_state=dict(unhealthy_conditions=["height"]))


def test_resume_skips_completed_cells_and_failed_baseline_stops(monkeypatch):
    spec = worker.stage_spec(2, True)
    expected = worker.cells(spec)
    baselines = {(c["model"], c["reset_seed"]): episode(c) for c in expected[:3]}
    calls, saved = [], []
    def evaluate(agents, cell, max_steps):
        calls.append(cell)
        return episode(cell)
    monkeypatch.setattr(worker, "evaluate_cell", evaluate)
    progress = dict(evaluations=[dict(episode(c), baseline_match=True if c["mode"] == "deterministic" else None)
                                for c in expected[:4]])
    worker.collect({}, baselines, spec, progress, lambda: saved.append(len(progress["evaluations"])))
    assert calls == expected[4:] and saved == [5, 6]
    worker.collect({}, baselines, spec, progress, lambda: pytest.fail("completed resume repeated work"))
    progress["evaluations"][0]["model"] = worker.MODELS[2]
    with pytest.raises(AssertionError):
        worker.collect({}, baselines, spec, progress, lambda: None)
    bad_baselines = deepcopy(baselines)
    bad_baselines[(worker.MODELS[0], 150000)]["length"] = 101
    fresh = dict(evaluations=[])
    with pytest.raises(AssertionError):
        worker.collect({}, bad_baselines, spec, fresh, lambda: None)
    assert not fresh["evaluations"]


def test_load_sources_uses_native_final_actors_and_correct_arm(tmp_path, monkeypatch):
    monkeypatch.setattr(worker.v128.v125, "make_agent", small_agent)
    args = SimpleNamespace(seed=0, v123_runtime=tmp_path/"v123", v126_runtime=tmp_path/"v126",
        v128_runtime=tmp_path/"v128")
    env = gym.make("Hopper-v4")
    old_reference = None
    retention_path = None
    for model, age, arm in zip(worker.MODELS, (300000, 350000, 350000), (None, *worker.v128.ARMS)):
        agent = small_agent(env, "cpu", -2.)
        agent.training_steps = age-agent.config.batch_size+1
        agent.context_tracker._count = age
        final = tmp_path/f"{model}.pt"
        agent.save(str(final))
        references = [episode(dict(model=model, mode="deterministic", reset_seed=reset, noise_seed=None))
                      for reset in range(150000, 150005)]
        data = dict(revision=worker.v128.v123.REVISION if arm is None else worker.v128.REVISION,
            training_seed=0, smoke=False, training=dict(interactions=age), effective_beta=-2.,
            config=asdict(agent.config), checkpoint_selection="final-not-best", source_readout_refitted=False,
            periodic_evaluation_selects_checkpoint=False, runtime={}, checkpoint=str(final))
        if arm is None:
            path = args.v123_runtime/"run_results/seed0/source.json"
            old_reference = dict(age=age, checkpoint=str(final), evaluations=references,
                initial_context_normalizer=dict(count=age, mean=agent.context_tracker._mean.tolist(),
                    m2=agent.context_tracker._m2.tolist()))
        else:
            path = args.v128_runtime/f"run_results/seed0_{arm}/mean_retention.json"
            data.update(arm=arm, spec=worker.v128.stage_spec(arm), mean_retention_coefficient=worker.v128.ARMS[arm],
                final_evaluation=dict(protocol=worker.v128.evaluation_spec(), device="cpu",
                    groups=dict(deterministic=dict(episodes=references))))
            retention_path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    env.close()
    baseline_path = args.v126_runtime/"run_results/seed0_cpu/noise.json"
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(json.dumps(dict(revision=worker.v126.REVISION, spec=worker.v126.stage_spec(0),
        runtime={}, checkpoint_selection="final-not-best", models=[old_reference])))
    agents, sources, baselines = worker.load_sources(args, {})
    assert list(agents) == list(worker.MODELS) and len(baselines) == 15
    assert [s["age"] for s in sources] == [300000, 350000, 350000]
    assert all(not p.requires_grad for a in agents.values() for p in a.actor.parameters())
    wrong = json.loads(retention_path.read_text())
    wrong["arm"] = "unchanged-control"
    retention_path.write_text(json.dumps(wrong))
    with pytest.raises(AssertionError):
        worker.load_sources(args, {})


def formal_rows():
    rows = []
    for seed in (0, 1, 2):
        spec = worker.stage_spec(seed)
        lengths = dict(zip(worker.MODELS, (200, 100, (90, 300, 400)[seed])))
        evaluations = [dict(episode(c, lengths[c["model"]]),
            baseline_match=True if c["mode"] == "deterministic" else None) for c in worker.cells(spec)]
        sources = [dict(model=m, age=age, config={}, revision=revision, arm=arm) for m, age, revision, arm in zip(
            worker.MODELS, (300000, 350000, 350000), (worker.v128.v123.REVISION, worker.v128.REVISION,
                worker.v128.REVISION), (None, *worker.v128.ARMS))]
        rows.append(dict(revision=worker.REVISION, spec=spec, training_seed=seed, evaluation_device="cpu",
            sources=sources, evaluations=evaluations, deterministic_baseline_matches=15,
            optimizer_updates=0, new_training_interactions=0, source_readout_refitted=False,
            checkpoint_selection="predeclared-final-only", reward_or_physics_changes=[],
            evaluation_interactions=sum(e["length"] for e in evaluations)))
    return rows


def test_summary_preserves_noise_pairs_seeds_and_negative_outcomes():
    rows = formal_rows()
    result = merger.summarize(rows)
    assert result["evaluation_episodes"] == 945 and result["deterministic_baseline_matches"] == 45
    assert result["independent_training_seeds"] == 3 and result["frozen_models"] == 9
    assert result["optimizer_updates"] == result["new_training_interactions"] == 0
    assert len(result["per_noise"]) == 60
    assert result["per_training_seed"][0]["modes"]["sampled"]["retention_minus_control"]["mean_length"] == -10.
    assert result["per_noise"][0]["models"][worker.MODELS[0]]["episodes"] == 5
    with pytest.raises(AssertionError):
        merger.summarize(rows[:-1])
    for key, value in (("baseline_match", False), ("noise_scale", .25)):
        wrong = deepcopy(rows)
        wrong[0]["evaluations"][0][key] = value
        with pytest.raises(AssertionError):
            merger.summarize(wrong)
    wrong = deepcopy(rows)
    wrong[1]["evaluations"][-1]["noise_seed"] = 12800
    with pytest.raises(AssertionError):
        merger.summarize(wrong)
