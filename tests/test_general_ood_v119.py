from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from scripts import run_general_ood_v119_actor_feedback as worker
from scripts import summarize_general_ood_v119 as merger
from scripts import analyze_general_ood_v83_cross_seed_covariance as source
from scripts.v73_recursive_mode_readout import RecursiveModeReference, observe_likelihoods
from test_general_ood_v58 import shared_start
from test_general_ood_v95 import collection_start


def scores(query):
    value = np.asarray(query["next_states"])[:, 0]-np.asarray(query["states"])[:, 0]
    return np.column_stack((value, -value, .5+value, -1-value, np.zeros_like(value)))


def test_protocol_budget_and_disjoint_seeds():
    manifest = json.loads((Path(__file__).resolve().parents[1]/
        "experiments/general_ood_v119_actor_feedback_20261011/dispatch_manifest.json").read_text())
    population = [c for s in worker.SEEDS for a in worker.ARMS for c in worker.cells(s, a)]
    assert len(population) == manifest["paths"] == 60
    assert len(population)*worker.SPEC["steps"] == manifest["actual_transitions"] == manifest["optimizer_updates"] == 360000
    assert worker.SPEC == manifest["spec"] and list(worker.ARMS) == manifest["arms"]
    assert not set(worker.SEEDS) & set(worker.v118.SEEDS)
    assert len(worker.cells(worker.SMOKE_SEED, "all", True)) == 20
    with pytest.raises(AssertionError):
        worker.cells(worker.SEEDS[0], "all")


def test_summary_pairs_environment_seeds_and_retains_normal_cost():
    records = []
    for i, seed in enumerate(worker.SEEDS):
        for arm in worker.ARMS:
            for mode in worker.v118.v117.MODES:
                reward = 100.+i+(i+1 if arm == worker.ARMS[0] and mode != "normal" else 0.)
                if arm == worker.ARMS[0] and mode == "normal":
                    reward -= 5.
                records.append(dict(seed=seed, arm=arm, mode=mode, windows=dict(post=dict(mean_reward=reward)),
                    evaluation={m: dict(mean_return=10*reward) for m in {mode, "normal"}}))
    result = merger.summarize(records)["matched_minus_controls"]["incumbent_v98"]
    assert result["changed"]["post_mean_reward"]["per_seed_differences"] == [1., 2., 3.]
    assert result["changed"]["post_mean_reward"]["mean_difference"] == 2.
    assert result["normal"]["post_mean_reward"]["per_seed_differences"] == [-5., -5., -5.]
    assert result["normal"]["post_mean_reward"]["ci95"] == [-5., -5.]


def test_online_readout_matches_causal_replay_across_reset_and_resume(shared_start):
    phase = shared_start[4]
    rng = np.random.RandomState(11)
    data = dict(states=rng.normal(size=(20, 11)), next_states=rng.normal(size=(20, 11)),
                executed_actions=np.zeros((20, 3)), dones=np.arange(20) % 4 == 3)
    readout = worker.FeedbackReadout(scores, phase)
    series = {k: [] for k in ("scores", "mode_scores", "risk", "signal", "threshold")}
    for i in range(20):
        if i == 9:
            saved = deepcopy(readout.state())
            readout = worker.FeedbackReadout(scores, phase)
            readout.restore(saved)
        readout.observe(data["states"][i], data["executed_actions"][i], data["next_states"][i], i % 4)
        for key in series:
            series[key].append(readout.departure.departure.threshold if key == "threshold" else getattr(readout, key))
    reference = RecursiveModeReference()
    reference.phase = phase
    worker.verify_online(dict(data, **{k: np.asarray(v) for k, v in series.items()}), reference, scores(data))
    assert readout.departure.after_reset


def test_penalty_slot_does_not_change_critic_objective(shared_start):
    checkpoint, shared, *_ = shared_start
    env = gym.make("Hopper-v4")
    agents = [worker.ActorFeedbackAgent(env, arm, "cpu") for arm in worker.ARMS]
    for agent in agents:
        agent.start_shared(checkpoint, shared)
        agent.feedback = SimpleNamespace(risk=.4)
    state, action = torch.zeros(8, agents[0].actor_state_dim), torch.zeros(8, 3)
    calls = [a._compute_q_loss_bapr_fallback(state, action, torch.ones(8, 1), state, torch.zeros(8, 1),
        action, torch.zeros(8), torch.zeros(a.critic.num_critics), 8) for a in agents]
    assert [r[3] for r in calls] == [.4, .4, 0., 1.]
    for row in calls[1:]:
        for x, y in zip(calls[0][:3], row[:3]):
            assert torch.equal(x, y)
    env.close()


@pytest.mark.parametrize("arm", ["matched_classifier", "fixed-minus3"])
def test_real_driver_exact_resume_prefixes_and_completed_cell_reuse(collection_start, shared_start, tmp_path, monkeypatch, arm):
    checkpoint, shared_dir = collection_start
    shared = torch.load(shared_dir/"shared.pt", map_location="cpu", weights_only=False)
    monkeypatch.setattr(source.v76.v69, "SOURCE_NAME", checkpoint.name)
    monkeypatch.setattr(worker, "CHECKPOINT_STEPS", 4)
    make = gym.make
    def short_env(*args, **kwargs):
        env = make(*args, **kwargs)
        env._max_episode_steps = 3
        return env
    monkeypatch.setattr(gym, "make", short_env)
    seed = worker.SEEDS[0]
    population = worker.cells(seed, arm)[:2]
    spec = dict(seeds=[seed], steps=12, switch_step=6, evaluation_episodes=1, evaluation_max_steps=4)
    args = SimpleNamespace(source_dir=checkpoint.parent, device="cpu", resume_from=None, checkpoint_dir=tmp_path/"direct")
    factory = lambda a: None if a.startswith("fixed-") else worker.FeedbackReadout(scores, shared_start[4])
    expected = worker.collect_paths(args, source, population, spec, shared, factory)
    args.checkpoint_dir = tmp_path/"resumed"
    save = source.v76.v68.v53.save_progress
    def interrupt(agent, env, progress, loop, path):
        save(agent, env, progress, loop, path)
        if progress["next_cell"] == 1 and loop is not None and loop["step"] == 8:
            raise InterruptedError("mid-path interruption after a reset and switch")
    monkeypatch.setattr(source.v76.v68.v53, "save_progress", interrupt)
    with pytest.raises(InterruptedError):
        worker.collect_paths(args, source, population, spec, shared, factory)
    monkeypatch.setattr(source.v76.v68.v53, "save_progress", save)
    args.resume_from = args.checkpoint_dir/"latest.pt"
    actual = worker.collect_paths(args, source, population, spec, shared, factory)
    assert expected == actual
    for cell in population:
        name = f"{cell['arm']}_{cell['mode']}.npz"
        with np.load(tmp_path/"direct"/name) as x, np.load(args.checkpoint_dir/name) as y:
            for key in x:
                np.testing.assert_array_equal(x[key], y[key])
            expected_beta = -2.-x["risk"]
            np.testing.assert_array_equal(x["effective_beta"], expected_beta)
    monkeypatch.setattr(worker, "ActorFeedbackAgent", lambda *a: pytest.fail("completed paths rerun"))
    assert worker.collect_paths(args, source, population, spec, shared, factory) == actual
