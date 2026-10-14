from copy import deepcopy
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

from csbapr.belief.departure import DepartureReadout, NormalAgeReference
from csbapr.utils import ReplayBuffer
from scripts import run_general_ood_v52 as worker
from scripts.summarize_general_ood_v52 import summarize
from scripts.v44_belief import SETTINGS
from scripts.v49_belief import NoiseLikelihoodStudent
from scripts.v52_agent import ContinuationAgent, NormalDynamics, TransitionReadouts, VARIANTS


def reference_data(seed):
    rng = np.random.RandomState(seed)
    states, actions = rng.normal(size=(256, 11)), rng.normal(size=(256, 3))
    next_states = states + .05 * states + .02 * actions.mean(axis=1, keepdims=True) + rng.normal(0, .01, states.shape)
    return dict(states=states, next_states=next_states, executed_actions=actions,
                episode_steps=np.tile(np.arange(32), 8))


@pytest.fixture
def shared_start(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(3)
    fitting, calibration = reference_data(2), reference_data(3)
    model = NormalDynamics(2, True).fit_references(fitting, calibration)
    scores, thresholds = model.calibrations["global-centered"].scores(model.errors(calibration), calibration["episode_steps"])
    reference = NormalAgeReference.fit_normal(np.log1p(scores / thresholds), calibration["episode_steps"])
    env = gym.make("Hopper-v4")
    source = ContinuationAgent(env, "fixed-minus2", "cpu")
    source.training_steps = 99745
    source.context_tracker._count = 100000
    checkpoint = tmp_path / "source.pt"
    source.save(str(checkpoint))
    replay = ReplayBuffer(source.config.buffer_size)
    context = np.zeros(source.context_dim, dtype=np.float32)
    for i in range(256):
        replay.push(fitting["states"][i], np.tanh(fitting["executed_actions"][i]), 1., fitting["next_states"][i],
                    0., context, context)
    shared = dict(model=model, reference=reference, startup_noise=2., replay=replay, context=source.snapshot_context())
    env.close()
    return checkpoint, shared


def test_online_readouts_match_causal_offline_and_use_executed_actions(shared_start):
    _, shared = shared_start
    trace = reference_data(8)
    scores, thresholds = shared["model"].calibrations["global-centered"].scores(shared["model"].errors(trace), trace["episode_steps"])
    signal = np.log1p(scores / thresholds)
    online = TransitionReadouts(shared["model"], shared["reference"], 2.)
    departure, age = DepartureReadout(shared["reference"]), NoiseLikelihoodStudent(**SETTINGS)
    before = deepcopy(shared["model"].metadata()), deepcopy(shared["reference"].metadata())
    for i, episode_age in enumerate(trace["episode_steps"]):
        online.observe(trace["states"][i], trace["executed_actions"][i], trace["next_states"][i], int(episode_age))
        departure.update(signal[i], episode_age)
        age.update(signal[i], 2. if episode_age == 0 else 0.)
        assert online.signal == pytest.approx(signal[i], abs=1e-12)
        assert online.departure.risk == pytest.approx(departure.risk, abs=1e-12)
        np.testing.assert_allclose(online.age.belief, age.belief, atol=1e-12)
    assert before == (shared["model"].metadata(), shared["reference"].metadata())


def test_shared_initial_networks_replay_and_actor_only_penalty(shared_start):
    checkpoint, shared = shared_start
    env = gym.make("Hopper-v4")
    agents = [ContinuationAgent(env, v, "cpu") for v in VARIANTS]
    for agent in agents:
        agent.start_shared(checkpoint, shared)
    for agent in agents[1:]:
        for name in ("actor", "critic", "target_critic"):
            for key, tensor in getattr(agents[0], name).state_dict().items():
                assert torch.equal(tensor, getattr(agent, name).state_dict()[key])
        for first, second in zip(agents[0].replay_buffer.buffer, agent.replay_buffer.buffer):
            for x, y in zip(first, second):
                np.testing.assert_array_equal(x, y)
    state = torch.zeros((8, agents[0].actor_state_dim))
    action, reward, done = torch.zeros(8, 3), torch.ones(8, 1), torch.zeros(8, 1)
    calls = [a._compute_q_loss_bapr_fallback(state, action, reward, state, done, action, torch.zeros(8),
              torch.zeros(a.critic.num_critics), 8) for a in agents]
    for result in calls[1:]:
        for x, y in zip(calls[0][:3], result[:3]):
            assert torch.equal(x, y)
    assert [c[3] for c in calls[:2]] == [0., .3]
    assert agents[0].replay_buffer is not agents[1].replay_buffer
    env.close()


def test_full_training_resume_preserves_physics_rng_replay_and_updates(shared_start, tmp_path, monkeypatch):
    source, shared = shared_start
    def fresh():
        env = gym.make("Hopper-v4")
        agent = ContinuationAgent(env, "age-departure", "cpu")
        agent.start_shared(source, shared)
        return agent, env
    monkeypatch.setitem(worker.BUDGETS, "preflight", 8)
    direct, env = fresh()
    worker.train(direct, env, "preflight", tmp_path / "direct.pt")
    env.close()
    monkeypatch.setitem(worker.BUDGETS, "preflight", 4)
    partial, env = fresh()
    checkpoint = tmp_path / "resumed.pt"
    worker.train(partial, env, "preflight", checkpoint)
    env.close()
    monkeypatch.setitem(worker.BUDGETS, "preflight", 8)
    resumed, env = fresh()
    result = worker.train(resumed, env, "preflight", checkpoint, checkpoint)
    assert result["resumed_at"] == 4 and result["optimizer_updates"] == 8
    for name in ("actor", "critic", "target_critic"):
        for key, value in getattr(direct, name).state_dict().items():
            assert torch.equal(value, getattr(resumed, name).state_dict()[key]), (name, key)
    assert direct.training_steps == resumed.training_steps and direct.replay_buffer.size == resumed.replay_buffer.size
    for x, y in zip(direct.replay_buffer.buffer[-1], resumed.replay_buffer.buffer[-1]):
        np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal(direct.readouts.age.belief, resumed.readouts.age.belief)
    assert direct.readouts.departure.risk == resumed.readouts.departure.risk
    env.close()


def test_completed_worker_matches_shared_start_and_rejects_wrong_penalty(tmp_path):
    directory = Path(__file__).resolve().parents[1] / "results" / worker.REVISION / "preflight"
    paths = [directory / f"{worker.REVISION}_{v}.json" for v in VARIANTS]
    if not all(p.exists() for p in paths):
        pytest.skip("preflight outputs not collected")
    result = summarize(directory, "preflight")
    assert result["matrix_complete"] and result["added_training_interactions"] == 24000
    for p in paths:
        data = json.loads(p.read_text())
        if data["variant"] == "age-departure":
            data["training"]["diagnostic_blocks"][0]["effective_beta"] += .01
        (tmp_path / p.name).write_text(json.dumps(data))
    with pytest.raises(AssertionError, match="wrong-actor-penalty"):
        summarize(tmp_path, "preflight")
