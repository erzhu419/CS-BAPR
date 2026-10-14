from copy import deepcopy
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

from csbapr.belief.departure import NormalAgeReference
from csbapr.utils import ReplayBuffer
from scripts import run_general_ood_v58 as worker
from scripts.v52_agent import ContinuationAgent, NormalDynamics
from scripts.v54_window_readout import SignedWindowReference
from scripts.v57_phase_readout import PhaseAgeReference, PhaseDepartureReadout
from scripts.probe_general_ood_v55_learning_calibration import signals_and_ages
from scripts.v58_agent import ClosedLoopAgent, OnlinePhaseReadout, VARIANTS


def trace(seed):
    rng = np.random.RandomState(seed)
    states, actions = rng.normal(size=(256, 11)), rng.normal(size=(256, 3))
    next_states = states + .05 * states + .02 * actions.mean(axis=1, keepdims=True) + rng.normal(0, .01, states.shape)
    return dict(states=states, next_states=next_states, executed_actions=actions,
                episode_steps=np.tile(np.arange(32), 8), dones=np.arange(256) % 32 == 31)


@pytest.fixture(scope='module')
def shared_start(tmp_path_factory):
    torch.set_num_threads(1)
    torch.manual_seed(3)
    fitting, calibration = trace(2), trace(3)
    model = NormalDynamics(2, True).fit_references(fitting, calibration)
    pooled = SignedWindowReference.fit_normal(model, [fitting], [calibration])
    phase = PhaseAgeReference(NormalAgeReference([0, 999], [.5, .5], [1, 1], 2),
                              NormalAgeReference([0, 999], [.1, .1], [1, 1], 2))
    env = gym.make('Hopper-v4')
    source = ContinuationAgent(env, 'fixed-minus2', 'cpu')
    source.training_steps, source.context_tracker._count = 99745, 100000
    checkpoint = tmp_path_factory.mktemp('v58_source') / 'source.pt'
    source.save(str(checkpoint))
    replay = ReplayBuffer(source.config.buffer_size)
    context = np.zeros(source.context_dim, dtype=np.float32)
    for i in range(256):
        replay.push(fitting['states'][i], np.tanh(fitting['executed_actions'][i]), 1., fitting['next_states'][i],
                    0., context, context)
    shared = dict(model=model, reference=NormalAgeReference.global95(), startup_noise=2.,
                  replay=replay, context=source.snapshot_context())
    env.close()
    return checkpoint, shared, model, pooled, phase


def test_budget_seeded_schedule_and_evaluation_contract():
    manifest = json.loads((Path(__file__).resolve().parents[1] /
        'experiments/general_ood_v58_closed_loop_20261008/dispatch_manifest.json').read_text())
    assert list(VARIANTS) == manifest['variants']
    assert list(worker.EVALUATION_STARTS) == manifest['continuation_seeds']
    assert len(VARIANTS) * len(worker.EVALUATION_STARTS) * worker.BUDGET == manifest['added_training_interactions'] == 48000
    for seed in worker.EVALUATION_STARTS:
        events = worker.mode_events(seed)
        assert [e['step'] for e in events] == [0, 800, 1800, 2800, 3800, 4800]
        assert {e['mode'] for e in events[1:5]} == set(worker.v53.MODES) - {'normal'}
        durations = {mode: 0 for mode in worker.v53.MODES}
        for e, end in zip(events, [e['step'] for e in events[1:]] + [worker.BUDGET]):
            durations[e['mode']] += end - e['step']
        assert durations == dict(normal=2000, heavy=1000, slippery=1000, high_grav=1000, compound=1000)
    assert worker.EVALUATION_EPISODES == 2
    assert worker.EVALUATION_STARTS[116001] - worker.EVALUATION_STARTS[116000] == 1000


def test_online_phase_readout_matches_offline_and_keeps_fixed_reference(shared_start):
    _, _, model, pooled, phase = shared_start
    data = trace(8)
    signals, ages = signals_and_ages(model, pooled, data)
    online, offline = OnlinePhaseReadout(model, pooled, phase), PhaseDepartureReadout(phase)
    before = deepcopy(pooled.metadata()), deepcopy(phase.metadata()), deepcopy(model.metadata())
    for i, age in enumerate(ages):
        risk = online.observe(data['states'][i], data['executed_actions'][i], data['next_states'][i], int(age))
        assert online.signal == pytest.approx(signals[i], abs=1e-12)
        assert risk == pytest.approx(offline.update(signals[i], age), abs=1e-12)
    assert online.departure.after_reset
    assert before == (pooled.metadata(), phase.metadata(), model.metadata())


def test_shared_start_and_feedback_only_changes_actor_beta_slot(shared_start):
    checkpoint, shared, model, pooled, phase = shared_start
    env = gym.make('Hopper-v4')
    agents = [ClosedLoopAgent(env, variant, 'cpu') for variant in VARIANTS]
    for agent in agents:
        agent.start_shared(checkpoint, shared, model, pooled, phase)
    for agent in agents[1:]:
        for name in ('actor', 'critic', 'target_critic'):
            for key, value in getattr(agents[0], name).state_dict().items():
                assert torch.equal(value, getattr(agent, name).state_dict()[key])
        for a, b in zip(agents[0].replay_buffer.buffer, agent.replay_buffer.buffer):
            for x, y in zip(a, b):
                np.testing.assert_array_equal(x, y)
        assert agent.replay_buffer is not agents[0].replay_buffer
    agents[2].readout_coefficient, agents[3].readout_coefficient = .2, .4
    state = torch.zeros((8, agents[0].actor_state_dim))
    action, reward, done = torch.zeros(8, 3), torch.ones(8, 1), torch.zeros(8, 1)
    calls = [a._compute_q_loss_bapr_fallback(state, action, reward, state, done, action, torch.zeros(8),
              torch.zeros(a.critic.num_critics), 8) for a in agents]
    for result in calls[1:]:
        for x, y in zip(calls[0][:3], result[:3]):
            assert torch.equal(x, y)
    assert [c[3] for c in calls] == [0., .3, .2, .4]
    env.close()


def test_full_resume_preserves_phase_history_rng_and_native_resets(shared_start, tmp_path, monkeypatch):
    checkpoint, shared, model, pooled, phase = shared_start
    def fresh():
        env = gym.make('Hopper-v4')
        env._max_episode_steps = 3
        agent = ClosedLoopAgent(env, 'phase-beta', 'cpu')
        agent.start_shared(checkpoint, shared, model, pooled, phase)
        return agent, env
    monkeypatch.setattr(worker, 'BUDGET', 8)
    direct, env = fresh()
    worker.train(direct, env, 116000, tmp_path / 'direct.pt')
    env.close()
    monkeypatch.setattr(worker, 'BUDGET', 4)
    partial, env = fresh()
    resume = tmp_path / 'resumed.pt'
    worker.train(partial, env, 116000, resume)
    env.close()
    monkeypatch.setattr(worker, 'BUDGET', 8)
    restored, env = fresh()
    result = worker.train(restored, env, 116000, resume, resume)
    assert result['resumed_at'] == 4 and result['optimizer_updates'] == 8
    for name in ('actor', 'critic', 'target_critic'):
        for key, value in getattr(direct, name).state_dict().items():
            assert torch.equal(value, getattr(restored, name).state_dict()[key]), (name, key)
    assert direct.readouts.departure.after_reset == restored.readouts.departure.after_reset
    assert direct.readout_coefficient == restored.readout_coefficient
    np.testing.assert_array_equal(direct.readouts.history, restored.readouts.history)
    np.testing.assert_array_equal(direct.readouts.departure.departure.history, restored.readouts.departure.departure.history)
    assert direct.training_steps == restored.training_steps and direct.replay_buffer.size == restored.replay_buffer.size
    env.close()
