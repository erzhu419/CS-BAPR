from copy import deepcopy

import numpy as np
import pytest
import torch

from scripts import probe_general_ood_v59_normal_attribution as worker
from scripts import run_general_ood_v58 as v58
from scripts.v58_agent import ClosedLoopAgent
from test_general_ood_v58 import shared_start, trace


def test_observers_match_online_without_changing_reference(shared_start):
    _, _, model, pooled, phase = shared_start
    data = trace(8)
    data.update(actions=np.tanh(data['executed_actions']), rewards=np.ones(len(data['states'])))
    before = deepcopy(worker.observe(data, model, pooled, phase))
    from scripts.v54_window_readout import SignedWindowReadout
    from scripts.v58_agent import OnlinePhaseReadout
    sensors = dict(pooled=SignedWindowReadout(model, pooled), phase=OnlinePhaseReadout(model, pooled, phase))
    for name, sensor in sensors.items():
        risks = [sensor.observe(data['states'][i], data['executed_actions'][i], data['next_states'][i],
                                int(data['episode_steps'][i])) for i in range(len(data['states']))]
        assert before['windows']['full'][f'{name}_coefficient'] == pytest.approx(np.mean(risks), abs=1e-12)
    assert before == worker.observe(data, model, pooled, phase)


def test_normal_collect_matches_v58_fixed_arm_and_resume(shared_start, tmp_path, monkeypatch):
    import gymnasium as gym
    checkpoint, shared, model, pooled, phase = shared_start
    def fresh():
        env = gym.make('Hopper-v4')
        env._max_episode_steps = 3
        agent = ClosedLoopAgent(env, 'fixed-minus2', 'cpu')
        agent.start_shared(checkpoint, shared, model, pooled, phase)
        return agent, env
    monkeypatch.setattr(v58, 'BUDGET', 8)
    direct, env = fresh()
    reference = v58.train(direct, env, 116000, tmp_path / 'v58.pt')
    env.close()
    cell = dict(split='normal-updating', seed=116000, updating=True, mode='normal')
    collected, env = fresh()
    data, _ = worker.v53.collect(collected, env, cell, dict(steps=8))
    assert worker.first_transition(data) == reference['first_transition']
    for name in ('actor', 'critic', 'target_critic'):
        for key, value in getattr(direct, name).state_dict().items():
            assert torch.equal(value, getattr(collected, name).state_dict()[key])
    env.close()
    partial, env = fresh()
    progress = dict(seed=116000, next_cell=0, records=[])
    saved = tmp_path / 'v59.pt'
    monkeypatch.setattr(worker.v53, 'CHECKPOINT_STEPS', 4)
    worker.v53.collect(partial, env, cell, dict(steps=4), checkpoint=lambda loop:
                       worker.v53.save_progress(partial, env, progress, loop, saved))
    env.close()
    restored, env = fresh()
    extra = restored.load(str(saved))
    worker.v53.apply_mode_to_env(env, 'Hopper-v4', worker.v53.MUJOCO_MODE_PROFILES['Hopper-v4']['normal'])
    env.reset(seed=116000)
    worker.v53.restore_environment(env, extra['environment'])
    resumed, record = worker.v53.collect(restored, env, cell, dict(steps=8), extra['loop'])
    assert record['optimizer_updates'] == 8
    for key, values in data.items():
        np.testing.assert_array_equal(values, resumed[key])
    for key, value in collected.actor.state_dict().items():
        assert torch.equal(value, restored.actor.state_dict()[key])
    env.close()


def test_budget_and_paired_seed_contract():
    import json
    from pathlib import Path
    manifest = json.loads((Path(__file__).resolve().parents[1] /
        'experiments/general_ood_v59_normal_attribution_20261008/dispatch_manifest.json').read_text())
    assert list(worker.SEEDS) == manifest['continuation_seeds'] == list(v58.EVALUATION_STARTS)
    assert list(worker.ARMS) == manifest['arms']
    assert worker.BUDGET == manifest['interactions_per_branch'] == 6000
    assert len(worker.SEEDS) * len(worker.ARMS) * worker.BUDGET == manifest['added_interactions'] == 24000
    assert len(worker.SEEDS) * worker.BUDGET == manifest['added_optimizer_updates'] == 12000


def test_complete_worker_handles_source_before_reward_ema_initialization(shared_start, tmp_path, monkeypatch):
    from dataclasses import asdict
    import argparse
    import json
    import shutil
    import gymnasium as gym
    from scripts.v52_agent import ContinuationAgent

    checkpoint, original, model, pooled, phase = shared_start
    shared = deepcopy(original)
    replay = shared['replay']
    initial = deepcopy(replay.buffer)
    while replay.size < 6144:
        replay.push(*initial[replay.size % len(initial)])
    monkeypatch.setattr(worker, 'BUDGET', 8)
    monkeypatch.setattr(worker.v53, 'CHECKPOINT_STEPS', 4)
    monkeypatch.setattr(worker.torch, 'set_num_interop_threads', lambda _: None)
    monkeypatch.setattr(worker, 'ContinuationAgent', lambda env, variant: ContinuationAgent(env, variant, 'cpu'))
    source = tmp_path / worker.v53.SOURCE_NAME
    shutil.copy2(checkpoint, source)
    env = gym.make('Hopper-v4')
    baseline_agent = worker.ContinuationAgent(env, 'fixed-minus2')
    baseline_agent.start_shared(source, shared)
    assert not hasattr(baseline_agent, '_reward_ema_var')
    config = asdict(baseline_agent.config)
    data, _ = worker.v53.collect(baseline_agent, env,
        dict(split='normal-frozen', seed=116000, updating=False, mode='normal'), dict(steps=1))
    env.close()
    baseline = dict(continuation_seed=116000, variant='fixed-minus2', config=config, initial_reward_ema_var=1.,
        readout_metadata=dict(model=model.metadata(), pooled=pooled.metadata(), phase=phase.metadata()),
        training=dict(first_transition=worker.first_transition(data)))
    (tmp_path / 'baseline.json').write_text(json.dumps(baseline))
    torch.save(shared, tmp_path / 'shared.pt')
    torch.save(dict(extra_state=dict(progress=dict(stage='preflight', next_cell=9, models=dict(broad=(model, None))))),
               tmp_path / 'dynamics.pt')
    torch.save(dict(extra_state=dict(progress=dict(stage='formal', next_cell=22,
        references=dict(matched_pooled=pooled, phase_calibrated=phase)))), tmp_path / 'references.pt')
    worker.run(argparse.Namespace(seed=116000, source_dir=tmp_path, shared_dir=tmp_path,
        dynamics_checkpoint=tmp_path / 'dynamics.pt', reference_checkpoint=tmp_path / 'references.pt',
        v58_result=tmp_path / 'baseline.json', checkpoint_dir=tmp_path / 'checkpoints',
        result_dir=tmp_path / 'results', resume_from=None))
    result = json.loads((tmp_path / 'results/normal_attribution.json').read_text())
    assert result['environment_interactions'] == 16 and result['optimizer_updates'] == 8
    assert result['records'][0]['actor_parameter_l2_change'] == 0.
    assert [r['first_transition'] for r in result['records']] == [baseline['training']['first_transition']] * 2
