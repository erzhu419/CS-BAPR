from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from scripts import probe_general_ood_v76_behavior_coverage as worker
from scripts.v68_mode_likelihood import ModeLikelihoodBank
from scripts.v72_conditional_likelihood import ConditionalDensityBank
from test_general_ood_v58 import shared_start
from test_general_ood_v68 import trace as synthetic_trace


def test_protocol_budgets_and_seed_separation():
    root = Path(__file__).resolve().parents[1]
    m = json.loads((root/'experiments/general_ood_v76_behavior_coverage_20261008/dispatch_manifest.json').read_text())
    worker.check_seeds()
    assert m['spec'] == worker.SPEC
    refs, tests = worker.reference_cells(worker.SPEC), worker.test_cells(worker.SPEC)
    assert len(refs) == 8 and len(tests) == 10
    assert (len(refs)+len(tests))*6000 == m['actual_transitions'] == 108000
    assert sum(c['updating'] for c in refs+tests)*6000 == m['optimizer_updates'] == 84000
    assert m['environment_step_calls'] == 648000 and m['counterfactual_step_calls'] == 540000
    assert m['added_labels_per_mode_per_condition'] == 4400
    assert worker.CONDITIONS == tuple(m['conditions'])


def test_mixed_extensions_match_profiles_times_and_counterfactual_targets(tmp_path):
    spec = dict(worker.SPEC, steps=20, switch_step=10)
    for cell in worker.reference_cells(spec):
        states = np.tile(np.arange(20)[:, None], (1, 11)).astype(float)
        states += 100*cell['updating']+10*(cell['mode']=='slippery')
        actions = np.zeros((20, 3))
        oracle = np.stack([states+i for i in range(5)], axis=1)
        np.savez_compressed(worker.trace_path(tmp_path, cell), states=states, executed_actions=actions,
                            next_states=states, oracle_observations=oracle)
    frozen, updating = [worker.extensions(tmp_path, spec, u) for u in (False, True)]
    for i, mode in enumerate(worker.MODES):
        assert len(frozen[mode]['states']) == len(updating[mode]['states']) == 20
        np.testing.assert_array_equal(updating[mode]['states']-frozen[mode]['states'], 100.)
        for data in (frozen, updating):
            np.testing.assert_array_equal(data[mode]['states'], data['normal']['states'])
            np.testing.assert_array_equal(data[mode]['next_states'], data[mode]['states']+i)
            np.testing.assert_array_equal(data[mode]['states'][:5, 0] % 10, np.arange(0, 10, 2))


def fresh(shared_start):
    checkpoint, shared, *_ = shared_start
    env = gym.make('Hopper-v4', max_episode_steps=3)
    shadows = worker.v69.make_shadows(env)
    agent = worker.v69.ContinuationAgent(env, 'fixed-minus2', 'cpu')
    agent.start_shared(checkpoint, shared)
    return agent, env, shadows


def close(env, shadows):
    for physical in (env, *shadows.values()):
        physical.close()


@pytest.mark.parametrize('mode', ['high_grav', 'compound'])
def test_updating_exact_replay_prefix_and_resume(shared_start, tmp_path, monkeypatch, mode):
    cell, spec = dict(role='test', seed=135179, mode=mode, updating=True), dict(steps=12, switch_step=4)
    direct, env, shadows = fresh(shared_start)
    actual, record = worker.collect_updating(direct, env, shadows, cell, spec)
    close(env, shadows)
    assert record['optimizer_updates'] == record['exact_true_mode_replays'] == 12
    assert direct.replay_buffer.size == shared_start[1]['replay'].size+12
    normal, env, shadows = fresh(shared_start)
    baseline, _ = worker.collect_updating(normal, env, shadows, dict(cell, mode='normal'), spec)
    close(env, shadows)
    assert worker.v68.v53.paired_prefix_equal(actual, baseline, 4)
    partial, env, shadows = fresh(shared_start)
    path = tmp_path/'resume.pt'
    monkeypatch.setattr(worker, 'CHECKPOINT_STEPS', 8)
    def interrupt(loop):
        worker.v68.v53.save_progress(partial, env, {}, loop, path)
        raise InterruptedError('test interruption')
    with pytest.raises(InterruptedError):
        worker.collect_updating(partial, env, shadows, cell, spec, checkpoint=interrupt)
    close(env, shadows)
    restored, env, shadows = fresh(shared_start)
    extra = restored.load(str(path))
    worker.v70.apply_mode(env, worker.v70.active_mode(cell, extra['loop']['step']-1, spec))
    env.reset(seed=cell['seed'])
    worker.v69.restore_step_state(env, extra['environment'])
    recovered, resumed_record = worker.collect_updating(restored, env, shadows, cell, spec, extra['loop'])
    close(env, shadows)
    assert record == resumed_record and restored.replay_buffer.size == direct.replay_buffer.size
    for key in actual:
        np.testing.assert_array_equal(actual[key], recovered[key])
    for name in ('actor', 'critic', 'target_critic'):
        for key, value in getattr(direct, name).state_dict().items():
            assert torch.equal(value, getattr(restored, name).state_dict()[key])


def test_coverage_preserves_forward_scales_and_base_residuals(tmp_path):
    torch.set_num_threads(1)
    fitting = {m: synthetic_trace(3, m, steps=160) for m in worker.MODES}
    calibration = {m: synthetic_trace(4, m, steps=160) for m in worker.MODES}
    original = ConditionalDensityBank.fit_density(ModeLikelihoodBank.fit(fitting, calibration), calibration)
    before = deepcopy(original.__dict__)
    extra = {m: synthetic_trace(5, m, steps=80) for m in worker.MODES}
    extended = worker.BehaviorCoverageBank.extend(original, extra, 'updating_mixed')
    for m in worker.MODES:
        np.testing.assert_array_equal(extended.coefficients[m], before['coefficients'][m])
        np.testing.assert_array_equal(extended.residuals[m][:160], before['residuals'][m])
        np.testing.assert_array_equal(original.residuals[m], before['residuals'][m])
    for key in ('input_mean', 'input_scale'):
        np.testing.assert_array_equal(getattr(extended.features, key), getattr(original.features, key))
    np.testing.assert_array_equal(extended.target_scale, before['target_scale'])
    assert 'matched-seed' in extended.metadata()['geometry_source']
    assert extended.__class__.__module__ == 'scripts.v76_behavior_coverage'
    path = tmp_path/'bank.pt'
    torch.save(extended, path)
    restored = torch.load(path, map_location='cpu', weights_only=False)
    np.testing.assert_array_equal(restored.tree.data, extended.tree.data)


def test_test_phase_resume_survives_empty_reference_phase(tmp_path):
    extra = dict(loop=dict(step=1000), progress=dict(next_cell=8))
    args = SimpleNamespace(checkpoint_dir=tmp_path)
    assert worker.collect_cells(args, None, extra['progress'], extra, None, 8) is extra
