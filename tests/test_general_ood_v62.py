from copy import deepcopy
import json
from pathlib import Path
import pickle

import numpy as np
import pytest
import torch

from csbapr.belief.departure import NormalAgeReference
from scripts import probe_general_ood_v62_causal_innovation as worker
from scripts.v62_innovation_readout import InnovationReference, OnlineInnovationReadout, window_features
from test_general_ood_v58 import shared_start, trace as short_trace
from test_general_ood_v57 import trace


@pytest.fixture(scope='module')
def reference(shared_start):
    _, _, model, pooled, _ = shared_start
    return InnovationReference.fit_normal(model, pooled, [trace(10,6000),trace(11,6000)],
                                          [trace(12,6000),trace(13,6000)])


def test_new_seed_budget_and_unchanged_gates():
    manifest = json.loads((Path(__file__).resolve().parents[1]/
        'experiments/general_ood_v62_causal_innovation_20261008/dispatch_manifest.json').read_text())
    assert worker.SPEC == manifest['stage_spec']
    cells = worker.v60.trajectory_specs(worker.SPEC)
    assert len(cells) == manifest['cells'] == 16
    assert len(cells)*worker.SPEC['steps'] == manifest['added_interactions'] == 96000
    assert sum(c['updating'] for c in cells)*worker.SPEC['steps'] == manifest['added_optimizer_updates'] == 72000
    assert not set(sum([worker.SPEC[k] for k in ('fit','calibration','test')],[])) & {132000,132010,132020,132021}
    assert worker.SPEC['switch_step'] == 3800 and worker.NORMAL_WINDOWS == worker.v60.NORMAL_WINDOWS
    assert worker.CHANGE_WINDOWS == worker.v60.CHANGE_WINDOWS
    assert list(worker.OBSERVERS) == manifest['observers']
    assert not manifest['known_boundary_used_by_readout'] and not manifest['readout_controls_actor']


def test_adjacent_windows_are_causal_and_two_sided(reference):
    values = np.random.RandomState(1).normal(size=(200,11))
    moments, innovations = window_features(values)
    assert np.count_nonzero(innovations[:31]) == 0
    np.testing.assert_allclose(innovations[31], values[16:32].mean(axis=0)-values[:16].mean(axis=0))
    changed = values.copy()
    changed[100:] += 100.
    after = window_features(changed)
    np.testing.assert_array_equal(moments[:100], after[0][:100])
    np.testing.assert_array_equal(innovations[:100], after[1][:100])
    vector = np.ones((1,11))
    np.testing.assert_array_equal(reference.innovation_scores(vector), reference.innovation_scores(-vector))
    far, near = reference.pooled.center[None]+vector, reference.pooled.center[None]+.5*vector
    first = reference.signals(far,np.zeros_like(vector))
    second = reference.signals(near,.5*vector)
    assert second['level_ratio'][0] < first['level_ratio'][0]
    assert second['innovation_ratio'][0] > first['innovation_ratio'][0]


def test_joint_normal_calibration_keeps_level_and_dynamics_frozen(shared_start,reference):
    _, _, model, pooled, _ = shared_start
    before = deepcopy(model.metadata()), deepcopy(pooled.metadata())
    data = [trace(12,6000),trace(13,6000)]
    signals = [reference.signals(*window_features(worker.normalized_residuals(model,t))) for t in data]
    ages = np.concatenate([worker.v53.v43.episode_steps(t['dones']) for t in data])
    from scripts.probe_general_ood_v56_reset_calibration import episode_ordinals
    ordinals = np.concatenate([episode_ordinals(t['dones']) for t in data])
    for name in worker.NAMES:
        values = np.concatenate([s[name] for s in signals])
        for curve, mask in ((reference.phases[name].initial,ordinals==0),
                            (reference.phases[name].after_reset,ordinals>0)):
            assert curve.metadata() == NormalAgeReference.fit_normal(values[mask],ages[mask]).metadata()
            assert curve.ages[0] == 0 and curve.ages[-1] == 999
    assert reference.fit_transitions == reference.calibration_transitions == 12000
    assert reference.pooled is pooled
    assert before == (model.metadata(),pooled.metadata())


def test_online_matches_batch_across_resets_and_serialized_resume(shared_start,reference):
    _, _, model, _, _ = shared_start
    data = short_trace(8)
    values = reference.signals(*window_features(worker.normalized_residuals(model,data)))
    ages = worker.v53.v43.episode_steps(data['dones'])
    before = deepcopy(reference.metadata())
    sensors = {name:OnlineInnovationReadout(model,reference,name) for name in worker.NAMES}
    risks = {name:[] for name in worker.NAMES}
    resumed = None
    for i, age in enumerate(ages):
        for name,sensor in sensors.items():
            args = (data['states'][i],data['executed_actions'][i],data['next_states'][i],int(age))
            risks[name].append(sensor.observe(*args))
            assert sensor.signal == pytest.approx(values[name][i],abs=1e-10)
            assert sensor.innovation_ratio == pytest.approx(values['innovation_ratio'][i],abs=1e-10)
            if resumed is not None:
                assert resumed[name].observe(*args) == pytest.approx(risks[name][-1],abs=1e-12)
        if i == 123:
            resumed = pickle.loads(pickle.dumps(sensors))
    observed = worker.observe(data,model,reference,128)
    for name,sensor in sensors.items():
        assert len(sensor.history) == 32 and sensor.departure.after_reset
        assert observed[name]['first_post50']['mean_coefficient'] == pytest.approx(np.mean(risks[name][128:178]),abs=1e-12)
    assert reference.metadata() == before


def test_any_normal_failure_or_nonpositive_change_keeps_hold():
    records = []
    for cell in worker.v60.trajectory_specs(worker.SPEC):
        r = dict(cell)
        if cell['split'] == 'test':
            r['observers'] = {n:{w:dict(mean_coefficient=.01) for w in worker.NORMAL_WINDOWS} for n in worker.OBSERVERS}
            if cell['mode'] != 'normal':
                r['change_minus_nochange'] = {n:{w:.2 for w in worker.CHANGE_WINDOWS} for n in worker.OBSERVERS}
        records.append(r)
    assert all(x['positive_pairs']==dict(first_post50=8,last64=8) and x['usable_normal_null'] for x in worker.summarize(records).values())
    records[4]['observers']['level_plus_innovation']['first800']['mean_coefficient'] = .11
    assert worker.summarize(records)['level_plus_innovation']['decision'] == 'hold-causal-innovation'
    records[4]['observers']['level_plus_innovation']['first800']['mean_coefficient'] = .01
    records[-1]['change_minus_nochange']['level_plus_innovation']['first_post50'] = 0.
    assert worker.summarize(records)['level_plus_innovation']['positive_pairs']['first_post50'] == 7


def test_complete_worker_retains_traces_freezes_before_test_and_resumes(shared_start,reference,tmp_path,monkeypatch):
    import argparse
    import shutil
    from scripts.v52_agent import ContinuationAgent
    checkpoint,original,model,pooled,phase = shared_start
    shared = deepcopy(original)
    initial = deepcopy(shared['replay'].buffer)
    while shared['replay'].size < 6144:
        shared['replay'].push(*initial[shared['replay'].size%len(initial)])
    monkeypatch.setattr(worker,'SPEC',dict(worker.SPEC,steps=8,switch_step=4))
    monkeypatch.setattr(worker.torch,'set_num_interop_threads',lambda _:None)
    monkeypatch.setattr(worker,'ContinuationAgent',lambda env,variant:ContinuationAgent(env,variant,'cpu'))
    calls = []
    def fit(m,p,f,c):
        calls.append((len(f),len(c)))
        assert len(f) == len(c) == 2
        return reference
    monkeypatch.setattr(worker.InnovationReference,'fit_normal',fit)
    shutil.copy2(checkpoint,tmp_path/worker.v53.SOURCE_NAME)
    torch.save(shared,tmp_path/'shared.pt')
    torch.save(dict(extra_state=dict(progress=dict(revision=worker.v60.REVISION,next_cell=16,
        references=dict(long_normal_refit=(model,pooled,phase))))),tmp_path/'reference.pt')
    args = argparse.Namespace(source_dir=tmp_path,shared_dir=tmp_path,reference_checkpoint=tmp_path/'reference.pt',
        checkpoint_dir=tmp_path/'checkpoints',result_dir=tmp_path/'results',resume_from=None)
    worker.run(args)
    result = json.loads((tmp_path/'results/causal_innovation.json').read_text())
    assert calls == [(2,2)] and len(result['records']) == 16
    assert result['environment_interactions'] == 128 and result['optimizer_updates'] == 96
    assert sum(r.get('paired_prefix_equal',False) for r in result['records']) == 10
    assert len(list((tmp_path/'checkpoints/traces').glob('*.npz'))) == 16
    args.resume_from = tmp_path/'checkpoints/preflight_latest.pt'
    worker.run(args)
    assert calls == [(2,2)] and json.loads((tmp_path/'results/causal_innovation.json').read_text()) == result
