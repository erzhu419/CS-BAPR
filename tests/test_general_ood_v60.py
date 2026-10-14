from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import probe_general_ood_v60_long_normal as worker
from scripts.v58_agent import OnlinePhaseReadout
from test_general_ood_v58 import shared_start
from test_general_ood_v57 import trace


def test_budget_separate_roles_and_late_change_contract():
    manifest = json.loads((Path(__file__).resolve().parents[1] /
        'experiments/general_ood_v60_long_normal_20261008/dispatch_manifest.json').read_text())
    assert manifest['stage_spec'] == worker.SPEC
    cells = worker.trajectory_specs(worker.SPEC)
    assert len(cells) == manifest['cells'] == 16
    assert len(cells) * worker.SPEC['steps'] == manifest['added_interactions'] == 96000
    assert sum(c['updating'] for c in cells) * worker.SPEC['steps'] == manifest['added_optimizer_updates'] == 72000
    assert worker.SPEC['switch_step'] % 1000 == 800
    assert list(worker.OBSERVERS) == manifest['observers']
    assert list(worker.NORMAL_WINDOWS) == manifest['normal_windows']
    assert list(worker.CHANGE_WINDOWS) == manifest['change_windows']


def test_long_calibration_control_keeps_old_model_and_moments(shared_start):
    _, _, model, pooled, phase = shared_start
    before = deepcopy(worker.reference_metadata({'v57_frozen': (model, pooled, phase)}))
    fitting, calibration = [trace(10, 6000), trace(11, 6000)], [trace(12, 6000), trace(13, 6000)]
    refs = worker.build_references(model, pooled, phase, fitting, calibration)
    assert list(refs) == list(worker.OBSERVERS)
    old_model, age_pooled, age_phase = refs['long_calibration_only']
    assert old_model is model
    for name in ('center', 'covariance', 'precision'):
        np.testing.assert_array_equal(getattr(age_pooled, name), getattr(pooled, name))
    assert age_pooled.score95 == pooled.score95 and age_pooled.ridge == pooled.ridge
    refitted, new_pooled, new_phase = refs['long_normal_refit']
    assert refitted is not model and new_pooled.fit_transitions == new_pooled.calibration_transitions == 12000
    for p in (age_phase, new_phase):
        assert p.initial.normal_start_count == 2 and p.after_reset.normal_start_count == 10
        assert p.initial.ages[0] == p.after_reset.ages[0] == 0
        assert p.initial.ages[-1] == p.after_reset.ages[-1] == 999
    assert before == worker.reference_metadata({'v57_frozen': (model, pooled, phase)})
    data = trace(14, 6000)
    sensor = OnlinePhaseReadout(refitted, new_pooled, new_phase)
    risks = [sensor.observe(data['states'][i], data['executed_actions'][i], data['next_states'][i], i % 1000)
             for i in range(6000)]
    windows = worker.observe(data, refitted, new_pooled, new_phase, 3800)
    assert windows['last1000']['mean_coefficient'] == pytest.approx(np.mean(risks[-1000:]), abs=1e-12)
    assert windows['first_post50']['mean_coefficient'] == pytest.approx(np.mean(risks[3800:3850]), abs=1e-12)


def test_gate_preserves_both_normal_branches_and_every_changed_tail():
    cells = worker.trajectory_specs(worker.SPEC)
    records = []
    for cell in cells:
        r = dict(cell)
        if cell['split'] == 'test':
            r['observers'] = {name: {w: dict(mean_coefficient=.01) for w in worker.NORMAL_WINDOWS} for name in worker.OBSERVERS}
            if cell['mode'] != 'normal':
                r['change_minus_nochange'] = {name: {w: .2 for w in worker.CHANGE_WINDOWS} for name in worker.OBSERVERS}
        records.append(r)
    assert all(s['usable_normal_null'] and s['positive_pairs']['last64'] == 8 for s in worker.summarize(records).values())
    records[4]['observers']['long_normal_refit']['first800']['mean_coefficient'] = .11
    assert not worker.summarize(records)['long_normal_refit']['usable_normal_null']
    records[4]['observers']['long_normal_refit']['first800']['mean_coefficient'] = .01
    records[-1]['change_minus_nochange']['long_normal_refit']['last64'] = 0.
    assert worker.summarize(records)['long_normal_refit']['positive_pairs']['last64'] == 7


def test_complete_worker_writes_all_cells_without_test_refitting(shared_start, tmp_path, monkeypatch):
    import argparse
    import shutil
    from scripts.v52_agent import ContinuationAgent
    checkpoint, original, model, pooled, phase = shared_start
    shared = deepcopy(original)
    initial = deepcopy(shared['replay'].buffer)
    while shared['replay'].size < 6144:
        shared['replay'].push(*initial[shared['replay'].size % len(initial)])
    monkeypatch.setattr(worker, 'SPEC', dict(worker.SPEC, steps=8, switch_step=4))
    monkeypatch.setattr(worker.torch, 'set_num_interop_threads', lambda _: None)
    monkeypatch.setattr(worker, 'ContinuationAgent', lambda env, variant: ContinuationAgent(env, variant, 'cpu'))
    calls = []
    def build(old_model, old_pooled, old_phase, fitting, calibration):
        calls.append((len(fitting), len(calibration)))
        assert (len(fitting), len(calibration)) == (2,2)
        return {name: (old_model, old_pooled, old_phase) for name in worker.OBSERVERS}
    monkeypatch.setattr(worker, 'build_references', build)
    shutil.copy2(checkpoint, tmp_path / worker.v53.SOURCE_NAME)
    torch.save(shared, tmp_path / 'shared.pt')
    torch.save(dict(extra_state=dict(progress=dict(stage='preflight', next_cell=9, models=dict(broad=(model,None))))), tmp_path / 'dynamics.pt')
    torch.save(dict(extra_state=dict(progress=dict(stage='formal', next_cell=22,
        references=dict(matched_pooled=pooled, phase_calibrated=phase)))), tmp_path / 'references.pt')
    args = argparse.Namespace(source_dir=tmp_path, shared_dir=tmp_path,
        dynamics_checkpoint=tmp_path / 'dynamics.pt', reference_checkpoint=tmp_path / 'references.pt',
        checkpoint_dir=tmp_path / 'checkpoints', result_dir=tmp_path / 'results', resume_from=None)
    worker.run(args)
    result = json.loads((tmp_path / 'results/long_normal.json').read_text())
    assert calls == [(2,2)] and len(result['records']) == 16
    assert result['environment_interactions'] == 128 and result['optimizer_updates'] == 96
    assert sum(r.get('paired_prefix_equal', False) for r in result['records']) == 10
    args.resume_from = tmp_path / 'checkpoints/preflight_latest.pt'
    worker.run(args)
    assert calls == [(2,2)]
    assert json.loads((tmp_path / 'results/long_normal.json').read_text()) == result
