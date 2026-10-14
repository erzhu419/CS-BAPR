from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import probe_general_ood_v61_early_response as worker
from test_general_ood_v58 import shared_start, trace


def test_targeted_budget_and_offline_anchor_contract():
    m=json.loads((Path(__file__).resolve().parents[1]/
        'experiments/general_ood_v61_early_response_20261008/dispatch_manifest.json').read_text())
    assert worker.SEED == m['continuation_seed'] == 132020
    assert worker.SPEC == dict(steps=m['steps'],switch_step=m['switch_step'])
    assert list(worker.v53.MODES) == m['modes']
    assert len(worker.v53.MODES)*worker.SPEC['steps'] == m['added_interactions'] == m['added_optimizer_updates'] == 30000
    assert worker.ANCHOR_STEPS == m['diagnostic_anchor_pre_steps'] == 64
    assert worker.REPRODUCTION_TOLERANCE == m['reproduction_tolerance']


def test_movement_towards_center_reduces_level_but_not_displacement(shared_start):
    _,_,_,pooled,_=shared_start
    anchor=pooled.center+np.ones(11)
    values,base,error=worker.geometry(np.asarray([pooled.center+.5*np.ones(11)]),pooled,anchor)
    assert values['level_score'][0] < base
    assert values['cross_term'][0] < 0 and values['displacement'][0] > 0
    assert values['level_score'][0] == pytest.approx(base+values['cross_term'][0]+values['displacement'][0])
    assert error < 1e-9


def test_diagnostic_risk_matches_v60_and_paired_identity(shared_start):
    _,_,model,pooled,phase=shared_start
    data=trace(8)
    anchor=worker.rolling_means(worker.normalized_residuals(model,data))[64:128].mean(axis=0)
    record=dict(diagnostic=worker.diagnose(data,model,pooled,phase,anchor,128))
    expected=worker.v60.observe(data,model,pooled,phase,128)
    for w in ('first_post50','last64'):
        assert record['diagnostic']['windows'][w]['risk'] == pytest.approx(expected[w]['mean_coefficient'],abs=1e-12)
        assert record['diagnostic']['windows'][w]['signal'] == pytest.approx(expected[w]['mean_signal'],abs=1e-12)
    changed=deepcopy(data)
    changed['next_states'][128:] += .05
    other=dict(diagnostic=worker.diagnose(changed,model,pooled,phase,anchor,128))
    differences=worker.paired_differences(other,record)
    assert differences['pre64']['level_score'] == 0.
    for values in differences.values():
        assert values['level_score'] == pytest.approx(values['cross_term']+values['displacement'],abs=1e-9)


def test_complete_worker_reproduces_keeps_traces_and_resumes(shared_start,tmp_path,monkeypatch):
    import argparse
    import shutil
    import gymnasium as gym
    from scripts.v52_agent import ContinuationAgent
    checkpoint,original,model,pooled,phase=shared_start
    shared=deepcopy(original)
    initial=deepcopy(shared['replay'].buffer)
    while shared['replay'].size < 6144:
        shared['replay'].push(*initial[shared['replay'].size%len(initial)])
    monkeypatch.setattr(worker,'SPEC',dict(steps=8,switch_step=4))
    monkeypatch.setattr(worker.torch,'set_num_interop_threads',lambda _:None)
    monkeypatch.setattr(worker,'ContinuationAgent',lambda env,variant:ContinuationAgent(env,variant,'cpu'))
    torch.set_num_threads(2)
    source=tmp_path/worker.v53.SOURCE_NAME
    shutil.copy2(checkpoint,source)
    expected=[]
    for mode in worker.v53.MODES:
        env=gym.make('Hopper-v4')
        agent=worker.ContinuationAgent(env,'fixed-minus2')
        agent.start_shared(source,shared)
        cell=dict(split='test',seed=worker.SEED,updating=True,mode=mode)
        data,record=worker.v53.collect(agent,env,cell,worker.SPEC)
        record['observers']=dict(long_normal_refit=worker.v60.observe(data,model,pooled,phase,4))
        expected.append(record)
        env.close()
    metadata=worker.v60.reference_metadata(dict(long_normal_refit=(model,pooled,phase)))
    baseline=dict(revision=worker.v60.REVISION,stage_spec=worker.SPEC,reference_metadata=metadata,records=expected)
    (tmp_path/'baseline.json').write_text(json.dumps(baseline))
    torch.save(shared,tmp_path/'shared.pt')
    torch.save(dict(extra_state=dict(progress=dict(revision=worker.v60.REVISION,next_cell=16,
        references=dict(long_normal_refit=(model,pooled,phase))))),tmp_path/'reference.pt')
    args=argparse.Namespace(source_dir=tmp_path,shared_dir=tmp_path,reference_checkpoint=tmp_path/'reference.pt',
        baseline_result=tmp_path/'baseline.json',checkpoint_dir=tmp_path/'checkpoints',result_dir=tmp_path/'results',resume_from=None)
    worker.run(args)
    result=json.loads((tmp_path/'results/early_response.json').read_text())
    assert result['decision'] == 'reproduced-localization-ready'
    assert result['environment_interactions'] == result['optimizer_updates'] == 40
    assert len(list((tmp_path/'checkpoints/traces').glob('*.npz'))) == 5
    for r in result['records']:
        saved=np.load(r['server_trace'])
        assert len(saved['states']) == 8 and r['paired_prefix_equal']
        saved.close()
    args.resume_from=tmp_path/'checkpoints/preflight_latest.pt'
    worker.run(args)
    assert json.loads((tmp_path/'results/early_response.json').read_text()) == result
