from copy import deepcopy
import json
from pathlib import Path
import pickle

import numpy as np
import pytest
import torch

from csbapr.belief.departure import NormalAgeReference
from scripts import analyze_general_ood_v66_covariance_mix as worker
from scripts import v66_covariance_mix_readout as readout
from scripts.v65_state_action_readout import StateActionReference
from test_general_ood_v65 import trace
from test_general_ood_v58 import shared_start, trace as short_trace


@pytest.fixture(scope='module')
def original(shared_start):
    _,_,model,pooled,_ = shared_start
    return StateActionReference.fit_normal(model,pooled,[trace(10),trace(11)],[trace(12),trace(13)])


@pytest.fixture(scope='module')
def reference(shared_start,original):
    _,_,model,_,_ = shared_start
    return readout.MixedStateActionReference.fit_normal(model,original,[trace(10),trace(11)],[trace(12),trace(13)])


def test_contract_preserves_budget_normal_calibration_and_gates():
    m = json.loads((Path(__file__).resolve().parents[1]/
        'experiments/general_ood_v66_covariance_mix_20261008/dispatch_manifest.json').read_text())
    assert m['added_interactions']==m['added_optimizer_updates']==0
    assert m['main_candidate']==readout.NAME and m['readout_replay_cells']==17
    assert m['fit_normal_transitions']==m['calibration_normal_transitions']==12000
    assert m['validation_block_steps']==readout.BLOCK_STEPS==1000
    assert m['validation_folds']==6 and m['paired_branches_held_out_together'] and m['purge_overlapping_training_windows']
    assert m['normal_fit_seeds']==worker.v65.v64.v63.v62.SPEC['fit']
    assert m['normal_calibration_seeds']==worker.v65.v64.v63.v62.SPEC['calibration']
    assert not set(m['normal_fit_seeds'])&set(m['normal_calibration_seeds'])
    assert m['observers']==list(worker.OBSERVERS)
    assert m['normal_windows']==list(worker.v65.v64.v63.v62.NORMAL_WINDOWS)
    assert m['change_windows']==list(worker.v65.v64.v63.v62.CHANGE_WINDOWS)


def test_normal_nll_selects_local_global_and_interior_endpoints():
    local = np.tile(np.eye(2),(40,1,1))
    global_covariance = 4*np.eye(2)
    directions = np.tile([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]],(10,1))
    for variance,expected in ((1.,0.),(2.,1/3),(4.,1.)):
        offsets = directions*np.sqrt(2*variance)
        weight,selection = readout.select_weight(offsets,local,global_covariance)
        assert weight==pytest.approx(expected,abs=1e-5)
        assert selection['optimizer_success']
        assert selection['selected_normal_nll']<=min(selection['local_only_normal_nll'],selection['global_only_normal_nll'])
        assert readout.normal_nll(weight,offsets,local,global_covariance)==pytest.approx(np.log(variance)+1,abs=1e-9)


def test_coupled_folds_purge_overlap_and_refit_scaling_only_on_training(shared_start,monkeypatch):
    _,_,model,_,_ = shared_start
    fitting = [trace(10),trace(11)]
    inputs,moments,ends = readout.fit_windows(model,fitting)
    calls = []
    original_geometry = readout.normal_geometry
    def geometry(tree,values,mean,scale,queries):
        a = len(calls)*1000
        train = (ends<a)|(ends-15>=a+1000)
        validation = (ends-15>=a)&(ends<a+1000)
        np.testing.assert_array_equal(values,moments[train])
        np.testing.assert_array_equal(queries,inputs[validation])
        np.testing.assert_allclose(mean,inputs[train].mean(axis=0),rtol=0,atol=1e-12)
        np.testing.assert_allclose(scale,np.maximum(inputs[train].std(axis=0),1e-6),rtol=0,atol=1e-12)
        np.testing.assert_allclose(tree.data,(inputs[train]-mean)/scale,rtol=0,atol=1e-12)
        assert not np.any(train&validation)
        calls.append(int(validation.sum()))
        return original_geometry(tree,values,mean,scale,queries)
    monkeypatch.setattr(readout,'normal_geometry',geometry)
    offsets,covariances,folds = readout.validation_samples(model,fitting)
    assert len(calls)==len(folds)==6 and len(offsets)==sum(calls)==744
    assert covariances.shape==(744,11,11)
    assert all(f['paired_branches_held_out_together'] for f in folds)
    assert [f['purged_boundary_windows'] for f in folds]==[2]*6


def test_mixture_endpoints_match_v65_and_calibration_is_exact(shared_start,original,reference):
    _,_,model,_,_ = shared_start
    data = short_trace(8)
    inputs = worker.rolling_means(model.inputs(data))
    centers,precision,_ = original.geometry(inputs)
    candidate = deepcopy(reference)
    for weight,expected in ((0.,precision),(1.,np.broadcast_to(original.global_precision,precision.shape))):
        candidate.global_weight = weight
        observed,actual,_ = candidate.geometry(inputs)
        np.testing.assert_array_equal(observed,centers)
        np.testing.assert_allclose(actual,expected,rtol=1e-9,atol=1e-9)
    calibration = [trace(12),trace(13)]
    moments = np.concatenate([worker.rolling_means(worker.normalized_residuals(model,t)) for t in calibration])
    inputs = np.concatenate([worker.rolling_means(model.inputs(t)) for t in calibration])
    ages = np.concatenate([readout.v43.episode_steps(t['dones']) for t in calibration])
    phases = np.concatenate([readout.episode_ordinals(t['dones'])>0 for t in calibration])
    scores,_ = reference.scores(moments,inputs)
    assert reference.score95[readout.NAME]==pytest.approx(np.quantile(scores[readout.NAME],.95),abs=1e-12)
    signals = np.log1p(scores[readout.NAME]/reference.score95[readout.NAME])
    for phase,curve in ((False,reference.phases[readout.NAME].initial),(True,reference.phases[readout.NAME].after_reset)):
        assert curve.metadata()==NormalAgeReference.fit_normal(signals[phases==phase],ages[phases==phase]).metadata()
    assert not reference.metadata()['calibration_used_for_weight_selection']
    assert not reference.metadata()['ood_used_for_weight_selection']


def test_weight_frozen_before_any_calibration_residual_is_read(shared_start,original,monkeypatch):
    _,_,model,_,_ = shared_start
    fitting,calibration = [trace(10),trace(11)],[trace(14),trace(15)]
    calls = []
    old_errors,old_select = model.errors,readout.select_weight
    def errors(data):
        if any(data is t for t in calibration):
            assert calls, 'calibration-read-before-weight-selection'
        return old_errors(data)
    def select(*args):
        result = old_select(*args)
        calls.append(result[0])
        return result
    monkeypatch.setattr(model,'errors',errors)
    monkeypatch.setattr(readout,'select_weight',select)
    changed = readout.MixedStateActionReference.fit_normal(model,original,fitting,calibration)
    assert len(calls)==1
    offsets,local,_ = readout.validation_samples(model,fitting)
    expected,_ = old_select(offsets,local,np.linalg.inv(original.global_precision))
    assert changed.global_weight==expected


def test_online_batch_causality_reset_persistence_resume_and_frozen_geometry(shared_start,reference):
    _,_,model,_,_ = shared_start
    data = short_trace(8)
    ages = readout.v43.episode_steps(data['dones'])
    def evaluate(t):
        return reference.evaluate(worker.rolling_means(worker.normalized_residuals(model,t)),worker.rolling_means(model.inputs(t)))[0][readout.NAME]
    expected = evaluate(data)
    future = deepcopy(data)
    future['states'][128:]+=10.
    future['next_states'][128:]+=20.
    future['executed_actions'][128:]+=10.
    np.testing.assert_array_equal(expected[:128],evaluate(future)[:128])
    frozen = reference.moments.copy(),reference.tree.data.copy(),deepcopy(reference.metadata())
    online = readout.OnlineMixedStateActionReadout(model,reference)
    resumed,risks = None,[]
    for i,age in enumerate(ages):
        args = (data['states'][i],data['executed_actions'][i],data['next_states'][i],int(age))
        risks.append(online.observe(*args))
        assert online.signal==pytest.approx(expected[i],rel=1e-10,abs=1e-10)
        if i==32:
            assert online.departure.after_reset and len(online.history)==len(online.input_history)==16
        if resumed is not None:
            assert resumed.observe(*args)==pytest.approx(risks[-1],abs=1e-12)
        if i==123:
            resumed = pickle.loads(pickle.dumps(online))
    assert worker.observe(data,model,reference,128)[readout.NAME]['first_post50']['mean_coefficient']==pytest.approx(np.mean(risks[128:178]))
    np.testing.assert_array_equal(frozen[0],reference.moments)
    np.testing.assert_array_equal(frozen[1],reference.tree.data)
    assert frozen[2]==reference.metadata()


def test_full_worker_preserves_controls_gates_and_server_only_storage(shared_start,original,tmp_path,monkeypatch):
    import argparse
    _,_,model,_,_ = shared_start
    monkeypatch.setattr(worker.v65.v64.v63.v62,'SPEC',dict(worker.v65.v64.v63.v62.SPEC,steps=1200,switch_step=800))
    v61,v62,v65 = [tmp_path/name for name in ('v61','v62','v65')]
    stored = v62/'run_checkpoints/preflight/traces'
    stored.mkdir(parents=True)
    inputs = []
    for split,seeds in (('fit',[10,11]),('calibration',[12,13])):
        for updating,seed in zip((False,True),seeds):
            cell = dict(split=split,seed=134000 if split=='fit' else 134010,updating=updating,mode='normal')
            kind = 'updating' if updating else 'frozen'
            np.savez_compressed(stored/f'{split}_seed{cell["seed"]}_{kind}_normal.npz',**trace(seed))
            inputs.append(cell)
    result = v62/'run_results/preflight'
    result.mkdir(parents=True)
    spec = worker.v65.v64.v63.v62.SPEC
    (result/'causal_innovation.json').write_text(json.dumps(dict(revision=worker.v65.v64.v63.v62.REVISION,stage_spec=spec,records=inputs)))
    old_reference = v65/'run_checkpoints/reference'
    old_reference.mkdir(parents=True)
    torch.save(dict(revision=worker.v65.REVISION,model=model,reference=original,reference_metadata=original.metadata(),
        normal_fit_seeds=spec['fit'],normal_calibration_seeds=spec['calibration']),old_reference/'state_action_reference.pt')
    legacy = v61/'run_checkpoints/preflight/traces'
    legacy.mkdir(parents=True)
    records = []
    cases = worker.v65.v64.v63.v62.v60.trajectory_specs(spec)[4:]
    cases += [dict(seed=132020,updating=True,mode=mode) for mode in worker.v65.v64.v63.v62.v53.MODES]
    for cell in cases:
        group = 'v61_development_replay' if cell['seed']==132020 else 'v62_test'
        data = trace(cell['seed'],1200)
        index = list(worker.v65.v64.v63.v62.v53.MODES).index(cell['mode'])
        data['next_states'][800:]+=.02*index
        if group=='v62_test':
            kind = 'updating' if cell['updating'] else 'frozen'
            path = stored/f'test_seed{cell["seed"]}_{kind}_{cell["mode"]}.npz'
        else:
            path = legacy/f'seed{cell["seed"]}_{cell["mode"]}.npz'
        np.savez_compressed(path,**data)
        records.append(dict(group=group,seed=cell['seed'],updating=cell['updating'],mode=cell['mode'],
            observers={n:{w:dict(mean_coefficient=.01+.01*index,mean_signal=.2+.01*index)
                for w in worker.v65.v64.window_ranges(1200,800)} for n in worker.v65.OBSERVERS}))
    result = v65/'run_results/screen'
    result.mkdir(parents=True)
    (result/'state_action.json').write_text(json.dumps(dict(revision=worker.v65.REVISION,
        environment_interactions=0,optimizer_updates=0,reference_metadata=original.metadata(),records=records)))
    calls = []
    old_fit = readout.MixedStateActionReference.fit_normal
    def fit(m,o,f,c):
        calls.append((len(f),len(c)))
        return old_fit(m,o,f,c)
    monkeypatch.setattr(worker.MixedStateActionReference,'fit_normal',fit)
    args = argparse.Namespace(v61_dir=v61,v62_dir=v62,v65_dir=v65,checkpoint_dir=tmp_path/'new_reference',result_dir=tmp_path/'result')
    worker.run(args)
    output = json.loads((args.result_dir/'covariance_mix.json').read_text())
    assert calls==[(2,2)] and len(output['records'])==17
    assert output['environment_interactions']==output['optimizer_updates']==0 and output['replayed_saved_transitions']==20400
    for old,new in zip(records,output['records']):
        assert all(old['observers'][n]==new['observers'][n] for n in worker.v65.OBSERVERS)
    metadata = output['reference_metadata']
    assert metadata['normal_reference_windows']==750 and metadata['weight_selection']['normal_validation_windows']==744
    assert not any(k in metadata for k in ('moments','precision','global_covariance','inputs','offsets'))
    saved = torch.load(output['server_reference_checkpoint'],map_location='cpu',weights_only=False)
    assert saved['reference_metadata']==metadata and saved['reference'].global_covariance.shape==(11,11)
    gate_records = deepcopy(output['records'])
    for r in gate_records:
        if r['mode']=='normal':
            for w in worker.v65.v64.v63.v62.NORMAL_WINDOWS:
                r['observers'][readout.NAME][w]['mean_coefficient']=.01
        else:
            for w in worker.v65.v64.v63.v62.CHANGE_WINDOWS:
                r['change_minus_nochange'][readout.NAME][w]=.1
    assert worker.decision(worker.summarize(gate_records)).startswith('retrospective-covariance-mix-screen-pass')
    gate_records[0]['observers'][readout.NAME]['first800']['mean_coefficient']=.11
    assert worker.decision(worker.summarize(gate_records))=='hold-covariance-mix-screen'
    gate_records[0]['observers'][readout.NAME]['first800']['mean_coefficient']=.01
    gate_records[-1]['change_minus_nochange'][readout.NAME]['last64']=0.
    assert worker.decision(worker.summarize(gate_records))=='hold-covariance-mix-screen'
