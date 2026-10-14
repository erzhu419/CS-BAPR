from copy import deepcopy
import json
from pathlib import Path
import pickle

import numpy as np
import pytest
from sklearn.covariance import oas
import torch

from csbapr.belief.departure import NormalAgeReference, SETTINGS
from scripts import analyze_general_ood_v65_state_action as worker
from scripts.v65_state_action_readout import StateActionReference, OnlineStateActionReadout, NEIGHBORS
from test_general_ood_v58 import shared_start, trace as short_trace
from test_general_ood_v57 import trace as old_trace


def trace(seed,steps=6000):
    data = old_trace(seed,steps)
    data['executed_actions'] = np.random.RandomState(seed+100).normal(size=(steps,3))
    return data


@pytest.fixture(scope='module')
def reference(shared_start):
    _,_,model,pooled,_ = shared_start
    return StateActionReference.fit_normal(model,pooled,[trace(10),trace(11)],[trace(12),trace(13)])


def test_contract_preserves_gates_and_normal_only_zero_rl_budget():
    m = json.loads((Path(__file__).resolve().parents[1]/
        'experiments/general_ood_v65_state_action_20261008/dispatch_manifest.json').read_text())
    assert m['added_interactions']==m['added_optimizer_updates']==0
    assert m['readout_replay_cells']==17 and m['main_candidate']==worker.MAIN
    assert m['fit_normal_transitions']==m['calibration_normal_transitions']==12000
    assert m['normal_reference_windows']==12000//m['window_steps']==750
    assert m['neighbors']==NEIGHBORS==64
    assert m['normal_fit_seeds']==worker.v64.v63.v62.SPEC['fit']
    assert m['normal_calibration_seeds']==worker.v64.v63.v62.SPEC['calibration']
    assert not set(m['normal_fit_seeds'])&set(m['normal_calibration_seeds'])
    assert list(worker.OBSERVERS)==m['observers']
    assert m['normal_windows']==list(worker.v64.v63.v62.NORMAL_WINDOWS)
    assert m['change_windows']==list(worker.v64.v63.v62.CHANGE_WINDOWS)
    assert m['knot_interval']==SETTINGS['knot_interval']==m['window_steps']==16
    assert m['age_radius']==SETTINGS['age_radius']==8


def test_exact_normal_neighbors_oas_geometry_and_separate_calibration(shared_start,reference):
    _,_,model,pooled,_ = shared_start
    fitting = [trace(10),trace(11)]
    inputs = np.concatenate([worker.rolling_means(model.inputs(t))[15::16] for t in fitting])
    moments = np.concatenate([worker.rolling_means(worker.normalized_residuals(model,t))[15::16] for t in fitting])
    np.testing.assert_array_equal(reference.moments,moments)
    np.testing.assert_array_equal(reference.global_precision,pooled.precision)
    np.testing.assert_array_equal(reference.tree.data,(inputs-reference.input_mean)/reference.input_scale)
    queries = inputs[[20,80,700]]+.001
    centers,precisions,diagnostic = reference.geometry(queries)
    for i,query in enumerate(queries):
        distances = np.linalg.norm((inputs-query)/reference.input_scale,axis=1)
        selected = np.argsort(distances)[:64]
        np.testing.assert_allclose(centers[i],moments[selected].mean(axis=0),atol=1e-12)
        covariance,weight = oas(moments[selected])
        expected = np.linalg.inv(covariance+.001*np.trace(covariance)/11*np.eye(11))
        np.testing.assert_allclose(precisions[i],expected,rtol=1e-10,atol=1e-10)
        assert diagnostic['shrinkage'][i]==pytest.approx(weight)
        assert diagnostic['neighbor_radius'][i]==pytest.approx(distances[selected[-1]])
        assert np.linalg.eigvalsh(precisions[i]).min()>0
    scores,_ = reference.scores(centers,queries)
    assert all(np.max(s)==0. for s in scores.values())
    moved,_ = reference.scores(centers+.01,queries)
    assert all(np.min(s)>0. for s in moved.values())
    calibration = [trace(12),trace(13)]
    moments = np.concatenate([worker.rolling_means(worker.normalized_residuals(model,t)) for t in calibration])
    inputs = np.concatenate([worker.rolling_means(model.inputs(t)) for t in calibration])
    ages = np.concatenate([worker.v64.v63.v62.v53.v43.episode_steps(t['dones']) for t in calibration])
    phases = np.concatenate([worker.v64.v63.episode_ordinals(t['dones'])>0 for t in calibration])
    scores,diagnostic = reference.scores(moments,inputs)
    assert reference.neighbor_radius95==pytest.approx(np.quantile(diagnostic['neighbor_radius'],.95))
    for name in worker.NAMES:
        assert reference.score95[name]==pytest.approx(np.quantile(scores[name],.95),abs=1e-12)
        values = np.log1p(scores[name]/reference.score95[name])
        for phase,curve in ((False,reference.phases[name].initial),(True,reference.phases[name].after_reset)):
            assert curve.metadata()==NormalAgeReference.fit_normal(values[phases==phase],ages[phases==phase]).metadata()
    assert reference.fit_transitions==reference.calibration_transitions==12000


def test_conditioning_ignores_next_state_and_does_not_fit_query_residual(shared_start,reference):
    _,_,model,_,_ = shared_start
    data = short_trace(8)
    changed = deepcopy(data)
    changed['next_states']+=10.
    inputs = worker.rolling_means(model.inputs(data))
    np.testing.assert_array_equal(inputs,worker.rolling_means(model.inputs(changed)))
    center,precision,_ = reference.geometry(inputs)
    altered_center,altered_precision,_ = reference.geometry(worker.rolling_means(model.inputs(changed)))
    np.testing.assert_array_equal(center,altered_center)
    np.testing.assert_array_equal(precision,altered_precision)
    original,_ = reference.evaluate(worker.rolling_means(worker.normalized_residuals(model,data)),inputs)
    altered,_ = reference.evaluate(worker.rolling_means(worker.normalized_residuals(model,changed)),inputs)
    assert all(np.any(original[n]!=altered[n]) for n in worker.NAMES)


def test_online_batch_causality_reset_persistence_resume_and_frozen_reference(shared_start,reference):
    _,_,model,_,_ = shared_start
    data = short_trace(8)
    ages = worker.v64.v63.v62.v53.v43.episode_steps(data['dones'])
    moments = worker.rolling_means(worker.normalized_residuals(model,data))
    inputs = worker.rolling_means(model.inputs(data))
    expected,_ = reference.evaluate(moments,inputs)
    future = deepcopy(data)
    future['states'][128:]+=20.
    future['executed_actions'][128:]+=20.
    future['next_states'][128:]+=30.
    altered,_ = reference.evaluate(worker.rolling_means(worker.normalized_residuals(model,future)),
                                   worker.rolling_means(model.inputs(future)))
    frozen = reference.moments.copy(),reference.tree.data.copy(),deepcopy(reference.metadata())
    for name in worker.NAMES:
        np.testing.assert_array_equal(expected[name][:128],altered[name][:128])
        sensor = OnlineStateActionReadout(model,reference,name)
        resumed,risks = None,[]
        for i,age in enumerate(ages):
            args = (data['states'][i],data['executed_actions'][i],data['next_states'][i],int(age))
            risks.append(sensor.observe(*args))
            assert sensor.signal==pytest.approx(expected[name][i],rel=1e-10,abs=1e-10)
            if i==32:
                assert len(sensor.history)==len(sensor.input_history)==16 and sensor.departure.after_reset
                np.testing.assert_allclose(np.mean(sensor.input_history,axis=0),inputs[i])
            if resumed is not None:
                assert resumed.observe(*args)==pytest.approx(risks[-1],abs=1e-12)
            if i==123:
                resumed = pickle.loads(pickle.dumps(sensor))
        observed = worker.observe(data,model,reference,128)
        assert observed[name]['first_post50']['mean_coefficient']==pytest.approx(np.mean(risks[128:178]),abs=1e-12)
    np.testing.assert_array_equal(frozen[0],reference.moments)
    np.testing.assert_array_equal(frozen[1],reference.tree.data)
    assert frozen[2]==reference.metadata()


def test_all_pair_gate_does_not_promote_ablation_or_drop_frozen_normal():
    records = []
    for group,seeds in (('v62_test',[134020,134021]),('v61_development_replay',[132020])):
        for seed in seeds:
            for updating,modes in ((False,['normal']),(True,worker.v64.v63.v62.v53.MODES)):
                for mode in modes:
                    records.append(dict(group=group,seed=seed,mode=mode,updating=updating,
                        observers={n:{w:dict(mean_coefficient=.01) for w in worker.v64.v63.v62.NORMAL_WINDOWS} for n in worker.OBSERVERS},
                        change_minus_nochange={n:{w:0. if mode=='normal' else .2 for w in worker.v64.v63.v62.CHANGE_WINDOWS} for n in worker.OBSERVERS}))
    assert worker.decision(worker.summarize(records)).startswith('retrospective-state-action-screen-pass')
    records[0]['observers'][worker.MAIN]['first800']['mean_coefficient']=.11
    summary = worker.summarize(records)
    assert worker.decision(summary)=='hold-state-action-screen'
    assert summary['v62_test']['state_action_center']['decision'].startswith('retrospective-screen-pass')
    records[0]['observers'][worker.MAIN]['first800']['mean_coefficient']=.01
    records[-1]['change_minus_nochange'][worker.MAIN]['last64']=0.
    assert worker.summarize(records)['v61_development_replay'][worker.MAIN]['positive_pairs']['last64']==3
    assert worker.decision(worker.summarize(records))=='hold-state-action-screen'


def test_full_worker_fits_once_inherits_controls_and_keeps_reference_server_side(shared_start,tmp_path,monkeypatch):
    import argparse
    _,_,model,pooled,phase = shared_start
    monkeypatch.setattr(worker.v64.v63.v62,'SPEC',dict(worker.v64.v63.v62.SPEC,steps=1200,switch_step=800))
    v60,v61,v62,v64 = [tmp_path/name for name in ('v60','v61','v62','v64')]
    old = v60/'run_checkpoints/preflight'
    old.mkdir(parents=True)
    torch.save(dict(extra_state=dict(progress=dict(revision=worker.v64.v63.v62.v60.REVISION,next_cell=16,
        references=dict(long_normal_refit=(model,pooled,phase))))),old/'preflight_latest.pt')
    stored = v62/'run_checkpoints/preflight/traces'
    stored.mkdir(parents=True)
    inputs = []
    for split,seeds in (('fit',[10,11]),('calibration',[12,13])):
        for updating,seed in zip((False,True),seeds):
            cell = dict(split=split,seed=134000 if split=='fit' else 134010,updating=updating,mode='normal')
            kind = 'updating' if updating else 'frozen'
            np.savez_compressed(stored/f'{split}_seed{cell["seed"]}_{kind}_normal.npz',**trace(seed))
            inputs.append(cell)
    inherited = worker.v64.v63.v62.v60.reference_metadata(dict(v60_frozen=(model,pooled,phase)))
    result = v62/'run_results/preflight'
    result.mkdir(parents=True)
    (result/'causal_innovation.json').write_text(json.dumps(dict(revision=worker.v64.v63.v62.REVISION,
        stage_spec=worker.v64.v63.v62.SPEC,records=inputs,inherited_reference_metadata=inherited)))
    legacy = v61/'run_checkpoints/preflight/traces'
    legacy.mkdir(parents=True)
    records = []
    cases = worker.v64.v63.v62.v60.trajectory_specs(worker.v64.v63.v62.SPEC)[4:]
    cases += [dict(seed=132020,updating=True,mode=mode) for mode in worker.v64.v63.v62.v53.MODES]
    for cell in cases:
        group = 'v61_development_replay' if cell['seed']==132020 else 'v62_test'
        data = trace(cell['seed'],1200)
        index = list(worker.v64.v63.v62.v53.MODES).index(cell['mode'])
        data['next_states'][800:]+=.02*index
        if group=='v62_test':
            kind = 'updating' if cell['updating'] else 'frozen'
            path = stored/f'test_seed{cell["seed"]}_{kind}_{cell["mode"]}.npz'
        else:
            path = legacy/f'seed{cell["seed"]}_{cell["mode"]}.npz'
        np.savez_compressed(path,**data)
        records.append(dict(group=group,seed=cell['seed'],updating=cell['updating'],mode=cell['mode'],
            observers={n:{w:dict(mean_coefficient=.01+.01*index,mean_signal=.2+.01*index)
                for w in worker.v64.window_ranges(1200,800)} for n in worker.v64.OBSERVERS}))
    result = v64/'run_results/screen'
    result.mkdir(parents=True)
    (result/'phase_vector.json').write_text(json.dumps(dict(revision=worker.v64.REVISION,
        environment_interactions=0,optimizer_updates=0,records=records)))
    calls = []
    original = StateActionReference.fit_normal
    def fit(m,p,f,c):
        calls.append((len(f),len(c)))
        return original(m,p,f,c)
    monkeypatch.setattr(worker.StateActionReference,'fit_normal',fit)
    args = argparse.Namespace(v60_dir=v60,v61_dir=v61,v62_dir=v62,v64_dir=v64,
        checkpoint_dir=tmp_path/'new_reference',result_dir=tmp_path/'result')
    worker.run(args)
    output = json.loads((args.result_dir/'state_action.json').read_text())
    assert calls==[(2,2)] and len(output['records'])==17
    assert output['environment_interactions']==output['optimizer_updates']==0
    assert output['replayed_saved_transitions']==20400
    for old,new in zip(records,output['records']):
        for name in worker.v64.OBSERVERS:
            assert old['observers'][name]==new['observers'][name]
    metadata = output['reference_metadata']
    assert metadata['fit_transitions']==metadata['calibration_transitions']==12000
    assert metadata['normal_reference_windows']==750 and not metadata['neighbor_distance_controls_risk']
    assert not any(k in metadata for k in ('moments','centers','precision','inputs'))
    saved = torch.load(output['server_reference_checkpoint'],map_location='cpu',weights_only=False)
    assert saved['reference'].moments.shape==(750,11) and saved['reference'].tree.m==14
    assert saved['reference_metadata']==metadata
