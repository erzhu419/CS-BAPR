from copy import deepcopy
import json
from pathlib import Path
import pickle

import numpy as np
import pytest
from sklearn.covariance import oas
import torch

from csbapr.belief.departure import NormalAgeReference
from scripts import analyze_general_ood_v67_forecast as worker
from scripts import v67_forecast_readout as readout
from scripts.v65_state_action_readout import StateActionReference
from scripts.v66_covariance_mix_readout import MixedStateActionReference, OnlineMixedStateActionReadout
from test_general_ood_v65 import trace
from test_general_ood_v58 import shared_start, trace as short_trace


@pytest.fixture(scope='module')
def level(shared_start):
    _,_,model,pooled,_ = shared_start
    fitting,calibration = [trace(10),trace(11)],[trace(12),trace(13)]
    old = StateActionReference.fit_normal(model,pooled,fitting,calibration)
    return MixedStateActionReference.fit_normal(model,old,fitting,calibration)


@pytest.fixture(scope='module')
def reference(shared_start,level):
    _,_,model,_,_ = shared_start
    return readout.ForecastReference.fit_normal(model,level,[trace(10),trace(11)],[trace(12),trace(13)])


def test_contract_preserves_budget_gates_and_inherited_level():
    m = json.loads((Path(__file__).resolve().parents[1]/
        'experiments/general_ood_v67_forecast_20261008/dispatch_manifest.json').read_text())
    assert m['added_interactions']==m['added_optimizer_updates']==0
    assert m['main_candidate']==readout.MAIN and m['readout_replay_cells']==17
    assert m['fit_normal_transitions']==m['calibration_normal_transitions']==12000
    assert m['forecast_input_dimension']==26 and m['forecast_feature_count']==27
    assert m['forecast_degree']==1 and m['forecast_ridge_mean_loss_coefficient']==readout.RIDGE==.001
    assert not m['inherited_level_refitted']
    assert m['normal_fit_seeds']==worker.v66.v65.v64.v63.v62.SPEC['fit']
    assert m['normal_calibration_seeds']==worker.v66.v65.v64.v63.v62.SPEC['calibration']
    assert not set(m['normal_fit_seeds'])&set(m['normal_calibration_seeds'])
    assert m['observers']==list(worker.OBSERVERS)
    assert m['normal_windows']==list(worker.v66.v65.v64.v63.v62.NORMAL_WINDOWS)
    assert m['change_windows']==list(worker.v66.v65.v64.v63.v62.CHANGE_WINDOWS)
    assert m['window_steps']==16


def test_forecast_inputs_strictly_exclude_current_and_future_residual(shared_start,reference):
    _,_,model,_,_ = shared_start
    data = short_trace(8)
    residuals = readout.normalized_residuals(model,data)
    inputs = readout.forecast_inputs(model,data,residuals)
    assert inputs.shape==(256,26)
    for i in range(len(inputs)):
        expected = residuals[max(0,i-16):i].mean(axis=0) if i else np.zeros(11)
        np.testing.assert_allclose(inputs[i,14:25],expected,rtol=0,atol=1e-12)
        assert inputs[i,-1]==min(i,16)/16
    changed = deepcopy(data)
    changed['next_states'][128:]+=10.
    altered = readout.forecast_inputs(model,changed,readout.normalized_residuals(model,changed))
    np.testing.assert_array_equal(inputs[:129],altered[:129])
    assert np.any(inputs[129]!=altered[129])
    np.testing.assert_array_equal(reference.forecast.predict(inputs[:129]),reference.forecast.predict(altered[:129]))


def test_exact_ridge_normal_covariance_and_separate_calibration(shared_start,level,reference):
    _,_,model,_,_ = shared_start
    fitting,calibration = [trace(10),trace(11)],[trace(12),trace(13)]
    residuals = [readout.normalized_residuals(model,t) for t in fitting]
    inputs = np.concatenate([readout.forecast_inputs(model,t,r) for t,r in zip(fitting,residuals)])
    targets = np.concatenate(residuals)
    forecast = reference.forecast
    design = forecast.design(inputs)
    penalty = len(inputs)*.001*np.eye(27)
    penalty[0,0] = 0.
    expected = np.linalg.solve(design.T@design+penalty,design.T@(targets/forecast.target_scale))
    np.testing.assert_allclose(forecast.coefficients,expected,rtol=0,atol=1e-12)
    moments = np.concatenate([readout.rolling_means(forecast.errors(model,t)) for t in fitting])
    np.testing.assert_array_equal(reference.center,moments.mean(axis=0))
    covariance,weight = oas(moments)
    expected = np.linalg.inv(covariance+.001*np.trace(covariance)/11*np.eye(11))
    np.testing.assert_allclose(reference.precision,expected,rtol=1e-10,atol=1e-10)
    assert reference.shrinkage==weight and np.linalg.eigvalsh(reference.precision).min()>0
    moments = np.concatenate([readout.rolling_means(forecast.errors(model,t)) for t in calibration])
    scores = reference.scores(moments)
    assert reference.score95==pytest.approx(np.quantile(scores,.95),abs=1e-12)
    ages = np.concatenate([readout.v43.episode_steps(t['dones']) for t in calibration])
    phases = np.concatenate([readout.episode_ordinals(t['dones'])>0 for t in calibration])
    signals = np.log1p(scores/reference.score95)
    for phase,curve in ((False,reference.phase.initial),(True,reference.phase.after_reset)):
        assert curve.metadata()==NormalAgeReference.fit_normal(signals[phases==phase],ages[phases==phase]).metadata()
    assert reference.level is level
    assert reference.fit_transitions==reference.calibration_transitions==forecast.fit_transitions==12000


def test_forecast_frozen_before_calibration_and_level_not_refitted(shared_start,level,monkeypatch):
    _,_,model,_,_ = shared_start
    fitting,calibration = [trace(10),trace(11)],[trace(14),trace(15)]
    before = deepcopy(level.metadata()),level.moments.copy(),level.tree.data.copy()
    calls = []
    old_errors,old_fit = model.errors,readout.NormalResidualForecast.fit_normal
    def errors(data):
        if any(data is t for t in calibration):
            assert calls,'calibration-read-before-forecast-fit'
        return old_errors(data)
    def fit(m,ts):
        assert all(t is f for t,f in zip(ts,fitting))
        result = old_fit(m,ts)
        calls.append(result.coefficients.copy())
        return result
    monkeypatch.setattr(model,'errors',errors)
    monkeypatch.setattr(readout.NormalResidualForecast,'fit_normal',fit)
    changed = readout.ForecastReference.fit_normal(model,level,fitting,calibration)
    assert len(calls)==1
    np.testing.assert_array_equal(changed.forecast.coefficients,calls[0])
    assert before[0]==level.metadata()
    np.testing.assert_array_equal(before[1],level.moments)
    np.testing.assert_array_equal(before[2],level.tree.data)


def test_online_batch_future_causality_reset_resume_and_level_floor(shared_start,reference):
    _,_,model,_,_ = shared_start
    data = short_trace(8)
    ages = readout.v43.episode_steps(data['dones'])
    expected = readout.observe_series(data,model,reference)
    future = deepcopy(data)
    future['states'][128:]+=10.
    future['next_states'][128:]+=20.
    future['executed_actions'][128:]+=10.
    altered = readout.observe_series(future,model,reference)
    for key in expected:
        np.testing.assert_array_equal(expected[key][:128],altered[key][:128])
    frozen = deepcopy(reference.metadata()),reference.forecast.coefficients.copy(),reference.precision.copy()
    for name in readout.NAMES:
        sensor = readout.OnlineForecastReadout(model,reference,name)
        base = OnlineMixedStateActionReadout(model,reference.level)
        resumed,risks = None,[]
        for i,age in enumerate(ages):
            args = (data['states'][i],data['executed_actions'][i],data['next_states'][i],int(age))
            risks.append(sensor.observe(*args))
            old = base.observe(*args)
            assert sensor.level_risk==pytest.approx(old,abs=1e-12)
            key = 'combined_risk' if name==readout.MAIN else 'forecast_risk'
            assert risks[-1]==pytest.approx(expected[key][i],rel=1e-10,abs=1e-10)
            assert sensor.forecast_signal==pytest.approx(expected['forecast_signal'][i],rel=1e-10,abs=1e-10)
            if name==readout.MAIN:
                assert risks[-1]>=old
            if i==32:
                assert sensor.level_sensor.after_reset and sensor.forecast_sensor.after_reset
                assert len(sensor.history)==len(sensor.input_history)==len(sensor.error_history)==16
            if resumed is not None:
                assert resumed.observe(*args)==pytest.approx(risks[-1],abs=1e-12)
            if i==123:
                resumed = pickle.loads(pickle.dumps(sensor))
        observed,_ = worker.observe(data,model,reference,128)
        assert observed[name]['first_post50']['mean_coefficient']==pytest.approx(np.mean(risks[128:178]),abs=1e-12)
    assert frozen[0]==reference.metadata()
    np.testing.assert_array_equal(frozen[1],reference.forecast.coefficients)
    np.testing.assert_array_equal(frozen[2],reference.precision)


def test_full_worker_inherits_controls_reproduces_v66_and_keeps_matrices_server_side(shared_start,level,tmp_path,monkeypatch):
    import argparse
    _,_,model,_,_ = shared_start
    monkeypatch.setattr(worker.v66.v65.v64.v63.v62,'SPEC',dict(worker.v66.v65.v64.v63.v62.SPEC,steps=1200,switch_step=800))
    v61,v62,v66 = [tmp_path/name for name in ('v61','v62','v66')]
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
    spec = worker.v66.v65.v64.v63.v62.SPEC
    (result/'causal_innovation.json').write_text(json.dumps(dict(revision=worker.v66.v65.v64.v63.v62.REVISION,stage_spec=spec,records=inputs)))
    old_reference = v66/'run_checkpoints/reference'
    old_reference.mkdir(parents=True)
    torch.save(dict(revision=worker.v66.REVISION,model=model,reference=level,reference_metadata=level.metadata(),
        normal_fit_seeds=spec['fit'],normal_calibration_seeds=spec['calibration']),old_reference/'covariance_mix_reference.pt')
    legacy = v61/'run_checkpoints/preflight/traces'
    legacy.mkdir(parents=True)
    records = []
    cases = worker.v66.v65.v64.v63.v62.v60.trajectory_specs(spec)[4:]
    cases += [dict(seed=132020,updating=True,mode=mode) for mode in worker.v66.v65.v64.v63.v62.v53.MODES]
    for cell in cases:
        group = 'v61_development_replay' if cell['seed']==132020 else 'v62_test'
        data = trace(cell['seed'],1200)
        index = list(worker.v66.v65.v64.v63.v62.v53.MODES).index(cell['mode'])
        data['next_states'][800:]+=.02*index
        if group=='v62_test':
            kind = 'updating' if cell['updating'] else 'frozen'
            path = stored/f'test_seed{cell["seed"]}_{kind}_{cell["mode"]}.npz'
        else:
            path = legacy/f'seed{cell["seed"]}_{cell["mode"]}.npz'
        np.savez_compressed(path,**data)
        observers = {n:{w:dict(mean_coefficient=.01+.01*index,mean_signal=.2+.01*index)
            for w in worker.v66.v65.v64.window_ranges(1200,800)} for n in worker.v66.OBSERVERS}
        observers.update(worker.v66.observe(data,model,level,800))
        records.append(dict(group=group,seed=cell['seed'],updating=cell['updating'],mode=cell['mode'],observers=observers))
    result = v66/'run_results/screen'
    result.mkdir(parents=True)
    (result/'covariance_mix.json').write_text(json.dumps(dict(revision=worker.v66.REVISION,
        environment_interactions=0,optimizer_updates=0,reference_metadata=level.metadata(),records=records)))
    calls = []
    old_fit = readout.ForecastReference.fit_normal
    def fit(m,l,f,c):
        calls.append((len(f),len(c)))
        return old_fit(m,l,f,c)
    monkeypatch.setattr(worker.ForecastReference,'fit_normal',fit)
    args = argparse.Namespace(v61_dir=v61,v62_dir=v62,v66_dir=v66,checkpoint_dir=tmp_path/'new_reference',result_dir=tmp_path/'result')
    worker.run(args)
    output = json.loads((args.result_dir/'forecast.json').read_text())
    assert calls==[(2,2)] and len(output['records'])==17
    assert output['environment_interactions']==output['optimizer_updates']==0 and output['replayed_saved_transitions']==20400
    assert output['maximum_v66_reproduction_error']<1e-10
    for old,new in zip(records,output['records']):
        assert all(old['observers'][n]==new['observers'][n] for n in worker.v66.OBSERVERS)
    metadata = output['reference_metadata']
    assert metadata['forecast']['input_dimension']==26 and metadata['forecast']['feature_count']==27
    assert not any(k in metadata for k in ('coefficients','precision','covariance','inputs'))
    saved = torch.load(output['server_reference_checkpoint'],map_location='cpu',weights_only=False)
    assert saved['reference_metadata']==metadata and saved['reference'].forecast.coefficients.shape==(27,11)
    gate_records = deepcopy(output['records'])
    for r in gate_records:
        if r['mode']=='normal':
            for w in worker.v66.v65.v64.v63.v62.NORMAL_WINDOWS:
                r['observers'][readout.MAIN][w]['mean_coefficient']=.01
        else:
            for w in worker.v66.v65.v64.v63.v62.CHANGE_WINDOWS:
                r['change_minus_nochange'][readout.MAIN][w]=.1
    assert worker.decision(worker.summarize(gate_records)).startswith('retrospective-forecast-screen-pass')
    gate_records[0]['observers'][readout.MAIN]['first800']['mean_coefficient']=.11
    assert worker.decision(worker.summarize(gate_records))=='hold-forecast-screen'
    gate_records[0]['observers'][readout.MAIN]['first800']['mean_coefficient']=.01
    gate_records[-1]['change_minus_nochange'][readout.MAIN]['last64']=0.
    assert worker.decision(worker.summarize(gate_records))=='hold-forecast-screen'
