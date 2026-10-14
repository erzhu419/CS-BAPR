from copy import deepcopy
import json
from pathlib import Path
import pickle

import numpy as np
import pytest
import torch

from csbapr.belief.departure import NormalAgeReference, SETTINGS
from scripts import analyze_general_ood_v64_phase_vector as worker
from scripts.v64_phase_vector_readout import PhaseVectorReference, OnlinePhaseVectorReadout
from test_general_ood_v58 import shared_start, trace as short_trace
from test_general_ood_v57 import trace


@pytest.fixture(scope='module')
def reference(shared_start):
    _,_,model,pooled,_=shared_start
    return PhaseVectorReference.fit_normal(model,pooled,[trace(10,6000),trace(11,6000)],
                                           [trace(12,6000),trace(13,6000)])


def test_contract_has_no_added_rl_budget_or_ood_fitting():
    m=json.loads((Path(__file__).resolve().parents[1]/
        'experiments/general_ood_v64_phase_vector_20261008/dispatch_manifest.json').read_text())
    assert m['added_interactions'] == m['added_optimizer_updates'] == 0
    assert m['readout_replay_cells'] == 17 and m['main_candidate'] == 'phase_center_covariance'
    assert m['fit_normal_transitions'] == m['calibration_normal_transitions'] == 12000
    assert not set(m['normal_fit_seeds'])&set(m['normal_calibration_seeds'])
    assert m['normal_fit_seeds'] == worker.v63.v62.SPEC['fit']
    assert m['normal_calibration_seeds'] == worker.v63.v62.SPEC['calibration']
    assert list(worker.OBSERVERS) == m['observers']
    assert SETTINGS['knot_interval'] == m['knot_interval'] == m['window_steps'] == 16
    assert SETTINGS['age_radius'] == m['age_radius'] == 8
    assert m['normal_windows'] == list(worker.v63.v62.NORMAL_WINDOWS)
    assert m['change_windows'] == list(worker.v63.v62.CHANGE_WINDOWS)


def test_geometry_and_calibration_use_exact_separate_normal_data(shared_start,reference):
    _,_,model,pooled,_=shared_start
    fitting=[trace(10,6000),trace(11,6000)]
    moments=np.concatenate([worker.rolling_means(worker.normalized_residuals(model,t)) for t in fitting])
    ages=np.concatenate([worker.v63.v62.v53.v43.episode_steps(t['dones']) for t in fitting])
    phases=np.concatenate([worker.v63.episode_ordinals(t['dones'])>0 for t in fitting])
    np.testing.assert_array_equal(reference.global_precision,pooled.precision)
    for phase,curve in reference.curves.items():
        for i,age in enumerate(curve.ages):
            selected=moments[(phases==phase)&(np.abs(ages-age)<=8)]
            np.testing.assert_allclose(curve.centers[i],selected.mean(axis=0),rtol=0,atol=1e-12)
            assert curve.counts[i] == len(selected)
            assert np.linalg.eigvalsh(curve.precisions[i]).min() > 0
    calibration=[trace(12,6000),trace(13,6000)]
    moments=np.concatenate([worker.rolling_means(worker.normalized_residuals(model,t)) for t in calibration])
    ages=np.concatenate([worker.v63.v62.v53.v43.episode_steps(t['dones']) for t in calibration])
    phases=np.concatenate([worker.v63.episode_ordinals(t['dones'])>0 for t in calibration])
    scores=reference.scores(moments,ages,phases)
    for name in worker.NAMES:
        assert reference.score95[name] == pytest.approx(np.quantile(scores[name],.95),abs=1e-12)
        values=np.log1p(scores[name]/reference.score95[name])
        for phase,curve in ((False,reference.phases[name].initial),(True,reference.phases[name].after_reset)):
            assert curve.metadata() == NormalAgeReference.fit_normal(values[phases==phase],ages[phases==phase]).metadata()
    assert reference.fit_transitions == reference.calibration_transitions == 12000


def test_towards_global_center_can_depart_from_conditional_center(shared_start,reference):
    _,_,_,pooled,_=shared_start
    curve=reference.curves[True]
    i=list(curve.ages).index(800)
    normal=curve.centers[i][None]
    moved=(normal+pooled.center[None])/2
    age,phase=np.array([800]),np.array([True])
    assert pooled.scores(moved)[0] < pooled.scores(normal)[0]
    initial,changed=reference.scores(normal,age,phase),reference.scores(moved,age,phase)
    for name in worker.NAMES:
        assert initial[name][0] == 0. and changed[name][0] > 0.


def test_online_causal_batch_equivalence_reset_history_and_resume(shared_start,reference):
    _,_,model,_,_=shared_start
    data=short_trace(8)
    ages=worker.v63.v62.v53.v43.episode_steps(data['dones'])
    phases=worker.v63.episode_ordinals(data['dones'])>0
    moments=worker.rolling_means(worker.normalized_residuals(model,data))
    expected=reference.signals(moments,ages,phases)
    future=deepcopy(data)
    future['next_states'][128:]+=20.
    altered=reference.signals(worker.rolling_means(worker.normalized_residuals(model,future)),ages,phases)
    frozen=[(c.centers.copy(),c.precisions.copy()) for c in reference.curves.values()]
    for name in worker.NAMES:
        np.testing.assert_array_equal(expected[name][:128],altered[name][:128])
        sensor=OnlinePhaseVectorReadout(model,reference,name)
        resumed=None
        risks=[]
        for i,age in enumerate(ages):
            args=(data['states'][i],data['executed_actions'][i],data['next_states'][i],int(age))
            risks.append(sensor.observe(*args))
            assert sensor.signal == pytest.approx(expected[name][i],rel=1e-10,abs=1e-10)
            if resumed is not None:
                assert resumed.observe(*args) == pytest.approx(risks[-1],abs=1e-12)
            if i==123:
                resumed=pickle.loads(pickle.dumps(sensor))
        assert sensor.departure.after_reset and len(sensor.history)==16
        observed=worker.observe(data,model,reference,128)
        assert observed[name]['first_post50']['mean_coefficient'] == pytest.approx(np.mean(risks[128:178]),abs=1e-12)
    for old,curve in zip(frozen,reference.curves.values()):
        np.testing.assert_array_equal(old[0],curve.centers)
        np.testing.assert_array_equal(old[1],curve.precisions)


def test_gate_requires_every_pair_and_does_not_substitute_ablation():
    records=[]
    for group,seeds in (('v62_test',[134020,134021]),('v61_development_replay',[132020])):
        for seed in seeds:
            for mode in worker.v63.v62.v53.MODES:
                records.append(dict(group=group,seed=seed,mode=mode,updating=True,
                    observers={n:{w:dict(mean_coefficient=.01) for w in worker.v63.v62.NORMAL_WINDOWS} for n in worker.OBSERVERS},
                    change_minus_nochange={n:{w:0. if mode=='normal' else .2 for w in worker.v63.v62.CHANGE_WINDOWS} for n in worker.OBSERVERS}))
    assert worker.summarize(records)['v62_test']['phase_center_covariance']['positive_pairs']==dict(first_post50=8,last64=8)
    records[0]['observers']['phase_center_covariance']['first800']['mean_coefficient']=.11
    summary=worker.summarize(records)['v62_test']
    assert summary['phase_center_covariance']['decision']=='hold-phase-vector-screen'
    assert summary['phase_center']['decision']=='retrospective-screen-pass-not-confirmation'
    records[0]['observers']['phase_center_covariance']['first800']['mean_coefficient']=.01
    records[-1]['change_minus_nochange']['phase_center_covariance']['last64']=0.
    assert worker.summarize(records)['v61_development_replay']['phase_center_covariance']['positive_pairs']['last64']==3


def test_full_replay_worker_fits_once_keeps_matrices_server_side(shared_start,tmp_path,monkeypatch):
    import argparse
    _,_,model,pooled,phase=shared_start
    torch.set_num_threads(2)
    monkeypatch.setattr(worker.v63.v62,'SPEC',dict(worker.v63.v62.SPEC,steps=1200,switch_step=800))
    v60,v61,v62,v63=[tmp_path/name for name in ('v60','v61','v62','v63')]
    old=v60/'run_checkpoints/preflight'
    old.mkdir(parents=True)
    torch.save(dict(extra_state=dict(progress=dict(revision=worker.v63.v62.v60.REVISION,next_cell=16,
        references=dict(long_normal_refit=(model,pooled,phase))))),old/'preflight_latest.pt')
    stored=v62/'run_checkpoints/preflight/traces'
    stored.mkdir(parents=True)
    inputs=[]
    for split,seeds in (('fit',[10,11]),('calibration',[12,13])):
        for updating,seed in zip((False,True),seeds):
            cell=dict(split=split,seed=134000 if split=='fit' else 134010,updating=updating,mode='normal')
            kind='updating' if updating else 'frozen'
            np.savez_compressed(stored/f'{split}_seed{cell["seed"]}_{kind}_normal.npz',**trace(seed,6000))
            inputs.append(cell)
    inherited=worker.v63.v62.v60.reference_metadata(dict(v60_frozen=(model,pooled,phase)))
    result=v62/'run_results/preflight'
    result.mkdir(parents=True)
    (result/'causal_innovation.json').write_text(json.dumps(dict(revision=worker.v63.v62.REVISION,
        stage_spec=worker.v63.v62.SPEC,records=inputs,inherited_reference_metadata=inherited)))
    legacy=v61/'run_checkpoints/preflight/traces'
    legacy.mkdir(parents=True)
    records=[]
    cases=worker.v63.v62.v60.trajectory_specs(worker.v63.v62.SPEC)[4:]
    cases += [dict(seed=132020,updating=True,mode=mode) for mode in worker.v63.v62.v53.MODES]
    for cell in cases:
        group='v61_development_replay' if cell['seed']==132020 else 'v62_test'
        data=trace(cell['seed'],1200)
        index=list(worker.v63.v62.v53.MODES).index(cell['mode'])
        data['next_states'][800:]+=.02*index
        if group=='v62_test':
            kind='updating' if cell['updating'] else 'frozen'
            path=stored/f'test_seed{cell["seed"]}_{kind}_{cell["mode"]}.npz'
        else:
            path=legacy/f'seed{cell["seed"]}_{cell["mode"]}.npz'
        np.savez_compressed(path,**data)
        records.append(dict(group=group,seed=cell['seed'],updating=cell['updating'],mode=cell['mode'],
            diagnostic={w:dict(observers={n:dict(mean_coefficient=.01+.01*index,mean_signal=.2+.01*index)
                for n in worker.v63.v62.OBSERVERS}) for w in worker.window_ranges(1200,800)}))
    result=v63/'run_results/attribution'
    result.mkdir(parents=True)
    (result/'readout_attribution.json').write_text(json.dumps(dict(revision=worker.v63.REVISION,
        maximum_reproduction_error=0.,records=records)))
    calls=[]
    original=PhaseVectorReference.fit_normal
    def fit(m,p,f,c):
        calls.append((len(f),len(c)))
        return original(m,p,f,c)
    monkeypatch.setattr(worker.PhaseVectorReference,'fit_normal',fit)
    args=argparse.Namespace(v60_dir=v60,v61_dir=v61,v62_dir=v62,v63_dir=v63,
        checkpoint_dir=tmp_path/'new_reference',result_dir=tmp_path/'result')
    worker.run(args)
    output=json.loads((args.result_dir/'phase_vector.json').read_text())
    assert calls==[(2,2)] and len(output['records'])==17
    assert output['environment_interactions']==output['optimizer_updates']==0
    assert output['replayed_saved_transitions']==20400
    assert output['reference_metadata']['fit_transitions']==output['reference_metadata']['calibration_transitions']==12000
    assert 'centers' not in output['reference_metadata']['vector_curves']['initial']
    saved=torch.load(output['server_reference_checkpoint'],map_location='cpu',weights_only=False)
    assert saved['reference'].curves[True].precisions.shape[1:]==(11,11)
    assert saved['reference_metadata']==output['reference_metadata']
