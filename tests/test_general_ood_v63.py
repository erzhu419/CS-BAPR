from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import analyze_general_ood_v63_readout_attribution as worker
from test_general_ood_v58 import shared_start, trace as short_trace
from test_general_ood_v57 import trace
from test_general_ood_v62 import reference


def test_replay_only_contract():
    m=json.loads((Path(__file__).resolve().parents[1]/
        'experiments/general_ood_v63_readout_attribution_20261008/dispatch_manifest.json').read_text())
    assert m['added_interactions'] == m['added_optimizer_updates'] == 0
    assert not m['reference_refitted'] and not m['threshold_modified']
    assert m['readout_replay_cells'] == 17 and m['normal_fit_calibration_input_traces'] == 4
    assert m['resource_request']['vram_mb'] == 0
    assert worker.TOLERANCE == m['reproduction_tolerance'] == 1e-8


def test_reproduction_and_clipping_statistics(shared_start,reference):
    _,_,model,pooled,phase=shared_start
    data=short_trace(8)
    series=worker.arrays(data,model,reference,phase)
    diagnostic=worker.diagnose(series,128,1.)
    expected=dict(v60_frozen=worker.v62.v60.observe(data,model,pooled,phase,128),
                  **worker.v62.observe(data,model,reference,128))
    assert worker.reproduction_error(diagnostic,expected) < 1e-10
    incorrect=deepcopy(expected)
    incorrect['level_plus_innovation']['first_post50']['mean_coefficient']+=.01
    assert worker.reproduction_error(diagnostic,incorrect) > worker.TOLERANCE
    for values in series['observers'].values():
        values['threshold']=values['signal']+.1
        values['instant']=np.zeros(len(values['signal']))
    clipped=worker.window_statistics(series,128,178,1.)
    for values in clipped['observers'].values():
        assert values['above_threshold_fraction'] == values['mean_instant_coefficient'] == 0.
        assert values['mean_margin'] == pytest.approx(-.1,abs=1e-12)
        assert values['max_margin'] < 0.


def test_calibration_knots_match_actual_normal_quantiles(shared_start,reference):
    _,_,model,_,phase=shared_start
    before=deepcopy(reference.metadata())
    knots=worker.calibration_knots([trace(12,6000),trace(13,6000)],model,reference,phase)
    assert [k['age'] for k in knots] == [800,816,832,848,944,960,976,992,999]
    assert all(k['count'] > 0 for k in knots)
    for knot in knots:
        for name in worker.v62.NAMES:
            v=knot['observers'][name]
            assert v['threshold'] == pytest.approx(v['normal_calibration95'],abs=1e-10)
            assert v['normal_calibration_max'] >= v['normal_calibration95']
    assert reference.metadata() == before


def test_matched_signed_displacement_does_not_become_online_risk(shared_start,reference):
    _,_,model,pooled,phase=shared_start
    data=short_trace(8)
    normal=dict(diagnostic=worker.diagnose(worker.arrays(data,model,reference,phase),128,1.))
    changed=deepcopy(data)
    changed['next_states'][128:]+=.05
    record=dict(diagnostic=worker.diagnose(worker.arrays(changed,model,reference,phase),128,1.))
    differences=worker.paired_differences(record,normal,pooled)
    assert differences['pre64']['signed_mean_displacement_energy'] == 0.
    assert differences['first_post50']['signed_mean_displacement_energy'] > 0.
    assert len(differences['first_post50']['signed_mean_displacement']) == 11
    for name in worker.v62.OBSERVERS:
        assert differences['first_post50']['observers'][name]['mean_coefficient'] == pytest.approx(
            record['diagnostic']['first_post50']['observers'][name]['mean_coefficient']-
            normal['diagnostic']['first_post50']['observers'][name]['mean_coefficient'])
