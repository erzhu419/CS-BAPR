from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from csbapr.belief.departure import NormalAgeReference
from scripts import probe_general_ood_v55_learning_calibration as v55
from scripts import probe_general_ood_v57_phase_calibration as worker
from scripts.probe_general_ood_v56_reset_calibration import episode_ordinals
from scripts.v54_window_readout import SignedWindowReference
from scripts.v57_phase_readout import PhaseAgeReference, PhaseDepartureReadout, fit_references


class Dynamics:
    calibrations = {'global-centered': SimpleNamespace(means=np.zeros((1, 11)), scales=np.ones((1, 11)))}

    def errors(self, trace):
        return trace['next_states'] - trace['states']


def trace(seed, steps=2000):
    rng = np.random.RandomState(seed)
    states = rng.normal(size=(steps, 11))
    errors = rng.normal(scale=.1, size=(steps, 11))
    errors[:1000] += 1.
    errors[1000:] += .1
    return dict(states=states, next_states=states + errors, executed_actions=np.zeros((steps, 3)),
                dones=np.arange(steps) % 1000 == 999)


@pytest.fixture
def original():
    model = Dynamics()
    reference = SignedWindowReference.fit_normal(model, [trace(1), trace(2)], [trace(3), trace(4)])
    return model, reference


def test_protocol_budgets_and_new_seed_roles():
    manifest = json.loads((Path(__file__).resolve().parents[1] /
        'experiments/general_ood_v57_phase_calibration_20261007/dispatch_manifest.json').read_text())
    seeds = []
    for stage in ('preflight', 'formal'):
        spec = worker.stage_spec(stage)
        assert spec == manifest['stage_specs'][stage]
        cells = worker.trajectory_specs(spec)
        budget = sum(spec['calibration_steps'] if c['split'] == 'calibration' else spec['steps'] for c in cells)
        assert budget == manifest['added_budgets'][stage]['interactions'] == manifest['added_budgets'][stage]['updates']
        assert all(c['updating'] for c in cells)
        assert spec['calibration_steps'] == 2000 and spec['steps'] == 1200
        seeds.extend(spec['calibration'] + spec['test'])
    assert len(seeds) == len(set(seeds))
    assert not set(seeds) & {98600, 98601, 98620, 98621, 98700, 98701, 98720, 98721, 98722, 98723}


def test_same_data_pooled_control_and_exact_phase_curves(original):
    model, reference = original
    before = deepcopy(reference.metadata())
    traces = [trace(5), trace(6)]
    pooled, phase = fit_references(model, reference, traces)
    assert pooled.metadata() == v55.calibrate_learning(model, reference, traces).metadata()
    for name in ('center', 'covariance', 'precision'):
        np.testing.assert_array_equal(getattr(pooled, name), getattr(reference, name))
    assert pooled.score95 == reference.score95 and pooled.ridge == reference.ridge
    values = [(*v55.signals_and_ages(model, reference, t), episode_ordinals(t['dones'])) for t in traces]
    signals, ages, ordinals = (np.concatenate([v[i] for v in values]) for i in range(3))
    for curve, mask in ((phase.initial, ordinals == 0), (phase.after_reset, ordinals > 0)):
        assert curve.metadata() == NormalAgeReference.fit_normal(signals[mask], ages[mask]).metadata()
        assert curve.ages[0] == 0 and curve.ages[-1] == 999
        assert curve.normal_start_count == 2
    assert pooled.calibration_transitions == 4000
    assert reference.metadata() == before


def test_actual_reset_switches_curve_without_clearing_departure_history():
    phase = PhaseAgeReference(NormalAgeReference([0, 999], [.5, .5], [1, 1], 2),
                              NormalAgeReference([0, 999], [.02, .02], [1, 1], 2))
    readout = PhaseDepartureReadout(phase)
    for age in range(20):
        readout.update(.6, age)
        assert not readout.after_reset and readout.departure.reference is phase.initial
    risk = readout.update(.03, 0)
    assert readout.after_reset and readout.departure.reference is phase.after_reset
    assert len(readout.departure.history) == 16
    assert risk == pytest.approx((15 * -np.expm1(-.1) - np.expm1(-.01)) / 16)
    readout.update(.03, 1)
    assert readout.after_reset


def test_previous_1200_step_calibration_cannot_silently_extrapolate_reset_curve(original):
    model, reference = original
    with pytest.raises(AssertionError, match='incomplete-normal-phase-age-coverage'):
        fit_references(model, reference, [trace(5, 1200), trace(6, 1200)])


def test_frozen_reference_progress_serializes(original, tmp_path):
    model, reference = original
    pooled, phase = fit_references(model, reference, [trace(5), trace(6)])
    progress = dict(stage='preflight', next_cell=2, calibration=[],
                    references=dict(matched_pooled=pooled, phase_calibrated=phase))
    path = tmp_path / 'progress.pt'
    torch.save(progress, path)
    restored = torch.load(path, map_location='cpu', weights_only=False)
    assert restored['references']['matched_pooled'].metadata() == pooled.metadata()
    assert restored['references']['phase_calibrated'].metadata() == phase.metadata()
    np.testing.assert_array_equal(restored['references']['matched_pooled'].precision, reference.precision)


def test_existing_gates_require_all_early_and_tail_pairs():
    windows = ('full', 'prefix', 'first_post50', 'post', 'last64')
    records = [dict(split='test', mode=mode,
        observers={name: {w: dict(mean_coefficient=.01) for w in windows} for name in worker.OBSERVERS},
        change_minus_nochange={name: {w: 0. if mode == 'normal' else .2 for w in windows}
                              for name in worker.OBSERVERS}) for mode in worker.v53.MODES]
    assert worker.summarize(records)['decision'] == 'phase-calibration-diagnostic-pass-not-RL-confirmation'
    records[-1]['change_minus_nochange']['phase_calibrated']['last64'] = 0.
    assert worker.summarize(records)['decision'] == 'hold-phase-calibration'
    records[-1]['change_minus_nochange']['phase_calibrated']['last64'] = .2
    records[0]['observers']['phase_calibrated']['last64']['mean_coefficient'] = .11
    assert worker.summarize(records)['decision'] == 'hold-phase-calibration'
