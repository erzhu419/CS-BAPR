from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from scripts import analyze_general_ood_v90_coverage_replay as worker
from scripts import analyze_general_ood_v83_cross_seed_covariance as source
from scripts.v83_cross_seed_covariance import CrossSeedDensityBank, cross_seed_errors
from scripts.v78_local_linear_mean import LocalLinearMean
from test_general_ood_v68 import trace
from test_general_ood_v78 import banks
from test_general_ood_v89 import sources


def test_expanded_means_keep_original_covariance_bank_and_physical_densities(banks, sources):
    bank = banks['updating']
    expanded, _ = worker.v89.expand_banks(bank, np.repeat([135050, *worker.v89.SEEDS], 80), sources)
    errors = cross_seed_errors(bank, np.repeat([1, 2, 3], 80))
    densities = worker.make_densities(bank, expanded, errors, .24619454443573643)
    query = trace(100, 'slippery', 4)
    inputs = bank.features.inputs(query)
    baseline = CrossSeedDensityBank(bank, errors, .24619454443573643)
    _, covariance, _ = baseline.geometry(inputs)
    for condition, density in densities.items():
        assert density.bank is bank and density.errors is errors
        assert density.correction.bank is expanded[condition]
        centers, actual_covariance, _ = density.geometry(inputs)
        direct = LocalLinearMean(expanded[condition]).predict(inputs)
        terms = density.terms(query)[0]
        np.testing.assert_array_equal(terms['logdet'], baseline.terms(query)[0]['logdet'])
        for c, mode in enumerate(source.MODES):
            np.testing.assert_array_equal(centers[mode], direct[mode])
            np.testing.assert_array_equal(actual_covariance[mode], covariance[mode])
            offsets = (bank.targets(query)-bank.design(query)@bank.coefficients[mode]-direct[mode])*bank.target_scale
            for row in range(4):
                physical = covariance[mode][row]*bank.target_scale[:, None]*bank.target_scale[None, :]
                assert terms['likelihoods'][row, c] == pytest.approx(multivariate_normal.logpdf(offsets[row], cov=physical), abs=1e-8)


def test_volume_check_uses_fixed_pairs_not_changed_best_other_selection():
    current = {'first_post50': dict(mean_covariance_volume_advantage=100.,
        fixed_competitor={'heavy': dict(mean_covariance_volume_advantage=1.)})}
    expected = deepcopy(current)
    expected['first_post50']['mean_covariance_volume_advantage'] = -100.
    worker.verify_frozen_volume(current, expected)
    expected['first_post50']['fixed_competitor']['heavy']['mean_covariance_volume_advantage'] += .1
    with pytest.raises(AssertionError):
        worker.verify_frozen_volume(current, expected)


def test_new_metadata_reports_changed_mean_and_frozen_covariance(banks, sources):
    bank = banks['updating']
    expanded, _ = worker.v89.expand_banks(bank, np.repeat([135050, *worker.v89.SEEDS], 80), sources)
    density = worker.make_densities(bank, expanded, bank.residuals, .24619454443573643)[worker.CONDITIONS[0]]
    ref = SimpleNamespace(bank=density, metadata=lambda: dict(bank=density.metadata(), phase={}))
    metadata = worker.reference_metadata(ref)['bank']
    assert metadata['conditional_mean_changed'] and not metadata['conditional_covariance_changed']
    assert metadata['mean_fitting_rows'] == bank.tree.n+2200
    assert metadata['covariance_reference_rows'] == bank.tree.n


def test_resume_reuses_complete_prefix_and_rejects_different_weights(tmp_path):
    cell = dict(group='v76_fresh_updating', seed=135180, mode='normal', updating=True)
    saved = dict(revision=worker.REVISION, spec=source.v76.SPEC, raw_weight=.24619454443573643,
        conditions=list(worker.CONDITIONS), records=[source.v78.replay_cell(cell)])
    path = tmp_path/'progress.json'
    path.write_text(json.dumps(saved))
    args = SimpleNamespace(resume_from=path)
    densities = {c: SimpleNamespace(raw_weight=saved['raw_weight']) for c in worker.CONDITIONS}
    assert worker.evaluate(args, [(cell, None)], {}, densities, {}, source) == saved['records']
    densities[worker.CONDITIONS[0]].raw_weight = .5
    with pytest.raises(AssertionError):
        worker.evaluate(args, [(cell, None)], {}, densities, {}, source)


def test_same_null_population_is_used_for_both_conditions_and_cached(tmp_path, monkeypatch):
    import scripts.v73_recursive_mode_readout as readout

    args = SimpleNamespace(checkpoint_dir=tmp_path, v69_dir=tmp_path)
    nulls = [dict(states=np.zeros((6000, 11))) for _ in range(2)]
    loaded, calibrated = [], []

    def load(path):
        loaded.append(path)
        return nulls[len(loaded)-1]

    def calibrate(cls, density, population):
        assert population[0] is nulls[0] and population[1] is nulls[1]
        calibrated.append(density)
        return SimpleNamespace(phase={'density': density.name}, normal_null_calibration_transitions=12000)

    monkeypatch.setattr(source.v76.v68, 'load_trace', load)
    monkeypatch.setattr(readout.RecursiveModeReference, 'calibrate', classmethod(calibrate))
    densities = {c: SimpleNamespace(name=c, raw_weight=.24619454443573643) for c in worker.CONDITIONS}
    result = worker.freeze_references(args, densities, source)
    assert len(loaded) == len(calibrated) == 2 and set(result) == set(worker.CONDITIONS)
    result = worker.freeze_references(args, densities, source)
    assert len(loaded) == len(calibrated) == 2
    assert all(r.normal_null_calibration_transitions == 12000 for r in result.values())


def test_native_smoke_checks_both_means_and_saved_baseline(banks, sources):
    bank = banks['updating']
    expanded, _ = worker.v89.expand_banks(bank, np.repeat([135050, *worker.v89.SEEDS], 80), sources)
    errors = cross_seed_errors(bank, np.repeat([1, 2, 3], 80))
    densities = worker.make_densities(bank, expanded, errors, .24619454443573643)
    cells = [dict(group=g, seed=135180 if g=='v76_fresh_updating' else 1, mode=m, updating=True)
        for g,m in [('v76_fresh_updating','compound'),('v61_development_replay','slippery'),('v61_development_replay','heavy')]]
    trajectories = [trace(101+i, c['mode'], 128) for i,c in enumerate(cells)]
    baseline = CrossSeedDensityBank(bank, errors, .24619454443573643)
    scores = {}
    windows = lambda: (('first_post50',0,50),('last64',64,128))
    for cell,t in zip(cells,trajectories):
        terms = baseline.terms(t)[0]
        scores[source.v78.v77.cell_key(source.v78.replay_cell(cell))] = dict(attribution={source.MIX_CONDITION:
            {w: source.v78.v77.attribution(t,terms,bank,cell['mode'],a,b) for w,a,b in windows()}})
    from unittest.mock import patch
    with patch.object(source.v78.v77,'windows',windows), patch.object(source.v76.v68,'load_trace',side_effect=trajectories):
        result = worker.native_smoke([(c,None) for c in cells],bank,expanded,densities,errors,.24619454443573643,scores,source)
    assert result['passed'] and result['checked_transitions']==342 and result['gaussian_checks']==300
