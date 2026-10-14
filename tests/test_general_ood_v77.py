from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import multivariate_normal
import torch

from scripts import analyze_general_ood_v77_mean_covariance as worker
from scripts.v68_mode_likelihood import ModeLikelihoodBank
from scripts.v72_conditional_likelihood import ConditionalDensityBank
from scripts.v76_behavior_coverage import BehaviorCoverageBank
from scripts.v77_mean_covariance import COMBINATIONS, MODES, MeanCovarianceBank, check_shared_forward, factorial_terms
from test_general_ood_v68 import trace


@pytest.fixture(scope="module")
def banks():
    torch.set_num_threads(1)
    fitting, calibration = [{m: trace(s, m, 160) for m in MODES} for s in (3, 4)]
    base = ConditionalDensityBank.fit_density(ModeLikelihoodBank.fit(fitting, calibration), calibration)
    extra = {m: trace(5, m, 80) for m in MODES}
    return {"frozen": base, "updating": BehaviorCoverageBank.extend(base, extra, "updating_mixed")}


def test_shared_forward_rejects_changed_coefficients(banks):
    check_shared_forward(banks)
    changed = deepcopy(banks)
    changed['updating'].coefficients['normal'][0, 0] += .1
    with pytest.raises(AssertionError):
        check_shared_forward(changed)


def test_factorial_densities_match_physical_gaussians_and_saved_diagonals(banks):
    query = trace(8, 'heavy', 8)
    output = factorial_terms(query, banks)
    for condition, (mean_source, cov_source) in COMBINATIONS.items():
        np.testing.assert_allclose(output[condition]['likelihoods'],
            MeanCovarianceBank(banks, condition).log_likelihoods(query), rtol=1e-12, atol=1e-9)
        for i, mode in enumerate(MODES):
            mean_bank, cov_bank = banks[mean_source], banks[cov_source]
            center = mean_bank.local_centers(mean_bank.features.inputs(query))[mode]
            covariance = cov_bank.geometry(cov_bank.features.inputs(query), mode)[1]
            prediction = mean_bank.design(query)@mean_bank.coefficients[mode]+center
            error = (mean_bank.targets(query)-prediction)*mean_bank.target_scale
            for row in range(len(error)):
                physical_cov = covariance[row]*mean_bank.target_scale[:, None]*mean_bank.target_scale[None, :]
                expected = multivariate_normal.logpdf(error[row], cov=physical_cov)
                assert output[condition]['likelihoods'][row, i] == pytest.approx(expected, abs=1e-8)
        if mean_source == cov_source:
            np.testing.assert_allclose(output[condition]['likelihoods'], banks[mean_source].log_likelihoods(query), atol=1e-9)


def test_attribution_reconstructs_fixed_and_best_competitor_margins(banks):
    query = trace(8, 'heavy', 8)
    query['oracle_observations'] = np.stack([query['next_states']+i*.03 for i in range(5)], axis=1)
    query['oracle_observations'][:, MODES.index('heavy')] = query['next_states']
    terms = factorial_terms(query, banks)
    for condition in COMBINATIONS:
        result = worker.attribution(query, terms[condition], banks['frozen'], 'heavy', 0, 8)
        assert result['mean_energy_advantage']+result['mean_covariance_volume_advantage'] == pytest.approx(result['mean_true_minus_best_other_log_density'])
        for pair in result['fixed_competitor'].values():
            assert pair['mean_energy_advantage']+pair['mean_covariance_volume_advantage'] == pytest.approx(pair['mean_log_density_margin'])
        assert sum(result['best_other_mode_fractions'].values()) == pytest.approx(1)
        assert result['best_other_mode_fractions']['heavy'] == 0


def test_protocol_is_offline_and_keeps_all_development_paths():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root/'experiments/general_ood_v77_mean_covariance_20261009/dispatch_manifest.json').read_text())
    assert manifest['evaluation_paths'] == 65
    assert manifest['conditions'] == {k: list(v) for k, v in COMBINATIONS.items()}
    assert manifest['environment_steps'] == manifest['optimizer_updates'] == 0
    assert not manifest['temporal_rules_changed'] and not manifest['readout_controls_actor']
    assert manifest['resource_request']['vram_mb'] == 0


def test_historical_traces_keep_density_attribution_without_oracle_templates(banks):
    query = trace(8, 'heavy', 8)
    terms = factorial_terms(query, banks)
    result = worker.attribution(query, terms['frozen_mean_frozen_cov'], banks['frozen'], 'heavy', 0, 8)
    assert not result['observation_template_gap_available']
    assert result['mean_nearest_other_observation_template_gap'] is None
    assert result['prediction_error_exceeds_template_gap_fraction'] is None
    assert np.isfinite(result['mean_true_minus_best_other_log_density'])
    assert len(result['fixed_competitor']) == 4


def test_scalar_resume_keeps_completed_prefix(tmp_path):
    from types import SimpleNamespace
    cell = dict(group='v76_fresh_updating', seed=135180, mode='normal', updating=True)
    record = dict(cell, group='v76_development_replay')
    progress = tmp_path/'progress.json'
    progress.write_text(json.dumps(dict(revision=worker.REVISION, spec=worker.v76.SPEC, records=[record])))
    args = SimpleNamespace(resume_from=progress)
    assert worker.evaluate(args, [(cell, Path('/unused'))], {}, {}, {}) == [record]
    changed = dict(cell, seed=135181)
    with pytest.raises(AssertionError):
        worker.evaluate(args, [(changed, Path('/unused'))], {}, {}, {})
