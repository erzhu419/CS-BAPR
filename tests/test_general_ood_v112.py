from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from scripts import analyze_general_ood_v112_matched_errors as worker
from scripts import analyze_general_ood_v77_mean_covariance as window_source
from scripts.v83_cross_seed_covariance import LocalLinearMean, subset_bank, error_covariance
from test_general_ood_v58 import shared_start
from test_general_ood_v68 import trace
from test_general_ood_v78 import banks
from test_general_ood_v89 import sources
from test_general_ood_v91 import expanded
from test_general_ood_v94 import fitted
from test_general_ood_v98 import expanded_coverage, fitted_coverage
from test_general_ood_v106 import family


def test_error_support_is_fixed_but_mean_fit_uses_all_other_source_seeds(family, tmp_path, monkeypatch):
    errors, fits = worker.fit_errors(family, tmp_path)
    assert sum(f["error_rows"] for f in fits) == family.incumbent.mean.tree.n
    assert set(f["seed"] for f in fits) == set(family.groups)
    assert 135230 not in set(family.groups) and 135230 in set(family.mean_groups)
    for fit in fits:
        rows = np.flatnonzero(family.groups == fit["seed"])[[0, -1]]
        keep = family.mean_groups != fit["seed"]
        assert np.all(keep[family.mean_groups == 135230])
        inputs = family.mean.tree.data[rows]*family.mean.features.input_scale+family.mean.features.input_mean
        prediction = LocalLinearMean(subset_bank(family.mean, keep)).predict(inputs)
        changed = deepcopy(family.mean)
        changed.residuals = {m: a.copy() for m, a in changed.residuals.items()}
        for m in worker.MODES:
            np.testing.assert_allclose(errors[m][rows], family.mean.residuals[m][rows]-prediction[m])
            changed.residuals[m][~keep] += 100
        changed_errors, _ = worker.v98.heldout_errors(changed, family.mean_groups, fit["seed"], rows)
        for m in worker.MODES:
            np.testing.assert_allclose(changed.residuals[m][rows]-changed_errors[m], prediction[m], atol=1e-12)
    def unexpected_fit(*args):
        raise AssertionError("Completed fold should be reused")
    monkeypatch.setattr(worker.v98, "heldout_errors", unexpected_fit)
    resumed, resumed_fits = worker.fit_errors(family, tmp_path)
    assert resumed_fits == fits
    for m in worker.MODES:
        np.testing.assert_array_equal(errors[m], resumed[m])


def test_only_error_second_moment_changes_with_raw_and_old_disagreement_fixed(family):
    inputs = family.bank.features.inputs(trace(135252, "normal", 3))
    errors = {m: 1.5*a for m, a in family.incumbent.errors.items()}
    old, new, old_cov, matched = worker.geometry(family, errors, inputs)
    direct_new, members, direct_old_cov, _ = family.geometry(inputs)
    np.testing.assert_array_equal(new, direct_new)
    for c in worker.CONDITIONS:
        np.testing.assert_array_equal(old_cov[c], direct_old_cov[c])
    mean = family.incumbent.mean
    _, indices = mean.tree.query((inputs-mean.features.input_mean)/mean.features.input_scale, k=64)
    expected = np.stack([family.raw_weight*family.bank.geometry(inputs, m)[1]+(1-family.raw_weight)*
        np.asarray([error_covariance(e)[0] for e in errors[m][indices]]) for m in worker.MODES], axis=1)
    np.testing.assert_allclose(matched[worker.CONDITIONS[0]], expected)
    moment = worker.v98.v94.v93.displacement_second_moment(old, members)
    np.testing.assert_allclose(matched[worker.CONDITIONS[1]], expected+moment)
    np.testing.assert_allclose(np.trace(matched[worker.CONDITIONS[2]], axis1=-2, axis2=-1),
        np.trace(expected+moment, axis1=-2, axis2=-1))
    assert not np.allclose(matched[worker.CONDITIONS[0]], old_cov[worker.CONDITIONS[0]])


def test_identity_control_production_likelihoods_and_physical_gaussian(family):
    mode, seed = "compound", 135252
    query = trace(seed, mode, 3)
    query["oracle_observations"] = np.stack([trace(seed, m, 3)["next_states"] for m in worker.MODES], axis=1)
    diagnostic, terms = worker.diagnose(query, family, family.incumbent.errors, mode, identity=True)
    for c in worker.CONDITIONS:
        for a, model in (("old", family.incumbent), ("new", family)):
            direct, _ = model.terms(query)
            for k in direct[c]:
                np.testing.assert_allclose(terms[a][c][k], direct[c][k], rtol=1e-10, atol=1e-8)
        np.testing.assert_allclose(terms["matched"][c]["likelihoods"], terms["new"][c]["likelihoods"])
        for a in worker.ARMS:
            stats = diagnostic["densities"][c][a]
            assert stats["mean_energy_advantage"]+stats["mean_volume_advantage"] == pytest.approx(stats["mean_true_minus_best_other_log_density"])
        assert diagnostic["densities"][c]["oracle_matched_cov"]["true_prediction_error_rms"] == pytest.approx(0., abs=1e-12)
    errors = {m: 1.5*a for m, a in family.incumbent.errors.items()}
    _, terms = worker.diagnose(query, family, errors, mode)
    _, fixed, _, covariance = worker.geometry(family, errors, family.bank.features.inputs(query))
    design, target, scale = family.bank.design(query), family.bank.targets(query), family.bank.target_scale
    for c in worker.CONDITIONS:
        for i, m in enumerate(worker.MODES):
            error = target[0]-design[0]@family.bank.coefficients[m]-fixed[0, i]
            physical = covariance[c][0, i]*scale[:, None]*scale[None, :]
            expected = multivariate_normal.logpdf(error*scale, cov=physical)
            assert terms["matched"][c]["likelihoods"][0, i] == pytest.approx(expected, abs=1e-8)


def test_smoke_has_no_partial_candidate_errors(family, tmp_path):
    errors, fits = worker.fit_errors(family, tmp_path, smoke=True)
    assert errors is family.incumbent.errors
    assert sum(f["error_rows"] for f in fits) == 2*len(fits)
    assert not list(tmp_path.glob("*.pt"))


def test_historical_trace_without_oracles_keeps_real_candidate_controls(family):
    query = trace(134020, "slippery", 3)
    diagnostic, terms = worker.diagnose(query, family, family.incumbent.errors, "slippery", identity=True)
    assert set(terms) == {"old", "new", "matched"}
    record = dict(origin="v98", group="v62_development_replay", updating=True, mode="slippery",
                  old_coverage_failure=False, diagnostics=diagnostic)
    result = worker.summarize([record])["historical_slippery"]
    for c in worker.CONDITIONS:
        assert set(result["conditions"][c]) == {"old", "new", "matched"}
        assert result["conditions"][c]["matched"]["windows"] == 1


def test_population_includes_current_and_earlier_failures_with_controls():
    root = Path(__file__).resolve().parents[1]/"results"
    old = json.loads((root/"general_ood_v98_regime_replay/replay_diverse_recipe/regime_replay.json").read_text())["records"]
    fresh = [r for s in worker.v111.v110.SEEDS for r in json.loads(
        (root/f"general_ood_v110_fresh_mixture/seed{s}/fresh_seed.json").read_text())["records"]]
    ranges = {w: (a, b) for w, a, b in window_source.windows()}
    for smoke, expected in ((False, (76, 4248)), (True, (6, 328))):
        population = worker.population(old, fresh, smoke)
        assert (len(population), sum(ranges[w][1]-ranges[w][0] for _, _, w in population)) == expected
