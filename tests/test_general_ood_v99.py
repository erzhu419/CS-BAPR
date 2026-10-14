from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import analyze_general_ood_v99_coverage_attribution as worker
from scripts import analyze_general_ood_v83_cross_seed_covariance as source
from scripts import analyze_general_ood_v79_linear_support as support_worker
from scripts import summarize_general_ood_v98 as summarizer
from test_general_ood_v68 import trace
from test_general_ood_v78 import banks
from test_general_ood_v89 import sources
from test_general_ood_v91 import expanded
from test_general_ood_v94 import fitted
from test_general_ood_v98 import expanded_coverage, fitted_coverage


def test_summary_and_selection_keep_failures_repairs_and_recipe_tradeoff():
    root = Path(__file__).resolve().parents[1]/"results/general_ood_v98_regime_replay"
    data = {r: json.loads((root/f"replay_{r}/regime_replay.json").read_text()) for r in worker.RECIPES}
    summary = summarizer.summarize(data)
    assert summary["populations"]["all"]["diverse_recipe"][worker.CANDIDATE]["changed_windows"] == 144
    population = worker.select_windows(data)
    assert len(population) == 20
    assert sum(50 if w == "first_post50" else 64 for _, w, _ in population) == 1084
    assert sum(role == "original_v95_failure_pair" for role, _, _ in population) == 18
    assert population[-2][0] == "matched_recipe_tradeoff" and population[-1][0] == "matched_tradeoff_normal"
    repaired = [p for _, w, p in population if w == "first_post50" and p["diverse_recipe"]["updating"]
                and p["diverse_recipe"]["seed"] == 135190 and p["diverse_recipe"]["mode"] in ("slippery", "high_grav")]
    assert len(repaired) == 2
    assert all(p["diverse_recipe"]["observers"][worker.CANDIDATE]["first_post50"]["mean_true_minus_best_other_score"] > 0 for p in repaired)
    altered = deepcopy(data)
    altered["diverse_recipe"]["records"][0]["observers"]["v94_unchanged__"+worker.CANDIDATE]["full"]["mode_accuracy"] = -1
    with pytest.raises(AssertionError):
        summarizer.summarize(altered)


def test_neighbor_partition_distinguishes_extension_from_old_suffix():
    result = worker.neighbor_partition(np.array([[0, 3, 4, 6, 7, 8]]), 4, 8)
    assert result == dict(original_raw=2/6, v97_extension=3/6, v94_suffix=1/6)
    assert sum(result.values()) == pytest.approx(1.)


def query_and_previous(family, oracle_available=True):
    query = trace(135221, "slippery", 3)
    oracle = np.repeat(query["next_states"][:, None, :], 5, axis=1)
    truth = worker.v98.v94.MODES.index("slippery")
    for i in range(5):
        if i != truth:
            oracle[:, i] += .1*(i+1)
    if oracle_available:
        query["oracle_observations"] = oracle
    terms, _ = family.terms(query)
    previous = dict(mode="slippery", attribution={c: dict(first_post50=source.v78.v77.attribution(
        query, terms[c], family.bank, "slippery", 0, 3)) for c in worker.v98.v94.CONDITIONS})
    return query, previous


@pytest.mark.parametrize("oracle_available", [True, False])
def test_support_reproduces_saved_prediction_rms_and_source_partition(fitted_coverage, expanded_coverage, oracle_available):
    family = fitted_coverage[1]
    query, previous = query_and_previous(family, oracle_available)
    actual, fixed = worker.support(query, family.bank, family.mean, family.groups,
        expanded_coverage[0].bank.tree.n, "slippery", previous, "first_post50", support_worker)
    expected = family.geometry(family.bank.features.inputs(query))[0]
    np.testing.assert_allclose(fixed, expected, rtol=1e-10, atol=1e-10)
    assert sum(actual["neighbor_partition"].values()) == pytest.approx(1.)
    assert sum(actual["neighbor_seed_fractions"].values()) == pytest.approx(1.)
    assert actual["oracle_observations_available"] == oracle_available
    assert (actual["error_directions"] is not None) == oracle_available
    json.dumps(actual, allow_nan=False)


@pytest.mark.parametrize("oracle_available", [True, False])
def test_crossed_controls_preserve_mean_and_swap_covariance(fitted_coverage, oracle_available):
    family = fitted_coverage[1]
    query, previous = query_and_previous(family, oracle_available)
    fixed, _, covariance, _ = family.geometry(family.bank.features.inputs(query))
    centers = dict(old_recipe=fixed, diverse_recipe=fixed+.07)
    covariances = dict(old_recipe=covariance, diverse_recipe={c: 2*v for c, v in covariance.items()})
    families = {r: family for r in worker.RECIPES}
    saved = {}
    for r in worker.RECIPES:
        terms = worker.v98.v94.terms_for_covariances(query, family.bank, centers[r], covariances[r])
        saved[r] = dict(mode="slippery", attribution={c: dict(first_post50=source.v78.v77.attribution(
            query, terms[c], family.bank, "slippery", 0, 3)) for c in worker.v98.v94.CONDITIONS})
    actual = worker.crossed_scores(query, families, centers, covariances, saved, "first_post50", source)
    for c in worker.v98.v94.CONDITIONS:
        old = actual["old_recipe_mean__old_recipe_covariance"][c]
        crossed = actual["old_recipe_mean__diverse_recipe_covariance"][c]
        assert crossed["mean_true_prediction_error_rms"] == pytest.approx(old["mean_true_prediction_error_rms"])
        assert crossed["mean_energy_advantage"] == pytest.approx(old["mean_energy_advantage"]/2)
        assert crossed["mean_covariance_volume_advantage"] == pytest.approx(old["mean_covariance_volume_advantage"])
        assert crossed["observation_template_gap_available"] == oracle_available
    json.dumps(actual, allow_nan=False)
