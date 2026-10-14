from copy import deepcopy
from itertools import product
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import analyze_general_ood_v102_failure_attribution as worker
from scripts import summarize_general_ood_v102 as summary
from scripts import analyze_general_ood_v83_cross_seed_covariance as source
from scripts.v73_recursive_mode_readout import temporal_scores
from test_general_ood_v68 import trace
from test_general_ood_v78 import banks
from test_general_ood_v89 import sources
from test_general_ood_v91 import expanded
from test_general_ood_v94 import fitted
from test_general_ood_v98 import expanded_coverage, fitted_coverage
from test_general_ood_v99 import query_and_previous


def test_actual_selection_keeps_all_three_new_failures_and_matched_controls():
    root = Path(__file__).resolve().parents[1]
    data = {"incumbent_suffix": json.loads((root/"results/general_ood_v98_regime_replay/replay_diverse_recipe/regime_replay.json").read_text()),
        "age_balanced": json.loads((root/"results/general_ood_v101_age_replay/replay/regime_replay.json").read_text())}
    population = worker.select_windows(data)
    assert len(population) == 12 and sum(50 if w == "first_post50" else 64 for _, w, _ in population) == 656
    assert sum(role == "new_failure_pair" and p["age_balanced"]["updating"] for role, _, p in population) == 3
    assert sum(role == "matched_normal" for role, _, _ in population) == 6
    assert len({worker.v99.record_key(p["age_balanced"]) for _, _, p in population}) == 10


def test_prefix_history_restores_full_filter_without_reset_or_future_input():
    ll = np.random.default_rng(102).normal(size=(30, 5))
    expected = temporal_scores(ll, "recursive")
    actual = worker.posterior_from_prefix(ll[10:20], expected[9])
    np.testing.assert_array_equal(actual, expected[10:20])
    changed = ll.copy(); changed[20:] += 1000
    np.testing.assert_array_equal(actual, worker.posterior_from_prefix(changed[10:20], temporal_scores(changed, "recursive")[9]))
    other_prefix = np.log(np.array([.96, .01, .01, .01, .01]))
    assert not np.allclose(actual, worker.posterior_from_prefix(ll[10:20], other_prefix))


def test_crossed_density_and_prefix_diagonals_restore_saved_windows(fitted_coverage):
    family = fitted_coverage[1]
    query, previous = query_and_previous(family)
    fixed, _, cov, _ = family.geometry(family.bank.features.inputs(query))
    centers = dict(incumbent_suffix=fixed, age_balanced=fixed+.07)
    covariances = dict(incumbent_suffix=cov, age_balanced={c: 2*v for c, v in cov.items()})
    families = {r: family for r in worker.RECIPES}
    prefixes, saved, cached = {}, {}, {}
    for i, recipe in enumerate(worker.RECIPES):
        terms = worker.v98.v94.terms_for_covariances(query, family.bank, centers[recipe], covariances[recipe])
        prefixes[recipe] = {c: np.log(np.array([.6, .1, .1, .1, .1]) if i else np.full(5, .2)) for c in worker.v98.v94.CONDITIONS}
        cached[recipe] = {c: terms[c]["likelihoods"] for c in worker.v98.v94.CONDITIONS}
        saved[recipe] = dict(mode="slippery", attribution={}, observers={})
        for c in worker.v98.v94.CONDITIONS:
            saved[recipe]["attribution"][c] = dict(first_post50=source.v78.v77.attribution(query, terms[c], family.bank, "slippery", 0, 3))
            saved[recipe]["observers"][c] = dict(first_post50=worker.posterior_statistics(
                worker.posterior_from_prefix(terms[c]["likelihoods"], prefixes[recipe][c]), "slippery"))
    instant, posterior = worker.crossed_scores(query, families, centers, covariances, prefixes, saved, cached, "first_post50", source)
    assert len(instant) == 4 and len(posterior) == 8
    c = worker.CANDIDATE
    a = instant["incumbent_suffix_mean__incumbent_suffix_covariance"][c]
    b = instant["incumbent_suffix_mean__age_balanced_covariance"][c]
    assert b["mean_true_prediction_error_rms"] == pytest.approx(a["mean_true_prediction_error_rms"])
    assert b["mean_energy_advantage"] == pytest.approx(a["mean_energy_advantage"]/2)
    altered = deepcopy(cached); altered["age_balanced"][c] = altered["age_balanced"][c]+1
    with pytest.raises(AssertionError):
        worker.crossed_scores(query, families, centers, covariances, prefixes, saved, altered, "first_post50", source)
    json.dumps(dict(instant=instant, posterior=posterior), allow_nan=False)


def test_three_factor_contributions_preserve_total_with_interaction():
    scores = {(h, m, c): 1+2*h+3*m+4*c+6*h*c for h, m, c in product((0, 1), repeat=3)}
    actual = summary.decompose_three(scores)
    assert actual["history_effect"] == pytest.approx(5.)
    assert actual["mean_effect"] == pytest.approx(3.)
    assert actual["covariance_effect"] == pytest.approx(7.)
    assert actual["total"] == pytest.approx(15.)
    assert sum(actual[k] for k in ("history_effect", "mean_effect", "covariance_effect")) == pytest.approx(actual["total"])
