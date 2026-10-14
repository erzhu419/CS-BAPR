from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import analyze_general_ood_v100_reference_age as worker
from scripts import analyze_general_ood_v79_linear_support as support_worker
from scripts import summarize_general_ood_v99 as summary_worker
from scripts import summarize_general_ood_v100 as age_summary
from scripts.v83_cross_seed_covariance import LocalLinearMean
from test_general_ood_v68 import trace
from test_general_ood_v78 import banks
from test_general_ood_v89 import sources
from test_general_ood_v91 import expanded
from test_general_ood_v94 import fitted
from test_general_ood_v99 import query_and_previous


def collection():
    output = {}
    for seed in worker.v98.v97.SEEDS:
        oracle = np.stack([trace(seed, m, 64)["next_states"] for m in worker.v98.v94.MODES], axis=1)
        output[seed] = {m: dict(trace(seed, m, 64), oracle_observations=oracle.copy())
                        for m in ("normal", "high_grav", "compound")}
    return output


def test_equal_budgets_and_all_five_targets_keep_exact_phase_selection():
    traces = collection()
    spec = dict(steps=64, switch_step=32, stride=2)
    payloads, groups, sources = worker.extensions(traces, spec)
    for recipe in worker.RECIPES:
        assert payloads[recipe]["normal"]["states"].shape == (64, 11)
        np.testing.assert_array_equal(groups[recipe], np.repeat(worker.v98.v97.SEEDS, 32))
        assert sum(s["rows"] for s in sources[recipe]) == 64
    seed = worker.v98.v97.SEEDS[0]
    for i, m in enumerate(worker.v98.v94.MODES):
        np.testing.assert_array_equal(payloads["late_prefix"][m]["next_states"][:32], traces[seed]["normal"]["oracle_observations"][:32, i])
        np.testing.assert_array_equal(payloads["age_balanced"][m]["next_states"][:16], traces[seed]["normal"]["oracle_observations"][:32:2, i])
        np.testing.assert_array_equal(payloads["age_balanced"][m]["next_states"][16:24], traces[seed]["high_grav"]["oracle_observations"][32:64:4, i])
        np.testing.assert_array_equal(payloads["age_balanced"][m]["next_states"][24:32], traces[seed]["compound"]["oracle_observations"][32:64:4, i])
        np.testing.assert_array_equal(payloads["diverse_suffix"][m]["next_states"][:16], traces[seed]["high_grav"]["oracle_observations"][32:64:2, i])
    altered = deepcopy(traces)
    altered[seed]["normal"]["oracle_observations"][0, 0, 0] += 1
    with pytest.raises(AssertionError):
        worker.extensions(altered, spec)


@pytest.mark.parametrize("oracle_available", [True, False])
def test_mean_support_keeps_direct_prediction_and_optional_oracle(fitted, oracle_available):
    original = fitted[0]
    payloads, groups, _ = worker.extensions(collection(), dict(steps=64, switch_step=32, stride=2))
    raw, mean, labels = worker.v98.expand(original, payloads["age_balanced"], groups["age_balanced"], "age_balanced")
    query, _ = query_and_previous(original, oracle_available)
    actual = worker.evaluate_window(query, raw, mean, labels, original.bank.tree.n, "slippery", support_worker)
    prediction = LocalLinearMean(mean).predict(raw.features.inputs(query))["slippery"]
    error = raw.design(query)@raw.coefficients["slippery"]+prediction-raw.targets(query)
    assert actual["prediction_error_rms"] == pytest.approx(np.sqrt(np.mean(error**2, axis=1)).mean())
    assert sum(actual["neighbor_partition"].values()) == pytest.approx(1.)
    assert sum(actual["neighbor_seed_fractions"].values()) == pytest.approx(1.)
    assert actual["oracle_observations_available"] == oracle_available
    assert (actual["nearest_oracle_gap_rms"] is not None) == oracle_available


def test_symmetric_score_decomposition_preserves_total_and_interaction():
    result = summary_worker.decompose([1., 2., 4., 8.])
    assert result["mean_effect"] == 2.5 and result["covariance_effect"] == 4.5
    assert result["interaction"] == 3.
    assert result["mean_effect"]+result["covariance_effect"] == result["total"] == 7.


def test_actual_replay_cohort_excludes_other_v62_seeds_and_frozen_controls():
    path = Path(__file__).resolve().parents[1]/"results/general_ood_v98_regime_replay/replay_diverse_recipe/regime_replay.json"
    data = json.loads(path.read_text())
    population = worker.evaluation_windows(data["records"], (("first_post50", 3800, 3850), ("last64", 5936, 6000)))
    assert len(population) == 62 and sum(b-a for _, _, a, b in population) == 3548
    controls = [(c["seed"], c["mode"], c["updating"], w) for c, w, _, _ in population
                if c["group"] == "v62_development_replay"]
    assert controls == [(134020, "normal", True, "last64"), (134020, "slippery", True, "last64")]
    assert sum(c["group"].startswith("v95_development_") for c, _, _, _ in population) == 60


def test_age_summary_uses_equal_windows_and_keeps_regressions():
    rows = []
    for transitions, baseline, candidate in ((50, 1., 2.), (64, 3., 1.)):
        rows.append(dict(transitions=transitions, diagnostics={recipe: dict(
            prediction_error_rms=baseline if recipe == "diverse_suffix" else candidate,
            neighbor_partition={"v97_extension": 0.25},
            support={"nearest_radius": {"mean": 0.5}}) for recipe in age_summary.RECIPES}))
    result = age_summary.aggregate(rows)["recipes"]
    assert result["diverse_suffix"]["prediction_error_rms"] == 2.
    assert result["age_balanced"]["prediction_error_rms"] == 1.5
    assert result["age_balanced"]["mean_error_change"] == -0.5
    assert result["age_balanced"]["improved_windows"] == 1
    assert result["age_balanced"]["worsened_windows"] == 1
