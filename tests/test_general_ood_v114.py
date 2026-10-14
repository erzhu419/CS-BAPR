from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from scripts import analyze_general_ood_v114_classifier_attribution as worker
from test_general_ood_v113 import family, shared_start, trace, banks, sources, expanded, fitted, expanded_coverage, fitted_coverage


def frozen_classifier():
    rng = np.random.RandomState(14)
    scaler = StandardScaler().fit(rng.normal(size=(100, 110))*3+4)
    model = LogisticRegression()
    model.classes_ = np.arange(5)
    model.coef_, model.intercept_ = rng.normal(size=(5, 110)), rng.normal(size=5)
    return scaler, model


def test_decomposition_reconstructs_score_with_varying_competitors_without_mutation():
    fitted = frozen_classifier()
    before = deepcopy(fitted)
    rng = np.random.RandomState(15)
    observed, predicted = rng.normal(size=(40, 11)), rng.normal(size=(40, 5, 11))
    actual = worker.center_diagnostics(fitted, observed, predicted, "compound")
    worker.check_original(actual, dict(candidates={a: worker.v113.statistics(p, 4)
        for a, p in worker.v113.scores(fitted, observed, predicted).items()}))
    for comparison in ("best_other", "versus_normal"):
        parts = actual[comparison]
        assert parts["mean_margin"] == pytest.approx(parts["intercept"]+parts["linear_residual"]+parts["squared_residual"])
        assert parts["linear_residual"] == pytest.approx(sum(parts["linear_by_reference_mode"].values()))
        assert parts["squared_residual"] == pytest.approx(sum(parts["squared_by_reference_mode"].values()))
    assert sum(v > 0 for v in actual["best_other"]["competitor_fractions"].values()) > 1
    for obj, previous, fields in ((fitted[0], before[0], ("mean_", "scale_")),
                                  (fitted[1], before[1], ("coef_", "intercept_"))):
        for field in fields:
            np.testing.assert_array_equal(getattr(obj, field), getattr(previous, field))


def test_oracle_uses_complete_normalized_predictions_and_checks_actual_transition(family):
    query = trace(135252, "compound", 8)
    query["oracle_observations"] = np.stack([trace(135252, m, 8)["next_states"] for m in worker.MODES], axis=1)
    predicted = worker.oracle_predictions(query, family.bank, "compound")
    direct = np.stack([family.bank.targets(dict(query, next_states=query["oracle_observations"][:, i])) for i in range(5)], axis=1)
    np.testing.assert_allclose(predicted, direct, atol=1e-12)
    actual = worker.center_diagnostics(frozen_classifier(), family.bank.targets(query), predicted, "compound")
    assert actual["true_prediction_error_rms"] == pytest.approx(0., abs=1e-12)
    assert actual["candidates"]["nearest_mean"]["instantaneous_mode_accuracy"] == 1.
    query["oracle_observations"][0, 4, 0] += 1
    with pytest.raises(AssertionError):
        worker.oracle_predictions(query, family.bank, "compound")


def test_full_population_and_smoke_retain_old_failures_and_frozen_regression():
    root = Path(__file__).resolve().parents[1]
    baseline = json.loads((root/"results/general_ood_v113_discriminative_evidence/formal/discriminative_evidence.json").read_text())
    rows = baseline["records"]
    selected = [r for r in rows if worker.window_key(r) in worker.SMOKE_WINDOWS]
    assert len(rows) == 76 and sum(r["transitions"] for r in rows) == 4248
    assert len(selected) == 7 and sum(r["transitions"] for r in selected) == 364
    populations = worker.cohorts(rows)
    assert len(populations["current_failed_updating"]) == 4
    assert len(populations["current_normal"]) == 6
    assert len(populations["all_frozen_changed"]) == 28
    assert len(populations["earlier_six_failures"]) == 6
    assert (135191, "slippery", False, "first_post50") in map(worker.window_key, selected)
    assert (135192, "high_grav", True, "first_post50") in map(worker.window_key, selected)


def test_summary_uses_paired_oracle_denominator_not_real_only_controls():
    fitted = worker.center_diagnostics(frozen_classifier(), np.zeros((3, 11)), np.ones((3, 5, 11)), "slippery")
    oracle = deepcopy(fitted)
    oracle["candidates"]["quadratic_logistic"]["instantaneous_mode_accuracy"] = 1.
    base = dict(mode="slippery", origin="v98", updating=True, old_coverage_failure=False, fitted=fitted)
    summary = worker.summarize([dict(base, oracle=oracle), dict(base, oracle=None)])
    result = summary["all_updating_changed"]
    assert result["fitted"]["windows"] == 2 and result["oracle"]["windows"] == 1
    assert result["paired_oracle_minus_fitted"]["windows"] == 1
    expected = 1.-fitted["candidates"]["quadratic_logistic"]["instantaneous_mode_accuracy"]
    assert result["paired_oracle_minus_fitted"]["candidates"]["quadratic_logistic"]["instantaneous_mode_accuracy"] == expected
