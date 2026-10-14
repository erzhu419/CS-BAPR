from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import analyze_general_ood_v96_mean_support as worker
from scripts import analyze_general_ood_v79_linear_support as support_worker
from scripts import analyze_general_ood_v83_cross_seed_covariance as source
from test_general_ood_v68 import trace
from test_general_ood_v78 import banks
from test_general_ood_v89 import sources
from test_general_ood_v91 import expanded
from test_general_ood_v94 import fitted


def test_selection_pairs_exact_completed_failure_windows():
    root = Path(__file__).resolve().parents[1]
    records = []
    for seed in worker.v95.SEEDS:
        records.extend(json.loads((root/f"results/general_ood_v95_fresh_seed/seed{seed}/fresh_seed.json").read_text())["records"])
    population = worker.select_failures(records[::-1])
    assert [(s, m, w) for s, m, w, _ in population] == list(worker.FAILURES)
    assert sum(2*(50 if w == "first_post50" else 64) for _, _, w, _ in population) == 956
    for seed, mode, _, pair in population:
        assert all(r["seed"] == seed and r["mode"] == mode and r["updating"] == u for u, r in pair.items())
    changed = deepcopy(records)
    bad = next(r for r in changed if r["seed"] == 135190 and r["mode"] == "slippery" and r["updating"])
    bad["observers"][worker.CANDIDATE]["first_post50"]["mean_true_minus_best_other_score"] = 1
    with pytest.raises(AssertionError, match="failure population changed"):
        worker.select_failures(changed)


def test_projection_sign_and_actual_zero_gap_rows():
    error = np.array([[2., 0.], [-1., 0.], [0., 0.]])
    gap = np.array([[1., 0.], [1., 0.], [0., 0.]])
    actual = worker.error_directions(error, dict(normal=gap, airborne=np.zeros_like(gap)))
    assert actual["normal"]["nonzero_gap_fraction"] == 2/3
    assert actual["normal"]["window_error_to_gap_rms_ratio"] == pytest.approx(np.sqrt(2.5))
    assert actual["normal"]["gap_energy_weighted_projection"] == .5
    assert actual["normal"]["window_cosine_toward_competitor"] == pytest.approx(1/np.sqrt(10))
    assert actual["airborne"] == dict(nonzero_gap_fraction=0., oracle_gap_rms=0., minimum_oracle_gap_rms=0.,
        window_error_to_gap_rms_ratio=None, gap_energy_weighted_projection=None, window_cosine_toward_competitor=None)
    json.dumps(actual, allow_nan=False)


def test_window_direction_does_not_amplify_roundoff_sized_airborne_gaps():
    actual = worker.error_directions(np.array([[1., 0.], [1., 0.]]),
                                    dict(normal=np.array([[1., 0.], [1e-16, 0.]])))["normal"]
    assert actual["gap_energy_weighted_projection"] == pytest.approx(1.)
    assert actual["window_error_to_gap_rms_ratio"] == pytest.approx(np.sqrt(2.))
    assert actual["window_cosine_toward_competitor"] == pytest.approx(1/np.sqrt(2.))


def test_diagnosis_reproduces_saved_scores_and_keeps_geometry(fitted):
    family = fitted[0]
    query = trace(992, "slippery", 3)
    oracle = np.repeat(query["next_states"][:, None, :], 5, axis=1)
    truth = worker.v94.MODES.index("slippery")
    for i in range(5):
        if i != truth:
            oracle[:, i] += .1*(i+1)
    query["oracle_observations"] = oracle
    terms, _ = family.terms(query)
    previous = dict(mode="slippery", attribution={}, observers={})
    for c in worker.v94.CONDITIONS:
        previous["attribution"][c] = dict(first_post50=source.v78.v77.attribution(query, terms[c], family.bank, "slippery", 0, 3))
        previous["observers"][c] = dict(first_post50=dict(mode_accuracy=.2, mean_true_minus_best_other_score=-1.))
    before = deepcopy({c: family.metadata(c) for c in worker.v94.CONDITIONS})
    actual = worker.diagnose(query, family, previous, "first_post50", source, support_worker)
    inputs = family.bank.features.inputs(query)
    direct = support_worker.local_prediction(family.mean, inputs)
    base_error = family.bank.targets(query)-family.bank.design(query)@family.bank.coefficients["slippery"]
    expected_constant_rms = np.sqrt(np.mean((base_error-direct["constant"]["slippery"])**2, axis=1)).mean()
    expected_global_rms = np.sqrt(np.mean(base_error**2, axis=1)).mean()
    for c in worker.v94.CONDITIONS:
        worker.v94.v90.verify_baseline(actual["conditions"][c]["linear"], previous["attribution"][c]["first_post50"])
        assert actual["conditions"][c]["constant"]["mean_true_prediction_error_rms"] == pytest.approx(expected_constant_rms)
        assert actual["conditions"][c]["global_forward"]["mean_true_prediction_error_rms"] == pytest.approx(expected_global_rms)
    assert sum(actual["neighbor_seed_fractions"].values()) == pytest.approx(1.)
    assert before == {c: family.metadata(c) for c in worker.v94.CONDITIONS}
    paired = dict(seed=1, mode="slippery", window="first_post50", frozen=actual, updating=deepcopy(actual))
    other = deepcopy(paired)
    other["updating"]["conditions"][worker.CANDIDATE]["linear"]["raw_mode_accuracy"] = 1.
    summary = worker.summarize([paired, other])
    assert summary["updating"]["conditions"][worker.CANDIDATE]["linear"]["raw_mode_accuracy"] == pytest.approx(
        (actual["conditions"][worker.CANDIDATE]["linear"]["raw_mode_accuracy"]+1)/2)
    json.dumps(dict(records=[paired], summary=summary), allow_nan=False)
