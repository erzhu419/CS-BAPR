from copy import deepcopy

import numpy as np
import pytest
import torch

from csbapr.belief.tracker import BeliefTracker
from scripts.v44_belief import AdaptiveStudent, SETTINGS, make_filters, replay


def test_predictive_density_matches_student_distribution():
    tracker = AdaptiveStudent(**SETTINGS)
    for history in ([], [.2, .4, .8], [5.] * 19):
        tracker.reset()
        tracker.history.extend(history)
        location, scale2, df = tracker.predictive_parameters()
        oracle = torch.distributions.StudentT(torch.tensor(df), torch.tensor(location), torch.tensor(np.sqrt(scale2)))
        np.testing.assert_allclose(tracker._compute_log_likelihood(2.7), oracle.log_prob(torch.tensor(2.7, dtype=torch.float64)).numpy())
    tracker.reset()
    _, scale2, df = tracker.predictive_parameters()
    np.testing.assert_allclose(scale2 * df / (df - 2), SETTINGS["base_variance"])


def test_known_history_conjugate_parameters_and_causal_update():
    tracker = AdaptiveStudent(**SETTINGS)
    tracker.history.extend([2., 4.])
    loc, scale2, df = tracker.predictive_parameters()
    np.testing.assert_allclose(loc[:3], [0., 2., 2.])
    np.testing.assert_allclose(df[:3], [4., 5., 6.])
    # n=2: beta=.5 + SSE/2 + n*mean^2/(2*kappa)=.5+1+3=4.5.
    assert scale2[2] == pytest.approx(2.)
    prior = np.zeros(20)
    prior[0] = .02
    prior[1:] = .98 * tracker.belief[:-1]
    prior[-1] += .98 * tracker.belief[-1]
    logw = np.log(prior) + tracker._compute_log_likelihood(8.)
    expected = np.exp(logw - logw.max())
    expected /= expected.sum()
    tracker.update(8.)
    np.testing.assert_allclose(tracker.belief, expected)
    assert list(tracker.history) == [2., 4., 8.]
    tracker.reset()
    assert not tracker.history
    np.testing.assert_allclose(tracker.belief, np.full(20, .05))


def test_half_normal_control_is_unchanged_and_cap_is_finite():
    filters = make_filters()
    original = BeliefTracker(**SETTINGS)
    for value in [.1] * 25 + [14.] * 100:
        original.update(value)
        for tracker in filters.values():
            tracker.update(value)
            assert np.isfinite(tracker.belief).all()
            assert tracker.belief.sum() == pytest.approx(1.)
        np.testing.assert_array_equal(filters["half-normal-fixed"].belief, original.belief)
    assert len(filters["student-adaptive"].history) == 19


def test_persistent_shift_can_recover_without_erasing_signal():
    scores = np.expm1(np.r_[np.full(100, .2), np.full(100, 5.)])
    signal, traces = replay(scores, np.ones(200))
    assert signal[-1] == pytest.approx(5.)
    adaptive = traces["student-adaptive"]
    assert adaptive[100, 0] > .5
    assert adaptive[-1, 1] > 17.
    assert adaptive[-1, 1] > traces["half-normal-fixed"][-1, 1] + 10.
    _, changed_future = replay(np.r_[scores[:120], np.zeros(80)], np.ones(200))
    for name in traces:
        np.testing.assert_array_equal(traces[name][:120], changed_future[name][:120])


def test_summary_rejects_source_score_drift(tmp_path):
    from scripts.summarize_general_ood_v44 import validate_run
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    destination = root / "results/general_ood_v44_frozen_belief/preflight"
    if not destination.exists():
        pytest.skip("worker preflight fixture not available yet")
    data = json.loads((destination / "general_ood_v44_frozen_belief_fixed-minus2.json").read_text())
    manifest = json.loads((root / "experiments/general_ood_v44_frozen_belief_20261006/dispatch_manifest.json").read_text())
    validate_run(data, manifest, "preflight", root)
    altered = deepcopy(data)
    altered["records"][0]["v43_score_check"]["first_post16"]["mean_score"] += 1
    with pytest.raises(AssertionError, match="v43-score-changed"):
        validate_run(altered, manifest, "preflight", root)
