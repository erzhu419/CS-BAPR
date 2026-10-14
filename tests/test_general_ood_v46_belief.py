import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.v44_belief import AdaptiveStudent, SETTINGS, student_log_density
from scripts.v46_belief import RobustAdaptiveStudent


def test_tail_change_preserves_location_variance_and_fresh_prior():
    original = AdaptiveStudent(**SETTINGS)
    robust = RobustAdaptiveStudent(**SETTINGS)
    for history in ([], [.1] * 19, [.2, 1., .4, .8]):
        original.reset()
        robust.reset()
        original.history.extend(history)
        robust.history.extend(history)
        mu, scale2, df = original.predictive_parameters()
        new_mu, new_scale2, new_df = robust.predictive_parameters()
        np.testing.assert_array_equal(new_mu, mu)
        np.testing.assert_allclose(new_scale2 * new_df / (new_df - 2), scale2 * df / (df - 2))
        np.testing.assert_array_equal(new_df, np.full(20, 4.))
        oracle = torch.distributions.StudentT(torch.tensor(new_df), torch.tensor(new_mu), torch.tensor(np.sqrt(new_scale2)))
        np.testing.assert_allclose(robust._compute_log_likelihood(1.05), oracle.log_prob(torch.tensor(1.05, dtype=torch.float64)).numpy())
        if not history:
            np.testing.assert_array_equal(original._compute_log_likelihood(1.05), robust._compute_log_likelihood(1.05))


def test_isolated_normal_spike_is_softened_but_sustained_shift_remains_visible():
    original, robust = AdaptiveStudent(**SETTINGS), RobustAdaptiveStudent(**SETTINGS)
    for _ in range(100):
        original.update(.1)
        robust.update(.1)
    original.update(1.05)
    robust.update(1.05)
    assert original.belief[0] > .5
    assert robust.belief[0] < .5
    assert robust.effective_window > original.effective_window
    robust.reset()
    for _ in range(100):
        robust.update(.1)
    robust.update(5.)
    assert robust.belief[0] > .5
    for _ in range(100):
        robust.update(5.)
    assert robust.effective_window > 17.
    robust.reset()
    assert not robust.history


def test_four_observed_normal_spikes_get_less_reset_vs_mature_likelihood_advantage():
    root = Path(__file__).resolve().parents[1]
    path = root / "results/general_ood_v45_reset_localization/formal/general_ood_v45_reset_localization_relative-adaptive.json"
    if not path.exists():
        pytest.skip("v45 diagnostic fixture not available")
    data = json.loads(path.read_text())
    events = [e for row in data["records"] if row["target_mode"] == "normal" for e in row["event_window"]
              if 768 <= e["step"] < 1024 and e["reset_probability"] > .5 and not e["calibration_changed"]]
    assert len(events) == 4
    for event in events:
        value = event["signal"]
        fresh = student_log_density(value, np.array([0.]), np.array([.5]), np.array([4.]))[0]
        mature = student_log_density(value, np.array([event["mature_predictive_mean"]]),
                                     np.array([event["mature_predictive_variance"] / 2]), np.array([4.]))[0]
        assert fresh - mature < event["log_likelihood_reset_minus_cap"]
