import numpy as np

from csbapr.belief.tracker import BeliefTracker
from scripts.v38_legacy_belief import BeliefTracker as LegacyBeliefTracker


def penalty(tracker):
    return float(tracker.belief @ np.exp(-0.1 * np.arange(tracker.max_H)))


def test_surprise_spike_shortens_window_and_increases_penalty():
    readings = []
    for surprise in (0.0, 1.0, 3.0, 6.0):
        tracker = BeliefTracker(hazard_rate=0.02, base_variance=1.0)
        tracker.update(surprise)
        readings.append((tracker.effective_window, penalty(tracker), tracker.belief[0]))
    assert all(b[0] < a[0] and b[1] > a[1] and b[2] > a[2]
               for a, b in zip(readings, readings[1:]))


def test_legacy_control_reproduces_opposite_direction():
    calm = LegacyBeliefTracker(hazard_rate=0.02, base_variance=1.0)
    shock = LegacyBeliefTracker(hazard_rate=0.02, base_variance=1.0)
    calm.update(0.0)
    shock.update(3.0)
    assert shock.effective_window > calm.effective_window
    assert penalty(shock) < penalty(calm)


def test_stable_run_grows_shock_resets_and_stability_recovers():
    tracker = BeliefTracker(hazard_rate=0.02, base_variance=1.0)
    for _ in range(100):
        tracker.update(0.0)
    mature_window, mature_penalty = tracker.effective_window, penalty(tracker)
    assert mature_window > 17
    tracker.update(6.0)
    assert tracker.effective_window < mature_window
    assert penalty(tracker) > mature_penalty
    assert tracker.belief[0] > tracker.hazard
    for _ in range(100):
        tracker.update(0.0)
    assert tracker.effective_window > 17


def test_cap_preserves_mass_without_spurious_resets():
    tracker = BeliefTracker(hazard_rate=0.02, base_variance=1.0, variance_growth=0.0)
    tracker.belief[:] = 0.0
    tracker.belief[-1] = 1.0
    tracker.update(0.0)
    assert np.isclose(tracker.belief[0], 0.02)
    assert np.isclose(tracker.belief[-1], 0.98)
    assert np.isclose(tracker.belief.sum(), 1.0)


def test_large_q_std_spike_does_not_underflow_to_uniform():
    tracker = BeliefTracker(hazard_rate=0.02, base_variance=1.0)
    tracker.update(1000.0)
    assert np.isfinite(tracker.belief).all()
    assert np.isclose(tracker.belief.sum(), 1.0)
    assert tracker.belief[0] > 0.99


def test_matches_transition_then_half_normal_bayes_update():
    tracker = BeliefTracker(max_run_length=4, hazard_rate=0.1, base_variance=1.0)
    tracker.belief = np.array([0.1, 0.2, 0.3, 0.4])
    prior = np.array([0.1, 0.09, 0.18, 0.63])
    variance = 1.0 / (1 + 0.05 * np.arange(4))
    weights = prior / np.sqrt(variance) * np.exp(-2.0**2 / (2 * variance))
    tracker.update(2.0)
    np.testing.assert_allclose(tracker.belief, weights / weights.sum())
