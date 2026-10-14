from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from csbapr.belief.departure import DepartureReadout, NormalAgeReference
from scripts import probe_general_ood_v50 as v50
from scripts import probe_general_ood_v51 as probe
from scripts.summarize_general_ood_v51 import validate_run
from tests.test_general_ood_v48 import reference_fixture
from tests.test_general_ood_v50 import trace_with_rewards


def test_normal_only_threshold_fit_and_smooth_age_interpolation():
    ages = np.tile(np.arange(65), 4)
    signal = np.exp(-ages / 12.)
    reference = NormalAgeReference.fit_normal(signal, ages)
    assert reference.normal_start_count == 4
    assert reference.threshold(0) > reference.threshold(64)
    assert reference.threshold(8) == (reference.threshold(0) + reference.threshold(16)) / 2
    assert reference.metadata()["counts"][0] == 36


def test_persistent_departure_and_observed_reset_keep_physical_history():
    reference = NormalAgeReference([0, 16, 64], [3., .2, .1], [1, 1, 1], 1)
    tracker = DepartureReadout(reference)
    for age in range(100):
        tracker.update(1.1, age)
    assert tracker.risk == pytest.approx(1 - np.exp(-1.)) and len(tracker.history) == 16
    before = tracker.risk
    tracker.update(3., 0)
    assert tracker.instant_risk == 0 and tracker.risk == pytest.approx(before * 15 / 16)
    assert tracker.beta == -2 - tracker.risk
    for age in range(1, 100):
        tracker.update(0., age)
    assert tracker.risk == 0 and tracker.beta == -2


def test_readout_is_causal_and_never_changes_signal_or_belief():
    model, calibration = reference_fixture()
    trace = trace_with_rewards()
    observed = v50.observe(trace, model, 32.)
    reference = NormalAgeReference.fit_normal(np.log1p(calibration["errors"][:, 0] ** 2), calibration["episode_steps"])
    snapshot = deepcopy(reference.metadata())
    histories = deepcopy(observed["histories"])
    first = probe.observe_readouts(observed, reference)
    changed = deepcopy(observed)
    changed["signal"][1536:] += 100
    second = probe.observe_readouts(changed, reference)
    for arm in probe.READOUTS:
        np.testing.assert_array_equal(first[arm][:1536], second[arm][:1536])
        assert np.isfinite(first[arm]).all() and np.all((first[arm][:, 0] >= 0) & (first[arm][:, 0] <= 1))
    for arm in v50.ARMS:
        np.testing.assert_array_equal(histories[arm], observed["histories"][arm])
    assert reference.metadata() == snapshot


def test_worker_source_null_seeds_and_readout_curve_accounting():
    root = Path(__file__).resolve().parents[1]
    p = root / "results/general_ood_v51_departure_penalty/preflight/general_ood_v51_departure_penalty_fixed-minus2.json"
    if not p.exists():
        pytest.skip("worker preflight not collected")
    data = json.loads(p.read_text())
    manifest = json.loads((root / "experiments/general_ood_v51_departure_penalty_20261006/dispatch_manifest.json").read_text())
    validate_run(data, manifest, "preflight", root)
    altered = deepcopy(data)
    row = next(r for r in altered["records"] if r["source_v50"]["scenario"] == "reset-normal->heavy")
    row["first50"]["reset"]["age-departure"][0] += .01
    with pytest.raises(AssertionError, match="readout-curve-mismatch"):
        validate_run(altered, manifest, "preflight", root)
