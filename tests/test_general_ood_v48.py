from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import probe_general_ood_v43 as v43
from scripts import probe_general_ood_v47 as v47
from scripts import probe_general_ood_v48 as probe
from scripts import v44_belief as belief
from scripts.v46_belief import RobustAdaptiveStudent
from scripts.summarize_general_ood_v48 import validate_run


def reference_fixture():
    ages = np.tile(np.arange(1024), 4)
    errors = np.tile([[-1.], [1.]], (2048, 1))
    errors[ages == 0] = 20.
    calibrations = {name: v43.ResidualCalibration(name, errors, ages, np.ones(1)) for name in v43.CALIBRATORS}
    model = SimpleNamespace(calibrations=calibrations, target_scale=np.ones(1), errors=lambda t: t["errors"])
    return model, dict(errors=errors, episode_steps=ages)


def trace_fixture():
    trace = dict(errors=np.full((1792, 1), .1), dones=np.zeros(1792, dtype=bool),
                 terminated=np.zeros(1792, dtype=bool), truncated=np.zeros(1792, dtype=bool))
    trace["errors"][[0, 1536]] = 20.
    trace["dones"][1535] = trace["truncated"][1535] = True
    return trace


def test_normal_only_offset_changes_startup_bias_not_other_scores_or_thresholds():
    model, calibration = reference_fixture()
    before = deepcopy(model.calibrations["global-centered"].metadata())
    offset, metadata = probe.fit_startup_offset(model, calibration)
    corrected = probe.correct_startup(calibration["errors"], calibration["episode_steps"], offset)
    assert metadata["normal_start_count"] == 4
    assert metadata["corrected_start_mean_signal"] < 1e-20
    assert metadata["original_start_mean_signal"] > 1.
    assert metadata["ongoing_scores_unchanged"]
    assert model.calibrations["global-centered"].metadata() == before
    np.testing.assert_array_equal(corrected[calibration["episode_steps"] > 0], calibration["errors"][calibration["episode_steps"] > 0])
    np.testing.assert_array_equal(calibration["errors"][calibration["episode_steps"] == 0], np.full((4, 1), 20.))


def test_counterfactual_uses_same_past_and_corrected_arm_preserves_real_startup_deviations():
    model, calibration = reference_fixture()
    offset, _ = probe.fit_startup_offset(model, calibration)
    trace = trace_fixture()
    arms, histories, signals, startups, _ = probe.observe(trace, model, offset)
    old_arms, old_histories, _, _ = v47.observe(trace, model)
    np.testing.assert_array_equal(histories["global-robust"], old_histories["global-robust"])
    for phase in v47.PHASES:
        assert arms["global-robust"]["phases"][phase] == old_arms["global-robust"]["phases"][phase]
    reset = startups[-1]
    assert reset["step"] == 1536 and reset["previous_truncated"] and not reset["previous_terminated"]
    assert reset["original_reset_probability"] > .5
    assert reset["same_past_corrected_reset_probability"] < .5 and reset["corrected_reset_probability"] < .5
    tracker = RobustAdaptiveStudent(**belief.SETTINGS)
    for value in signals["global-robust"][:1536]:
        tracker.update(float(value))
    tracker.update(float(signals["reset-centered-robust"][1536]))
    assert tracker.belief[0] == reset["same_past_corrected_reset_probability"]
    changed = deepcopy(trace)
    changed["errors"][1536] = 100.
    _, _, new_signals, new_starts, _ = probe.observe(changed, model, offset)
    assert new_signals["reset-centered-robust"][1536] > signals["reset-centered-robust"][1536]
    assert new_starts[-1]["corrected_reset_probability"] > reset["corrected_reset_probability"]
    for arm in probe.ARMS:
        np.testing.assert_array_equal(signals[arm][:1536], new_signals[arm][:1536])


def test_worker_source_and_first50_curve_accounting():
    root = Path(__file__).resolve().parents[1]
    path = root / "results/general_ood_v48_reset_center/preflight/general_ood_v48_reset_center_fixed-minus2.json"
    if not path.exists():
        pytest.skip("worker preflight not collected")
    data = json.loads(path.read_text())
    manifest = json.loads((root / "experiments/general_ood_v48_reset_center_20261006/dispatch_manifest.json").read_text())
    validate_run(data, manifest, "preflight", root)
    altered = deepcopy(data)
    altered["records"][0]["paired_first50_penalty"]["global-robust"][0] += .1
    with pytest.raises(AssertionError, match="first50-curve-mismatch"):
        validate_run(altered, manifest, "preflight", root)
