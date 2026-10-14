from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import probe_general_ood_v43 as v43
from scripts import probe_general_ood_v44 as v44
from scripts import probe_general_ood_v45 as v45
from scripts import v44_belief as belief


def fixture():
    errors = np.concatenate([np.tile([[-s], [s]], (128, 1)) for s in (10., 10., 10., .03)])
    calibrations = {name: v43.ResidualCalibration(name, errors, np.arange(1024), np.ones(1))
                    for name in v45.CHANNELS}
    model = SimpleNamespace(calibrations=calibrations, target_scale=np.ones(1), errors=lambda t: t["errors"])
    trace = dict(errors=np.ones((1536, 1)), dones=np.zeros(1536, dtype=bool),
                 terminated=np.zeros(1536, dtype=bool), truncated=np.zeros(1536, dtype=bool))
    return model, trace


def test_boundary_counterfactual_uses_current_error_and_past_filter():
    model, trace = fixture()
    result = v45.diagnose(trace, model)
    rows = {r["step"]: r for r in result["event_window"]}
    boundary = rows[768]
    assert boundary["episode_age"] == 768 and boundary["calibration_changed"]
    assert boundary["reset_probability"] > .5
    assert boundary["previous_group_current_error_reset"] < .5
    assert boundary["signal"] > boundary["previous_group_current_error_signal"]
    assert rows[767]["target_scaled_residual_rms"] == boundary["target_scaled_residual_rms"] == 1.
    assert boundary["mature_predictive_mean"] < .01
    assert result["calibration_boundary_steps"] == [256, 512, 768]


def test_localization_preserves_v44_replay_and_causal_predictive_parameters():
    model, trace = fixture()
    result = v45.diagnose(trace, model)
    scores, thresholds = model.calibrations["episode-centered"].scores(model.errors(trace), v43.episode_steps(trace["dones"]))
    signal, replay = belief.replay(scores, thresholds)
    for phase, (start, end) in v44.PHASES.items():
        assert result["channel_statistics"]["episode-centered"][phase] == v44.phase_statistics(replay["student-adaptive"][start:end])
    tracker = belief.AdaptiveStudent(**belief.SETTINGS)
    for value in signal[:768]:
        tracker.update(float(value))
    location, scale2, df = tracker.predictive_parameters()
    actual = next(r for r in result["event_window"] if r["step"] == 768)
    assert actual["mature_predictive_mean"] == location[-1]
    assert actual["mature_predictive_variance"] == scale2[-1] * df[-1] / (df[-1] - 2)
    assert len(result["event_window"]) < 100
    assert "states" not in result and "errors" not in result


def test_episode_reset_boundary_is_distinguished_from_age_increment():
    model, trace = fixture()
    trace["dones"][900] = trace["terminated"][900] = True
    result = v45.diagnose(trace, model)
    rows = {r["step"]: r for r in result["event_window"]}
    assert rows[901]["previous_transition_ended"] and rows[901]["calibration_changed"]
    assert rows[901]["episode_age"] == 0
    assert not rows[768]["previous_transition_ended"]


def test_summary_checks_source_and_retains_all_major_reset_locations():
    from scripts.summarize_general_ood_v45 import validate_run
    root = Path(__file__).resolve().parents[1]
    path = root / "results/general_ood_v45_reset_localization/preflight/general_ood_v45_reset_localization_fixed-minus2.json"
    if not path.exists():
        pytest.skip("preflight fixture not yet collected")
    data = json.loads(path.read_text())
    manifest = json.loads((root / "experiments/general_ood_v45_reset_localization_20261006/dispatch_manifest.json").read_text())
    validate_run(data, manifest, "preflight", root)
    altered = deepcopy(data)
    altered["records"][0]["channel_statistics"]["episode-centered"]["first_post16"]["effective_age"]["mean"] += 1
    with pytest.raises(AssertionError, match="v44-belief-changed"):
        validate_run(altered, manifest, "preflight", root)
