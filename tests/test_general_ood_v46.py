from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import probe_general_ood_v43 as v43
from scripts import probe_general_ood_v44 as v44
from scripts import probe_general_ood_v46 as v46
from scripts import v44_belief as belief
from tests.test_general_ood_v45 import fixture


def test_original_control_is_unchanged_and_whole_trace_counts_include_boundary():
    model, trace = fixture()
    arms, signal, _ = v46.observe(trace, model)
    scores, thresholds = model.calibrations["episode-centered"].scores(model.errors(trace), v43.episode_steps(trace["dones"]))
    _, replay = belief.replay(scores, thresholds)
    for phase, (start, end) in v46.PHASES.items():
        assert arms["episode-original"]["phases"][phase] == v44.phase_statistics(replay["student-adaptive"][start:end])
    assert arms["episode-original"]["major_reset_transition_counts"]["full1536"] > 0
    assert any(e["step"] == 768 and e["calibration_changed"] for e in arms["episode-original"]["major_reset_events"])
    for name in ("global-original", "global-robust"):
        assert not arms[name]["major_reset_events"]
    assert set(signal) == {"episode-centered", "global-centered"}


def test_all_four_arms_preserve_prefix_when_future_residuals_change():
    model, trace = fixture()
    original, _, _ = v46.observe(trace, model)
    changed = deepcopy(trace)
    changed["errors"][1024:] *= 10
    modified, _, _ = v46.observe(changed, model)
    for name in original:
        assert original[name]["phases"]["prefix1024"] == modified[name]["phases"]["prefix1024"]
        assert [e for e in original[name]["major_reset_events"] if e["step"] < 1024] == [e for e in modified[name]["major_reset_events"] if e["step"] < 1024]


def test_summary_rejects_changed_control_and_missing_whole_trace_events():
    from scripts.summarize_general_ood_v46 import validate_run
    root = Path(__file__).resolve().parents[1]
    path = root / "results/general_ood_v46_crossed_tails/preflight/general_ood_v46_crossed_tails_fixed-minus2.json"
    if not path.exists():
        pytest.skip("worker preflight fixture not yet collected")
    data = json.loads(path.read_text())
    manifest = json.loads((root / "experiments/general_ood_v46_crossed_tails_20261006/dispatch_manifest.json").read_text())
    validate_run(data, manifest, "preflight", root)
    changed = deepcopy(data)
    changed["records"][0]["arms"]["global-original"]["phases"]["validation256"]["penalty"]["mean"] += 1
    with pytest.raises(AssertionError, match="v45-global-belief-changed"):
        validate_run(changed, manifest, "preflight", root)
    changed = deepcopy(data)
    changed["records"][0]["arms"]["global-robust"]["major_reset_transition_counts"]["full1536"] += 1
    with pytest.raises(AssertionError, match="major-reset-count-mismatch"):
        validate_run(changed, manifest, "preflight", root)


def test_missing_runs_are_pending_not_success(tmp_path):
    from scripts.summarize_general_ood_v46 import summarize
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "experiments/general_ood_v46_crossed_tails_20261006/dispatch_manifest.json").read_text())
    result = summarize(manifest, tmp_path, "formal")
    assert not result["matrix_complete"]
    assert result["decision"] == "pending-missing-policies"
