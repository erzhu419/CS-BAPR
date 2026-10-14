from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import probe_general_ood_v49 as old
from scripts import probe_general_ood_v50 as probe
from scripts.summarize_general_ood_v50 import validate_run
from tests.test_general_ood_v48 import reference_fixture, trace_fixture


def trace_with_rewards():
    trace = trace_fixture()
    trace["rewards"] = np.zeros(1792)
    for key in ("states", "executed_actions", "contexts"):
        trace[key] = np.zeros((1792, 1))
    return trace


def test_preserved_filter_signal_and_source_statistics_match_v49():
    model, calibration = reference_fixture()
    noise, _ = old.fit_startup_noise(model, calibration)
    trace = trace_with_rewards()
    observed = probe.observe(trace, model, noise)
    arms, histories, signal, _, _ = old.observe(trace, model, noise)
    np.testing.assert_array_equal(observed["signal"], signal)
    for arm in probe.ARMS:
        np.testing.assert_array_equal(observed["histories"][arm], histories[arm])
        assert observed["source"]["arms"][arm] == arms[arm]


def test_signed_direction_is_retained_when_scalar_energy_is_identical():
    cal = SimpleNamespace(means=np.zeros((1, 2)), scales=np.ones((1, 2)))
    cal.scores = lambda errors, ages: (np.mean(errors ** 2, axis=1), np.ones(len(ages)))
    model = SimpleNamespace(calibrations={"global-centered": cal}, errors=lambda trace: trace["errors"])
    first = trace_with_rewards()
    first["errors"] = np.tile([2., -2.], (1792, 1))
    second = deepcopy(first)
    second["errors"] *= -1
    a, b = probe.observe(first, model, 0.), probe.observe(second, model, 0.)
    np.testing.assert_array_equal(a["signal"], b["signal"])
    for arm in probe.ARMS:
        np.testing.assert_array_equal(a["histories"][arm], b["histories"][arm])
    wa, wb = probe.windows(first, a, 1536)["first50"], probe.windows(second, b, 1536)["first50"]
    assert wa["signed_mean"] == [2., -2.] and wb["signed_mean"] == [-2., 2.]
    assert wa["squared_mean"] == wb["squared_mean"] == [4., 4.]
    assert wa["mean_score_ratio"] == wb["mean_score_ratio"] == 4.


def test_windows_and_filter_do_not_read_future_or_assume_fixed_restart():
    model, calibration = reference_fixture()
    noise, _ = old.fit_startup_noise(model, calibration)
    trace = trace_with_rewards()
    trace["dones"][:] = trace["truncated"][:] = False
    trace["dones"][776] = trace["truncated"][776] = True
    observed = probe.observe(trace, model, noise)
    changed = deepcopy(trace)
    changed["errors"][793:] += 100.
    other = probe.observe(changed, model, noise)
    for arm in probe.ARMS:
        np.testing.assert_array_equal(observed["histories"][arm][:793], other["histories"][arm][:793])
    assert probe.windows(trace, observed, 777)["first16"] == probe.windows(changed, other, 777)["first16"]
    assert probe.windows(trace, observed, 777)["first50"] != probe.windows(changed, other, 777)["first50"]
    row = probe.make_record(trace, observed, trace, observed, 99000, "reset-normal->heavy", 777)
    assert row["assessed_switch_step"] == 777 and row["restart_previous_truncated"]
    assert row["diagnostics"]["reset"]["first16"]["startup_count"] == 1


def test_worker_source_and_reset_curve_accounting():
    root = Path(__file__).resolve().parents[1]
    p = root / "results/general_ood_v50_residual_readout/preflight/general_ood_v50_residual_readout_fixed-minus2.json"
    if not p.exists():
        pytest.skip("worker preflight not collected")
    data = json.loads(p.read_text())
    manifest = json.loads((root / "experiments/general_ood_v50_residual_readout_20261006/dispatch_manifest.json").read_text())
    validate_run(data, manifest, "preflight", root)
    altered = deepcopy(data)
    row = next(r for r in altered["records"] if r["scenario"] == "reset-normal->heavy")
    row["first50"]["reset"]["signal"][0] += .1
    with pytest.raises(AssertionError, match="first50-curve-mismatch"):
        validate_run(altered, manifest, "preflight", root)
