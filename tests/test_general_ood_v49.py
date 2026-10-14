from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import probe_general_ood_v48 as v48
from scripts import probe_general_ood_v49 as probe
from scripts import v44_belief as belief
from scripts.v46_belief import RobustAdaptiveStudent
from scripts.v49_belief import NoiseLikelihoodStudent, NoiseWeightedStudent
from scripts.summarize_general_ood_v49 import validate_run
from tests.test_general_ood_v48 import reference_fixture, trace_fixture


def test_zero_noise_matches_existing_filter_and_keeps_physical_cap_hazard():
    old, noisy, weighted = (t(**belief.SETTINGS) for t in (RobustAdaptiveStudent, NoiseLikelihoodStudent, NoiseWeightedStudent))
    for value in [.1] * 30 + [5.7] + [.1] * 30:
        old.update(value)
        noisy.update(value, 0.)
        weighted.update(value, 0.)
        np.testing.assert_array_equal(old.belief, noisy.belief)
        np.testing.assert_array_equal(old.belief, weighted.belief)
        for a, b in zip(old.predictive_parameters(), weighted.predictive_parameters()):
            np.testing.assert_array_equal(a, b)
    assert len(weighted.history) == len(weighted.precisions) == 19
    assert all(w == 1. for w in weighted.precisions) and weighted.hazard == .02
    weighted.reset()
    assert not weighted.history and not weighted.precisions and weighted.observation_variance == 0.


def test_uncertainty_softens_current_spike_and_history_contamination_without_masking():
    old, noisy, weighted = (t(**belief.SETTINGS) for t in (RobustAdaptiveStudent, NoiseLikelihoodStudent, NoiseWeightedStudent))
    for _ in range(100):
        for t in (old, noisy, weighted):
            t.update(.1)
    old.update(5.7)
    noisy.update(5.7, 32.)
    weighted.update(5.7, 32.)
    assert old.belief[0] > .5 and noisy.belief[0] < .5 and weighted.belief[0] < .5
    assert weighted.history[-1] == 5.7 and weighted.precisions[-1] == 1 / 33
    assert weighted.predictive_parameters()[0][-1] < noisy.predictive_parameters()[0][-1]
    younger = deepcopy(weighted)
    weighted.update(.1)
    younger.update(5.7)
    assert younger.belief[0] > weighted.belief[0]
    for _ in range(100):
        weighted.update(5.)
    assert weighted.effective_window > 17.


def test_fit_is_normal_only_and_observer_control_and_startup_counterfactual_are_causal():
    model, calibration = reference_fixture()
    noise, metadata = probe.fit_startup_noise(model, calibration)
    assert metadata["normal_start_count"] == 4 and noise > 0 and 0 < metadata["startup_history_precision"] < 1
    trace = trace_fixture()
    observed, histories, signal, startups, _ = probe.observe(trace, model, noise)
    offset, _ = v48.fit_startup_offset(model, calibration)
    old, old_histories, old_signals, _, _ = v48.observe(trace, model, offset)
    np.testing.assert_array_equal(signal, old_signals["global-robust"])
    np.testing.assert_array_equal(histories["global-robust"], old_histories["global-robust"])
    assert observed["global-robust"] == old["global-robust"]
    reset = startups[-1]
    cf = NoiseLikelihoodStudent(**belief.SETTINGS)
    for value in signal[:1536]:
        cf.update(float(value))
    cf.update(float(signal[1536]), noise)
    assert cf.belief[0] == reset["same_past_noisy_reset_probability"]
    changed = deepcopy(trace)
    changed["errors"][1536:] += 100.
    _, altered, _, _, _ = probe.observe(changed, model, noise)
    for arm in probe.ARMS:
        np.testing.assert_array_equal(histories[arm][:1536], altered[arm][:1536])


def test_worker_checks_legacy_control_and_true_reset_switch_windows():
    root = Path(__file__).resolve().parents[1]
    p = root / "results/general_ood_v49_startup_noise/preflight/general_ood_v49_startup_noise_fixed-minus2.json"
    if not p.exists():
        pytest.skip("worker preflight not collected")
    data = json.loads(p.read_text())
    manifest = json.loads((root / "experiments/general_ood_v49_startup_noise_20261006/dispatch_manifest.json").read_text())
    validate_run(data, manifest, "preflight", root)
    altered = deepcopy(data)
    reset = next(r for r in altered["records"] if r["scenario"] == "reset-normal->heavy")
    reset["control_assessment"]["global-robust"]["first50"]["penalty"]["mean"] += .1
    with pytest.raises(AssertionError, match="unpaired-control-window"):
        validate_run(altered, manifest, "preflight", root)
