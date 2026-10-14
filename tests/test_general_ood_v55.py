from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from csbapr.belief.departure import NormalAgeReference
from scripts import probe_general_ood_v55_learning_calibration as worker
from scripts.v54_window_readout import SignedWindowReference


class Dynamics:
    calibrations = {"global-centered": SimpleNamespace(means=np.zeros((1, 11)), scales=np.ones((1, 11)))}

    def errors(self, trace):
        return trace["next_states"] - trace["states"]


def trace(seed, scale=1., late_offset=0.):
    rng = np.random.RandomState(seed)
    states = rng.normal(size=(1200, 11))
    errors = rng.normal(size=(1200, 11)) * scale
    errors[780:880] += late_offset
    return dict(states=states, next_states=states + errors, executed_actions=rng.normal(size=(1200, 3)),
                dones=np.arange(1200) == 999)


@pytest.fixture
def old_reference():
    model = Dynamics()
    fitting = [trace(1, late_offset=3.), trace(2, .1)]
    calibration = [trace(3, late_offset=3.), trace(4, .1)]
    reference = SignedWindowReference.fit_normal(model, fitting, calibration)
    return model, reference, calibration


def test_protocol_budgets_and_audit_excluded_from_new_calibration():
    manifest = json.loads((Path(__file__).resolve().parents[1] /
        "experiments/general_ood_v55_learning_calibration_20261007/dispatch_manifest.json").read_text())
    new_seeds = []
    for stage in ("preflight", "formal"):
        spec = worker.stage_spec(stage)
        assert spec == manifest["stage_specs"][stage]
        cells = worker.trajectory_specs(spec)
        assert len(cells) * spec["steps"] == manifest["added_budgets"][stage]["interactions"]
        assert sum(c["updating"] for c in cells) * spec["steps"] == manifest["added_budgets"][stage]["updates"]
        assert all(c["updating"] and c["mode"] == "normal" for c in cells if c["split"] == "calibration")
        new_seeds.extend(spec["calibration"] + spec["test"])
    assert len(new_seeds) == len(set(new_seeds))
    assert not set(new_seeds) & {98400, 98410, 98420, 98421}
    spec["audit"] = spec["calibration"]
    with pytest.raises(AssertionError):
        worker.trajectory_specs(spec)


def test_only_age_threshold_changes_and_old_reference_is_not_mutated(old_reference):
    model, original, _ = old_reference
    before = deepcopy(original.metadata())
    calibration = [trace(8, .1), trace(9, .1)]
    candidate = worker.calibrate_learning(model, original, calibration)
    for name in ("center", "covariance", "precision"):
        np.testing.assert_array_equal(getattr(candidate, name), getattr(original, name))
    assert candidate.score95 == original.score95 and candidate.ridge == original.ridge
    signals = [worker.signals_and_ages(model, original, t) for t in calibration]
    expected = NormalAgeReference.fit_normal(np.concatenate([v[0] for v in signals]),
                                             np.concatenate([v[1] for v in signals]))
    assert candidate.age_reference.metadata() == expected.metadata()
    assert candidate.age_reference.metadata() != original.age_reference.metadata()
    assert original.metadata() == before
    assert candidate.calibration_transitions == 2400


def test_old_calibration_audit_detects_population_contribution_and_mismatch(old_reference):
    model, reference, calibration = old_reference
    audit = worker.audit_pooled_calibration(model, reference, calibration)
    assert not audit["data_used_in_new_calibration"]
    assert [r["age"] for r in audit["first50_age_knots"]] == [800, 816, 832, 848]
    assert all(r["pooled_threshold_exceeds_learning_max"] for r in audit["first50_age_knots"])
    wrong = deepcopy(calibration)
    wrong[0]["next_states"][780:880] += 20.
    with pytest.raises(AssertionError):
        worker.audit_pooled_calibration(model, reference, wrong)


def test_calibration_progress_serializes_without_refitting_fixed_moments(old_reference, tmp_path):
    model, reference, _ = old_reference
    candidate = worker.calibrate_learning(model, reference, [trace(8, .1), trace(9, .1)])
    progress = dict(stage="preflight", next_cell=4, reference=candidate, audit=[], calibration=[],
                    audit_metadata={"data_used_in_new_calibration": False})
    path = tmp_path / "progress.pt"
    torch.save(progress, path)
    restored = torch.load(path, map_location="cpu", weights_only=False)
    assert restored["reference"].metadata() == candidate.metadata()
    np.testing.assert_array_equal(restored["reference"].precision, reference.precision)
    assert restored["audit"] == restored["calibration"] == []


def test_original_null_and_each_early_tail_condition_remain_required():
    windows = ("full", "prefix", "first_post50", "post", "last64")
    records = [dict(split="test", mode=mode,
        observers={name: {w: dict(mean_coefficient=.01) for w in windows} for name in worker.OBSERVERS},
        change_minus_nochange={"learning_calibrated": {w: .2 for w in windows}}) for mode in worker.v53.MODES]
    assert worker.summarize(records)["decision"] == "learning-calibration-diagnostic-pass-not-RL-confirmation"
    records[1]["change_minus_nochange"]["learning_calibrated"]["first_post50"] = 0.
    assert worker.summarize(records)["decision"] == "hold-learning-calibration"
    records[1]["change_minus_nochange"]["learning_calibrated"]["first_post50"] = .2
    records[0]["observers"]["learning_calibrated"]["last64"]["mean_coefficient"] = .11
    assert worker.summarize(records)["decision"] == "hold-learning-calibration"
