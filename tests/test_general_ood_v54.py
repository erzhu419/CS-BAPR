from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import probe_general_ood_v53_behavior_coverage as v53
from scripts import probe_general_ood_v54_signed_window as worker
from scripts.v54_window_readout import (COVARIANCE_RIDGE, WINDOW_STEPS, SignedWindowReadout,
    SignedWindowReference, normalized_residuals, observe_trace, rolling_means)


class Dynamics:
    calibrations = {"global-centered": SimpleNamespace(means=np.arange(11)[None] / 10.,
        scales=np.linspace(1., 2., 11)[None])}

    def errors(self, trace):
        return trace["next_states"] - trace["states"] - .25 * trace["executed_actions"].mean(axis=1, keepdims=True)


def trace(seed):
    rng = np.random.RandomState(seed)
    correlated = rng.normal(size=(512, 1)) + .2 * rng.normal(size=(512, 11))
    states, actions = rng.normal(size=(512, 11)), rng.normal(size=(512, 3))
    calibration = Dynamics.calibrations["global-centered"]
    return dict(states=states, executed_actions=actions,
        next_states=states + .25 * actions.mean(axis=1, keepdims=True) +
                    calibration.means[0] + correlated * calibration.scales[0],
        dones=np.arange(512) % 64 == 63)


def test_frozen_protocol_roles_and_budgets():
    manifest = json.loads((Path(__file__).resolve().parents[1] /
        "experiments/general_ood_v54_signed_window_20261006/dispatch_manifest.json").read_text())
    assert WINDOW_STEPS == manifest["window_steps"] and COVARIANCE_RIDGE == manifest["covariance_ridge_relative"]
    all_seeds = []
    for stage in ("preflight", "formal"):
        spec = worker.stage_spec(stage)
        assert spec == manifest["stage_specs"][stage]
        all_seeds.extend(s for role in ("fit", "calibration", "test") for s in spec[role])
        cells = v53.trajectory_specs(spec)
        assert len(cells) * spec["steps"] == manifest["added_budgets"][stage]["interactions"]
        assert sum(c["updating"] for c in cells) * spec["steps"] == manifest["added_budgets"][stage]["updates"]
        assert all(c["mode"] == "normal" for c in cells if c["split"] != "test")
    assert len(all_seeds) == len(set(all_seeds))
    assert not set(all_seeds) & {98200, 98210, 98220}


def test_normal_moment_fit_is_separate_from_threshold_calibration():
    model, fitting, calibration = Dynamics(), [trace(1), trace(2)], [trace(3)]
    reference = SignedWindowReference.fit_normal(model, fitting, calibration)
    moments = np.concatenate([rolling_means(normalized_residuals(model, t)) for t in fitting])
    center = moments.mean(axis=0)
    covariance = (moments - center).T @ (moments - center) / len(moments)
    np.testing.assert_allclose(reference.center, center, atol=1e-14)
    np.testing.assert_allclose(reference.covariance, covariance, atol=1e-14)
    expected_precision = np.linalg.inv(covariance + 1e-3 * np.trace(covariance) / 11 * np.eye(11))
    np.testing.assert_allclose(reference.precision, expected_precision, atol=1e-12)
    perturbed = deepcopy(calibration)
    perturbed[0]["next_states"][:, 0] += 10.
    alternative = SignedWindowReference.fit_normal(model, fitting, perturbed)
    np.testing.assert_array_equal(reference.center, alternative.center)
    np.testing.assert_array_equal(reference.covariance, alternative.covariance)
    assert alternative.score95 != reference.score95
    assert reference.fit_transitions == 1024 and reference.calibration_transitions == 512


def test_full_covariance_uses_normal_direction_not_coordinate_selection():
    reference = SignedWindowReference.fit_normal(Dynamics(), [trace(1), trace(2)], [trace(3)])
    common = np.ones(11) / np.sqrt(11)
    opposed = np.zeros(11)
    opposed[:2] = [1 / np.sqrt(2), -1 / np.sqrt(2)]
    assert reference.scores(np.asarray([reference.center + opposed]))[0] > reference.scores(
        np.asarray([reference.center + common]))[0]
    assert reference.metadata()["coordinate_selection"] == "all observation coordinates"


def test_online_offline_causal_identity_and_serialized_history_across_resets(tmp_path):
    model, testing = Dynamics(), trace(9)
    reference = SignedWindowReference.fit_normal(model, [trace(1), trace(2)], [trace(3)])
    metadata = deepcopy(reference.metadata())
    ages = v53.v43.episode_steps(testing["dones"])
    scores = reference.scores(rolling_means(normalized_residuals(model, testing)))
    online = SignedWindowReadout(model, reference)
    coefficients = []
    for i, age in enumerate(ages):
        if i == 128:
            path = tmp_path / "readout.pt"
            torch.save(online, path)
            online = torch.load(path, weights_only=False)
        coefficients.append(online.observe(testing["states"][i], testing["executed_actions"][i],
            testing["next_states"][i], age))
        assert online.score == pytest.approx(scores[i], abs=1e-10)
        assert len(online.history) == min(i + 1, 16)
    observed = observe_trace(testing, model, reference, 256)
    assert observed["full"]["mean_coefficient"] == pytest.approx(np.mean(coefficients), abs=1e-12)
    assert observed["last64"]["mean_coefficient"] == pytest.approx(np.mean(coefficients[-64:]), abs=1e-12)
    assert reference.metadata() == metadata
    future = deepcopy(testing)
    future["next_states"][256:] += 5.
    np.testing.assert_array_equal(rolling_means(normalized_residuals(model, future))[:256],
        rolling_means(normalized_residuals(model, testing))[:256])


def test_normal_null_and_each_tail_remain_required():
    windows = ("full", "prefix", "first_post50", "post", "last64")
    records = [dict(split="test", mode=mode,
        observers={name: {w: dict(mean_coefficient=.01) for w in windows} for name in ("v53_scalar", "signed_window")},
        change_minus_nochange={"signed_window": {w: .2 for w in windows}}) for mode in v53.MODES]
    assert worker.summarize(records)["decision"] == "signed-window-diagnostic-pass-not-RL-confirmation"
    records[1]["change_minus_nochange"]["signed_window"]["last64"] = -.001
    assert worker.summarize(records)["decision"] == "hold-signed-window"
    records[1]["change_minus_nochange"]["signed_window"]["last64"] = .2
    records[0]["observers"]["signed_window"]["last64"]["mean_coefficient"] = .11
    assert worker.summarize(records)["decision"] == "hold-signed-window"
