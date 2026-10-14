import json
from pathlib import Path

import numpy as np

from scripts import analyze_general_ood_v109_density_mixture as worker


def test_mixture_preserves_probability_normalization_and_control_columns():
    old = np.arange(1, 31, dtype=float).reshape(6, 5)
    new = old[::-1].copy()
    old, new = old/old.sum(axis=0), new/new.sum(axis=0)
    actual = worker.compose(np.log(old), np.log(new))
    for arm in worker.ARMS:
        np.testing.assert_allclose(np.exp(actual[arm]).sum(axis=0), 1., atol=1e-14)
    np.testing.assert_allclose(np.exp(actual["all_mode_density_mixture"]), .5*(old+new), atol=1e-14)
    np.testing.assert_allclose(np.exp(actual["normal_density_mixture"][:, 0]), .5*(old[:, 0]+new[:, 0]))
    np.testing.assert_allclose(np.exp(actual["normal_density_mixture"][:, 1:]), new[:, 1:])
    for arm in worker.CONTROLS:
        np.testing.assert_array_equal(actual[arm], worker.v108.compose(np.log(old), np.log(new))[arm])


def test_log_mixture_is_stable_and_not_a_log_density_average():
    old, new = np.full((3, 5), -1000.), np.full((3, 5), -100.)
    actual = worker.compose(old, new)["all_mode_density_mixture"]
    np.testing.assert_allclose(actual, -100.-np.log(2.))
    assert not np.allclose(actual, .5*(old+new))
    np.testing.assert_allclose(worker.compose(old, old)["all_mode_density_mixture"], old)


def test_shared_summary_accepts_mixture_arms_with_matched_controls():
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root/"results/general_ood_v108_modewise_mean/native_smoke/native_smoke.json").read_text())
    records = [dict(r, arms={a: r["arms"][a if a in worker.CONTROLS else "old"] for a in worker.ARMS})
               for r in data["records"]]
    actual = worker.v108.summarize(records)
    assert tuple(actual["populations"]["v95_updating"]) == worker.ARMS
    assert actual["populations"]["v95_updating"]["old"] == data["summary"]["populations"]["v95_updating"]["old"]
    assert actual["old_failure_windows"]["normal_density_mixture"]["windows"] == 2


def test_each_mixture_gets_its_own_both_history_calibration():
    nulls = [dict(dones=np.arange(2000) % 1000 == 999) for _ in range(2)]
    rng = np.random.default_rng(109)
    likelihoods = [{a: {c: rng.normal(size=(2000, 5))+np.array([3., 0., 0., 0., 0.])
        for c in worker.summary.CONDITIONS} for a in ("old", "new")} for _ in nulls]
    metadata = {a: {c: dict(arm=a, condition=c) for c in worker.summary.CONDITIONS} for a in worker.ARMS}
    actual = worker.v108.calibrate(nulls, likelihoods, metadata, assemble=worker.compose)
    for arm in worker.ARMS:
        for c in worker.summary.CONDITIONS:
            assert actual[arm][c].normal_null_calibration_transitions == 4000
            assert actual[arm][c].metadata()["bank"] == metadata[arm][c]
    c = worker.v108.PRIMARY
    assert actual["all_mode_density_mixture"][c].phase.metadata() != actual["old"][c].phase.metadata()
