import json
from pathlib import Path

import numpy as np

from scripts import analyze_general_ood_v108_modewise_mean as worker


def test_mode_sources_and_grafts_are_fixed_without_query_labels():
    old = np.arange(30, dtype=float).reshape(6, 5)
    new = old+100
    actual = worker.compose(old, new)
    np.testing.assert_array_equal(actual["old"], old)
    np.testing.assert_array_equal(actual["new"], new)
    for arm, normal, ood in (("old_normal_new_ood", old, new), ("new_normal_old_ood", new, old)):
        np.testing.assert_array_equal(actual[arm][:, 0], normal[:, 0])
        np.testing.assert_array_equal(actual[arm][:, 1:], ood[:, 1:])
    assert worker.MEAN_SOURCES["old_normal_new_ood"] == dict(normal="old", heavy="new",
        slippery="new", high_grav="new", compound="new")


def test_log_density_shift_splits_normal_and_compound_contributions():
    old = np.zeros((4, 5))
    new = old+np.array([2., 1., 4., -2., 3.])
    actual = worker.likelihood_shift(old, new)
    assert actual["compound_minus_normal_change"] == 1.
    assert actual["mean_log_density_change"]["normal"] == 2.
    assert actual["mean_log_density_change"]["compound"] == 3.


def test_full_and_smoke_summaries_have_matched_normal_controls():
    root = Path(__file__).resolve().parents[1]
    old = json.loads((root/"results/general_ood_v98_regime_replay/replay_diverse_recipe/regime_replay.json").read_text())
    new = json.loads((root/"results/general_ood_v106_standard_mean_replay/replay/regime_replay.json").read_text())
    records = [dict(**{k: a[k] for k in ("group", "seed", "mode", "updating")},
        arms=dict(old=a["observers"], new=b["observers"], old_normal_new_ood=a["observers"],
                  new_normal_old_ood=b["observers"])) for a, b in zip(old["records"], new["records"])]
    actual = worker.summarize(records)
    updating = actual["populations"]["updating"]
    assert updating["old"][worker.PRIMARY]["accuracy"] == .8138671875
    assert updating["new"][worker.PRIMARY]["accuracy"] == .846875
    assert updating["old"][worker.PRIMARY]["positive_departure_contrasts"] == 64
    assert actual["old_failure_windows"]["old"]["windows"] == 6
    smoke = worker.summarize(worker.v107.select_paths(records, True))
    assert smoke["populations"]["v95_frozen"]["old"][worker.PRIMARY]["changed_paths"] == 1
    assert smoke["old_failure_windows"]["old"]["windows"] == 2
    assert smoke["historical_slippery_tail"] is None


def test_calibration_uses_each_density_and_both_full_normal_histories():
    from scripts.v73_recursive_mode_readout import RecursiveModeReference

    nulls = [dict(dones=np.arange(6000) % 1000 == 999) for _ in range(2)]
    rng = np.random.default_rng(108)
    likelihoods = [{a: {c: rng.normal(size=(6000, 5)) + np.array([3., 0., 0., 0., 0.])
        for c in worker.summary.CONDITIONS} for a in ("old", "new")} for _ in nulls]
    metadata = {a: {c: dict(arm=a, condition=c) for c in worker.summary.CONDITIONS} for a in worker.ARMS}
    actual = worker.calibrate(nulls, likelihoods, metadata)
    for arm in worker.ARMS:
        for c in worker.summary.CONDITIONS:
            assert actual[arm][c].normal_null_calibration_transitions == 12000
            assert actual[arm][c].metadata()["bank"] == metadata[arm][c]
    from types import SimpleNamespace
    c = worker.PRIMARY
    expected = RecursiveModeReference.calibrate(SimpleNamespace(metadata=lambda: metadata["old"][c],
        log_likelihoods=lambda t: likelihoods[next(i for i, v in enumerate(nulls) if t is v)]["old"][c]), nulls)
    assert actual["old"][c].metadata() == expected.metadata()
    assert actual["old_normal_new_ood"][c].phase.metadata() != actual["old"][c].phase.metadata()
