from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import probe_general_ood_v118_frozen_classifier_confirmation as worker
from scripts import summarize_general_ood_v118 as merger
from scripts import analyze_general_ood_v83_cross_seed_covariance as source
from scripts import v73_recursive_mode_readout as temporal
from test_general_ood_v117 import null_population


@pytest.fixture(scope="module")
def frozen():
    nulls, values = null_population()
    references = worker.v117.calibrate_scores(nulls, values)
    for ref in references.values():
        ref.normal_null_calibration_transitions = 12000
    metadata = worker.v117.reference_metadata(references)
    null_cells = [dict(seed=135060, updating=u) for u in (False, True)]
    saved = dict(revision=worker.v117.REVISION, normal_null_cells=null_cells,
                 phases={a: r.phase for a, r in references.items()}, reference_metadata=metadata)
    development = dict(revision=worker.v117.REVISION, passed=True, smoke=False,
        checked_paths=125, checked_transitions=750000, unchanged_v116_score_windows_verified=76,
        normal_null_cells=null_cells, reference_metadata=metadata, records=[dict(seed=135252)])
    return saved, development


def test_frozen_references_match_exactly_without_calibration(frozen, monkeypatch):
    saved, development = frozen
    monkeypatch.setattr(temporal.RecursiveModeReference, "calibrate", lambda *a: pytest.fail("must not recalibrate"))
    before = deepcopy(saved["reference_metadata"])
    refs = worker.restore_score_references(saved, development)
    assert worker.v117.reference_metadata(refs) == before
    changed = deepcopy(development); changed["smoke"] = True
    with pytest.raises(AssertionError):
        worker.restore_score_references(saved, changed)
    changed = deepcopy(saved); changed["phases"][worker.ARMS[0]].initial.upper += .01
    with pytest.raises(AssertionError):
        worker.restore_score_references(changed, development)
    assert saved["reference_metadata"] == before


def test_fresh_protocol_keeps_all_modes_normal_controls_and_excludes_used_seeds(frozen):
    _, development = frozen
    family = SimpleNamespace(mean_groups=np.array([135230, 135231]))
    worker.verify_unused_seeds(development, family, source)
    with pytest.raises(AssertionError):
        worker.verify_unused_seeds(dict(development, records=[dict(seed=worker.SEEDS[0])]), family, source)
    with pytest.raises(AssertionError):
        worker.verify_unused_seeds(development, SimpleNamespace(mean_groups=np.array([worker.SMOKE_SEED])), source)
    all_cells = [c for s in worker.SEEDS for c in worker.cells(s)]
    assert len(all_cells) == 30 and sum(c["updating"] for c in all_cells) == 15
    assert sum(c["mode"] != "normal" for c in all_cells) == 24
    assert len(worker.cells(worker.SMOKE_SEED, True)) == 10
    for c in all_cells:
        assert dict(c, mode="normal") in all_cells
    with pytest.raises(AssertionError):
        worker.cells(worker.SMOKE_SEED)


def synthetic_records(seed):
    records = []
    for c in worker.cells(seed):
        normal = c["mode"] == "normal"
        windows = {w: dict(mode_accuracy=.95 if normal else .8, mean_coefficient=.01 if normal else .8,
            mean_true_minus_best_other_score=1., above_threshold_fraction=.06 if normal else 1.)
            for w in ("first800", "full", "last1000", "first_post50", "last64")}
        old = deepcopy(windows)
        for row in old.values():
            row.update(mode_accuracy=.98 if normal else .7, mean_coefficient=.005 if normal else .6,
                       above_threshold_fraction=.05 if normal else .9)
        records.append(dict(**c, arms={a: deepcopy(windows) for a in worker.ARMS},
            incumbent={k: deepcopy(old) for k in worker.CONDITIONS},
            normal_episodes={a: [dict(mean_risk=.02)] for a in worker.ARMS} if normal else None,
            incumbent_normal_episodes={k: [dict(mean_risk=.01)] for k in worker.CONDITIONS} if normal else None))
    return records


def test_summary_reports_incumbent_normal_exceedance_and_preserves_adverse_cases():
    records = synthetic_records(worker.SEEDS[0])
    changed = next(r for r in records if r["mode"] == "compound" and r["updating"])
    changed["arms"]["matched_classifier"]["first_post50"]["mode_accuracy"] = 0.
    summary = worker.summarize(records)
    updating = summary["updating"]
    assert updating["matched_classifier"]["early_accuracy"] == pytest.approx(.6)
    assert updating["matched_classifier"]["minimum_departure_contrast"] == pytest.approx(.79)
    assert updating["incumbent_v98"]["normal_full_above_threshold_fraction"] == .05
    assert updating["incumbent_v98"]["normal_full_mean_risk"] == .005
    assert summary["frozen"]["matched_classifier"]["early_accuracy"] == .8


def test_evaluation_uses_complete_history_for_all_six_readouts(frozen, tmp_path, monkeypatch):
    saved, development = frozen
    refs = worker.restore_score_references(saved, development)
    incumbent = {c: refs[worker.ARMS[0]] for c in worker.CONDITIONS}
    data = dict(dones=np.arange(8) % 4 == 3)
    values = np.tile([3., 0., 0., 0., 0.], (8, 1)); values[4:] = [0., 3., 0., 0., 0.]
    monkeypatch.setattr(source.v76.v68, "load_trace", lambda p: data)
    monkeypatch.setattr(worker, "score_trace", lambda *a: ({a: values for a in worker.ARMS}, {c: values for c in worker.CONDITIONS}))
    monkeypatch.setitem(source.v76.v73.SPEC, "switch_step", 4)
    ranges = dict(first800=(0, 4), full=(0, 8), last1000=(4, 8), first_post50=(4, 6), last64=(6, 8))
    monkeypatch.setattr(source.v76.v73.v68.v66.v65.v64, "window_ranges", lambda n, s: ranges)
    args = SimpleNamespace(seed=worker.SEEDS[0], checkpoint_dir=tmp_path, smoke=False)
    records, checked = worker.evaluate_paths(args, source, worker.cells(args.seed), dict(steps=8), None, None, None, refs, incumbent)
    assert len(records) == 10 and checked == 60
    series = temporal.observe_likelihoods(values, data["dones"], refs[worker.ARMS[0]], "recursive")
    expected = worker.v117.window_statistics(data, values, series, "normal", source)
    assert records[0]["arms"][worker.ARMS[0]] == expected
    assert records[0]["incumbent"][worker.PRIMARY] == expected
    assert worker.v117.reference_metadata(refs) == development["reference_metadata"]


def test_merge_uses_three_environment_seed_units_and_rejects_metadata_drift():
    parts = []
    for seed in worker.SEEDS:
        records = synthetic_records(seed)
        parts.append(dict(revision=worker.REVISION, passed=True, smoke=False, seed=seed, spec=worker.SPEC,
            arms=list(worker.ARMS), incumbent_conditions=list(worker.CONDITIONS), checked_path_readouts=60,
            fresh_paths=10, changed_windows=16, records=records, summary=worker.summarize(records),
            reference_metadata_unchanged=True, references_frozen_before_collection=True, reference_metadata=dict(frozen=True),
            actual_transitions=60000, counterfactual_step_calls=300000, environment_step_calls=360000,
            exact_true_mode_replays=60000, optimizer_updates=30000, limitations=[],
            **{f: False for f in ("reference_refitted", "normal_thresholds_recalibrated", "temporal_rules_changed",
                "classifier_settings_changed", "mean_settings_changed", "candidate_selected", "readout_controls_actor", "readout_adopted")}))
    result = merger.summarize(parts)
    assert result["paths"] == 30 and result["actual_transitions"] == 180000
    interval = result["paired_updating_matched_minus_v98"]["early_accuracy"]
    assert interval["deltas"] == pytest.approx([.1]*3)
    assert interval["confidence_interval_95"] == pytest.approx([.1, .1])
    parts[1]["reference_metadata"] = dict(frozen=False)
    with pytest.raises(AssertionError):
        merger.summarize(parts)
