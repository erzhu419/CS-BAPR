from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import probe_general_ood_v110_fresh_mixture as worker
from scripts import analyze_general_ood_v83_cross_seed_covariance as source
from test_general_ood_v58 import shared_start
from test_general_ood_v68 import trace
from test_general_ood_v78 import banks
from test_general_ood_v89 import sources
from test_general_ood_v91 import expanded
from test_general_ood_v94 import fitted
from test_general_ood_v98 import expanded_coverage, fitted_coverage
from test_general_ood_v106 import family


@pytest.fixture(scope="module")
def frozen():
    nulls = [dict(dones=np.arange(2000) % 1000 == 999) for _ in range(2)]
    rng = np.random.default_rng(110)
    ll = [{a: {c: rng.normal(size=(2000, 5))+np.array([3., 0., 0., 0., 0.])
        for c in worker.summary.CONDITIONS} for a in ("old", "new")} for _ in nulls]
    metadata = {a: {c: dict(arm=a, condition=c) for c in worker.summary.CONDITIONS} for a in worker.ARMS}
    refs = worker.v109.v108.calibrate(nulls, ll, metadata, assemble=worker.v109.compose)
    # Use the production metadata count while keeping small synthetic calibration curves.
    for arm in refs.values():
        for ref in arm.values():
            ref.normal_null_calibration_transitions = 12000
    development = dict(revision=worker.v109.REVISION, passed=True, smoke=False,
        checked_paths=95, checked_transitions=570000, mixture_weights=[.5, .5], mixture_weights_fitted=False,
        mode_density_sources=worker.v109.SOURCES,
        reference_metadata={a: {c: r.metadata() for c, r in arm.items()} for a, arm in refs.items()})
    saved = dict(revision=worker.v109.REVISION, normal_null_transitions=12000,
        mode_density_sources=worker.v109.SOURCES,
        phases={a: {c: r.phase for c, r in arm.items()} for a, arm in refs.items()})
    family = SimpleNamespace(metadata=lambda c: metadata["new"][c],
                             incumbent=SimpleNamespace(metadata=lambda c: metadata["old"][c]))
    return saved, development, family


def test_protocol_freezes_candidate_before_three_fresh_matched_seeds():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root/"experiments/general_ood_v110_fresh_mixture_20261010/dispatch_manifest.json").read_text())
    development = json.loads((root/"results/general_ood_v109_density_mixture/formal/density_mixture.json").read_text())
    population = [c for s in worker.SEEDS for c in worker.cells(s)]
    assert len(population) == manifest["paths"] == 30
    assert 2*sum(c["mode"] != "normal" for c in population) == manifest["changed_windows"] == 48
    assert set(worker.SEEDS+(worker.SMOKE_SEED,)).isdisjoint({r["seed"] for r in development["records"]})
    assert manifest["spec"] == worker.SPEC and tuple(manifest["arms"]) == worker.ARMS
    assert manifest["actual_transitions"] == 180000 and manifest["optimizer_updates"] == 90000
    assert len(worker.cells(worker.SMOKE_SEED, True)) == 10
    with pytest.raises(AssertionError):
        worker.cells(worker.SMOKE_SEED)


def test_shared_covariance_components_equal_both_existing_production_models(family, monkeypatch):
    query = trace(135249, "compound", 4)
    direct = family.incumbent.geometry
    calls = []
    def counted(inputs):
        calls.append(1)
        return direct(inputs)
    monkeypatch.setattr(family.incumbent, "geometry", counted)
    actual = worker.component_likelihoods(family, query)
    assert len(calls) == 1
    monkeypatch.setattr(family.incumbent, "geometry", direct)
    for arm, model in (("old", family.incumbent), ("new", family)):
        expected, _ = model.terms(query)
        for c in worker.summary.CONDITIONS:
            np.testing.assert_allclose(actual[arm][c], expected[c]["likelihoods"], rtol=1e-10, atol=1e-8)


def test_inherited_phases_match_exactly_and_are_not_refitted(frozen, monkeypatch):
    saved, development, family = frozen
    monkeypatch.setattr(source.v76.v73.RecursiveModeReference, "calibrate",
                        lambda *a: pytest.fail("Confirmation must not recalibrate"))
    actual = worker.frozen_references(saved, development, family)
    assert {a: {c: r.metadata() for c, r in arm.items()} for a, arm in actual.items()} == development["reference_metadata"]
    changed = deepcopy(development); changed["mixture_weights"] = [.4, .6]
    with pytest.raises(AssertionError):
        worker.frozen_references(saved, changed, family)
    changed = deepcopy(development); changed["reference_metadata"]["normal_density_mixture"][worker.summary.CONDITIONS[0]]["switch_hazard"] += .01
    with pytest.raises(AssertionError):
        worker.frozen_references(saved, changed, family)


def test_native_likelihood_cache_reuses_completed_scoring_and_rejects_other_cell(tmp_path, monkeypatch):
    monkeypatch.setitem(worker.SPEC, "steps", 8)
    values = {a: {c: np.zeros((8, 5)) for c in worker.summary.CONDITIONS} for a in ("old", "new")}
    monkeypatch.setattr(worker, "component_likelihoods", lambda *a: values)
    cell = worker.cells(worker.SEEDS[0])[0]
    args = SimpleNamespace(checkpoint_dir=tmp_path)
    worker.cached_likelihoods(args, None, None, cell)
    monkeypatch.setattr(worker, "component_likelihoods", lambda *a: pytest.fail("Completed terms must be reused"))
    actual = worker.cached_likelihoods(args, None, None, cell)
    np.testing.assert_array_equal(actual["old"][worker.summary.CONDITIONS[0]], np.zeros((8, 5)))
    with pytest.raises(AssertionError):
        worker.cached_likelihoods(args, None, None, dict(cell, seed=worker.SEEDS[1]))


def test_evaluation_preserves_complete_history_and_matched_normal_summaries(frozen, tmp_path, monkeypatch):
    saved, development, family = frozen
    refs = worker.frozen_references(saved, development, family)
    monkeypatch.setitem(worker.SPEC, "steps", 8)
    monkeypatch.setitem(source.v76.v73.SPEC, "switch_step", 4)
    ranges = dict(first800=(0, 4), full=(0, 8), last1000=(4, 8), first_post50=(4, 6), last64=(6, 8))
    monkeypatch.setattr(source.v76.v73.v68.v66.v65.v64, "window_ranges", lambda n, s: ranges)
    data = trace(135250, "normal", 8); data["dones"] = np.arange(8) % 4 == 3
    monkeypatch.setattr(source.v76.v68, "load_trace", lambda p: data)
    rng = np.random.default_rng(110)
    ll = {a: {c: rng.normal(size=(8, 5)) for c in worker.summary.CONDITIONS} for a in ("old", "new")}
    monkeypatch.setattr(worker, "cached_likelihoods", lambda *a: ll)
    args = SimpleNamespace(seed=worker.SEEDS[0], checkpoint_dir=tmp_path)
    records = worker.evaluate_paths(args, source, worker.cells(args.seed), None, refs)
    for r in records:
        for c in worker.summary.CONDITIONS:
            for a, values in worker.v109.compose(ll["old"][c], ll["new"][c]).items():
                if a not in worker.ARMS:
                    continue
                series = source.v76.v73.observe_likelihoods(values, data["dones"], refs[a][c], "recursive")
                assert r["arms"][a][c] == source.v76.v73.window_statistics(data, values, series, r["mode"])
    result = worker.summarize(records)
    for a in worker.ARMS:
        row = result["updating"][a][worker.summary.CONDITIONS[0]]
        assert row["changed_windows"] == 8 and row["normal_paths"] == 1
        assert result["all"][a][worker.summary.CONDITIONS[0]]["changed_windows"] == 16
    assert {a: {c: r.metadata() for c, r in arm.items()} for a, arm in refs.items()} == development["reference_metadata"]
