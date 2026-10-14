from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import probe_general_ood_v95_fresh_seed as worker
from scripts import analyze_general_ood_v83_cross_seed_covariance as source
from scripts.v73_recursive_mode_readout import RecursiveModeReference, observe_likelihoods
from scripts.v78_local_linear_mean import LocalLinearDensityBank, LocalLinearMean
from test_general_ood_v58 import shared_start
from test_general_ood_v68 import trace
from test_general_ood_v78 import banks
from test_general_ood_v89 import sources
from test_general_ood_v91 import expanded
from test_general_ood_v94 import fitted


def test_protocol_matched_cells_and_explicit_new_collection_budget():
    m = json.loads((Path(__file__).resolve().parents[1]/
        "experiments/general_ood_v95_fresh_seed_20261009/dispatch_manifest.json").read_text())
    population = [c for s in worker.SEEDS for c in worker.cells(s)]
    assert len(population) == m["paths"] == 30
    assert sum(c["mode"] != "normal" for c in population)*2 == m["changed_windows"] == 48
    assert len(population)*6000 == m["actual_transitions"] == 180000
    assert sum(c["updating"] for c in population)*6000 == m["optimizer_updates"] == 90000
    assert m["counterfactual_step_calls"] == 5*m["actual_transitions"]
    assert m["environment_step_calls"] == 6*m["actual_transitions"]
    assert m["spec"] == worker.SPEC and tuple(m["conditions"]) == worker.CONDITIONS
    assert not m["readout_controls_actor"] and not m["reference_refitted"]
    assert not m["normal_thresholds_recalibrated"] and not m["coefficient_tuned"]
    assert set(worker.SEEDS).isdisjoint(source.v76.SPEC["test"]+source.v76.SPEC["density_calibration"])
    assert len(worker.cells(worker.SMOKE_SEED, True)) == 10
    with pytest.raises(AssertionError):
        worker.cells(worker.SMOKE_SEED)


def test_four_readouts_reuse_exact_v78_and_v94_scores(fitted):
    family = fitted[0]
    baseline = LocalLinearDensityBank(dict(frozen=family.bank, updating=family.bank), LocalLinearMean(family.bank), "updating")
    query = trace(991, "heavy", 5)
    actual, geometry = worker.score_terms(family, baseline, query)
    expected, other_geometry = family.terms(query)
    assert geometry == other_geometry and tuple(actual) == worker.CONDITIONS
    np.testing.assert_allclose(actual[worker.BASELINE]["likelihoods"], baseline.log_likelihoods(query), rtol=1e-11, atol=1e-9)
    for c in worker.v94.CONDITIONS:
        for k in actual[c]:
            np.testing.assert_array_equal(actual[c][k], expected[c][k])


@pytest.fixture(scope="module")
def collection_start(shared_start, tmp_path_factory):
    checkpoint, original, *_ = shared_start
    shared = deepcopy(original)
    sample = shared["replay"].buffer[0]
    while shared["replay"].size < 6144:
        shared["replay"].push(*sample)
    directory = tmp_path_factory.mktemp("v95-shared")
    torch.save(shared, directory/"shared.pt")
    return checkpoint, directory


@pytest.mark.parametrize("interrupt_cell", [0, 5])
def test_real_driver_prefixes_and_exact_resume(collection_start, tmp_path, monkeypatch, interrupt_cell):
    checkpoint, shared = collection_start
    monkeypatch.setattr(source.v76.v69, "SOURCE_NAME", checkpoint.name)
    monkeypatch.setattr(source.v76, "CHECKPOINT_STEPS", 8)
    monkeypatch.setattr(source.v76.v70, "CHECKPOINT_STEPS", 8)
    seed = worker.SEEDS[0]
    spec = dict(seeds=[seed], steps=12, switch_step=4)
    population = worker.cells(seed)
    direct = SimpleNamespace(source_dir=checkpoint.parent, shared_dir=shared, seed=seed, device="cpu",
                             resume_from=None, checkpoint_dir=tmp_path/"direct")
    expected = worker.collect_paths(direct, source, population, spec)
    assert sum(r["optimizer_updates"] for r in expected) == 60
    resumed = SimpleNamespace(**vars(direct))
    resumed.checkpoint_dir = tmp_path/"resumed"
    save = source.v76.v68.v53.save_progress
    def interrupt(agent, env, progress, loop, path):
        save(agent, env, progress, loop, path)
        if progress["next_cell"] == interrupt_cell and loop is not None and loop["step"] == 8:
            raise InterruptedError("within-cell driver interruption")
    monkeypatch.setattr(source.v76.v68.v53, "save_progress", interrupt)
    with pytest.raises(InterruptedError):
        worker.collect_paths(resumed, source, population, spec)
    monkeypatch.setattr(source.v76.v68.v53, "save_progress", save)
    resumed.resume_from = resumed.checkpoint_dir/"latest.pt"
    actual = worker.collect_paths(resumed, source, population, spec)
    assert expected == actual
    for cell in population:
        a = source.v76.v68.load_trace(source.v76.trace_path(direct.checkpoint_dir/"traces", cell))
        b = source.v76.v68.load_trace(source.v76.trace_path(resumed.checkpoint_dir/"traces", cell))
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])
    def no_more_collection(*args, **kwargs):
        pytest.fail("Completed cells must not be collected again")
    monkeypatch.setattr(source.v76.v69, "ContinuationAgent", no_more_collection)
    assert worker.collect_paths(resumed, source, population, spec) == actual


def test_evaluation_uses_full_history_frozen_phases_and_completed_path_resume(fitted, shared_start, tmp_path, monkeypatch):
    family = fitted[0]
    baseline = LocalLinearDensityBank(dict(frozen=family.bank, updating=family.bank), LocalLinearMean(family.bank), "updating")
    spec = dict(seeds=list(worker.SEEDS), steps=8, switch_step=4)
    monkeypatch.setattr(worker, "SPEC", spec)
    monkeypatch.setitem(source.v76.v73.SPEC, "switch_step", 4)
    windows = dict(first800=(0, 4), full=(0, 8), last1000=(4, 8), first_post50=(4, 6), last64=(6, 8))
    monkeypatch.setattr(source.v76.v73.v68.v66.v65.v64, "window_ranges", lambda n, s: windows)
    monkeypatch.setattr(source.v78.v77, "WINDOWS", tuple(windows))
    monkeypatch.setattr(source.v78.v77, "windows", lambda: (("first_post50", 4, 6), ("last64", 6, 8)))
    references = {}
    for c in worker.CONDITIONS:
        ref = RecursiveModeReference()
        ref.bank = baseline if c == worker.BASELINE else SimpleNamespace(metadata=lambda c=c: family.metadata(c))
        ref.phase, ref.normal_null_calibration_transitions = deepcopy(shared_start[4]), 12000
        references[c] = ref
    before = deepcopy({c: r.metadata() for c, r in references.items()})
    args = SimpleNamespace(seed=worker.SEEDS[0], checkpoint_dir=tmp_path)
    population = worker.cells(args.seed)[:2]
    stored = tmp_path/"traces"
    stored.mkdir()
    queries = []
    for i, cell in enumerate(population):
        data = trace(996+i, cell["mode"], 8)
        queries.append(data)
        np.savez_compressed(source.v76.trace_path(stored, cell), **data)
    records = worker.evaluate_paths(args, source, population, family, baseline, references)
    for cell, data, record in zip(population, queries, records):
        terms, _ = worker.score_terms(family, baseline, data)
        for c in worker.CONDITIONS:
            observed = observe_likelihoods(terms[c]["likelihoods"], data["dones"], references[c], "recursive")
            assert record["observers"][c] == source.v76.v73.window_statistics(data, terms[c]["likelihoods"], observed, cell["mode"])
    assert before == {c: r.metadata() for c, r in references.items()}
    monkeypatch.setattr(source.v76.v68, "load_trace", lambda p: pytest.fail("Completed scoring must be reused"))
    assert worker.evaluate_paths(args, source, population, family, baseline, references) == records
    altered = SimpleNamespace(raw_weight=.5)
    with pytest.raises(AssertionError):
        worker.evaluate_paths(args, source, population, altered, baseline, references)
