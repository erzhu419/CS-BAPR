import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import multivariate_normal
import torch

from scripts import probe_general_ood_v74_density_coverage as worker
from scripts import v74_density_coverage as coverage
from scripts import v72_conditional_likelihood as density
from scripts import v73_recursive_mode_readout as temporal
from scripts import v68_mode_likelihood as original
from test_general_ood_v68 import trace


@pytest.fixture(scope="module")
def baseline():
    torch.set_num_threads(1)
    fitting = {m: trace(3, m, steps=400) for m in original.MODES}
    calibration = {m: trace(4, m, steps=400) for m in original.MODES}
    nulls = [trace(5), trace(6)]
    forward = original.ModeLikelihoodReference.fit(fitting, calibration, nulls)
    bank = density.ConditionalDensityBank.fit_density(forward.bank, calibration)
    return dict(window16=density.calibrate_bank(bank, nulls),
                recursive=temporal.RecursiveModeReference.calibrate(bank, nulls)), nulls


def test_protocol_counts_and_seed_partition():
    root = Path(__file__).resolve().parents[1]
    m = json.loads((root/"experiments/general_ood_v74_density_coverage_20261008/dispatch_manifest.json").read_text())
    assert m["spec"] == worker.SPEC
    worker.check_seed_partition()
    assert len(worker.reference_cells(worker.SPEC)) == m["reference_paths"] == 4
    assert len(worker.test_cells(worker.SPEC)) == m["fresh_test_paths"] == 10
    assert (4+10)*worker.SPEC["steps"] == m["actual_transitions"] == 84000
    assert m["environment_step_calls"] == 6*m["actual_transitions"] == 504000
    assert m["base_density_labels_per_mode"] == worker.BASE_DENSITY_TRANSITIONS == 18600
    assert m["added_density_labels_per_mode_per_condition"] == 2*(6000-3800) == 4400
    assert m["density_extension_labels_used"] == 4400*5*2 == 44000
    assert m["switch_hazard"] == temporal.SWITCH_HAZARD == .02
    assert m["neighbors"] == density.NEIGHBORS == 64 and m["optimizer_updates"] == 0
    assert not m["forward_models_refitted"] and not m["scaling_refitted"] and not m["temporal_rules_changed"]


def test_extension_preserves_forward_scaling_and_base_labels(baseline):
    refs, _ = baseline
    bank = refs["window16"].bank
    before = deepcopy(bank.__dict__)
    extra = {m: trace(7, m, steps=100) for m in original.MODES}
    extended = coverage.DensityCoverageBank.extend(bank, extra, "slippery_state")
    assert extended.density_transitions == 500 and extended.added_density_transitions == 100
    np.testing.assert_array_equal(extended.tree.data[:400], before["tree"].data)
    expected = (bank.features.inputs(extra["normal"])-bank.features.input_mean)/bank.features.input_scale
    np.testing.assert_array_equal(extended.tree.data[400:], expected)
    for m in original.MODES:
        np.testing.assert_array_equal(extended.coefficients[m], before["coefficients"][m])
        np.testing.assert_array_equal(extended.residuals[m][:400], before["residuals"][m])
        np.testing.assert_array_equal(bank.residuals[m], before["residuals"][m])
        expected = bank.targets(extra[m])-bank.design(extra[m])@bank.coefficients[m]
        np.testing.assert_allclose(extended.residuals[m][400:], expected, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(extended.target_scale, bank.target_scale)
    np.testing.assert_array_equal(extended.features.input_mean, bank.features.input_mean)
    np.testing.assert_array_equal(extended.features.input_scale, bank.features.input_scale)
    assert extended.metadata()["density_calibration_transitions"] == 500
    changed = deepcopy(extra)
    changed["heavy"]["states"][0, 0] += 1
    with pytest.raises(AssertionError):
        coverage.DensityCoverageBank.extend(bank, changed, "slippery_state")


def test_extended_density_keeps_physical_normalization_and_causal_geometry(baseline):
    refs, _ = baseline
    extra = {m: trace(7, m, steps=100) for m in original.MODES}
    bank = coverage.DensityCoverageBank.extend(refs["window16"].bank, extra, "normal_state")
    query = trace(8, "slippery", steps=8)
    ll = bank.log_likelihoods(query)
    for i, m in enumerate(original.MODES):
        centers, covariance, _ = bank.geometry(bank.features.inputs(query), m)
        means = query["states"]+(bank.design(query)@bank.coefficients[m]+centers)*bank.target_scale
        for j in range(8):
            physical = covariance[j]*bank.target_scale[:, None]*bank.target_scale[None, :]
            assert ll[j, i] == pytest.approx(multivariate_normal.logpdf(query["next_states"][j], mean=means[j], cov=physical), abs=1e-8)
    altered = deepcopy(query)
    altered["next_states"] += 20
    np.testing.assert_array_equal(bank.geometry(bank.features.inputs(query), "slippery")[1],
                                  bank.geometry(bank.features.inputs(altered), "slippery")[1])


def setup_traces(tmp_path, monkeypatch):
    spec = dict(worker.SPEC, steps=2000, switch_step=1000)
    monkeypatch.setattr(worker, "SPEC", spec)
    for module in (worker.v71, worker.v73):
        monkeypatch.setattr(module, "SPEC", dict(module.SPEC, steps=2000, switch_step=1000))
    stored = tmp_path/"checkpoint/traces"
    stored.mkdir(parents=True)
    cells = worker.reference_cells(spec)+worker.test_cells(spec)
    records = []
    for cell in cells:
        branches = [trace(cell["seed"], m, switch=0) for m in original.MODES]
        offset = np.zeros_like(branches[0]["states"])
        if cell["mode"] != "normal":
            offset[1000:, 0] = 2.
        oracle = np.stack([t["next_states"]+1.05*offset for t in branches], axis=1)
        actual = trace(cell["seed"])
        actual["states"] += offset
        truth = np.where(np.arange(2000) < 1000, 0, original.MODES.index(cell["mode"]))
        actual["next_states"] = oracle[np.arange(2000), truth]
        np.savez_compressed(worker.v71.trace_path(stored, cell), **actual, oracle_observations=oracle)
        records.append(dict(**cell, actual_transitions=2000, counterfactual_step_calls=10000,
            environment_step_calls=12000, exact_true_mode_replays=2000, optimizer_updates=0))
    return stored, records


def test_equal_count_extensions_use_only_reference_suffix_and_paired_mode_targets(tmp_path, monkeypatch):
    stored, _ = setup_traces(tmp_path, monkeypatch)
    normal = worker.extensions(stored, worker.SPEC, "normal_state")
    slippery = worker.extensions(stored, worker.SPEC, "slippery_state")
    for mode in original.MODES:
        for data in (normal, slippery):
            assert len(data[mode]["states"]) == 2000
            np.testing.assert_array_equal(data[mode]["states"], data["normal"]["states"])
            np.testing.assert_array_equal(data[mode]["executed_actions"], data["normal"]["executed_actions"])
        np.testing.assert_allclose(slippery[mode]["states"][:, 0]-normal[mode]["states"][:, 0], 2.)
        cell = dict(role="density_extension", seed=worker.SPEC["density_calibration"][0], mode="slippery")
        source = worker.v68.load_trace(worker.v71.trace_path(stored, cell))
        np.testing.assert_array_equal(slippery[mode]["next_states"][:1000], source["oracle_observations"][1000:, original.MODES.index(mode)])


def test_completed_worker_freezes_before_evaluation_and_keeps_server_only_models(baseline, tmp_path, monkeypatch):
    refs, nulls = baseline
    stored, records = setup_traces(tmp_path, monkeypatch)
    monkeypatch.setattr(worker, "BASE_DENSITY_TRANSITIONS", 400)
    old = tmp_path/"v73/run_checkpoints/preflight"
    old.mkdir(parents=True)
    torch.save(dict(revision=worker.v73.REVISION, spec=worker.v73.SPEC, references=refs), old/"temporal_references.pt")
    null_dir = tmp_path/"v69/run_checkpoints/reference/traces"
    null_dir.mkdir(parents=True)
    for cell, null in zip([c for c in worker.v69.reference_cells(worker.v69.SPEC) if c["role"] == "normal_null"], nulls):
        np.savez_compressed(worker.v69.trace_path(null_dir, cell), **null)
    resume = stored.parent/"preflight_latest.pt"
    progress = dict(revision=worker.REVISION, spec=worker.SPEC, next_cell=14, records=records)
    torch.save(dict(extra_state=dict(progress=progress, loop=None)), resume)
    shared = tmp_path/"shared"
    shared.mkdir()
    torch.save({}, shared/"shared.pt")
    def inputs(args, directory):
        assert (args.checkpoint_dir/"density_coverage_references.pt").is_file()
        return [(dict(group="v74_fresh_frozen", seed=c["seed"], mode=c["mode"], updating=False), worker.v71.trace_path(directory, c))
                for c in worker.test_cells(worker.SPEC)]
    monkeypatch.setattr(worker, "evaluation_inputs", inputs)
    args = argparse.Namespace(shared_dir=shared, v73_dir=old.parents[1], v69_dir=null_dir.parents[2],
                              checkpoint_dir=stored.parent, result_dir=tmp_path/"results", resume_from=resume, device="cpu")
    worker.run(args)
    output = json.loads((args.result_dir/"density_coverage.json").read_text())
    assert output["actual_transitions"] == 28000 and output["environment_step_calls"] == 168000
    assert output["exact_true_mode_replays"] == 28000 and output["optimizer_updates"] == 0
    assert output["added_density_labels_per_mode_per_condition"] == 2000 and output["density_extension_labels_used"] == 20000
    assert len(output["reference_records"]) == 4 and len(output["test_records"]) == 10 and len(output["records"]) == 10
    saved = torch.load(output["server_reference_checkpoint"], map_location="cpu", weights_only=False)["references"]
    for condition in worker.CONDITIONS:
        assert saved[condition]["window16"].bank is saved[condition]["recursive"].bank
        assert saved[condition]["recursive"].normal_null_calibration_transitions == 4000
        for m in original.MODES:
            np.testing.assert_array_equal(saved[condition]["recursive"].bank.coefficients[m], refs["window16"].bank.coefficients[m])
    assert saved["normal_state"]["window16"].bank.density_transitions == saved["slippery_state"]["window16"].bank.density_transitions == 2400
    query = worker.v68.load_trace(worker.v71.trace_path(stored, worker.test_cells(worker.SPEC)[0]))
    expected = original.observe_series(query, refs["window16"])
    assert output["records"][0]["observers"]["v72"]["window16"]["last64"]["mean_coefficient"] == expected["risk"][-64:].mean()
    assert saved["v72"]["window16"].metadata() == refs["window16"].metadata()
    assert not any(k in json.dumps(output) for k in ('"oracle_observations"', '"residuals"', '"coefficients"', '"precisions"'))
