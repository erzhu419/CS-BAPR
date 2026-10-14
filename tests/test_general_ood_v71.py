import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import probe_general_ood_v71_state_coverage as worker
from test_general_ood_v68 import trace


def test_protocol_equal_budget_selection_and_seed_isolation():
    path = Path(__file__).resolve().parents[1]/"experiments/general_ood_v71_state_coverage_20261008/dispatch_manifest.json"
    m = json.loads(path.read_text())
    assert m["spec"] == worker.SPEC
    worker.check_seed_partition(worker.SPEC)
    assert len(worker.reference_cells(worker.SPEC)) == m["reference_paths"] == 10
    assert len(worker.test_cells(worker.SPEC)) == m["fresh_test_paths"] == 6
    assert 16*worker.SPEC["steps"] == m["actual_transitions"] == 96000
    assert m["environment_step_calls"] == 6*m["actual_transitions"] == 576000
    assert 2*worker.v69.SPEC["steps"]+3*(worker.SPEC["steps"]-worker.SPEC["switch_step"]) == m["total_labels_per_mode_per_role"]
    normal = worker.selected_cells(worker.SPEC, "fit", "normal_only")
    shifted = worker.selected_cells(worker.SPEC, "fit", "shifted_state")
    assert [c["seed"] for c in normal] == [c["seed"] for c in shifted]
    assert [c["mode"] for c in normal] == ["normal"]*3
    assert [c["mode"] for c in shifted] == ["normal", "high_grav", "compound"]
    with pytest.raises(AssertionError, match="reused-reference-or-evaluation-seed"):
        worker.check_seed_partition(dict(worker.SPEC, test=worker.v70.SPEC["seeds"]))


def setup_traces(tmp_path, monkeypatch):
    spec = dict(worker.SPEC, steps=2000, switch_step=1000)
    monkeypatch.setattr(worker, "SPEC", spec)
    monkeypatch.setattr(worker.v69, "SPEC", dict(worker.v69.SPEC, steps=2000))
    old = tmp_path/"v69/run_checkpoints/reference/traces"
    stored = tmp_path/"new/traces"
    old.mkdir(parents=True)
    stored.mkdir(parents=True)
    for c in worker.v69.reference_cells(worker.v69.SPEC):
        seed = c["seed"]+int(c["updating"])
        np.savez_compressed(worker.v69.trace_path(old, c), **trace(seed))
        if c["role"] != "normal_null":
            for mode in worker.MODES:
                np.savez_compressed(worker.v69.trace_path(old, c, mode), **trace(seed, mode))
    records = []
    for c in worker.reference_cells(spec)+worker.test_cells(spec):
        actual = trace(c["seed"], c["mode"], switch=spec["switch_step"])
        shift = c["mode"] != "normal"
        if shift:
            actual["states"][1000:, 0] += 3.
            actual["next_states"][1000:, 0] += 3.
        oracle = np.stack([trace(c["seed"], mode)["next_states"] for mode in worker.MODES], axis=1)
        if shift:
            oracle[1000:, :, 0] += 3.
        np.savez_compressed(worker.trace_path(stored, c), **actual, oracle_observations=oracle)
        records.append(dict(**c, actual_transitions=2000, counterfactual_step_calls=10000,
                            environment_step_calls=12000, exact_true_mode_replays=2000, optimizer_updates=0))
    return old, stored, records


def test_equal_count_extensions_use_suffix_and_matched_mode_inputs(tmp_path, monkeypatch):
    old, stored, _ = setup_traces(tmp_path, monkeypatch)
    normal = worker.datasets(old, stored, worker.SPEC, "normal_only")
    shifted = worker.datasets(old, stored, worker.SPEC, "shifted_state")
    for role in normal:
        for mode in worker.MODES:
            for data in (normal, shifted):
                assert len(data[role][mode]["states"]) == 7000
                np.testing.assert_array_equal(data[role][mode]["states"], data[role]["normal"]["states"])
                np.testing.assert_array_equal(data[role][mode]["executed_actions"], data[role]["normal"]["executed_actions"])
            np.testing.assert_array_equal(normal[role][mode]["states"][:5000], shifted[role][mode]["states"][:5000])
            np.testing.assert_allclose(shifted[role][mode]["states"][5000:, 0]-normal[role][mode]["states"][5000:, 0], 3.)
            source = worker.v68.load_trace(worker.trace_path(stored, worker.selected_cells(worker.SPEC, role, "shifted_state")[1]))
            np.testing.assert_array_equal(shifted[role][mode]["next_states"][5000:6000], source["oracle_observations"][1000:, worker.MODES.index(mode)])


def test_completed_collection_worker_fits_before_evaluation_and_keeps_baseline(tmp_path, monkeypatch):
    old, stored, records = setup_traces(tmp_path, monkeypatch)
    reference = worker.v69.fit_reference(old, worker.v69.reference_cells(worker.v69.SPEC))
    before = deepcopy(reference.bank.coefficients)
    torch.save(dict(revision=worker.v69.REVISION, spec=worker.v69.SPEC, reference=reference), old.parent/"counterfactual_reference.pt")
    resume = stored.parent/"preflight_latest.pt"
    progress = dict(revision=worker.REVISION, spec=worker.SPEC, next_cell=16, records=records)
    torch.save(dict(extra_state=dict(progress=progress, loop=None)), resume)
    shared = tmp_path/"shared"
    shared.mkdir()
    torch.save({}, shared/"shared.pt")
    args = argparse.Namespace(source_dir=tmp_path/"unused_source", shared_dir=shared, v69_dir=tmp_path/"v69",
        v70_dir=tmp_path/"unused70", v61_dir=tmp_path/"unused61", v62_dir=tmp_path/"unused62",
        checkpoint_dir=stored.parent, result_dir=tmp_path/"results", resume_from=resume, device="cpu")
    def evaluation_inputs(args, stored):
        assert (args.checkpoint_dir/"coverage_references.pt").is_file()
        return [(dict(group="v71_fresh_frozen", seed=c["seed"], mode=c["mode"], updating=False), worker.trace_path(stored, c))
                for c in worker.test_cells(worker.SPEC)]
    monkeypatch.setattr(worker, "evaluation_inputs", evaluation_inputs)
    worker.run(args)
    output = json.loads((args.result_dir/"state_coverage.json").read_text())
    assert output["actual_transitions"] == 32000 and output["environment_step_calls"] == 192000
    assert output["optimizer_updates"] == 0 and output["added_labels_per_mode_per_role"] == 3000
    assert len(output["records"]) == 6 and output["normal_null_recalibrated_per_reference"]
    saved = torch.load(output["reference_checkpoint"], map_location="cpu", weights_only=False)
    for condition in worker.CONDITIONS:
        ref = saved["references"][condition]
        assert ref.normal_null_calibration_transitions == 4000
        for stats in ref.bank.statistics.values():
            assert stats["fit_transitions"] == stats["density_calibration_transitions"] == 7000
    for mode in worker.MODES:
        np.testing.assert_array_equal(saved["references"]["v69"].bank.coefficients[mode], before[mode])
    assert not any(k in json.dumps(output) for k in ('"oracle_observations"', '"coefficients"', '"precisions"'))


def test_detection_alone_does_not_pass_identification_gate():
    records = []
    for mode in ("normal", "high_grav"):
        windows = {w: dict(mean_coefficient=.01, mean_true_minus_best_other_log_likelihood=-1.) for w in worker.WINDOWS}
        records.append(dict(group="test", mode=mode, observers={n: deepcopy(windows) for n in worker.REFERENCES},
            change_minus_nochange={n: {w: .1 for w in worker.WINDOWS} for n in worker.REFERENCES}))
    summary = worker.summarize(records)["test"]
    assert all(not s["screen_pass"] and s["positive_risk_contrast_pairs"]["last64"] == 1
               and s["positive_true_mode_margin_pairs"]["last64"] == 0 for s in summary.values())
