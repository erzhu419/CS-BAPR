import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import probe_general_ood_v97_regime_coverage as worker
from scripts import analyze_general_ood_v83_cross_seed_covariance as source
from test_general_ood_v58 import shared_start
from test_general_ood_v95 import collection_start


def test_protocol_has_equal_collection_and_reference_budgets():
    manifest = json.loads((Path(__file__).resolve().parents[1]/
        "experiments/general_ood_v97_regime_coverage_20261009/dispatch_manifest.json").read_text())
    assert manifest["spec"] == worker.SPEC
    assert len(worker.cells(worker.SEEDS[0])) == 4
    assert len(worker.SEEDS)*4*6000 == manifest["actual_transitions"] == manifest["optimizer_updates"] == 48000
    assert manifest["counterfactual_step_calls"] == 240000 and manifest["environment_step_calls"] == 288000
    assert manifest["per_recipe_reference_inputs"] == 4400
    assert manifest["per_recipe_counterfactual_labels"] == 22000
    assert manifest["preferred_node"] is None and not manifest["readout_controls_actor"]


def test_suffix_payloads_have_equal_counts_and_all_five_paired_targets():
    traces = {}
    for j, cell in enumerate(worker.cells(135200)):
        states = np.repeat(np.arange(20)[:, None]+100*j, 11, axis=1).astype(float)
        oracle = np.stack([states+i for i in range(5)], axis=1)
        traces[cell["mode"]] = dict(states=states, executed_actions=np.zeros((20, 3)),
            next_states=oracle[:, source.MODES.index(cell["mode"])].copy(), oracle_observations=oracle)
    payload, provenance = worker.extensions(traces, dict(steps=20, switch_step=10, stride=2), source.MODES)
    for recipe, data in payload.items():
        assert len(data["normal"]["states"]) == 10
        assert provenance[recipe][1]["input_row_range"] == [5, 10]
        for i, mode in enumerate(source.MODES):
            np.testing.assert_array_equal(data[mode]["next_states"], data[mode]["states"]+i)
        np.testing.assert_array_equal(data["normal"]["states"][:5, 0] % 100, [10, 12, 14, 16, 18])
    traces["compound"]["next_states"][10, 0] += 1
    with pytest.raises(AssertionError):
        worker.extensions(traces, dict(steps=20, switch_step=10, stride=2), source.MODES)


@pytest.mark.parametrize("interrupt_cell", [0, 2])
def test_driver_exact_resume_and_prefixes(collection_start, tmp_path, monkeypatch, interrupt_cell):
    checkpoint, shared = collection_start
    monkeypatch.setattr(source.v76.v69, "SOURCE_NAME", checkpoint.name)
    monkeypatch.setattr(source.v76, "CHECKPOINT_STEPS", 8)
    spec = dict(seeds=list(worker.SEEDS), steps=12, switch_step=4, stride=2)
    args = SimpleNamespace(seed=worker.SEEDS[0], source_dir=checkpoint.parent, shared_dir=shared,
                           checkpoint_dir=tmp_path/"direct", resume_from=None, device="cpu")
    expected = worker.collect(args, source, spec)
    assert sum(r["optimizer_updates"] for r in expected) == 48
    resumed = SimpleNamespace(**vars(args))
    resumed.checkpoint_dir = tmp_path/"resumed"
    save = source.v76.v68.v53.save_progress
    def interrupt(agent, env, progress, loop, path):
        save(agent, env, progress, loop, path)
        if progress["next_cell"] == interrupt_cell and loop is not None and loop["step"] == 8:
            raise InterruptedError("within-reference-cell")
    monkeypatch.setattr(source.v76.v68.v53, "save_progress", interrupt)
    with pytest.raises(InterruptedError):
        worker.collect(resumed, source, spec)
    monkeypatch.setattr(source.v76.v68.v53, "save_progress", save)
    resumed.resume_from = resumed.checkpoint_dir/"latest.pt"
    assert worker.collect(resumed, source, spec) == expected
    for cell in worker.cells(args.seed):
        direct = source.v76.v68.load_trace(source.v76.trace_path(args.checkpoint_dir/"traces", cell))
        recovered = source.v76.v68.load_trace(source.v76.trace_path(resumed.checkpoint_dir/"traces", cell))
        for key in direct:
            np.testing.assert_array_equal(direct[key], recovered[key])
    monkeypatch.setattr(source.v76.v69, "ContinuationAgent", lambda *a: pytest.fail("completed paths recollected"))
    assert worker.collect(resumed, source, spec) == expected
