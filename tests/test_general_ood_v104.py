import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import probe_general_ood_v104_behavior_exploration as worker
from scripts import analyze_general_ood_v83_cross_seed_covariance as source
from test_general_ood_v58 import shared_start
from test_general_ood_v95 import collection_start


def test_protocol_budget_and_noise_only_after_switch():
    manifest = json.loads((Path(__file__).resolve().parents[1]/
        "experiments/general_ood_v104_behavior_exploration_20261010/dispatch_manifest.json").read_text())
    assert manifest["spec"] == worker.SPEC
    assert len(worker.SEEDS)*len(worker.cells(worker.SEEDS[0]))*6000 == manifest["actual_transitions"] == 48000
    assert manifest["optimizer_updates"] == 48000
    assert manifest["counterfactual_step_calls"] == 240000 and manifest["environment_step_calls"] == 288000
    assert manifest["per_behavior_reference_inputs"] == 4400
    assert manifest["per_behavior_used_extension_union_inputs"] == 8800
    assert manifest["preferred_node"] is None and manifest["preferred_gpu"] is None
    action = np.array([0.9, -0.9, 0.], dtype=np.float32)
    np.random.seed(29)
    for behavior, step in (("standard", 6000), ("exploratory", 3799)):
        np.testing.assert_array_equal(worker.reference_action(action, behavior, step, worker.SPEC), action)
    next_value = np.random.random()
    np.random.seed(29)
    assert np.random.random() == next_value, "matched prefix consumes extra RNG"
    actual = worker.reference_action(action, "exploratory", 3800, worker.SPEC)
    assert actual.dtype == action.dtype and np.all(np.abs(actual) <= 1.) and not np.array_equal(actual, action)


def test_payloads_are_paired_and_equal_budget():
    traces = {}
    for j, cell in enumerate(worker.cells(worker.SEEDS[0])):
        states = np.repeat(np.arange(20)[:, None]+100*j, 11, axis=1).astype(float)
        oracle = np.stack([states+i for i in range(5)], axis=1)
        traces[cell["group"], cell["mode"]] = dict(states=states, executed_actions=np.zeros((20, 3)),
            next_states=oracle[:, source.MODES.index(cell["mode"])].copy(), oracle_observations=oracle)
    spec = dict(steps=20, switch_step=10, stride=2)
    payload, provenance = worker.extensions(traces, spec, source.MODES)
    for behavior, data in payload.items():
        assert len(data["normal"]["states"]) == 10
        assert provenance[behavior][1]["input_row_range"] == [5, 10]
        for i, mode in enumerate(source.MODES):
            np.testing.assert_array_equal(data[mode]["next_states"], data[mode]["states"]+i)
    traces["exploratory", "compound"]["next_states"][10, 0] += 1
    with pytest.raises(AssertionError):
        worker.extensions(traces, spec, source.MODES)


@pytest.mark.parametrize("interrupt_cell", [0, 2])
def test_exact_resume_prefixes_and_actual_replay_actions(collection_start, tmp_path, monkeypatch, interrupt_cell):
    checkpoint, shared = collection_start
    monkeypatch.setattr(source.v76.v69, "SOURCE_NAME", checkpoint.name)
    monkeypatch.setattr(worker, "CHECKPOINT_STEPS", 8)
    spec = dict(worker.SPEC, steps=12, switch_step=4, stride=2)
    args = SimpleNamespace(seed=worker.SEEDS[0], source_dir=checkpoint.parent, shared_dir=shared,
                           checkpoint_dir=tmp_path/"direct", resume_from=None, device="cpu")
    actions = []
    original_store = source.v76.v69.ContinuationAgent.store_transition
    def store(agent, state, action, *remaining):
        actions.append(action.copy())
        return original_store(agent, state, action, *remaining)
    monkeypatch.setattr(source.v76.v69.ContinuationAgent, "store_transition", store)
    expected = worker.collect(args, source, spec)
    assert sum(r["optimizer_updates"] for r in expected) == 48
    for i, cell in enumerate(worker.cells(args.seed)):
        trace = source.v76.v68.load_trace(source.v76.trace_path(args.checkpoint_dir/"traces", cell))
        np.testing.assert_array_equal(trace["actions"], actions[i*12:(i+1)*12])
        np.testing.assert_array_equal(trace["actions"], trace["executed_actions"])
        np.testing.assert_array_equal(trace["actions"][:4], trace["policy_actions"][:4])
        if cell["group"] == "exploratory":
            assert np.any(trace["actions"][4:] != trace["policy_actions"][4:])
        else:
            np.testing.assert_array_equal(trace["actions"], trace["policy_actions"])
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
