from copy import deepcopy
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

from scripts import probe_general_ood_v70_physical_audit as worker
from test_general_ood_v58 import shared_start
from test_general_ood_v68 import trace as synthetic_trace
from scripts.v68_mode_likelihood import ModeLikelihoodReference, MODES


def fresh(shared_start):
    checkpoint, shared, *_ = shared_start
    env = gym.make("Hopper-v4", max_episode_steps=3)
    shadows = worker.v69.make_shadows(env)
    agent = worker.v69.ContinuationAgent(env, "fixed-minus2", "cpu")
    agent.start_shared(checkpoint, shared)
    return agent, env, shadows


def close(env, shadows):
    for physical in (env, *shadows.values()):
        physical.close()


def test_manifest_budget_and_fresh_seeds():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root/"experiments/general_ood_v70_physical_audit_20261008/dispatch_manifest.json").read_text())
    assert manifest["spec"] == worker.SPEC
    assert len(worker.audit_cells(worker.SPEC))*worker.SPEC["steps"] == manifest["actual_transitions"] == 36000
    assert manifest["environment_step_calls"] == 6*manifest["actual_transitions"] == 216000
    assert manifest["counterfactual_step_calls"] == 180000 and manifest["optimizer_updates"] == 0
    assert not set(worker.SPEC["seeds"]) & {worker.v69.SPEC[k] for k in worker.v69.SPEC if k.endswith("seed")}


@pytest.mark.parametrize("mode", ["high_grav", "compound"])
def test_true_ood_shadow_matched_prefix_frozen_agent_and_exact_resume(shared_start, tmp_path, monkeypatch, mode):
    spec = dict(steps=12, switch_step=4)
    cell = dict(seed=135070, mode=mode)
    direct, env, shadows = fresh(shared_start)
    before = {name: deepcopy(getattr(direct, name).state_dict()) for name in ("actor", "critic", "target_critic")}
    actual, record = worker.collect(direct, env, shadows, cell, spec)
    close(env, shadows)
    assert record["optimizer_updates"] == 0 and record["exact_true_mode_replays"] == 12
    assert record["time_limit_truncations"] == 4
    truth = [MODES.index(worker.active_mode(cell, i, spec)) for i in range(12)]
    np.testing.assert_array_equal(actual["oracle_observations"][np.arange(12), truth], actual["next_states"])
    for name, weights in before.items():
        for key, value in weights.items():
            assert torch.equal(value, getattr(direct, name).state_dict()[key])
    normal, env, shadows = fresh(shared_start)
    baseline, _ = worker.collect(normal, env, shadows, dict(cell, mode="normal"), spec)
    close(env, shadows)
    assert worker.v69.v68.v53.paired_prefix_equal(actual, baseline, 4)
    partial, env, shadows = fresh(shared_start)
    path = tmp_path/"resume.pt"
    monkeypatch.setattr(worker, "CHECKPOINT_STEPS", 8)
    def save(loop):
        worker.v69.v68.v53.save_progress(partial, env, {}, loop, path)
        raise RuntimeError("test interruption")
    with pytest.raises(RuntimeError, match="test interruption"):
        worker.collect(partial, env, shadows, cell, spec, checkpoint=save)
    close(env, shadows)
    restored, env, shadows = fresh(shared_start)
    extra = restored.load(str(path))
    worker.apply_mode(env, worker.active_mode(cell, extra["loop"]["step"]-1, spec))
    env.reset(seed=cell["seed"])
    worker.v69.restore_step_state(env, extra["environment"])
    recovered, resumed_record = worker.collect(restored, env, shadows, cell, spec, extra["loop"])
    close(env, shadows)
    assert record == resumed_record
    assert restored.replay_buffer.size == shared_start[1]["replay"].size
    for key in actual:
        np.testing.assert_array_equal(recovered[key], actual[key])


def test_oracle_gaps_aliasing_and_scalar_readout_do_not_refit():
    fitting = {m: synthetic_trace(2, m) for m in MODES}
    calibration = {m: synthetic_trace(3, m) for m in MODES}
    reference = ModeLikelihoodReference.fit(fitting, calibration, [synthetic_trace(4), synthetic_trace(5)])
    data = synthetic_trace(6, "high_grav", steps=2000, switch=1000)
    spec = dict(steps=2000, switch_step=1000)
    truth = np.where(np.arange(2000) < 1000, 0, MODES.index("high_grav"))
    oracle = np.repeat(data["next_states"][:, None, :], 5, axis=1)
    for i in range(5):
        oracle[:, i, 0] += (i-truth)*.05
    # An observation alias with a different hidden physical state.
    oracle[-64:, 0] = oracle[-64:, MODES.index("high_grav")]
    full = np.concatenate((oracle, np.broadcast_to(np.arange(5)[None, :, None], (2000, 5, 1))), axis=2)
    data.update(oracle_observations=oracle, oracle_physical_states=full,
        actual_physical_states=full[np.arange(2000), truth], clipped_input_velocities=np.zeros(2000, dtype=bool))
    coefficients = deepcopy(reference.bank.coefficients)
    report = worker.summarize_trace(data, dict(seed=135070, mode="high_grav"), reference, spec)
    assert report["first_post50"]["observable_unique_true_template_fraction"] == 1.
    assert report["last64"]["observable_unique_true_template_fraction"] == 0.
    assert report["last64"]["full_state_unique_true_template_fraction"] == 1.
    assert report["last64"]["nearest_physical_template_gap_rms_common_units"]["mean"] == 0.
    assert report["last64"]["model_error_over_nearest_template_gap"] is None
    json.dumps(report, allow_nan=False)
    for mode in MODES:
        np.testing.assert_array_equal(reference.bank.coefficients[mode], coefficients[mode])
