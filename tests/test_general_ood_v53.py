from copy import deepcopy
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

from csbapr.belief.departure import DepartureReadout
from csbapr.utils import ReplayBuffer
from scripts import probe_general_ood_v43 as v43
from scripts import probe_general_ood_v53_behavior_coverage as worker
from scripts.train_csbapr import apply_mode_to_env, MUJOCO_MODE_PROFILES
from scripts.v52_agent import ContinuationAgent


def reference_trace(seed):
    rng = np.random.RandomState(seed)
    states, actions = rng.normal(size=(256, 11)), rng.normal(size=(256, 3))
    next_states = states + .05 * states + .02 * actions.mean(axis=1, keepdims=True) + rng.normal(0, .01, states.shape)
    return dict(states=states, next_states=next_states, executed_actions=actions, dones=np.arange(256) % 32 == 31)


def test_disjoint_roles_and_frozen_protocol_budgets():
    manifest = json.loads((Path(__file__).resolve().parents[1] /
        "experiments/general_ood_v53_behavior_coverage_20261006/dispatch_manifest.json").read_text())
    for stage in ("preflight", "formal"):
        spec = worker.stage_spec(stage)
        assert spec == manifest["stage_specs"][stage]
        cells = worker.trajectory_specs(spec)
        assert len(cells) * spec["steps"] == manifest["added_budgets"][stage]["interactions"]
        assert sum(c["updating"] for c in cells) * spec["steps"] == manifest["added_budgets"][stage]["updates"]
        assert all(c["mode"] == "normal" for c in cells if c["split"] != "test")
    spec["calibration"] = spec["test"]
    with pytest.raises(AssertionError):
        worker.trajectory_specs(spec)


def test_broad_reference_preserves_causal_readout_without_test_refit():
    model, reference = worker.fit_reference([reference_trace(1), reference_trace(2)], [reference_trace(3)])
    trace = reference_trace(9)
    metadata = deepcopy(model.metadata()), deepcopy(reference.metadata())
    observed = worker.observe(trace, model, reference, 128)
    ages = v43.episode_steps(trace["dones"])
    scores, thresholds = model.calibrations["global-centered"].scores(model.errors(trace), ages)
    readout = DepartureReadout(reference)
    coefficients = [readout.update(s, a) for s, a in zip(np.log1p(scores / thresholds), ages)]
    assert observed["full"]["mean_coefficient"] == pytest.approx(np.mean(coefficients), abs=1e-12)
    assert observed["last64"]["mean_coefficient"] == pytest.approx(np.mean(coefficients[-64:]), abs=1e-12)
    assert metadata == (model.metadata(), reference.metadata())


def test_learning_trajectory_resume_keeps_physics_rng_replay_and_full_prefix(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(3)
    model, reference = worker.fit_reference([reference_trace(2)], [reference_trace(3)])
    env = gym.make("Hopper-v4")
    source = ContinuationAgent(env, "fixed-minus2", "cpu")
    source.training_steps, source.context_tracker._count = 99745, 100000
    replay = ReplayBuffer(source.config.buffer_size)
    trace = reference_trace(2)
    context = np.zeros(source.context_dim, dtype=np.float32)
    for i in range(256):
        replay.push(trace["states"][i], np.tanh(trace["executed_actions"][i]), 1., trace["next_states"][i], 0., context, context)
    shared = dict(model=model, reference=reference, startup_noise=2., replay=replay, context=source.snapshot_context())
    path = tmp_path / "source.pt"
    source.save(str(path))
    env.close()
    def fresh():
        env = gym.make("Hopper-v4")
        agent = ContinuationAgent(env, "fixed-minus2", "cpu")
        agent.start_shared(path, shared)
        return agent, env
    spec = dict(steps=8, switch_step=3)
    cell = dict(split="test", seed=100, updating=True, mode="high_grav")
    direct, env = fresh()
    expected, record = worker.collect(direct, env, cell, spec)
    env.close()
    partial, env = fresh()
    progress = dict(next_cell=0, stage="preflight")
    checkpoint = tmp_path / "latest.pt"
    monkeypatch.setattr(worker, "CHECKPOINT_STEPS", 4)
    class Interruption(Exception):
        pass
    def interrupt(loop):
        worker.save_progress(partial, env, progress, loop, checkpoint)
        raise Interruption
    with pytest.raises(Interruption):
        worker.collect(partial, env, cell, spec, checkpoint=interrupt)
    env.close()
    resumed, env = fresh()
    payload = resumed.load(str(checkpoint))
    apply_mode_to_env(env, "Hopper-v4", MUJOCO_MODE_PROFILES["Hopper-v4"]["high_grav"])
    env.reset(seed=cell["seed"])
    worker.restore_environment(env, payload["environment"])
    actual, resumed_record = worker.collect(resumed, env, cell, spec, payload["loop"])
    assert record == resumed_record
    for key in worker.TRACE_KEYS:
        np.testing.assert_array_equal(expected[key], actual[key])
    for name in ("actor", "critic", "target_critic"):
        for key, value in getattr(direct, name).state_dict().items():
            assert torch.equal(value, getattr(resumed, name).state_dict()[key])
    assert direct.replay_buffer.size == resumed.replay_buffer.size
    env.close()


def test_low_null_alone_cannot_pass_without_retained_ood_separation():
    windows = ("full", "prefix", "first_post50", "post", "last64")
    records = [dict(split="test", mode=mode,
        observers={name: {w: dict(mean_coefficient=.01) for w in windows} for name in ("old", "broad")},
        change_minus_nochange={"broad": {w: .2 for w in windows}}) for mode in worker.MODES]
    assert worker.summarize(records)["decision"] == "coverage-diagnostic-pass-not-RL-confirmation"
    records[1]["change_minus_nochange"]["broad"]["last64"] = -.001
    assert worker.summarize(records)["decision"] == "hold-reference-coverage"
    records[1]["change_minus_nochange"]["broad"]["last64"] = .2
    records[0]["observers"]["broad"]["last64"]["mean_coefficient"] = .11
    assert worker.summarize(records)["decision"] == "hold-reference-coverage"


def test_gravity_localization_retains_signed_direction_and_causal_window():
    from types import SimpleNamespace
    from scripts.probe_general_ood_v53_gravity_localization import residual_windows
    class ResidualModel:
        calibrations = {"global-centered": SimpleNamespace(means=np.full((1, 11), 3.), scales=np.full((1, 11), 2.))}
        def errors(self, trace):
            return trace["errors"]
    noise, bias = np.zeros((128, 11)), np.zeros((128, 11))
    noise[16:, 0] = np.tile([1., -1.], 56)
    bias[16:, 0] = 1.
    normal = residual_windows(dict(errors=3. + 2. * noise), ResidualModel(), 16)
    gravity = residual_windows(dict(errors=3. + 2. * bias), ResidualModel(), 16)
    assert normal["prefix"] == gravity["prefix"]
    assert normal["post"]["mean_normalized_square"] == gravity["post"]["mean_normalized_square"]
    assert normal["last64"]["rolling16_mean_squared"][0] == 0.
    assert gravity["last64"]["rolling16_mean_squared"][0] == 1.
    assert gravity["last64"]["signed_normalized_mean"][0] == 1.
    expected = np.mean([(n / 16.) ** 2 for n in range(1, 17)] + [1.] * 34)
    assert gravity["first_post50"]["rolling16_mean_squared"][0] == pytest.approx(expected)
