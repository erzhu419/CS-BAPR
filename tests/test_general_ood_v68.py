from copy import deepcopy
import argparse
import json
from pathlib import Path
import pickle

import gymnasium as gym
import numpy as np
import pytest
from scipy.stats import multivariate_normal
import torch

from scripts import probe_general_ood_v68_mode_likelihood as worker
from scripts import v68_mode_likelihood as readout
from scripts.run_general_ood_v52 import restore_environment
from scripts.v52_agent import ContinuationAgent
from test_general_ood_v58 import shared_start


def trace(seed, mode="normal", steps=2000, switch=0):
    rng = np.random.RandomState(seed)
    states, actions = rng.normal(size=(steps, 11)), rng.normal(size=(steps, 3))
    delta = .05*states+.02*actions.mean(axis=1, keepdims=True)+rng.normal(0, .01, states.shape)
    index = readout.MODES.index(mode)
    if index:
        delta[switch:, index] += .2
    dones = np.arange(steps) % 1000 == 999
    return dict(states=states, next_states=states+delta, executed_actions=actions, actions=actions,
        rewards=np.ones(steps), dones=dones, terminated=np.zeros(steps, dtype=bool), truncated=dones)


@pytest.fixture(scope="module")
def data():
    fitting = {m: trace(135000, m) for m in readout.MODES}
    density = {m: trace(135010, m) for m in readout.MODES}
    return fitting, density, [trace(135020)]


@pytest.fixture(scope="module")
def reference(data):
    return readout.ModeLikelihoodReference.fit(*data)


def test_contract_known_prior_and_separate_reference_budgets():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root/"experiments/general_ood_v68_mode_likelihood_20261008/dispatch_manifest.json").read_text())
    assert manifest["reference_spec"] == worker.SPEC
    cells = worker.reference_cells(worker.SPEC)
    assert len(cells) == manifest["reference_cells"] == 11
    assert len(cells)*worker.SPEC["steps"] == manifest["added_interactions"] == 66000
    assert manifest["added_optimizer_updates"] == 0 and not manifest["readout_controls_actor"]
    assert not manifest["RL_replay_augmented"] and "labelled" in manifest["known_mode_prior"]
    assert manifest["modes"] == list(readout.MODES)
    assert manifest["normal_windows"] == list(worker.BASE.NORMAL_WINDOWS)
    assert manifest["change_windows"] == list(worker.BASE.CHANGE_WINDOWS)
    assert manifest["window_steps"] == readout.WINDOW_STEPS == 16
    assert not {c["seed"] for c in cells} & set(worker.BASE.SPEC["test"])


def test_density_uses_common_physical_units_and_determinant(reference, data):
    bank = reference.bank
    fitting, density, _ = data
    pooled = np.concatenate([bank.features.inputs(t) for t in fitting.values()])
    np.testing.assert_array_equal(bank.features.input_mean, pooled.mean(axis=0))
    observed = trace(135030, "high_grav", 40)
    likelihoods = bank.log_likelihoods(observed)
    for index, mode in enumerate(readout.MODES):
        design, target = bank.design(fitting[mode]), bank.targets(fitting[mode])
        penalty = len(target)*.001*np.eye(120)
        penalty[0, 0] = 0.
        np.testing.assert_allclose(bank.coefficients[mode], np.linalg.solve(design.T@design+penalty, design.T@target), atol=1e-12)
        center = (bank.targets(density[mode])-bank.design(density[mode])@bank.coefficients[mode]).mean(axis=0)
        np.testing.assert_array_equal(bank.centers[mode], center)
        normalized_covariance = np.linalg.inv(bank.precisions[mode])
        covariance = normalized_covariance*bank.target_scale[:, None]*bank.target_scale[None, :]
        physical_errors = (bank.targets(observed)-bank.design(observed)@bank.coefficients[mode]-center)*bank.target_scale
        expected = multivariate_normal.logpdf(physical_errors, mean=np.zeros(11), cov=covariance)
        np.testing.assert_allclose(likelihoods[:, index], expected, rtol=1e-11, atol=1e-9)
    assert np.isfinite(likelihoods).all()
    assert np.argmax(likelihoods.mean(axis=0)) == readout.MODES.index("high_grav")


def test_density_and_null_calibration_do_not_change_predictors(data, reference):
    fitting, density, _ = data
    altered = deepcopy(density)
    for t in altered.values():
        t["next_states"] += .03
    bank = readout.ModeLikelihoodBank.fit(fitting, altered)
    for mode in readout.MODES:
        np.testing.assert_array_equal(bank.coefficients[mode], reference.bank.coefficients[mode])
        assert np.any(bank.centers[mode] != reference.bank.centers[mode])
    null = trace(135020)
    signals = readout.mode_evidence(readout.rolling_means(reference.bank.log_likelihoods(null)))[2]
    ages = readout.v43.episode_steps(null["dones"])
    phases = readout.episode_ordinals(null["dones"]) > 0
    for later, curve in ((False, reference.phase.initial), (True, reference.phase.after_reset)):
        expected = readout.NormalAgeReference.fit_normal(signals[phases == later], ages[phases == later])
        assert curve.metadata() == expected.metadata()


def test_online_batch_causality_reset_resume_and_frozen_reference(reference):
    before = pickle.dumps(reference)
    observed = trace(135030, "high_grav", switch=800)
    expected = readout.observe_series(observed, reference)
    future = deepcopy(observed)
    future["next_states"][1100:] += 20.
    altered = readout.observe_series(future, reference)
    for key in expected:
        np.testing.assert_array_equal(expected[key][:1100], altered[key][:1100])
    sensor = readout.OnlineModeLikelihoodReadout(reference)
    resumed = None
    for i, age in enumerate(readout.v43.episode_steps(observed["dones"])):
        args = observed["states"][i], observed["executed_actions"][i], observed["next_states"][i], int(age)
        risk = sensor.observe(*args)
        assert risk == pytest.approx(expected["risk"][i], abs=1e-10)
        np.testing.assert_allclose(sensor.mode_weights, expected["mode_weights"][i], atol=1e-10)
        if i == 1000:
            assert sensor.departure.after_reset and len(sensor.history) == 16
        if resumed is not None:
            assert resumed.observe(*args) == pytest.approx(risk, abs=1e-12)
        if i == 998:
            resumed = pickle.loads(pickle.dumps(sensor))
    assert pickle.dumps(reference) == before
    windows = worker.window_statistics(observed, reference, "high_grav", 800)
    assert windows["first800"]["mode_accuracy"] > .99
    assert windows["last64"]["mode_accuracy"] == 1.
    assert windows["first_post50"]["mean_true_minus_best_other_log_likelihood"] > 0


def test_frozen_collector_profiles_context_and_exact_resume(shared_start, tmp_path, monkeypatch):
    checkpoint, shared, _, _, _ = shared_start
    cell = dict(role="fit", seed=135000, mode="heavy")
    def fresh():
        env = gym.make("Hopper-v4", max_episode_steps=3)
        agent = ContinuationAgent(env, "fixed-minus2", "cpu")
        agent.start_shared(checkpoint, shared)
        return agent, env
    direct, env = fresh()
    before = deepcopy(direct.actor.state_dict())
    expected, record = worker.collect_reference(direct, env, cell, 8)
    env.close()
    partial, env = fresh()
    path = tmp_path/"resume.pt"
    def save(loop):
        worker.v53.save_progress(partial, env, {"cell": cell}, loop, path)
        raise RuntimeError("test interruption")
    # Save at a nonterminal tick so the native TimeLimit must also be restored.
    monkeypatch.setattr(worker, "CHECKPOINT_STEPS", 4)
    with pytest.raises(RuntimeError, match="test interruption"):
        worker.collect_reference(partial, env, cell, 8, checkpoint=save)
    env.close()
    restored, env = fresh()
    extra = restored.load(str(path))
    worker.apply_mode_to_env(env, "Hopper-v4", worker.MUJOCO_MODE_PROFILES["Hopper-v4"][cell["mode"]])
    env.reset(seed=cell["seed"])
    restore_environment(env, extra["environment"])
    actual, record = worker.collect_reference(restored, env, cell, 8, extra["loop"])
    env.close()
    for key in worker.v53.TRACE_KEYS:
        np.testing.assert_array_equal(actual[key], expected[key])
    assert record["optimizer_updates"] == 0 and direct.training_steps == restored.training_steps == 99745
    assert direct.replay_buffer.size == restored.replay_buffer.size == shared["replay"].size
    for key, value in before.items():
        assert torch.equal(value, direct.actor.state_dict()[key]) and torch.equal(value, restored.actor.state_dict()[key])


def test_full_worker_replay_and_small_result_storage(tmp_path, monkeypatch):
    spec = dict(worker.SPEC, steps=2000)
    monkeypatch.setattr(worker, "SPEC", spec)
    monkeypatch.setattr(worker.BASE, "SPEC", dict(worker.BASE.SPEC, steps=2000, switch_step=800))
    checkpoint_dir = tmp_path/"reference"
    stored = checkpoint_dir/"traces"
    stored.mkdir(parents=True)
    cells = worker.reference_cells(spec)
    reference_records = []
    for cell in cells:
        np.savez_compressed(worker.trace_path(stored, cell), **trace(cell["seed"], cell["mode"]))
        reference_records.append(dict(**cell, interactions=2000, optimizer_updates=0))
    resume = checkpoint_dir/"preflight_latest.pt"
    progress = dict(revision=worker.REVISION, spec=spec, next_cell=len(cells), records=reference_records)
    torch.save(dict(extra_state=dict(progress=progress, loop=None)), resume)
    v61, v62, v66, shared = [tmp_path/p for p in ("v61", "v62", "v66", "shared")]
    shared.mkdir()
    torch.save({}, shared/"shared.pt")
    (v62/"run_results/preflight").mkdir(parents=True)
    (v62/"run_results/preflight/causal_innovation.json").write_text(json.dumps(dict(revision=worker.BASE.REVISION, stage_spec=worker.BASE.SPEC)))
    cases = [dict(group="v62_test", seed=seed, mode=mode, updating=updating)
        for seed in worker.BASE.SPEC["test"] for mode, updating in [("normal", False)]+[(m, True) for m in readout.MODES]]
    cases += [dict(group="v61_development_replay", seed=132020, mode=mode, updating=True) for mode in readout.MODES]
    old_records = []
    for cell in cases:
        if cell["group"] == "v62_test":
            kind = "updating" if cell["updating"] else "frozen"
            path = v62/f"run_checkpoints/preflight/traces/test_seed{cell['seed']}_{kind}_{cell['mode']}.npz"
        else:
            path = v61/f"run_checkpoints/preflight/traces/seed{cell['seed']}_{cell['mode']}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **trace(cell["seed"], cell["mode"], switch=800))
        control = {w: dict(mean_coefficient=.02 if cell["mode"] == "normal" else .03, mean_signal=.3)
                   for w in worker.v66.v65.v64.window_ranges(2000, 800)}
        old_records.append(dict(**cell, observers={worker.CONTROL: control}))
    (v66/"run_results/screen").mkdir(parents=True)
    (v66/"run_results/screen/covariance_mix.json").write_text(json.dumps(dict(revision=worker.v66.REVISION,
        environment_interactions=0, optimizer_updates=0, records=old_records)))
    args = argparse.Namespace(source_dir=tmp_path/"unused_source", shared_dir=shared, v61_dir=v61, v62_dir=v62,
        v66_dir=v66, checkpoint_dir=checkpoint_dir, result_dir=tmp_path/"results", resume_from=resume)
    worker.run(args)
    output = json.loads((args.result_dir/"mode_likelihood.json").read_text())
    assert output["environment_interactions"] == 22000 and output["optimizer_updates"] == 0
    assert output["replayed_saved_transitions"] == 34000 and len(output["records"]) == 17
    assert not output["readout_controls_actor"] and not output["RL_replay_augmented"]
    for old, new in zip(old_records, output["records"]):
        assert old["observers"][worker.CONTROL] == new["observers"][worker.CONTROL]
    metadata = output["reference_metadata"]
    assert not any(k in json.dumps(metadata) for k in ('"coefficients"', '"precisions"', '"covariance"'))
    saved = torch.load(output["server_reference_checkpoint"], map_location="cpu", weights_only=False)
    assert saved["reference_metadata"] == metadata and saved["spec"] == spec
    assert set(saved["reference"].bank.coefficients) == set(readout.MODES)
