import argparse
from copy import deepcopy
import json
from pathlib import Path
import pickle

import numpy as np
import pytest
from scipy.special import logsumexp
import torch

from scripts import probe_general_ood_v73_recursive_mode as worker
from scripts import v73_recursive_mode_readout as readout
from scripts import v68_mode_likelihood as original


class ScoreBank:
    def log_likelihoods(self, trace):
        return np.asarray(trace["states"][:, :5], dtype=float)

    def metadata(self):
        return dict(test_bank="fixed scores", forward_refitted=False)


def trace(mode="normal", steps=2000, switch=1000):
    states = np.zeros((steps, 11))
    states[:, 0] = 8.
    states[switch:, 0] = 0.
    states[switch:, original.MODES.index(mode)] = 8.
    dones = np.arange(steps) % 1000 == 999
    return dict(states=states, executed_actions=np.zeros((steps, 3)), next_states=states.copy(), dones=dones)


@pytest.fixture(scope="module")
def references():
    bank = ScoreBank()
    nulls = [trace(), trace()]
    # The baseline calibration itself is unchanged.
    from scripts.v72_conditional_likelihood import calibrate_bank
    return dict(window16=calibrate_bank(bank, nulls), recursive=readout.RecursiveModeReference.calibrate(bank, nulls))


def test_recursive_update_matches_explicit_transition_matrix_and_extreme_densities():
    sensor = readout.RecursiveModeFilter()
    h = readout.SWITCH_HAZARD
    transition = (1-h)*np.eye(5)+h/5
    posterior = np.full(5, .2)
    rows = np.random.default_rng(6).normal(0, 3, (100, 5))
    for row in rows:
        posterior = (posterior@transition)*np.exp(row-row.max())
        posterior /= posterior.sum()
        np.testing.assert_allclose(np.exp(sensor.update(row)), posterior, atol=1e-14)
    for row in (np.array([-100000., 0., -100000., -100000., -100000.]), np.zeros(5)):
        scores = sensor.update(row)
        assert np.isfinite(scores).all()
        assert logsumexp(scores) == pytest.approx(0., abs=1e-12)


def test_switch_prior_releases_old_history_but_retains_weak_evidence():
    ll = np.zeros((102, 5))
    ll[:100, 0] = 8.
    ll[100, 4] = 8.
    ll[101, 0] = 2.
    scores = readout.temporal_scores(ll, "recursive")
    assert scores[100].argmax() == 4
    assert readout.temporal_scores(ll, "window16")[100].argmax() == 0
    assert scores[101].argmax() == 4
    shifted = readout.temporal_scores(ll+10000., "recursive")
    np.testing.assert_allclose(scores, shifted, atol=2e-12)


def test_online_batch_causality_persistent_reset_and_pickle_resume(references):
    reference = references["recursive"]
    query = trace("compound", switch=980)
    ll = reference.bank.log_likelihoods(query)
    expected = readout.observe_likelihoods(ll, query["dones"], reference, "recursive")
    altered = ll.copy()
    altered[1100:] += np.arange(5)*100
    changed = readout.observe_likelihoods(altered, query["dones"], reference, "recursive")
    for key in expected:
        np.testing.assert_array_equal(expected[key][:1100], changed[key][:1100])
    online = readout.OnlineRecursiveReadout(reference)
    resumed = None
    before = deepcopy(reference.metadata())
    for i, age in enumerate(original.v43.episode_steps(query["dones"])):
        values = query["states"][i], query["executed_actions"][i], query["next_states"][i], int(age)
        risk = online.observe(*values)
        assert risk == pytest.approx(expected["risk"][i], abs=1e-12)
        np.testing.assert_allclose(online.mode_scores, expected["mode_scores"][i], atol=1e-12)
        if resumed is not None:
            assert resumed.observe(*values) == pytest.approx(risk, abs=1e-12)
            np.testing.assert_array_equal(resumed.mode_scores, online.mode_scores)
        if i == 999:
            resumed = pickle.loads(pickle.dumps(online))
        if i == 1000:
            assert online.departure.after_reset
            assert online.mode_scores[4] > readout.temporal_scores(ll[1000:1001], "recursive")[0, 4]
    assert reference.metadata() == before


def test_baseline_is_unchanged_and_recursive_calibration_is_normal_only(references):
    control = references["window16"]
    candidate = references["recursive"]
    assert candidate.bank is control.bank
    assert candidate.normal_null_calibration_transitions == 4000
    query = trace("high_grav", switch=800)
    expected = original.observe_series(query, control)
    actual = readout.observe_likelihoods(control.bank.log_likelihoods(query), query["dones"], control, "window16")
    for new_key, old_key in (("mode_scores", "mean_log_likelihoods"), ("log_alternative_contrast", "log_bayes_factor"),
                             ("mode_weights", "mode_weights"), ("signal", "signal"), ("risk", "risk"), ("threshold", "threshold")):
        np.testing.assert_array_equal(actual[new_key], expected[old_key])
    null = trace()
    signals = original.mode_evidence(readout.temporal_scores(candidate.bank.log_likelihoods(null), "recursive"))[2]
    ages = original.v43.episode_steps(null["dones"])
    phases = original.episode_ordinals(null["dones"]) > 0
    for p, curve in ((False, candidate.phase.initial), (True, candidate.phase.after_reset)):
        fitted = original.NormalAgeReference.fit_normal(np.tile(signals[phases == p], 2), np.tile(ages[phases == p], 2))
        assert curve.metadata() == fitted.metadata()
    assert not candidate.metadata()["switch_step_used_online"]


def test_worker_contract_all_modes_ledger_and_completed_resume(references, tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root/"experiments/general_ood_v73_recursive_mode_20261008/dispatch_manifest.json").read_text())
    assert manifest["spec"] == worker.SPEC
    worker.check_seeds()
    assert len(worker.test_cells(worker.SPEC)) == manifest["fresh_test_paths"] == 10
    assert 10*worker.SPEC["steps"] == manifest["actual_transitions"] == 60000
    assert manifest["environment_step_calls"] == 6*manifest["actual_transitions"] == 360000
    assert manifest["switch_hazard"] == readout.SWITCH_HAZARD == .02
    assert manifest["optimizer_updates"] == manifest["new_reference_labels"] == 0
    assert not manifest["forward_models_refitted"] and not manifest["conditional_density_refitted"]
    spec = dict(worker.SPEC, steps=2000, switch_step=1000)
    monkeypatch.setattr(worker, "SPEC", spec)
    monkeypatch.setattr(worker, "freeze_references", lambda args: references)
    stored = tmp_path/"checkpoint/traces"
    stored.mkdir(parents=True)
    records = []
    cells = worker.test_cells(spec)
    for c in cells:
        np.savez_compressed(worker.v71.trace_path(stored, c), **trace(c["mode"]))
        records.append(dict(**c, actual_transitions=2000, counterfactual_step_calls=10000,
            environment_step_calls=12000, exact_true_mode_replays=2000, optimizer_updates=0))
    resume = stored.parent/"preflight_latest.pt"
    progress = dict(revision=worker.REVISION, spec=spec, next_cell=10, records=records)
    torch.save(dict(extra_state=dict(progress=progress, loop=None)), resume)
    shared = tmp_path/"shared"
    shared.mkdir()
    torch.save({}, shared/"shared.pt")
    def inputs(args, directory):
        assert (args.checkpoint_dir/"temporal_references.pt").is_file()
        return [(dict(group="v73_fresh_frozen", seed=c["seed"], mode=c["mode"], updating=False),
                 worker.v71.trace_path(directory, c)) for c in cells]
    monkeypatch.setattr(worker, "evaluation_inputs", inputs)
    args = argparse.Namespace(checkpoint_dir=stored.parent, result_dir=tmp_path/"results", resume_from=resume,
                              shared_dir=shared, device="cpu")
    worker.run(args)
    output = json.loads((args.result_dir/"recursive_mode.json").read_text())
    assert output["actual_transitions"] == 20000 and output["environment_step_calls"] == 120000
    assert output["exact_true_mode_replays"] == 20000 and output["optimizer_updates"] == 0
    assert len(output["records"]) == 10 and set(output["summary"]["v73_fresh_frozen"]) == set(readout.NAMES)
    assert not any(k in json.dumps(output) for k in ('"residuals"', '"coefficients"', '"precisions"', '"oracle_observations"'))
    baseline = original.observe_series(trace(), references["window16"])
    assert output["records"][0]["observers"]["window16"]["last64"]["mean_coefficient"] == baseline["risk"][-64:].mean()
