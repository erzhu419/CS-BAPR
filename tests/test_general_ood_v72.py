import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import multivariate_normal
import torch

from scripts import probe_general_ood_v72_conditional_density as worker
from scripts import v72_conditional_likelihood as density
from scripts import v68_mode_likelihood as original
from test_general_ood_v68 import trace


@pytest.fixture(scope="module")
def reference_data():
    torch.set_num_threads(1)
    fitting = {m: trace(3, m) for m in original.MODES}
    calibration = {m: trace(4, m) for m in original.MODES}
    nulls = [trace(5), trace(6)]
    global_ref = original.ModeLikelihoodReference.fit(fitting, calibration, nulls)
    return global_ref, calibration, nulls


def test_contract_reuses_density_data_and_isolates_fresh_seeds():
    path = Path(__file__).resolve().parents[1]/"experiments/general_ood_v72_conditional_density_20261008/dispatch_manifest.json"
    m = json.loads(path.read_text())
    assert m["spec"] == worker.SPEC
    worker.check_seeds()
    assert len(worker.test_cells(worker.SPEC)) == m["fresh_test_paths"] == 6
    assert 6*worker.SPEC["steps"] == m["actual_transitions"] == 36000
    assert m["environment_step_calls"] == 6*m["actual_transitions"] == 216000
    assert m["neighbors"] == density.NEIGHBORS == 64
    assert m["new_reference_labels"] == m["optimizer_updates"] == 0
    assert not m["forward_models_refitted"] and not m["readout_controls_actor"]


@pytest.mark.parametrize("cls", [density.ConditionalMeanBank, density.ConditionalDensityBank])
def test_physical_gaussian_normalization_and_frozen_forward(reference_data, cls):
    forward, calibration, _ = reference_data
    before = deepcopy(forward.bank.coefficients)
    bank = cls.fit_density(forward.bank, calibration)
    query = trace(7, "compound", steps=8)
    ll = bank.log_likelihoods(query)
    inputs = bank.features.inputs(query)
    for i, mode in enumerate(original.MODES):
        np.testing.assert_array_equal(bank.coefficients[mode], before[mode])
        if cls is density.ConditionalDensityBank:
            center, covariance, _ = bank.geometry(inputs, mode)
        else:
            center = bank.local_centers(inputs)[mode]
            covariance = np.repeat(np.linalg.inv(bank.precisions[mode])[None], 8, axis=0)
        prediction = query["states"]+(bank.design(query)@bank.coefficients[mode]+center)*bank.target_scale
        for j in range(8):
            physical_covariance = covariance[j]*bank.target_scale[:, None]*bank.target_scale[None, :]
            expected = multivariate_normal.logpdf(query["next_states"][j], mean=prediction[j], cov=physical_covariance)
            assert ll[j, i] == pytest.approx(expected, rel=1e-10, abs=1e-8)
    altered = deepcopy(query)
    altered["next_states"] += 1.
    np.testing.assert_array_equal(bank.local_centers(inputs)["compound"], bank.local_centers(bank.features.inputs(altered))["compound"])
    if cls is density.ConditionalDensityBank:
        np.testing.assert_array_equal(bank.geometry(inputs, "compound")[1], bank.geometry(bank.features.inputs(altered), "compound")[1])


def test_online_batch_causality_and_calibration_do_not_refit(reference_data):
    forward, calibration, nulls = reference_data
    bank = density.ConditionalDensityBank.fit_density(forward.bank, calibration)
    reference = density.calibrate_bank(bank, nulls)
    assert reference.normal_null_calibration_transitions == 4000
    before = deepcopy(reference.metadata())
    query = trace(8, "high_grav", steps=80, switch=20)
    expected = original.observe_series(query, reference)
    changed = deepcopy(query)
    changed["next_states"][50:] += 10.
    future = original.observe_series(changed, reference)
    for key in expected:
        np.testing.assert_array_equal(expected[key][:50], future[key][:50])
    online = original.OnlineModeLikelihoodReadout(reference)
    risks, ll = [], []
    for i in range(80):
        risks.append(online.observe(query["states"][i], query["executed_actions"][i], query["next_states"][i], i))
        ll.append(np.mean(online.history, axis=0))
    np.testing.assert_allclose(ll, expected["mean_log_likelihoods"], rtol=1e-11, atol=1e-8)
    np.testing.assert_allclose(risks, expected["risk"], rtol=1e-11, atol=1e-8)
    assert reference.metadata() == before
    for mode in original.MODES:
        np.testing.assert_array_equal(reference.bank.coefficients[mode], forward.bank.coefficients[mode])


def test_completed_worker_keeps_global_control_and_server_only_geometry(reference_data, tmp_path, monkeypatch):
    forward, calibration, nulls = reference_data
    refs = {"global": forward,
        "conditional_center": density.calibrate_bank(density.ConditionalMeanBank.fit_density(forward.bank, calibration), nulls),
        "conditional_density": density.calibrate_bank(density.ConditionalDensityBank.fit_density(forward.bank, calibration), nulls)}
    spec = dict(worker.SPEC, steps=2000, switch_step=1000)
    monkeypatch.setattr(worker, "SPEC", spec)
    monkeypatch.setattr(worker, "fit_references", lambda args: refs)
    stored = tmp_path/"checkpoint/traces"
    stored.mkdir(parents=True)
    cells = worker.test_cells(spec)
    records = []
    for c in cells:
        np.savez_compressed(worker.v71.trace_path(stored, c), **trace(c["seed"], c["mode"], switch=1000))
        records.append(dict(**c, actual_transitions=2000, counterfactual_step_calls=10000,
                            environment_step_calls=12000, exact_true_mode_replays=2000, optimizer_updates=0))
    resume = stored.parent/"preflight_latest.pt"
    progress = dict(revision=worker.REVISION, spec=spec, next_cell=6, records=records)
    torch.save(dict(extra_state=dict(progress=progress, loop=None)), resume)
    shared = tmp_path/"shared"
    shared.mkdir()
    torch.save({}, shared/"shared.pt")
    args = argparse.Namespace(checkpoint_dir=stored.parent, result_dir=tmp_path/"results", resume_from=resume,
                              shared_dir=shared, device="cpu")
    def inputs(args, directory):
        assert (args.checkpoint_dir/"conditional_references.pt").is_file()
        return [(dict(group="v72_fresh_frozen", seed=c["seed"], mode=c["mode"], updating=False), worker.v71.trace_path(directory, c)) for c in cells]
    monkeypatch.setattr(worker, "evaluation_inputs", inputs)
    worker.run(args)
    result = json.loads((args.result_dir/"conditional_density.json").read_text())
    assert result["actual_transitions"] == 12000 and result["environment_step_calls"] == 72000
    assert result["optimizer_updates"] == result["new_reference_labels"] == 0
    assert not result["forward_models_refitted"] and len(result["records"]) == 6
    assert set(result["summary"]["v72_fresh_frozen"]) == set(worker.NAMES)
    expected = worker.v68.window_statistics(trace(spec["test"][0]), forward, "normal", 1000)
    assert result["records"][0]["observers"]["global"]["last64"]["mean_coefficient"] == expected["last64"]["mean_coefficient"]
    assert not any(k in json.dumps(result) for k in ('"residuals"', '"coefficients"', '"precisions"', '"oracle_observations"'))
