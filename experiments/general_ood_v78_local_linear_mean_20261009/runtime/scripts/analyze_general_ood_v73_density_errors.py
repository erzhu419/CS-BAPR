"""Scalar-only density error attribution on already collected v73 traces."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from scripts import probe_general_ood_v73_recursive_mode as v73


def distribution(values):
    return dict(mean=float(np.mean(values)), minimum=float(np.min(values)),
                median=float(np.median(values)), p10=float(np.quantile(values, .1)),
                p90=float(np.quantile(values, .9)))


def analyze(trace, bank, mode, start, stop):
    query = {k: trace[k][start:stop] for k in ("states", "executed_actions", "next_states")}
    inputs = bank.features.inputs(query)
    design, targets = bank.design(query), bank.targets(query)
    energies, determinants, predictions = [], [], []
    for candidate in v73.MODES:
        centers, covariance, _ = bank.geometry(inputs, candidate)
        prediction = design@bank.coefficients[candidate]+centers
        errors = targets-prediction
        sign, logdet = np.linalg.slogdet(covariance)
        assert np.all(sign > 0)
        energy = np.einsum("ni,ni->n", errors, np.linalg.solve(covariance, errors[..., None])[..., 0])
        energies.append(energy)
        determinants.append(logdet)
        predictions.append(query["states"]+prediction*bank.target_scale)
    energies, determinants = np.column_stack(energies), np.column_stack(determinants)
    constant = len(bank.target_scale)*np.log(2*np.pi)+2*np.log(bank.target_scale).sum()
    likelihoods = -.5*(energies+determinants+constant)
    np.testing.assert_allclose(likelihoods, bank.log_likelihoods(query), rtol=1e-12, atol=1e-10)
    truth = v73.MODES.index(mode)
    others = likelihoods.copy()
    others[:, truth] = -np.inf
    competitor = others.argmax(axis=1)
    rows = np.arange(stop-start)
    energy_advantage = .5*(energies[rows, competitor]-energies[:, truth])
    volume_advantage = .5*(determinants[rows, competitor]-determinants[:, truth])
    margins = likelihoods[:, truth]-others.max(axis=1)
    np.testing.assert_allclose(energy_advantage+volume_advantage, margins, rtol=1e-11, atol=1e-10)
    oracle = trace["oracle_observations"][start:stop]
    gaps = np.sqrt(np.mean(((oracle-query["next_states"][:, None, :])/bank.target_scale)**2, axis=2))
    gaps[:, truth] = np.inf
    nearest_gap = gaps.min(axis=1)
    mean_error = np.sqrt(np.mean(((predictions[truth]-query["next_states"])/bank.target_scale)**2, axis=1))
    unique = np.sum(np.all(oracle == query["next_states"][:, None, :], axis=2), axis=1) == 1
    radius = bank.tree.query((inputs-bank.features.input_mean)/bank.features.input_scale, k=64)[0][:, -1]
    return dict(transitions=stop-start, raw_mode_accuracy=float(np.mean(likelihoods.argmax(axis=1) == truth)),
        true_minus_best_other_log_density=distribution(margins),
        energy_advantage=distribution(energy_advantage), covariance_volume_advantage=distribution(volume_advantage),
        best_other_mode_fractions={m: float(np.mean(competitor == i)) for i, m in enumerate(v73.MODES)},
        wrong_offsets=np.flatnonzero(likelihoods.argmax(axis=1) != truth).tolist(),
        unique_true_observation_template_fraction=float(unique.mean()),
        nearest_other_physical_gap=distribution(nearest_gap), conditional_mean_error=distribution(mean_error),
        mean_error_exceeds_nearest_physical_gap_fraction=float(np.mean(mean_error > nearest_gap)),
        neighbor_radius=distribution(radius))


def run(args):
    torch.set_num_threads(2)
    saved = torch.load(args.runtime/"run_checkpoints/preflight/temporal_references.pt", map_location="cpu", weights_only=False)
    assert saved["revision"] == v73.REVISION and saved["spec"] == v73.SPEC
    bank = saved["references"]["window16"].bank
    rows = []
    for seed in v73.SPEC["test"]:
        for mode in v73.MODES[1:]:
            path = v73.v71.trace_path(args.runtime/"run_checkpoints/preflight/traces", dict(role="test", seed=seed, mode=mode))
            trace = v73.v68.load_trace(path)
            windows = {name: analyze(trace, bank, mode, a, b) for name, a, b in (
                ("first_post50", v73.SPEC["switch_step"], v73.SPEC["switch_step"]+50),
                ("last64", v73.SPEC["steps"]-64, v73.SPEC["steps"]))}
            rows.append(dict(seed=seed, mode=mode, windows=windows))
    result = dict(revision="general_ood_v73_density_error_attribution", source_revision=v73.REVISION,
                  environment_steps=0, optimizer_updates=0, reference_refitted=False, records=rows,
                  use="development attribution, not a new confirmation test")
    args.result_dir.mkdir(parents=True, exist_ok=True)
    path = args.result_dir/"density_error_attribution.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] scalar attribution: {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    run(parser.parse_args())
