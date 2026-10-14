"""Separate mean-direction and covariance effects on fixed development windows."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

REVISION = "general_ood_v86_directional_errors"


def pair_diagnostics(error, delta, true_cov, other_cov, reference_errors, oracle_delta=None):
    true_direction = np.linalg.solve(true_cov, delta[..., None])[..., 0]
    separation = np.einsum("ni,ni->n", delta, true_direction)
    other_error = error+delta
    true_energy = np.einsum("ni,ni->n", error, np.linalg.solve(true_cov, error[..., None])[..., 0])
    other_energy = np.einsum("ni,ni->n", other_error, np.linalg.solve(other_cov, other_error[..., None])[..., 0])
    common_other_energy = np.einsum("ni,ni->n", other_error, np.linalg.solve(true_cov, other_error[..., None])[..., 0])
    common_margin = np.einsum("ni,ni->n", error, true_direction)+.5*separation
    shape = .5*(other_energy-common_other_energy)
    volume = .5*(np.linalg.slogdet(other_cov)[1]-np.linalg.slogdet(true_cov)[1])
    energy_margin = .5*(other_energy-true_energy)
    margin = energy_margin+volume
    np.testing.assert_allclose(common_margin+shape, energy_margin, rtol=1e-9, atol=1e-8)
    result = dict(mean_log_density_margin=float(margin.mean()),
        mean_energy_advantage=float(energy_margin.mean()), mean_covariance_volume_advantage=float(volume.mean()),
        positive_margin_fraction=float(np.mean(margin > 0)),
        mean_common_covariance_margin=float(common_margin.mean()),
        mean_covariance_shape_adjustment=float(shape.mean()),
        common_covariance_wrong_fraction=float(np.mean(common_margin < 0)),
        shape_flips_energy_fraction=float(np.mean((common_margin > 0)&(energy_margin < 0))),
        volume_flips_margin_fraction=float(np.mean((energy_margin > 0)&(margin < 0))),
        mean_separation_energy=float(separation.mean()))
    valid = separation > 0
    result["nonzero_separation_fraction"] = float(valid.mean())
    projection_keys = ("query_toward_other_mean", "reference_toward_other_mean", "query_projection_second_moment",
        "reference_projection_second_moment", "query_to_reference_second_moment_ratio",
        "query_exceeds_reference95_fraction", "reference_common_covariance_wrong_fraction")
    result.update({k: None for k in projection_keys})
    if np.any(valid):
        # Positive projection points from the true predicted mean toward the other mean.
        direction = -true_direction[valid]/np.sqrt(separation[valid, None])
        query_projection = np.einsum("ni,ni->n", error[valid], direction)
        reference_projection = np.einsum("nkd,nd->nk", reference_errors[valid], direction)
        query_second = float(np.mean(query_projection**2))
        reference_second = float(np.mean(reference_projection**2))
        result.update(query_toward_other_mean=float(query_projection.mean()),
            reference_toward_other_mean=float(reference_projection.mean()),
            query_projection_second_moment=query_second, reference_projection_second_moment=reference_second,
            query_to_reference_second_moment_ratio=query_second/reference_second if reference_second > 0 else None,
            query_exceeds_reference95_fraction=float(np.mean(query_projection > np.quantile(reference_projection, .95, axis=1))),
            reference_common_covariance_wrong_fraction=float(np.mean(
                reference_projection > .5*np.sqrt(separation[valid, None]))))
    result.update(mean_oracle_separation_energy=None, mean_predicted_oracle_direction_cosine=None)
    if oracle_delta is not None:
        oracle_energy = np.einsum("ni,ni->n", oracle_delta,
            np.linalg.solve(true_cov, oracle_delta[..., None])[..., 0])
        available = valid&(oracle_energy > 0)
        result["mean_oracle_separation_energy"] = float(oracle_energy.mean())
        if np.any(available):
            cosine = np.einsum("ni,ni->n", oracle_delta, true_direction)
            result["mean_predicted_oracle_direction_cosine"] = float(np.mean(
                cosine[available]/np.sqrt(separation[available]*oracle_energy[available])))
    return result


def verify_source(result, expected):
    for key in ("mean_true_prediction_error_rms", "mean_true_minus_best_other_log_density",
                "mean_energy_advantage", "mean_covariance_volume_advantage", "raw_mode_accuracy"):
        np.testing.assert_allclose(result[key], expected[key], rtol=1e-9, atol=1e-8)
    for mode, pair in result["pairs"].items():
        for key in ("mean_log_density_margin", "mean_energy_advantage", "mean_covariance_volume_advantage", "positive_margin_fraction"):
            np.testing.assert_allclose(pair[key], expected["fixed_competitor"][mode][key], rtol=1e-9, atol=1e-8)


def diagnose(query, bank, density, source_errors, mode, prior):
    from scripts.v77_mean_covariance import density_terms

    modes = prior.v78.v77.MODES
    inputs, design, targets = bank.features.inputs(query), bank.design(query), bank.targets(query)
    centers, mixed_covariance, _ = density.geometry(inputs)
    covariances = {"linear_mean_updating_cov": {m: bank.geometry(inputs, m)[1] for m in modes},
                   prior.MIX_CONDITION: mixed_covariance}
    errors = {m: targets-design@bank.coefficients[m]-centers[m] for m in modes}
    _, indices = bank.tree.query((inputs-bank.features.input_mean)/bank.features.input_scale, k=64)
    reference_errors = source_errors[mode][indices]
    truth = modes.index(mode)
    output = {}
    for condition, covariance in covariances.items():
        terms = [density_terms(errors[m], covariance[m], bank.target_scale) for m in modes]
        energy, logdet, ll = (np.column_stack([t[i] for t in terms]) for i in range(3))
        other = ll.copy()
        other[:, truth] = -np.inf
        competitor = other.argmax(axis=1)
        rows = np.arange(len(inputs))
        result = dict(mean_true_prediction_error_rms=float(np.sqrt(np.mean(errors[mode]**2, axis=1)).mean()),
            mean_true_minus_best_other_log_density=float((ll[:, truth]-other.max(axis=1)).mean()),
            mean_energy_advantage=float(.5*(energy[rows, competitor]-energy[:, truth]).mean()),
            mean_covariance_volume_advantage=float(.5*(logdet[rows, competitor]-logdet[:, truth]).mean()),
            raw_mode_accuracy=float(np.mean(ll.argmax(axis=1) == truth)),
            best_other_fractions={m: float(np.mean(competitor == i)) for i, m in enumerate(modes)}, pairs={})
        for c, candidate in enumerate(modes):
            if c == truth:
                continue
            oracle_delta = None
            if "oracle_observations" in query:
                oracle_delta = (query["oracle_observations"][:, truth]-query["oracle_observations"][:, c])/bank.target_scale
            result["pairs"][candidate] = pair_diagnostics(errors[mode], errors[candidate]-errors[mode],
                covariance[mode], covariance[candidate], reference_errors, oracle_delta)
        output[condition] = result
    return output


def run(args):
    sys.path.insert(0, str(args.v83_dir))
    from scripts import analyze_general_ood_v83_cross_seed_covariance as prior

    torch.set_num_threads(2)
    saved = torch.load(args.v76_dir/"run_checkpoints/formal/behavior_coverage_references.pt", map_location="cpu", weights_only=False)
    scores = json.loads((args.v83_dir/"run_results/cross-seed-covariance/cross_seed_covariance.json").read_text())
    assert saved["revision"] == prior.v76.REVISION and scores["revision"] == prior.REVISION
    assert saved["spec"] == scores["spec"] == prior.v76.SPEC
    bank = saved["references"]["updating_mixed"].bank
    cache = torch.load(args.v83_dir/"run_checkpoints/cross-seed-covariance/cross_seed_errors.pt", map_location="cpu", weights_only=False)
    assert cache["revision"] == prior.REVISION and cache["rows"] == bank.density_transitions
    weight = scores["reference_selection"]["selection"]["raw_weight"]
    assert scores["selected_condition"] == prior.MIX_CONDITION
    density = prior.CrossSeedDensityBank(bank, cache["errors"], weight)
    inputs = prior.v76.evaluation_inputs(args, args.v76_dir/"run_checkpoints/formal/traces")
    index = {prior.v78.v77.cell_key(r): r for r in scores["records"]}
    assert len(inputs) == len(index) == 65
    assert {prior.v78.v77.cell_key(prior.v78.replay_cell(c)) for c, _ in inputs} == set(index)
    records, checked = [], 0
    for cell, path in inputs:
        trace = prior.v76.v68.load_trace(path)
        previous = index[prior.v78.v77.cell_key(prior.v78.replay_cell(cell))]
        windows = {}
        for name, start, stop in prior.v78.v77.windows():
            query = {k: trace[k][start:stop] for k in prior.v76.v74.KEYS}
            if "oracle_observations" in trace:
                query["oracle_observations"] = trace["oracle_observations"][start:stop]
            conditions = diagnose(query, bank, density, cache["errors"], cell["mode"], prior)
            for c, values in conditions.items():
                verify_source(values, previous["attribution"][c][name])
                values["saved_recursive_accuracy"] = previous["observers"][c][name]["mode_accuracy"]
                values["saved_recursive_margin"] = previous["observers"][c][name]["mean_true_minus_best_other_score"]
            windows[name] = dict(transitions=stop-start, oracle_available="oracle_observations" in query, conditions=conditions)
            checked += stop-start
        records.append(dict(**prior.v78.replay_cell(cell), windows=windows))
        print(f"[DIRECTION] {len(records)}/{len(inputs)} {cell['group']} seed{cell['seed']} {cell['mode']}", flush=True)
    result = dict(revision=REVISION, source_revision=prior.REVISION, raw_weight=weight,
        evaluation_paths=len(records), windows=2*len(records), checked_transitions=checked,
        conditions=["linear_mean_updating_cov", prior.MIX_CONDITION], records=records,
        verification=dict(all_source_density_and_error_statistics_match=True, fixed_pair_comparisons=8*2*len(records)),
        reference_refitted=False, weight_refitted=False, temporal_rules_changed=False, readout_controls_actor=False,
        environment_steps=0, optimizer_updates=0, independent_confirmation=False,
        decision="development-directional-attribution-not-confirmation",
        limitations=["Previously inspected paths from one source training seed and known modes, not new RL return evidence.",
            "Neighbor reference errors come from smaller whole-seed-held-out mean fits; projecting them at a query is an error-transport diagnostic.",
            "Component algebra and associations do not identify a unique causal repair; reference quantiles are not confidence intervals."])
    args.result_dir.mkdir(parents=True, exist_ok=True)
    path = args.result_dir/"directional_errors.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] {REVISION}: {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("v83-dir", "v76-dir", "v74-dir", "v73-dir", "v72-dir", "v71-dir", "v69-dir", "v70-dir", "v61-dir", "v62-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    run(parser.parse_args())
