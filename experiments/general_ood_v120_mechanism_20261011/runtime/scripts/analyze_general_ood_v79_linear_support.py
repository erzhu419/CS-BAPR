"""Server-only attribution of the saved v78 local-linear predictions."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from scripts import analyze_general_ood_v78_local_linear_mean as v78
from scripts.probe_general_ood_v42 import RIDGE
from scripts.v65_state_action_readout import NEIGHBORS
from scripts.v77_mean_covariance import MODES, check_shared_forward, density_terms
from scripts.v78_local_linear_mean import LocalLinearMean

REVISION = "general_ood_v79_linear_support"


def local_prediction(bank, inputs):
    query = (inputs-bank.features.input_mean)/bank.features.input_scale
    distances, indices = bank.tree.query(query, k=NEIGHBORS)
    paired = np.concatenate([bank.residuals[m] for m in MODES], axis=1)
    constants, predictions, fit_errors, loo_errors, rows = [], [], [], [], []
    for q, neighbors, radius in zip(query, indices, distances):
        offsets, targets = bank.tree.data[neighbors]-q, paired[neighbors]
        displacement, center = offsets.mean(axis=0), targets.mean(axis=0)
        x, y = offsets-displacement, targets-center
        gram = x.T@x+NEIGHBORS*RIDGE*np.eye(x.shape[1])
        slopes = np.linalg.solve(gram, x.T@y)
        direction = np.linalg.solve(gram, displacement)
        weights = np.full(NEIGHBORS, 1/NEIGHBORS)-x@direction
        prediction = center-displacement@slopes
        np.testing.assert_allclose(prediction, weights@targets, rtol=1e-10, atol=1e-10)
        fitted_error = y-x@slopes
        hat_diagonal = 1/NEIGHBORS+np.einsum("ij,ji->i", x, np.linalg.solve(gram, x.T))
        constants.append(center)
        predictions.append(prediction)
        fit_errors.append(np.sqrt(np.mean(fitted_error.reshape(NEIGHBORS, len(MODES), -1)**2, axis=(0, 2))))
        loo_errors.append(np.sqrt(np.mean((fitted_error/(1-hat_diagonal[:, None])).reshape(NEIGHBORS, len(MODES), -1)**2, axis=(0, 2))))
        rows.append(dict(nearest_radius=radius[0], radius64=radius[-1],
            state_neighbor_rms=np.sqrt(np.mean(offsets[:, :11]**2)),
            action_neighbor_rms=np.sqrt(np.mean(offsets[:, 11:]**2)),
            centroid_distance=np.linalg.norm(displacement),
            ridge_query_leverage=1/NEIGHBORS+displacement@direction,
            prediction_weight_l1=np.abs(weights).sum(),
            negative_weight_fraction=np.mean(weights < 0), prediction_weight_squared_sum=weights@weights))
    return dict(constant=dict(zip(MODES, np.split(np.asarray(constants), len(MODES), axis=1))),
        linear=dict(zip(MODES, np.split(np.asarray(predictions), len(MODES), axis=1))),
        fit_rms=np.asarray(fit_errors), loo_rms=np.asarray(loo_errors),
        support={k: np.asarray([r[k] for r in rows]) for k in rows[0]})


def energy_change(error, correction, covariance):
    solved = np.linalg.solve(covariance, correction[..., None])[..., 0]
    alignment = 2*np.einsum("ni,ni->n", error, solved)
    penalty = np.einsum("ni,ni->n", correction, solved)
    before = density_terms(error, covariance, np.ones(error.shape[1]))[0]
    after = density_terms(error-correction, covariance, np.ones(error.shape[1]))[0]
    np.testing.assert_allclose(before-after, alignment-penalty, rtol=1e-9, atol=1e-8)
    return dict(alignment=alignment, penalty=penalty, gain=alignment-penalty,
        coordinate_gain=(2*error-correction)*solved)


def statistics(values):
    return dict(mean=float(np.mean(values)), median=float(np.median(values)),
        p90=float(np.quantile(values, .9)), maximum=float(np.max(values)))


def correlation(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def diagnose(query, banks, mode):
    bank = banks["updating"]
    inputs, design, targets = bank.features.inputs(query), bank.design(query), bank.targets(query)
    prediction = local_prediction(bank, inputs)
    truth = MODES.index(mode)
    error_constant = {m: targets-design@bank.coefficients[m]-prediction["constant"][m] for m in MODES}
    error_linear = {m: targets-design@bank.coefficients[m]-prediction["linear"][m] for m in MODES}
    correction = {m: prediction["linear"][m]-prediction["constant"][m] for m in MODES}
    details = {}
    for source in ("frozen", "updating"):
        covariances = {m: banks[source].geometry(inputs, m)[1] for m in MODES}
        terms = {kind: {} for kind in ("constant", "linear")}
        for kind, errors in (("constant", error_constant), ("linear", error_linear)):
            computed = [density_terms(errors[m], covariances[m], bank.target_scale) for m in MODES]
            terms[kind] = {k: np.column_stack([t[i] for t in computed])
                for i, k in enumerate(("energy", "logdet", "likelihoods"))}
        changes = {m: energy_change(error_constant[m], correction[m], covariances[m]) for m in MODES}
        ll = terms["linear"]["likelihoods"]
        other = ll.copy()
        other[:, truth] = -np.inf
        competitor, row = other.argmax(axis=1), np.arange(len(inputs))
        margin = ll[:, truth]-other.max(axis=1)
        condition = "linear_mean_"+source+"_cov"
        details[condition] = dict(
            mean_true_prediction_error_rms=float(np.sqrt(np.mean(error_linear[mode]**2, axis=1)).mean()),
            mean_constant_prediction_error_rms=float(np.sqrt(np.mean(error_constant[mode]**2, axis=1)).mean()),
            mean_true_minus_best_other_log_density=float(margin.mean()),
            mean_energy_advantage=float(.5*(terms["linear"]["energy"][row, competitor]-terms["linear"]["energy"][:, truth]).mean()),
            mean_covariance_volume_advantage=float(.5*(terms["linear"]["logdet"][row, competitor]-terms["linear"]["logdet"][:, truth]).mean()),
            raw_mode_accuracy=float(np.mean(ll.argmax(axis=1) == truth)),
            true_correction_rms=statistics(np.sqrt(np.mean(correction[mode]**2, axis=1))),
            local_reference_fit_rms=statistics(prediction["fit_rms"][:, truth]),
            local_reference_loo_rms=statistics(prediction["loo_rms"][:, truth]),
            true_energy_alignment=statistics(changes[mode]["alignment"]),
            true_energy_penalty=statistics(changes[mode]["penalty"]),
            true_energy_gain=statistics(changes[mode]["gain"]),
            energy_improved_fraction=float(np.mean(changes[mode]["gain"] > 0)),
            mean_coordinate_energy_gain=changes[mode]["coordinate_gain"].mean(axis=0).tolist(),
            support_correlation_with_energy_gain={k: correlation(v, changes[mode]["gain"])
                for k, v in prediction["support"].items()},
            fixed_competitor={m: dict(
                constant_mean_margin=float((terms["constant"]["likelihoods"][:, truth]-terms["constant"]["likelihoods"][:, i]).mean()),
                linear_mean_margin=float((ll[:, truth]-ll[:, i]).mean()),
                mean_true_energy_gain=float(changes[mode]["gain"].mean()),
                mean_competitor_energy_gain=float(changes[m]["gain"].mean()))
                for i, m in enumerate(MODES) if i != truth})
    return dict(transitions=len(inputs), support={k: statistics(v) for k, v in prediction["support"].items()},
        conditions=details), prediction


def verify_source(result, previous):
    for condition, values in result["conditions"].items():
        for key in ("mean_true_prediction_error_rms", "mean_true_minus_best_other_log_density",
                    "mean_energy_advantage", "mean_covariance_volume_advantage", "raw_mode_accuracy"):
            np.testing.assert_allclose(values[key], previous["attribution"][condition][key], rtol=1e-9, atol=1e-8)


def run(args):
    torch.set_num_threads(2)
    saved = torch.load(args.v76_dir/"run_checkpoints/formal/behavior_coverage_references.pt", map_location="cpu", weights_only=False)
    scores = json.loads((args.v78_dir/"run_results/mean-correction/local_linear_mean.json").read_text())
    assert saved["revision"] == v78.v76.REVISION and scores["revision"] == v78.REVISION
    assert saved["spec"] == scores["spec"] == v78.v76.SPEC
    banks = {"frozen": saved["references"]["frozen_mixed"].bank,
             "updating": saved["references"]["updating_mixed"].bank}
    check_shared_forward(banks)
    inputs = v78.v76.evaluation_inputs(args, args.v76_dir/"run_checkpoints/formal/traces")
    index = {v78.v77.cell_key(r): r for r in scores["records"]}
    assert len(inputs) == len(index) == 65
    assert {v78.v77.cell_key(v78.replay_cell(c)) for c, _ in inputs} == set(index)
    if args.smoke:
        inputs = [(c, p) for c, p in inputs if (c["group"] == "v76_fresh_updating" and c["mode"] == "compound" and c["seed"] == 135180)
                  or (c["group"] == "v61_development_replay" and c["mode"] == "heavy")]
        assert len(inputs) == 2
    records = []
    for cell, path in inputs:
        trace = v78.v76.v68.load_trace(path)
        assert len(trace["states"]) == v78.v76.SPEC["steps"]
        previous = index[v78.v77.cell_key(v78.replay_cell(cell))]
        windows = {}
        for window, a, b in v78.v77.windows():
            query = {k: trace[k][a:b] for k in v78.v76.v74.KEYS}
            result, prediction = diagnose(query, banks, cell["mode"])
            verify_source(result, {"attribution": {c: values[window] for c, values in previous["attribution"].items()}})
            if args.smoke:
                expected = LocalLinearMean(banks["updating"]).predict(banks["updating"].features.inputs(query))
                for mode in MODES:
                    np.testing.assert_allclose(prediction["linear"][mode], expected[mode], rtol=1e-10, atol=1e-10)
            for condition, values in result["conditions"].items():
                values["saved_recursive_mode_accuracy"] = previous["observers"][condition][window]["mode_accuracy"]
            windows[window] = result
        records.append(dict(**v78.replay_cell(cell), windows=windows))
        print(f"[SUPPORT] {len(records)}/{len(inputs)} {cell['group']} seed{cell['seed']} {cell['mode']} updating={cell['updating']}", flush=True)
    result = dict(revision=REVISION, source_revision=v78.REVISION, evaluation_paths=len(records),
        checked_transitions=sum(w["transitions"] for r in records for w in r["windows"].values()),
        passed=True, source_prediction_and_density_statistics_match=True,
        native_predictions_match_v78=True if args.smoke else None, environment_steps=0, optimizer_updates=0,
        reference_refitted=False, temporal_rules_changed=False, readout_controls_actor=False,
        ridge_alpha=NEIGHBORS*RIDGE, records=records,
        decision="development-support-attribution-not-confirmation",
        limitations=["Already inspected development replays, one source training seed and mild known modes.",
            "Query leverage and signed weights describe reference extrapolation, not causal proof of failure.",
            "Local reference leave-one-out errors reuse the query-selected neighborhood; they are not held-out deployment errors.",
            "No new model selection, calibration, actor feedback or RL return evidence."])
    args.result_dir.mkdir(parents=True, exist_ok=True)
    path = args.result_dir/("native_smoke.json" if args.smoke else "linear_support.json")
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] {REVISION}: {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("v78-dir", "v76-dir", "v74-dir", "v73-dir", "v72-dir", "v71-dir", "v69-dir", "v70-dir", "v61-dir", "v62-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
