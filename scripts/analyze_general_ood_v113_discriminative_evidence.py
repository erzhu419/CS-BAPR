"""Paired-mode residual classification with a complete-source held-out split."""
import argparse
import json
from pathlib import Path
import sys
import warnings

import numpy as np
from scipy.special import logsumexp
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_general_ood_v112_matched_errors as v112

v106, v98 = v112.v106, v112.v98
MODES, CONDITIONS = v112.MODES, v112.CONDITIONS
REVISION = "general_ood_v113_discriminative_evidence"
ARMS = ("quadratic_logistic", "nearest_mean")
SETTINGS = dict(C=1., max_iter=1000, tol=1e-4, solver="lbfgs")
SMOKE_WINDOWS = {(135191, "slippery", True, "first_post50"), (135192, "normal", False, "first_post50"),
    (135192, "compound", True, "last64"), (134020, "slippery", True, "last64"),
    (135252, "normal", True, "first_post50"), (135252, "compound", True, "first_post50")}


def selected_rows(groups, smoke):
    if not smoke:
        return np.arange(len(groups))
    return np.sort(np.concatenate([rows[np.linspace(0, len(rows)-1, min(32, len(rows)), dtype=int)]
        for seed in np.unique(groups) for rows in [np.flatnonzero(groups == seed)]]))


def trace_cell(record):
    if record["origin"] == "v110":
        cell = next(c for c in v112.v111.v110.cells(record["seed"])
                    if c["mode"] == record["mode"] and c["updating"] == record["updating"])
        assert cell["group"] == record["group"]
        return cell
    return {k: record[k] for k in ("group", "seed", "mode", "updating")}


def reference_targets(bank, rows):
    inputs = bank.tree.data[rows]*bank.features.input_scale+bank.features.input_mean
    design = bank.features.features(inputs)
    return np.stack([design@bank.coefficients[m]+bank.residuals[m][rows] for m in MODES], axis=1)


def features(observed, predicted):
    residual = observed[:, None, :]-predicted
    flat = residual.reshape(len(residual), -1)
    return np.concatenate((flat, flat**2), axis=1)


def paired_features(targets, predicted):
    n, modes, width = targets.shape
    assert (modes, width) == (5, 11) and predicted.shape == targets.shape
    return features(targets.reshape(n*5, 11), np.repeat(predicted, 5, axis=0)), np.tile(np.arange(5), n)


def seed_weights(groups):
    seeds, counts = np.unique(groups, return_counts=True)
    return np.asarray([len(groups)/(len(seeds)*count) for g in groups for count in [counts[np.searchsorted(seeds, g)]]])


def fit_classifier(x, y, groups):
    assert len(x) == len(y) == 5*len(groups)
    np.testing.assert_array_equal(y, np.tile(np.arange(5), len(groups)))
    weights = np.repeat(seed_weights(groups), 5)
    scaler = StandardScaler().fit(x, sample_weight=weights)
    model = LogisticRegression(**SETTINGS)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(scaler.transform(x), y, sample_weight=weights)
    np.testing.assert_array_equal(model.classes_, np.arange(5))
    metadata = dict(reference_inputs=len(groups), paired_mode_labels=len(y), feature_dimension=x.shape[1],
        source_seeds=np.unique(groups).astype(int).tolist(), iterations=model.n_iter_.tolist(),
        settings=SETTINGS, weighted_mode_fractions=[float(weights[y == i].sum()/weights.sum()) for i in range(5)],
        weighted_seed_fractions={str(int(s)): float(weights[np.repeat(groups == s, 5)].sum()/weights.sum()) for s in np.unique(groups)})
    return (scaler, model), metadata


def scores(fitted, observed, predicted):
    scaler, model = fitted
    logits = model.decision_function(scaler.transform(features(observed, predicted)))
    logits -= logits.max(axis=1, keepdims=True)
    learned = logits-logsumexp(logits, axis=1, keepdims=True)
    nearest = -.5*np.sum((observed[:, None, :]-predicted)**2, axis=2)
    nearest -= nearest.max(axis=1, keepdims=True)
    nearest -= logsumexp(nearest, axis=1, keepdims=True)
    return dict(quadratic_logistic=learned, nearest_mean=nearest)


def statistics(logp, labels):
    labels = np.broadcast_to(np.asarray(labels, dtype=int), (len(logp),))
    rows = np.arange(len(logp))
    other = logp.copy(); other[rows, labels] = -np.inf
    prediction = logp.argmax(axis=1)
    return dict(instantaneous_mode_accuracy=float(np.mean(prediction == labels)),
        mean_true_minus_best_other_score=float(np.mean(logp[rows, labels]-other.max(axis=1))),
        classification_nll=float(-np.mean(logp[rows, labels])),
        mean_true_mode_probability=float(np.exp(logp[rows, labels]).mean()),
        predicted_mode_fractions={m: float(np.mean(prediction == i)) for i, m in enumerate(MODES)})


def expanded_errors(args, family, rows):
    support = family.incumbent.mean.tree.n
    directory = args.v112_dir/"run_checkpoints/formal"
    assert all((directory/f"errors_seed{s}.pt").is_file() for s in np.unique(family.groups))
    old, _ = v112.fit_errors(family, directory)
    errors = {m: np.empty((len(rows), 11)) for m in MODES}
    fits = []
    for seed in np.unique(family.mean_groups):
        selected = np.flatnonzero(family.mean_groups[rows] == seed)
        indices = rows[selected]
        if np.all(indices < support):
            values = {m: old[m][indices] for m in MODES}
        else:
            assert np.all(indices >= support)
            path = args.checkpoint_dir/f"added_errors_seed{seed}.pt"
            if path.is_file():
                saved = torch.load(path, map_location="cpu", weights_only=False)
                assert saved["revision"] == REVISION and saved["smoke"] == args.smoke
                np.testing.assert_array_equal(saved["rows"], indices)
                np.testing.assert_array_equal(saved["mean_groups"], family.mean_groups)
                values = saved["errors"]
            else:
                values, _ = v98.heldout_errors(family.mean, family.mean_groups, seed, indices)
                torch.save(dict(revision=REVISION, smoke=args.smoke, rows=indices,
                    mean_groups=family.mean_groups, errors=values), path)
        for m in MODES:
            assert values[m].shape == (len(indices), 11) and np.isfinite(values[m]).all()
            errors[m][selected] = values[m]
        fits.append(dict(seed=int(seed), error_rows=len(indices),
            mean_training_rows=int(np.sum(family.mean_groups != seed)),
            excluded_mean_rows=int(np.sum(family.mean_groups == seed))))
        print(f"[FEATURE ERRORS] seed{seed}: {len(indices)} rows", flush=True)
    return np.stack([errors[m] for m in MODES], axis=1), fits


def heldout_validation(family, smoke):
    from scripts.v83_cross_seed_covariance import LocalLinearMean

    old = family.incumbent
    rows = selected_rows(family.groups, smoke)
    targets = reference_targets(old.mean, rows)
    predicted = targets-np.stack([old.errors[m][rows] for m in MODES], axis=1)
    x, y = paired_features(targets, predicted)
    fitted, metadata = fit_classifier(x, y, family.groups[rows])
    support = old.mean.tree.n
    validation_rows = selected_rows(family.mean_groups[support:], smoke)+support
    inputs = family.mean.tree.data[validation_rows]*family.mean.features.input_scale+family.mean.features.input_mean
    correction = LocalLinearMean(old.mean).predict(inputs)
    design = family.mean.features.features(inputs)
    predicted = np.stack([design@family.mean.coefficients[m]+correction[m] for m in MODES], axis=1)
    targets = reference_targets(family.mean, validation_rows)
    observed, repeated = targets.reshape(-1, 11), np.repeat(predicted, 5, axis=0)
    evidence = scores(fitted, observed, repeated)
    labels = np.tile(np.arange(5), len(targets))
    groups = family.mean_groups[validation_rows]
    assert not set(groups) & set(family.groups)
    by_seed = {str(int(s)): {a: statistics(logp[np.repeat(groups == s, 5)], labels[np.repeat(groups == s, 5)])
        for a, logp in evidence.items()} for s in np.unique(groups)}
    return dict(training=metadata, validation_source_seeds=np.unique(groups).astype(int).tolist(),
        validation_inputs=len(groups), validation_mode_labels=len(labels),
        heldout_seeds_excluded_from_classifier_scaler_and_all_local_mean_fits=True,
        per_seed=by_seed, aggregate={a: statistics(logp, labels) for a, logp in evidence.items()}), fitted


def summarize(records):
    populations = dict(
        current_failed_updating=[r for r in records if r["origin"] == "v110" and r["updating"] and r["mode"] != "normal"],
        current_normal=[r for r in records if r["origin"] == "v110" and r["mode"] == "normal"],
        current_matched_frozen=[r for r in records if r["origin"] == "v110" and not r["updating"] and r["mode"] != "normal"],
        earlier_six_failures=[r for r in records if r["old_coverage_failure"]],
        earlier_updating_changed=[r for r in records if r["origin"] == "v98" and r["group"].startswith("v95_") and r["updating"] and r["mode"] != "normal"],
        earlier_frozen_changed=[r for r in records if r["origin"] == "v98" and r["group"].startswith("v95_") and not r["updating"] and r["mode"] != "normal"],
        earlier_normal=[r for r in records if r["origin"] == "v98" and r["mode"] == "normal"],
        historical_slippery=[r for r in records if r["group"] == "v62_development_replay" and r["mode"] == "slippery"])
    fields = ("instantaneous_mode_accuracy", "mean_true_minus_best_other_score", "classification_nll", "mean_true_mode_probability")
    return {name: dict(windows=len(rows), candidates={a: {f: float(np.mean([r["candidates"][a][f] for r in rows]))
        for f in fields} for a in ARMS}, controls={c: {a: float(np.mean([r["controls"][c][a]["instantaneous_mode_accuracy"]
        for r in rows])) for a in ("old", "new", "matched")} for c in CONDITIONS})
        for name, rows in populations.items() if rows}


def run(args):
    sys.path.insert(0, str(args.v83_dir))
    from scripts import analyze_general_ood_v83_cross_seed_covariance as source

    torch.set_num_threads(2)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    family, _ = v106.load_family(args, source)
    baseline = json.loads((args.v112_dir/"run_results/formal/matched_errors.json").read_text())
    assert baseline["revision"] == v112.REVISION and baseline["passed"] and not baseline["smoke"]
    assert (baseline["checked_windows"], baseline["checked_transitions"]) == (76, 4248)
    selected = [r for r in baseline["records"] if not args.smoke or
        (r["seed"], r["mode"], r["updating"], r["window"]) in SMOKE_WINDOWS]
    assert len(selected) == (6 if args.smoke else 76)
    assert not set(family.mean_groups) & {r["seed"] for r in selected}
    print("[FIT] source-held-out validation classifier", flush=True)
    validation, validation_model = heldout_validation(family, args.smoke)
    print(f"[VALIDATE] {validation['validation_mode_labels']} labels: {validation['aggregate']}", flush=True)
    rows = selected_rows(family.mean_groups, args.smoke)
    errors, fits = expanded_errors(args, family, rows)
    targets = reference_targets(family.mean, rows)
    x, y = paired_features(targets, targets-errors)
    print(f"[FIT] final classifier: {len(y)} paired mode labels", flush=True)
    fitted, training = fit_classifier(x, y, family.mean_groups[rows])
    torch.save(dict(revision=REVISION, smoke=args.smoke, settings=SETTINGS,
        validation_model=validation_model, final_model=fitted), args.checkpoint_dir/"classifier_models.pt")
    paths = {v112.v105.key(c): p for c, p, _ in v98.evaluation_inputs(args, source)}
    ranges = {w: (a, b) for w, a, b in source.v78.v77.windows()}
    records = []
    for previous in selected:
        cell = trace_cell(previous)
        directory = args.v110_dir/f"run_checkpoints/seed{cell['seed']}/traces"
        path = source.v76.trace_path(directory, cell) if previous["origin"] == "v110" else paths[v112.v105.key(cell)]
        trace = source.v76.v68.load_trace(path)
        start, stop = ranges[previous["window"]]
        query = {k: trace[k][start:stop] for k in v98.v97.KEYS}
        correction = family.correction.predict(family.bank.features.inputs(query))
        predicted = np.stack([family.bank.design(query)@family.bank.coefficients[m]+correction[m] for m in MODES], axis=1)
        observed = family.bank.targets(query)
        truth = MODES.index(cell["mode"])
        rms = float(np.sqrt(np.mean((observed-predicted[:, truth])**2, axis=1)).mean())
        for c in CONDITIONS:
            np.testing.assert_allclose(rms, previous["diagnostics"]["densities"][c]["new"]["true_prediction_error_rms"], rtol=1e-9, atol=1e-8)
        evidence = scores(fitted, observed, predicted)
        records.append(dict(**cell, origin=previous["origin"], window=previous["window"], transitions=stop-start,
            old_coverage_failure=previous["old_coverage_failure"],
            candidates={a: statistics(logp, truth) for a, logp in evidence.items()},
            controls={c: {a: previous["diagnostics"]["densities"][c][a] for a in ("old", "new", "matched")} for c in CONDITIONS}))
        print(f"[PILOT] {len(records)}/{len(selected)} {v112.v105.key(cell)} {previous['window']}", flush=True)
    transitions = sum(r["transitions"] for r in records)
    assert transitions == (328 if args.smoke else 4248)
    result = dict(revision=REVISION, passed=True, smoke=args.smoke, checked_windows=len(records),
        checked_transitions=transitions, validation=validation, training=training, feature_error_fits=fits,
        unchanged_expanded_mean_rms_comparisons=3*len(records), control_source_revision=v112.REVISION,
        controls_reused_from_scalar_result=True, control_likelihoods_recomputed=False,
        records=records, summary=summarize(records),
        score_units="normalized discriminative log probabilities; not physical Gaussian densities",
        feature_formula="concatenated target-scale-normalized residuals against five expanded local means, then their coordinate squares",
        training_weights="equal source seed, equal mode within every matched state/action",
        reference_inputs=family.mean.tree.n, reference_seed_count=len(np.unique(family.mean_groups)),
        environment_steps=0, rl_optimizer_updates=0, classifier_fits=2, hyperparameter_search=False,
        oracle_observations_used_for_query_scores=False, posterior_replayed=False, normal_thresholds_recalibrated=False,
        readout_controls_actor=False, readout_adopted=False,
        decision="partial-training-smoke-only" if args.smoke else "development-discriminative-pilot-not-adoption",
        limitations=["Known simulator-mode counterfactual labels are available in reference data, not deployment queries.",
            "Validation holds out two reference source seeds; inspected evaluation windows are development evidence.",
            "Frozen global forward model and target scale are retained, not cross-refitted.",
            "No full-history belief, normal-null calibration, actor or return evaluation."])
    args.result_dir.mkdir(parents=True, exist_ok=True)
    path = args.result_dir/("native_smoke.json" if args.smoke else "discriminative_evidence.json")
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] {REVISION}: {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("v112-dir", "v110-dir", "v104-dir", "v98-dir", "v97-dir", "v95-dir", "v94-dir", "v83-dir",
                 "v78-dir", "v76-dir", "v74-dir", "v73-dir", "v72-dir", "v71-dir", "v69-dir", "v70-dir",
                 "v61-dir", "v62-dir", "checkpoint-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
