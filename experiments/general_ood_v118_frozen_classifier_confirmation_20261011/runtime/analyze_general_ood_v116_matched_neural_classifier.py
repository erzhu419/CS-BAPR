"""Match v113's classifier to frozen v115 means using source-cross-fitted errors."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_general_ood_v115_shared_residual_mean as v115

v114, v113, MODES = v115.v114, v115.v113, v115.MODES
REVISION = "general_ood_v116_matched_neural_classifier"
ARMS = ("matched_classifier", "original_classifier", "nearest_mean")


def crossfit_errors(bank, groups, device, smoke, directory):
    from scripts.v83_cross_seed_covariance import subset_bank

    directory.mkdir(parents=True, exist_ok=True)
    rows = v113.selected_rows(groups, smoke)
    errors = np.empty((len(rows), 5, 11))
    fits = []
    for seed in np.unique(groups):
        selected = np.flatnonzero(groups[rows] == seed)
        indices, keep = rows[selected], groups != seed
        path = directory/f"errors_seed{seed}.pt"
        cached = path.is_file()
        if cached:
            saved = torch.load(path, map_location="cpu", weights_only=False)
            assert saved["revision"] == REVISION and saved["smoke"] == smoke
            assert saved["mean_settings"] == v115.SETTINGS
            np.testing.assert_array_equal(saved["groups"], groups)
            np.testing.assert_array_equal(saved["rows"], indices)
            values, training = saved["errors"], saved["training"]
        else:
            reduced = subset_bank(bank, keep)
            model, training = v115.fit_mean(reduced, groups[keep], device, smoke, f"{directory.name}_exclude{seed}")
            inputs = bank.tree.data[indices]*bank.features.input_scale+bank.features.input_mean
            prediction = v115.residual_prediction(model, bank, inputs)
            values = np.stack([bank.residuals[m][indices] for m in MODES], axis=1)-prediction
            torch.save(dict(revision=REVISION, smoke=smoke, mean_settings=v115.SETTINGS,
                groups=groups, rows=indices, errors=values, training=training), path)
            del model
        assert values.shape == (len(indices), 5, 11) and np.isfinite(values).all()
        assert training["source_seeds"] == np.unique(groups[keep]).astype(int).tolist()
        assert int(seed) not in training["source_seeds"]
        assert training["reference_inputs"] == len(v113.selected_rows(groups[keep], smoke))
        errors[selected] = values
        fits.append(dict(seed=int(seed), heldout_rows=len(indices), training_inputs=training["reference_inputs"],
            training_source_seeds=training["source_seeds"], epochs=training["epochs"],
            optimizer_updates=training["optimizer_updates"], final_online_training_mse=training["final_epoch_training_mse"],
            reused_cache=cached))
        print(f"[CROSSFIT] {directory.name} seed{seed}: {len(indices)} excluded-source errors", flush=True)
    return rows, errors, fits


def fit_reference_classifier(bank, groups, device, smoke, directory):
    rows, errors, fits = crossfit_errors(bank, groups, device, smoke, directory)
    targets = v113.reference_targets(bank, rows)
    x, labels = v113.paired_features(targets, targets-errors)
    print(f"[CLASSIFIER] {directory.name}: {len(labels)} paired labels", flush=True)
    fitted, metadata = v113.fit_classifier(x, labels, groups[rows])
    return fitted, metadata, fits


def load_frozen_means(path, width):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert saved["revision"] == v115.REVISION and not saved["smoke"]
    assert saved["settings"] == v115.SETTINGS
    models = {}
    for name, training_name in (("validation_model", "validation_training"), ("final_model", "final_training")):
        assert saved[training_name]["epochs"] == v115.SETTINGS["epochs"]
        model = v115.SharedResidualMean(width)
        model.load_state_dict(saved[name])
        models[name] = model.eval()
    return models, saved


def score_arms(matched, original, observed, predicted):
    new, old = v113.scores(matched, observed, predicted), v113.scores(original, observed, predicted)
    np.testing.assert_array_equal(new["nearest_mean"], old["nearest_mean"])
    return dict(matched_classifier=new["quadratic_logistic"], original_classifier=old["quadratic_logistic"],
        nearest_mean=new["nearest_mean"])


def validation_scores(family, model, matched, original, smoke):
    support = family.incumbent.mean.tree.n
    rows = v113.selected_rows(family.mean_groups[support:], smoke)+support
    groups = family.mean_groups[rows]
    assert not set(groups) & set(family.groups)
    inputs = family.mean.tree.data[rows]*family.mean.features.input_scale+family.mean.features.input_mean
    design = family.mean.features.features(inputs)
    predicted = np.stack([design@family.mean.coefficients[m] for m in MODES], axis=1)
    predicted += v115.residual_prediction(model, family.mean, inputs)
    targets = v113.reference_targets(family.mean, rows)
    scores = score_arms(matched, original, targets.reshape(-1, 11), np.repeat(predicted, 5, axis=0))
    labels = np.tile(np.arange(5), len(rows))
    return dict(validation_source_seeds=np.unique(groups).astype(int).tolist(), validation_inputs=len(rows),
        validation_mode_labels=len(labels), aggregate={a: v113.statistics(p, labels) for a, p in scores.items()},
        per_seed={str(int(s)): {a: v113.statistics(p[np.repeat(groups == s, 5)], labels[np.repeat(groups == s, 5)])
            for a, p in scores.items()} for s in np.unique(groups)},
        mean_prediction_quality=v115.prediction_statistics(targets, predicted),
        heldout_seeds_excluded_from_classifier_scaler_and_all_crossfit_mean_training=True)


def summarize(records):
    fields = ("instantaneous_mode_accuracy", "mean_true_minus_best_other_score", "classification_nll")
    return {name: dict(windows=len(rows), arms={a: {f: float(np.mean([r["arms"][a][f] for r in rows]))
        for f in fields} for a in ARMS}, true_prediction_error_rms=float(np.mean([r["true_prediction_error_rms"] for r in rows])))
        for name, rows in v114.cohorts(records).items() if rows}


def run(args):
    sys.path.insert(0, str(args.v83_dir))
    from scripts import analyze_general_ood_v83_cross_seed_covariance as source

    torch.set_num_threads(2)
    family, _ = v113.v106.load_family(args, source)
    baseline = json.loads((args.v115_dir/"run_results/formal/shared_residual_mean.json").read_text())
    assert baseline["revision"] == v115.REVISION and baseline["passed"] and not baseline["smoke"]
    means, mean_saved = load_frozen_means(args.v115_dir/"run_checkpoints/formal/mean_models.pt", len(family.bank.features.input_mean))
    for name, groups in (("validation_training", family.groups), ("final_training", family.mean_groups)):
        assert mean_saved[name]["source_seeds"] == np.unique(groups).astype(int).tolist()
        assert mean_saved[name]["reference_inputs"] == len(groups)
    original = torch.load(args.v113_dir/"run_checkpoints/formal/classifier_models.pt", map_location="cpu", weights_only=False)
    assert original["revision"] == v113.REVISION and not original["smoke"]
    validation_model, validation_training, outer_fits = fit_reference_classifier(
        family.incumbent.mean, family.groups, args.device, args.smoke, args.checkpoint_dir/"outer")
    validation = validation_scores(family, means["validation_model"], validation_model, original["validation_model"], args.smoke)
    validation["training"] = validation_training
    assert not set(validation["validation_source_seeds"]) & set(validation_training["source_seeds"])
    if not args.smoke:
        previous = baseline["validation"]["aggregate"]["shared_residual"]
        for field in ("mean_prediction_mse", "mean_prediction_error_rms"):
            np.testing.assert_allclose(validation["mean_prediction_quality"][field], previous[field], rtol=1e-6, atol=1e-8)
    print(f"[VALIDATE] {validation['validation_mode_labels']} labels: {validation['aggregate']}", flush=True)
    final_model, training, final_fits = fit_reference_classifier(
        family.mean, family.mean_groups, args.device, args.smoke, args.checkpoint_dir/"final")
    assert (len(outer_fits), len(final_fits)) == (8, 10)
    torch.save(dict(revision=REVISION, smoke=args.smoke, mean_source_revision=v115.REVISION,
        settings=v113.SETTINGS, validation_model=validation_model, final_model=final_model),
        args.checkpoint_dir/"classifier_models.pt")
    selected = [r for r in baseline["records"] if not args.smoke or v114.window_key(r) in v114.SMOKE_WINDOWS]
    assert len(selected) == (7 if args.smoke else 76)
    assert not set(family.mean_groups) & {r["seed"] for r in selected}
    paths = {v113.v112.v105.key(c): p for c, p, _ in v113.v98.evaluation_inputs(args, source)}
    ranges = {w: (a, b) for w, a, b in source.v78.v77.windows()}
    records = []
    for previous in selected:
        cell = v113.trace_cell(previous)
        directory = args.v110_dir/f"run_checkpoints/seed{cell['seed']}/traces"
        path = source.v76.trace_path(directory, cell) if previous["origin"] == "v110" else paths[v113.v112.v105.key(cell)]
        trace = source.v76.v68.load_trace(path)
        start, stop = ranges[previous["window"]]
        query = {k: trace[k][start:stop] for k in v113.v98.v97.KEYS}
        inputs, design = family.bank.features.inputs(query), family.bank.design(query)
        predicted = np.stack([design@family.bank.coefficients[m] for m in MODES], axis=1)
        predicted += v115.residual_prediction(means["final_model"], family.bank, inputs)
        observed = family.bank.targets(query)
        unchanged = v115.query_statistics(original["final_model"], observed, predicted, cell["mode"])
        v114.check_original(unchanged, previous["arms"]["shared_residual"])
        np.testing.assert_allclose(unchanged["true_prediction_error_rms"],
            previous["arms"]["shared_residual"]["true_prediction_error_rms"], rtol=1e-9, atol=1e-8)
        truth = MODES.index(cell["mode"])
        scores = score_arms(final_model, original["final_model"], observed, predicted)
        records.append(dict(**cell, origin=previous["origin"], window=previous["window"], transitions=stop-start,
            old_coverage_failure=previous["old_coverage_failure"],
            true_prediction_error_rms=unchanged["true_prediction_error_rms"],
            arms={a: v113.statistics(p, truth) for a, p in scores.items()}))
        print(f"[PILOT] {len(records)}/{len(selected)} {v114.window_key(previous)}", flush=True)
    transitions = sum(r["transitions"] for r in records)
    assert transitions == (364 if args.smoke else 4248)
    result = dict(revision=REVISION, passed=True, smoke=args.smoke, validation=validation, training=training,
        checked_windows=len(records), checked_transitions=transitions, unchanged_v115_score_windows_verified=len(records),
        source_crossfit_folds=dict(outer=outer_fits, final=final_fits),
        crossfit_mean_fits_computed=sum(not f["reused_cache"] for f in outer_fits+final_fits),
        crossfit_mean_folds=18, classifier_fits=2, deployed_mean_fits=0, mean_source_revision=v115.REVISION,
        records=records, summary=summarize(records), classifier_settings=v113.SETTINGS, mean_settings=v115.SETTINGS,
        hyperparameter_search=False, environment_steps=0, rl_optimizer_updates=0,
        frozen_v115_means_loaded=True, global_forward_refitted=False, input_target_scaling_refitted=False,
        classifier_feature_scaler_refitted=True, oracle_observations_used_for_query_scores=False,
        posterior_replayed=False, normal_thresholds_recalibrated=False, readout_controls_actor=False, readout_adopted=False,
        decision="partial-crossfit-training-smoke-only" if args.smoke else "matched-neural-classifier-pilot-not-adoption",
        limitations=["Reference counterfactual labels are supervision, not deployment observations.",
            "Previously used source holdouts and query windows remain development evidence.",
            "Global forward models and input/target scales remain inherited rather than cross-refitted.",
            "Cross-fitted mean training sets are smaller than deployment mean training sets.",
            "Classifier probabilities are not physical Gaussian densities or normal-null calibrated belief updates.",
            "No full-history belief, actor or return evaluation."])
    args.result_dir.mkdir(parents=True, exist_ok=True)
    path = args.result_dir/("native_smoke.json" if args.smoke else "matched_neural_classifier.json")
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] {REVISION}: {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("v115-dir", "v113-dir", "v110-dir", "v104-dir", "v98-dir", "v97-dir", "v95-dir", "v94-dir", "v83-dir",
                 "v78-dir", "v76-dir", "v74-dir", "v73-dir", "v72-dir", "v71-dir", "v69-dir", "v70-dir",
                 "v61-dir", "v62-dir", "checkpoint-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
