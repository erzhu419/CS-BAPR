"""Matched-reference shared residual means, source holdout then fixed-score pilot."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_general_ood_v114_classifier_attribution as v114

v113 = v114.v113
MODES = v113.MODES
REVISION = "general_ood_v115_shared_residual_mean"
SETTINGS = dict(hidden_width=128, hidden_layers=2, activation="SiLU", epochs=100,
    batch_size=512, learning_rate=.001, weight_decay=.0001, training_seed=115)
ARMS = ("local_linear", "shared_residual")


def reference_arrays(bank, groups, smoke):
    rows = v113.selected_rows(groups, smoke)
    x = np.asarray(bank.tree.data[rows], dtype=np.float32)
    y = np.concatenate([bank.residuals[m][rows] for m in MODES], axis=1).astype(np.float32)
    return x, y, groups[rows]


class SharedResidualMean(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(width, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 55))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, x):
        return self.layers(x)


def fit_mean(bank, groups, device, smoke=False, stage="test"):
    x, y, selected_groups = reference_arrays(bank, groups, smoke)
    torch.manual_seed(SETTINGS["training_seed"])
    model = SharedResidualMean(x.shape[1]).to(device)
    x, y = torch.as_tensor(x, device=device), torch.as_tensor(y, device=device)
    weights = torch.as_tensor(v113.seed_weights(selected_groups), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=SETTINGS["learning_rate"], weight_decay=SETTINGS["weight_decay"])
    generator = torch.Generator().manual_seed(SETTINGS["training_seed"])
    epochs, batch_size = (2 if smoke else SETTINGS["epochs"]), SETTINGS["batch_size"]
    losses = []
    for epoch in range(epochs):
        order = torch.randperm(len(x), generator=generator).to(device)
        epoch_loss = 0.
        for indices in order.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            per_row = (model(x[indices])-y[indices]).square().mean(dim=1)
            loss = (per_row*weights[indices]).mean()
            assert torch.isfinite(loss).item()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()*len(indices)
        losses.append(epoch_loss/len(x))
        if epoch == 0 or (epoch+1) % 10 == 0 or epoch+1 == epochs:
            print(f"[TRAIN] {stage} epoch={epoch+1}/{epochs} weighted_mse={losses[-1]:.6g}", flush=True)
    model.eval()
    seed_weights = v113.seed_weights(selected_groups)
    return model, dict(reference_inputs=len(x), paired_mode_targets=5*len(x), source_seeds=np.unique(selected_groups).astype(int).tolist(),
        weighted_seed_fractions={str(int(s)): float(seed_weights[selected_groups == s].sum()/seed_weights.sum()) for s in np.unique(selected_groups)},
        weighted_mode_fractions=[.2]*5, input_dimension=x.shape[1], output_dimension=55,
        epochs=epochs, optimizer_updates=epochs*((len(x)+batch_size-1)//batch_size), settings=SETTINGS,
        epoch_online_training_mse=losses, final_epoch_training_mse=losses[-1], device=str(device))


def residual_prediction(model, bank, inputs):
    x = (inputs-bank.features.input_mean)/bank.features.input_scale
    device = next(model.parameters()).device
    with torch.inference_mode():
        values = model(torch.as_tensor(x, dtype=torch.float32, device=device)).cpu().numpy()
    assert values.shape == (len(inputs), 55) and np.isfinite(values).all()
    return values.astype(np.float64).reshape(-1, 5, 11)


def reference_predictions(model, bank, rows, correction):
    inputs = bank.tree.data[rows]*bank.features.input_scale+bank.features.input_mean
    design = bank.features.features(inputs)
    forward = np.stack([design@bank.coefficients[m] for m in MODES], axis=1)
    local = correction.predict(inputs)
    return dict(local_linear=forward+np.stack([local[m] for m in MODES], axis=1),
        shared_residual=forward+residual_prediction(model, bank, inputs))


def prediction_statistics(targets, predicted):
    assert targets.shape == predicted.shape and targets.shape[1:] == (5, 11)
    squared = np.mean((targets-predicted)**2, axis=2)
    distances = np.mean((targets[:, :, None, :]-predicted[:, None, :, :])**2, axis=3)
    winners = distances.argmin(axis=2)
    gaps = np.sqrt(np.mean((targets[:, :, None, :]-targets[:, None, :, :])**2, axis=3))
    gaps[:, np.arange(5), np.arange(5)] = np.inf
    nearest_gap = gaps.min(axis=2)
    return dict(mean_prediction_mse=float(squared.mean()), mean_prediction_error_rms=float(np.sqrt(squared).mean()),
        paired_nearest_mode_accuracy=float(np.mean(winners == np.arange(5))),
        error_exceeds_nearest_oracle_gap_fraction=float(np.mean(np.sqrt(squared) > nearest_gap)),
        zero_oracle_gap_fraction=float(np.mean(nearest_gap == 0)),
        per_mode={m: dict(mean_prediction_mse=float(squared[:, i].mean()),
            mean_prediction_error_rms=float(np.sqrt(squared[:, i]).mean()),
            paired_nearest_mode_accuracy=float(np.mean(winners[:, i] == i))) for i, m in enumerate(MODES)})


def heldout_validation(family, device, smoke=False):
    from scripts.v83_cross_seed_covariance import LocalLinearMean

    bank, groups = family.incumbent.mean, family.groups
    support = bank.tree.n
    validation_rows = v113.selected_rows(family.mean_groups[support:], smoke)+support
    validation_groups = family.mean_groups[validation_rows]
    assert not set(validation_groups) & set(groups)
    model, training = fit_mean(bank, groups, device, smoke, "source_holdout")
    predictions = reference_predictions(model, family.mean, validation_rows, LocalLinearMean(bank))
    targets = v113.reference_targets(family.mean, validation_rows)
    return dict(training=training, validation_source_seeds=np.unique(validation_groups).astype(int).tolist(),
        validation_inputs=len(targets), validation_mode_targets=5*len(targets),
        heldout_seeds_excluded_from_neural_and_local_mean_fits=True,
        aggregate={a: prediction_statistics(targets, p) for a, p in predictions.items()},
        per_seed={str(int(s)): {a: prediction_statistics(targets[validation_groups == s], p[validation_groups == s])
            for a, p in predictions.items()} for s in np.unique(validation_groups)}), model


def query_statistics(fitted, observed, predicted, mode):
    truth = MODES.index(mode)
    return dict(candidates={a: v113.statistics(logp, truth) for a, logp in v113.scores(fitted, observed, predicted).items()},
        true_prediction_error_rms=float(np.sqrt(np.mean((observed-predicted[:, truth])**2, axis=1)).mean()))


def summarize(records):
    fields = ("instantaneous_mode_accuracy", "mean_true_minus_best_other_score", "classification_nll")
    return {name: dict(windows=len(rows), arms={a: dict(
        true_prediction_error_rms=float(np.mean([r["arms"][a]["true_prediction_error_rms"] for r in rows])),
        candidates={score: {f: float(np.mean([r["arms"][a]["candidates"][score][f] for r in rows]))
            for f in fields} for score in v113.ARMS}) for a in ARMS})
        for name, rows in v114.cohorts(records).items() if rows}


def run(args):
    sys.path.insert(0, str(args.v83_dir))
    from scripts import analyze_general_ood_v83_cross_seed_covariance as source

    torch.set_num_threads(2)
    family, _ = v113.v106.load_family(args, source)
    baseline = json.loads((args.v114_dir/"run_results/formal/classifier_attribution.json").read_text())
    assert baseline["revision"] == v114.REVISION and baseline["passed"] and not baseline["smoke"]
    validation, validation_model = heldout_validation(family, args.device, args.smoke)
    print(f"[VALIDATE] {validation['validation_inputs']} paired reference inputs: {validation['aggregate']}", flush=True)
    model, training = fit_mean(family.mean, family.mean_groups, args.device, args.smoke, "final")
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(dict(revision=REVISION, smoke=args.smoke, settings=SETTINGS,
        validation_model=validation_model.cpu().state_dict(), final_model=model.cpu().state_dict(),
        validation_training=validation["training"], final_training=training), args.checkpoint_dir/"mean_models.pt")
    saved_classifier = torch.load(args.v113_dir/"run_checkpoints/formal/classifier_models.pt", map_location="cpu", weights_only=False)
    assert saved_classifier["revision"] == v113.REVISION and not saved_classifier["smoke"]
    fitted = saved_classifier["final_model"]
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
        inputs = family.bank.features.inputs(query)
        design = family.bank.design(query)
        forward = np.stack([design@family.bank.coefficients[m] for m in MODES], axis=1)
        local = family.correction.predict(inputs)
        predictions = dict(local_linear=forward+np.stack([local[m] for m in MODES], axis=1),
            shared_residual=forward+residual_prediction(model, family.bank, inputs))
        observed = family.bank.targets(query)
        stats = {a: query_statistics(fitted, observed, p, cell["mode"]) for a, p in predictions.items()}
        v114.check_original(stats["local_linear"], previous["fitted"])
        np.testing.assert_allclose(stats["local_linear"]["true_prediction_error_rms"],
            previous["fitted"]["true_prediction_error_rms"], rtol=1e-9, atol=1e-8)
        diagnostics = None
        if "oracle_observations" in trace:
            query["oracle_observations"] = trace["oracle_observations"][start:stop]
            oracle = v114.oracle_predictions(query, family.bank, cell["mode"])
            truth = MODES.index(cell["mode"])
            gaps = np.sqrt(np.mean((oracle-oracle[:, truth:truth+1])**2, axis=2))
            nearest_gap = np.delete(gaps, truth, axis=1).min(axis=1)
            diagnostics = dict(mean_nearest_oracle_gap_rms=float(nearest_gap.mean()),
                zero_oracle_gap_fraction=float(np.mean(nearest_gap == 0)),
                arms={a: dict(error_exceeds_nearest_oracle_gap_fraction=float(np.mean(
                    np.sqrt(np.mean((observed-p[:, truth])**2, axis=1)) > nearest_gap))) for a, p in predictions.items()})
        records.append(dict(**cell, origin=previous["origin"], window=previous["window"], transitions=stop-start,
            old_coverage_failure=previous["old_coverage_failure"], arms=stats, oracle_gap_diagnostics=diagnostics))
        print(f"[PILOT] {len(records)}/{len(selected)} {v114.window_key(previous)}", flush=True)
    transitions = sum(r["transitions"] for r in records)
    assert transitions == (364 if args.smoke else 4248)
    result = dict(revision=REVISION, passed=True, smoke=args.smoke, validation=validation, training=training,
        checked_windows=len(records), checked_transitions=transitions, unchanged_local_score_windows_verified=len(records),
        records=records, summary=summarize(records), mean_fits=2, hyperparameter_search=False,
        training_objective="equal-source-seed, equal-mode MSE of target-scale-normalized residual corrections",
        classifier_source="v113 formal final_model; frozen input-intervention diagnostic, not recalibrated",
        environment_steps=0, rl_optimizer_updates=0, classifier_fits=0, forward_refitted=False, scaling_refitted=False,
        oracle_used_for_query_predictions=False, posterior_replayed=False, normal_thresholds_recalibrated=False,
        readout_controls_actor=False, readout_adopted=False,
        decision="partial-training-smoke-only" if args.smoke else "matched-reference-mean-pilot-not-adoption",
        limitations=["Known-mode counterfactual targets are reference supervision, not deployment observations.",
            "Two previously used source holdouts and all inspected query windows are development evidence.",
            "Frozen global forward models and scales are retained, not cross-refitted.",
            "Fixed classifier scores change its input distribution and are not a newly calibrated likelihood.",
            "No full-history belief, actor or return evaluation."])
    args.result_dir.mkdir(parents=True, exist_ok=True)
    path = args.result_dir/("native_smoke.json" if args.smoke else "shared_residual_mean.json")
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] {REVISION}: {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("v114-dir", "v113-dir", "v110-dir", "v104-dir", "v98-dir", "v97-dir", "v95-dir", "v94-dir", "v83-dir",
                 "v78-dir", "v76-dir", "v74-dir", "v73-dir", "v72-dir", "v71-dir", "v69-dir", "v70-dir",
                 "v61-dir", "v62-dir", "checkpoint-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
