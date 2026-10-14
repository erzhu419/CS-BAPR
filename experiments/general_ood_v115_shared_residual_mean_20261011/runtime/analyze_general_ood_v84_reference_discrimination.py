"""Diagnose five-way reference discrimination from frozen v83 fold caches."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import logsumexp
import torch

MODES = ("normal", "heavy", "slippery", "high_grav", "compound")
REVISION = "general_ood_v84_reference_discrimination"


def reconstruct_fold(bank, groups, values):
    own_offsets, cross, raw, fold = values
    n = fold["validation_rows"]
    heldout = np.flatnonzero(groups == fold["seed"])
    selected = heldout[np.linspace(0, len(heldout)-1, n, dtype=int)]
    assert len(heldout) == fold["excluded_rows"]
    assert int((groups != fold["seed"]).sum()) == fold["training_rows"]
    inputs = bank.tree.data[selected]*bank.features.input_scale+bank.features.input_mean
    design = bank.features.features(inputs)
    targets = np.stack([design@bank.coefficients[m]+bank.residuals[m][selected] for m in MODES], axis=1)
    own = own_offsets.reshape(len(MODES), n, -1).transpose(1, 0, 2)
    predictions = targets-own
    # Cache errors are diagonal (true mode = candidate). Reconstruct all 25 pairs.
    offsets = targets[:, :, None, :]-predictions[:, None, :, :]
    dimension = targets.shape[-1]
    covariances = [a.reshape(len(MODES), n, dimension, dimension).transpose(1, 0, 2, 3)
                   for a in (cross, raw)]
    return inputs, predictions, offsets, *covariances


def scores(offsets, covariance):
    signs, logdet = np.linalg.slogdet(covariance)
    assert np.all(signs > 0)
    solved = np.linalg.solve(covariance, offsets.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)
    energy = np.einsum("ntcd,ntcd->ntc", offsets, solved)
    # The physical Gaussian constant is common to candidates and cancels in ratios.
    return -.5*(energy+logdet[:, None, :]), energy, logdet


def summarize(likelihoods, energy, logdet):
    truth = np.arange(len(MODES))
    diagonal = likelihoods[:, truth, truth]
    log_posterior = likelihoods-logsumexp(likelihoods, axis=2, keepdims=True)
    competitors = likelihoods.copy()
    competitors[:, truth, truth] = -np.inf
    margins = diagonal-competitors.max(axis=2)
    predicted = likelihoods.argmax(axis=2)
    classification_loss = -log_posterior[:, truth, truth]
    per_mode = {}
    for t, mode in enumerate(MODES):
        pairs = {}
        for c, candidate in enumerate(MODES):
            if c == t:
                continue
            pairs[candidate] = dict(
                mean_log_density_margin=float(np.mean(diagonal[:, t]-likelihoods[:, t, c])),
                nonpositive_margin_fraction=float(np.mean(diagonal[:, t] <= likelihoods[:, t, c])),
                mean_energy_advantage=float(.5*np.mean(energy[:, t, c]-energy[:, t, t])),
                mean_volume_advantage=float(.5*np.mean(logdet[:, c]-logdet[:, t])))
        per_mode[mode] = dict(accuracy=float(np.mean(predicted[:, t] == t)),
            classification_nll=float(classification_loss[:, t].mean()),
            true_density_nll_without_constant=float(-diagonal[:, t].mean()),
            mean_true_minus_best_other=float(margins[:, t].mean()),
            nonpositive_margin_fraction=float(np.mean(margins[:, t] <= 0)),
            predicted_mode_fractions={m: float(np.mean(predicted[:, t] == c)) for c, m in enumerate(MODES)},
            pairs=pairs)
    return dict(accuracy=float(np.mean(predicted == truth)),
        classification_nll=float(classification_loss.mean()),
        true_density_nll_without_constant=float(-diagonal.mean()),
        mean_true_minus_best_other=float(margins.mean()),
        nonpositive_margin_fraction=float(np.mean(margins <= 0)), per_mode=per_mode)


def run(args):
    sys.path.insert(0, str(args.v83_dir))
    from scripts import v83_cross_seed_covariance as v83
    from scripts.v78_local_linear_mean import LocalLinearMean

    torch.set_num_threads(2)
    source = args.v83_dir/"run_checkpoints/cross-seed-covariance"
    selection = json.loads((source/"nested_selection.json").read_text())
    assert selection["revision"] == v83.REVISION
    saved = torch.load(args.v76_dir/"run_checkpoints/formal/behavior_coverage_references.pt",
                       map_location="cpu", weights_only=False)
    bank = saved["references"]["updating_mixed"].bank
    assert tuple(bank.residuals) == MODES
    groups = v83.groups_from_blocks(selection["reference_blocks"], bank.tree.n)
    weight = selection["selection"]["raw_weight"]
    conditions = {"raw": 1., "cross_seed": 0., "selected_mix": weight}
    collected = {c: [] for c in conditions}
    folds = []
    checked = 0
    for expected in selection["selection"]["folds"]:
        seed = expected["seed"]
        cached = torch.load(source/f"validation/outer_seed{seed}.pt", map_location="cpu", weights_only=False)
        assert cached["revision"] == v83.REVISION and cached["ridge_alpha"] == v83.ALPHA
        values = cached["values"]
        assert values[3]["seed"] == seed
        assert values[3]["outer_seed_excluded_from_inner_fits"] and values[3]["inner_seed_excluded_from_its_error_fit"]
        inputs, predictions, offsets, cross, raw = reconstruct_fold(bank, groups, values)
        rows = np.linspace(0, len(inputs)-1, 8, dtype=int)
        reduced = v83.subset_bank(bank, groups != seed)
        direct = LocalLinearMean(reduced).predict(inputs[rows])
        design = bank.features.features(inputs[rows])
        for c, mode in enumerate(MODES):
            np.testing.assert_allclose(predictions[rows, c], design@bank.coefficients[mode]+direct[mode],
                                       rtol=1e-10, atol=1e-10)
        checked += len(rows)
        result = dict(values[3], conditions={})
        for name, w in conditions.items():
            ll, energy, logdet = scores(offsets, w*raw+(1-w)*cross)
            stats = summarize(ll, energy, logdet)
            loss_key = {"raw": "raw_only_nll", "cross_seed": "cross_seed_only_nll", "selected_mix": "selected_nll"}[name]
            np.testing.assert_allclose(stats["true_density_nll_without_constant"], expected[loss_key], rtol=1e-10, atol=1e-9)
            result["conditions"][name] = stats
            collected[name].append((ll, energy, logdet))
        folds.append(result)
        print(f"[FOLD] seed{seed}: {len(inputs)} inputs, five labels scored against five candidates", flush=True)
    assert len(folds) == 6 and all(f["validation_rows"] == 512 for f in folds)
    aggregate = {name: dict(raw_weight=conditions[name], **summarize(
        *(np.concatenate([v[i] for v in values]) for i in range(3)))) for name, values in collected.items()}
    for name, key in (("raw", "raw_only_nll"), ("cross_seed", "cross_seed_only_nll"), ("selected_mix", "selected_nll")):
        np.testing.assert_allclose(aggregate[name]["true_density_nll_without_constant"],
            selection["selection"][key], rtol=1e-10, atol=1e-9)
    result = dict(revision=REVISION, source_revision=v83.REVISION, modes=list(MODES),
        queries=3072, true_mode_labels=15360, candidate_scores=76800,
        conditions=aggregate, folds=folds,
        verification=dict(direct_mean_queries=checked, all_five_mode_means_match=True,
                          all_fold_and_aggregate_density_losses_match_v83=True),
        weighting="equal seed / true mode / row", prior="uniform, instantaneous posterior",
        temporal_readout_applied=False, weights_refitted=False, means_refitted=False, covariances_refitted=False,
        selection_uses_evaluation_paths=False, normal_null_loaded=False, readout_controls_actor=False,
        environment_steps=0, optimizer_updates=0,
        limitations=["Post-selection reference diagnostics on the same six weight-selection folds, not independent confirmation.",
            "Whole-seed mean/covariance fits exclude the outer seed; the shared mixture weight was selected using all folds.",
            "Instantaneous likelihood comparisons do not reproduce recursive query-path accuracy or RL returns."])
    args.result_dir.mkdir(parents=True, exist_ok=True)
    path = args.result_dir/"reference_discrimination.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] {REVISION}: {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("v83-dir", "v76-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    run(parser.parse_args())
