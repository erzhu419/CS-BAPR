"""Nested reference-seed cross-fitting for uncertainty under fixed v78 means."""
from copy import copy

import numpy as np
from scipy.spatial import cKDTree
import torch

from scripts.v65_state_action_readout import NEIGHBORS
from scripts.v66_covariance_mix_readout import normal_nll, select_weight
from scripts.v77_mean_covariance import MODES
from scripts.v78_local_linear_mean import LocalLinearMean
from scripts.v81_prediction_error_covariance import ALPHA, PredictionErrorDensityBank, error_covariance
from scripts.v82_reference_covariance_mix import VALIDATION_ROWS_PER_SEED

REVISION = "general_ood_v83_cross_seed_covariance"
ERROR_CONDITION = "linear_mean_cross_seed_error_cov"
MIX_CONDITION = "linear_mean_cross_seed_mix_cov"


def groups_from_blocks(blocks, rows):
    groups = np.empty(rows, dtype=np.int64)
    cursor = 0
    for block in blocks:
        start, stop = block["bank_row_range"]
        assert start == cursor and start < stop <= rows
        groups[start:stop] = block["seed"]
        cursor = stop
    assert cursor == rows
    return groups


def subset_bank(bank, keep):
    reduced = copy(bank)
    reduced.tree = cKDTree(bank.tree.data[keep])
    reduced.residuals = {m: bank.residuals[m][keep] for m in MODES}
    reduced.density_transitions = int(keep.sum())
    assert reduced.density_transitions >= NEIGHBORS
    return reduced


def cross_seed_errors(bank, groups):
    errors = {m: np.empty_like(bank.residuals[m]) for m in MODES}
    for seed in np.unique(groups):
        heldout = groups == seed
        reduced = subset_bank(bank, ~heldout)
        inputs = bank.tree.data[heldout]*bank.features.input_scale+bank.features.input_mean
        prediction = LocalLinearMean(reduced).predict(inputs)
        for mode in MODES:
            errors[mode][heldout] = bank.residuals[mode][heldout]-prediction[mode]
        print(f"[ERRORS] seed{seed}: {int(heldout.sum())} rows predicted from {reduced.density_transitions} other-seed rows", flush=True)
    return errors


class CrossSeedDensityBank(PredictionErrorDensityBank):
    def __init__(self, bank, errors, raw_weight, selection=None):
        self.bank, self.errors, self.raw_weight, self.selection = bank, errors, raw_weight, selection
        self.correction = LocalLinearMean(bank)
        assert all(errors[m].shape == bank.residuals[m].shape for m in MODES)

    def geometry(self, inputs):
        bank = self.bank
        means = self.correction.predict(inputs)
        _, indices = bank.tree.query((inputs-bank.features.input_mean)/bank.features.input_scale, k=NEIGHBORS)
        covariance, raw_traces, error_traces, rms, shrinkages = {}, [], [], [], []
        for mode in MODES:
            neighbors = self.errors[mode][indices]
            fitted = [error_covariance(values) for values in neighbors]
            error = np.asarray([c for c, _ in fitted])
            raw = bank.geometry(inputs, mode)[1]
            covariance[mode] = self.raw_weight*raw+(1-self.raw_weight)*error
            raw_traces.append(np.trace(raw, axis1=1, axis2=2))
            error_traces.append(np.trace(error, axis1=1, axis2=2))
            rms.append(np.sqrt(np.mean(neighbors**2, axis=(1, 2))))
            shrinkages.append([s for _, s in fitted])
        return means, covariance, dict(raw_covariance_trace=np.column_stack(raw_traces),
            error_covariance_trace=np.column_stack(error_traces), cross_seed_error_rms=np.column_stack(rms),
            error_covariance_shrinkage=np.column_stack(shrinkages))

    def metadata(self):
        return dict(modes=list(MODES), conditional_mean="unchanged v78 local linear ridge residual prediction",
            conditional_mean_source="updating_mixed", neighbors=NEIGHBORS, ridge_alpha=ALPHA,
            intercept_penalized=False, covariance_source="same reference inputs and 64-neighbor rule",
            covariance_estimator="OAS second moment of whole-reference-seed-held-out prediction errors, mixed with raw covariance",
            covariance_formula="raw_weight*regularized_OAS(raw residuals)+(1-raw_weight)*regularized_OAS(cross-seed prediction errors, assume_centered=True)",
            covariance_ridge_relative=.001, error_second_moment_assume_centered=True,
            raw_weight=self.raw_weight, weight_selection=self.selection,
            error_bank_rows=self.bank.density_transitions, error_source_seed_excluded_from_mean_fit=True,
            nested_selection_excludes_outer_seed_from_all_inner_fits=True,
            selection_uses_evaluation_paths=False, selection_uses_normal_null=False,
            query_residual_used_for_geometry=False, conditional_mean_changed=False,
            conditional_covariance_changed=True, forward_refitted=False, scaling_refitted=False,
            likelihood_units="physical delta-state Gaussian density including log determinant")


def validation_fold(bank, groups, seed, rows_per_seed):
    keep = groups != seed
    reduced = subset_bank(bank, keep)
    # Outer seed is absent before any inner error-bank fit or neighborhood query.
    errors = cross_seed_errors(reduced, groups[keep])
    density = CrossSeedDensityBank(reduced, errors, 0.)
    heldout = np.flatnonzero(~keep)
    selected = heldout[np.linspace(0, len(heldout)-1, rows_per_seed, dtype=int)]
    assert len(np.unique(selected)) == rows_per_seed
    query = bank.tree.data[selected]*bank.features.input_scale+bank.features.input_mean
    means, error_covariances, _ = density.geometry(query)
    offsets = np.concatenate([bank.residuals[m][selected]-means[m] for m in MODES])
    raw = np.concatenate([reduced.geometry(query, m)[1] for m in MODES])
    cross = np.concatenate([error_covariances[m] for m in MODES])
    fold = dict(seed=int(seed), excluded_rows=len(heldout), training_rows=reduced.density_transitions,
        error_bank_rows=reduced.density_transitions, validation_rows=rows_per_seed, validation_labels=len(offsets),
        inner_seeds=[int(s) for s in np.unique(groups[keep])], outer_seed_excluded_from_inner_fits=True,
        inner_seed_excluded_from_its_error_fit=True, coupled_streams_held_out_together=True)
    return offsets, cross, raw, fold


def fit_selection(bank, groups, checkpoint_dir, rows_per_seed=VALIDATION_ROWS_PER_SEED):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    folds = []
    for seed in np.unique(groups):
        path = checkpoint_dir/f"outer_seed{seed}.pt"
        if path.is_file():
            saved = torch.load(path, map_location="cpu", weights_only=False)
            assert saved["revision"] == REVISION and saved["ridge_alpha"] == ALPHA
            values = saved["values"]
            assert values[3]["seed"] == seed and values[3]["validation_rows"] == rows_per_seed
            assert values[3]["training_rows"] == int((groups != seed).sum())
        else:
            print(f"[OUTER] validating seed{seed}", flush=True)
            values = validation_fold(bank, groups, seed, rows_per_seed)
            torch.save(dict(revision=REVISION, ridge_alpha=ALPHA, values=values), path)
        folds.append(values)
        print(f"[VALIDATE] seed{seed}: nested error bank complete", flush=True)
    offsets, cross, raw = (np.concatenate([v[i] for v in folds]) for i in range(3))
    weight, diagnostic = select_weight(offsets, cross, raw)
    details = [dict(v[3], selected_nll=normal_nll(weight, *v[:3]),
        cross_seed_only_nll=normal_nll(0., *v[:3]), raw_only_nll=normal_nll(1., *v[:3])) for v in folds]
    per_mode = {m: {name: float(np.mean([normal_nll(w,
        *(a[i*rows_per_seed:(i+1)*rows_per_seed] for a in v[:3])) for v in folds]))
        for name, w in (("selected_nll", weight), ("cross_seed_only_nll", 0.), ("raw_only_nll", 1.))}
        for i, m in enumerate(MODES)}
    return dict(raw_weight=weight, rows_per_seed=rows_per_seed, folds=details, per_mode=per_mode,
        validation_queries=len(offsets)//len(MODES), validation_labels=len(offsets),
        selected_nll=diagnostic["selected_normal_nll"], cross_seed_only_nll=diagnostic["local_only_normal_nll"],
        raw_only_nll=diagnostic["global_only_normal_nll"], bounded_solution=diagnostic["bounded_solution"],
        optimizer_evaluations=diagnostic["optimizer_evaluations"], solver=diagnostic["solver"],
        objective="equal-seed, equal-known-mode Gaussian NLL; constants independent of weight omitted",
        outer_seed_excluded_from_inner_fits=True, inner_seed_excluded_from_its_error_fit=True,
        full_reference_used_after_selection=True, selection_uses_evaluation_paths=False, selection_uses_normal_null=False)
