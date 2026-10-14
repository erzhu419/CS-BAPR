"""Reference-seed-held-out covariance mixing with the unchanged v78 mean rule."""
from copy import copy

import numpy as np
from scipy.spatial import cKDTree

from scripts import probe_general_ood_v76_behavior_coverage as v76
from scripts.v66_covariance_mix_readout import normal_nll, select_weight
from scripts.v77_mean_covariance import MODES
from scripts.v81_prediction_error_covariance import PredictionErrorDensityBank

CONDITION = "linear_mean_reference_mix_cov"
VALIDATION_ROWS_PER_SEED = 512


def reference_groups(args, bank):
    """Recover row provenance and verify it against the saved, paired bank."""
    inputs, residuals, groups, blocks = [], {m: [] for m in MODES}, [], []

    def append(seed, origin, traces, **details):
        x = bank.features.inputs(traces["normal"])
        start = sum(len(v) for v in inputs)
        inputs.append((x-bank.features.input_mean)/bank.features.input_scale)
        groups.append(np.full(len(x), seed, dtype=np.int64))
        for mode in MODES:
            np.testing.assert_array_equal(bank.features.inputs(traces[mode]), x)
            residuals[mode].append(bank.targets(traces[mode])-bank.design(traces[mode])@bank.coefficients[mode])
        blocks.append(dict(seed=seed, origin=origin, bank_row_range=[start, start+len(x)], **details))

    old = args.v69_dir/"run_checkpoints/reference/traces"
    for cell in v76.v69.reference_cells(v76.v69.SPEC):
        if cell["role"] == "density_calibration":
            traces = {m: v76.v68.load_trace(v76.v69.trace_path(old, cell, m)) for m in MODES}
            append(cell["seed"], "v69", traces, updating=cell["updating"], source_steps=[0, 6000], stride=1)
    stored = args.v71_dir/"run_checkpoints/preflight/traces"
    for cell in v76.v74.v71.selected_cells(v76.v74.v71.SPEC, "density_calibration", "shifted_state"):
        trace = v76.v68.load_trace(v76.v74.v71.trace_path(stored, cell))
        suffix = slice(3800, 6000)
        traces = {m: dict(states=trace["states"][suffix], executed_actions=trace["executed_actions"][suffix],
                         next_states=trace["oracle_observations"][suffix, i]) for i, m in enumerate(MODES)}
        append(cell["seed"], "v71", traces, source_mode=cell["mode"], source_steps=[3800, 6000], stride=1)
    stored = args.v76_dir/"run_checkpoints/formal/traces"
    for cell in v76.reference_cells(v76.SPEC):
        if cell["updating"]:
            trace = v76.v68.load_trace(v76.trace_path(stored, cell))
            suffix = slice(3800, 6000, 2)
            traces = {m: dict(states=trace["states"][suffix], executed_actions=trace["executed_actions"][suffix],
                             next_states=trace["oracle_observations"][suffix, i]) for i, m in enumerate(MODES)}
            append(cell["seed"], "v76", traces, source_mode=cell["mode"], updating=True,
                   source_steps=[3800, 6000], stride=2)
    np.testing.assert_allclose(np.concatenate(inputs), bank.tree.data, rtol=0, atol=1e-12)
    for mode in MODES:
        np.testing.assert_allclose(np.concatenate(residuals[mode]), bank.residuals[mode], rtol=0, atol=1e-12)
    labels = np.concatenate(groups)
    assert len(labels) == 23000 and len(np.unique(labels)) == 6
    return labels, blocks


class ReferenceMixedDensityBank(PredictionErrorDensityBank):
    def __init__(self, bank, raw_weight, selection=None):
        super().__init__(bank)
        self.raw_weight, self.selection = raw_weight, selection

    def geometry(self, inputs):
        means, loo, diagnostic = super().geometry(inputs)
        raw = {m: self.bank.geometry(inputs, m)[1] for m in MODES}
        covariance = {m: self.raw_weight*raw[m]+(1-self.raw_weight)*loo[m] for m in MODES}
        for name, values in (("raw_covariance_trace", raw), ("loo_covariance_trace", loo)):
            diagnostic[name] = np.column_stack([np.trace(values[m], axis1=1, axis2=2) for m in MODES])
        return means, covariance, diagnostic

    def metadata(self):
        meta = super().metadata()
        meta.update(covariance_estimator="reference-seed-held-out mixture of raw residual covariance and LOO error second moment",
            covariance_formula="raw_weight*regularized_OAS(raw residuals) + (1-raw_weight)*regularized_OAS(LOO errors, assume_centered=True)",
            raw_weight=self.raw_weight, weight_selection=self.selection,
            selection_uses_evaluation_paths=False, selection_uses_normal_null=False)
        return meta


def validation_fold(bank, groups, seed, rows_per_seed):
    heldout = np.flatnonzero(groups == seed)
    selected = heldout[np.linspace(0, len(heldout)-1, rows_per_seed, dtype=int)]
    assert len(np.unique(selected)) == rows_per_seed
    train = groups != seed
    reduced = copy(bank)
    reduced.tree = cKDTree(bank.tree.data[train])
    reduced.residuals = {m: bank.residuals[m][train] for m in MODES}
    reduced.density_transitions = int(train.sum())
    # All labels and coupled physical streams of this seed leave both neighborhoods.
    query = bank.tree.data[selected]*bank.features.input_scale+bank.features.input_mean
    density = PredictionErrorDensityBank(reduced)
    means, loo, _ = density.geometry(query)
    errors = np.concatenate([bank.residuals[m][selected]-means[m] for m in MODES])
    raw = np.concatenate([reduced.geometry(query, m)[1] for m in MODES])
    loo = np.concatenate([loo[m] for m in MODES])
    fold = dict(seed=int(seed), excluded_rows=len(heldout), training_rows=int(train.sum()),
        validation_rows=rows_per_seed, validation_labels=len(errors), coupled_streams_held_out_together=True)
    return errors, loo, raw, fold


def fit_selection(bank, groups, rows_per_seed=VALIDATION_ROWS_PER_SEED):
    folds = []
    for seed in np.unique(groups):
        values = validation_fold(bank, groups, seed, rows_per_seed)
        folds.append(values)
        print(f"[VALIDATE] held out seed{seed}: {values[3]['excluded_rows']} excluded, {rows_per_seed} queries", flush=True)
    errors, loo, raw = (np.concatenate([v[i] for v in folds]) for i in range(3))
    weight, diagnostic = select_weight(errors, loo, raw)
    details = [dict(v[3], selected_nll=normal_nll(weight, *v[:3]),
        loo_only_nll=normal_nll(0., *v[:3]), raw_only_nll=normal_nll(1., *v[:3])) for v in folds]
    per_mode = {m: dict(selected_nll=float(np.mean([
        normal_nll(weight, *(a[i*rows_per_seed:(i+1)*rows_per_seed] for a in v[:3])) for v in folds])))
        for i, m in enumerate(MODES)}
    return dict(raw_weight=weight, rows_per_seed=rows_per_seed, folds=details, per_mode=per_mode,
        validation_queries=len(errors)//len(MODES), validation_labels=len(errors),
        selected_nll=diagnostic["selected_normal_nll"], loo_only_nll=diagnostic["local_only_normal_nll"],
        raw_only_nll=diagnostic["global_only_normal_nll"], bounded_solution=diagnostic["bounded_solution"],
        optimizer_evaluations=diagnostic["optimizer_evaluations"], solver=diagnostic["solver"],
        objective="equal-seed, equal-known-mode Gaussian NLL; constants independent of weight omitted",
        full_reference_used_after_selection=True, global_forward_and_scaling_fixed=True,
        selection_uses_evaluation_paths=False, selection_uses_normal_null=False)
