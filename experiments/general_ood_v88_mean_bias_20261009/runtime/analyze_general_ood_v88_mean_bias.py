"""Nested reference-only mean-bias correction with frozen v83 covariances."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_general_ood_v87_shared_covariance as prior

v84 = prior.v84
REVISION = "general_ood_v88_mean_bias"
CONDITION = "cross_seed_bias_mix"
NEIGHBORS = 64


def neighborhood_bias(bank, errors, inputs):
    normalized = (inputs-bank.features.input_mean)/bank.features.input_scale
    _, indices = bank.tree.query(normalized, k=NEIGHBORS)
    return np.stack([errors[m][indices].mean(axis=1) for m in v84.MODES], axis=1)


def fit_bias(bank, groups, seed, inputs):
    from scripts import v83_cross_seed_covariance as v83

    keep = groups != seed
    reduced = v83.subset_bank(bank, keep)
    errors = v83.cross_seed_errors(reduced, groups[keep])
    return neighborhood_bias(reduced, errors, inputs), dict(
        outer_seed=int(seed), error_bank_rows=reduced.density_transitions,
        inner_seeds=[int(s) for s in np.unique(groups[keep])],
        outer_seed_excluded_from_all_inner_fits=True, inner_seed_excluded_from_its_error_fit=True)


def corrected_offsets(offsets, bias):
    # Prediction gains the estimated target-minus-prediction error, so offsets lose it.
    return offsets-bias[:, None, :, :]


def freeze_bias(path, bank, groups, seed, inputs, weight):
    if path.is_file():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        assert saved["revision"] == REVISION and saved["raw_weight"] == weight
        assert saved["metadata"]["outer_seed"] == seed
        assert saved["bias"].shape == (len(inputs), len(v84.MODES), len(bank.target_scale))
        print(f"[REUSE] nested bias seed{seed}", flush=True)
        return saved["bias"], saved["metadata"]
    bias, metadata = fit_bias(bank, groups, seed, inputs)
    torch.save(dict(revision=REVISION, raw_weight=weight, bias=bias, metadata=metadata), path)
    return bias, metadata


def run(args):
    sys.path.insert(0, str(args.v83_dir))
    from scripts import v83_cross_seed_covariance as v83

    torch.set_num_threads(2)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    source = args.v83_dir/"run_checkpoints/cross-seed-covariance"
    selection = json.loads((source/"nested_selection.json").read_text())
    previous = json.loads((args.v84_dir/"run_results/diagnostic/reference_discrimination.json").read_text())
    assert selection["revision"] == v83.REVISION and previous["revision"] == v84.REVISION
    bank = torch.load(args.v76_dir/"run_checkpoints/formal/behavior_coverage_references.pt",
                      map_location="cpu", weights_only=False)["references"]["updating_mixed"].bank
    assert tuple(bank.residuals) == v84.MODES
    groups = v83.groups_from_blocks(selection["reference_blocks"], bank.tree.n)
    weight = selection["selection"]["raw_weight"]
    assert weight == previous["conditions"]["selected_mix"]["raw_weight"]
    expected_folds = selection["selection"]["folds"]
    assert len(expected_folds) == len(previous["folds"]) == 6
    if args.smoke:
        expected_folds = expected_folds[:1]
    weights = dict(raw=1., cross_seed=0., selected_mix=weight)
    collected = {c: [] for c in (*weights, CONDITION)}
    records, changes = [], []
    checked = 0
    for expected, old in zip(expected_folds, previous["folds"]):
        seed = expected["seed"]
        cached = torch.load(source/f"validation/outer_seed{seed}.pt", map_location="cpu", weights_only=False)
        assert cached["revision"] == v83.REVISION and cached["ridge_alpha"] == v83.ALPHA
        values = cached["values"]
        assert values[3]["seed"] == old["seed"] == seed and values[3]["validation_rows"] == 512
        assert values[3]["outer_seed_excluded_from_inner_fits"] and values[3]["inner_seed_excluded_from_its_error_fit"]
        inputs, predictions, offsets, cross, raw = v84.reconstruct_fold(bank, groups, values)
        reduced = v83.subset_bank(bank, groups != seed)
        rows = np.linspace(0, len(inputs)-1, 8, dtype=int)
        direct = v83.LocalLinearMean(reduced).predict(inputs[rows])
        design = bank.features.features(inputs[rows])
        for c, mode in enumerate(v84.MODES):
            np.testing.assert_allclose(predictions[rows, c], design@bank.coefficients[mode]+direct[mode], rtol=1e-10, atol=1e-10)
        checked += len(rows)
        conditions = {}
        for name, w in weights.items():
            terms = v84.scores(offsets, w*raw+(1-w)*cross)
            stats = v84.summarize(*terms)
            prior.verify_control(stats, old["conditions"][name])
            collected[name].append(terms)
            conditions[name] = prior.compact(stats)
        print(f"[VERIFY] seed{seed}: means and three old controls match", flush=True)
        bias, metadata = freeze_bias(args.checkpoint_dir/f"outer_seed{seed}_bias.pt", bank, groups, seed, inputs, weight)
        shifted = corrected_offsets(offsets, bias)
        terms = v84.scores(shifted, weight*raw+(1-weight)*cross)
        stats = v84.summarize(*terms)
        collected[CONDITION].append(terms)
        conditions[CONDITION] = prior.compact(stats)
        diagnostics = {}
        for c, mode in enumerate(v84.MODES):
            new, old_mode = stats["per_mode"][mode], old["conditions"]["selected_mix"]["per_mode"][mode]
            changes.append(dict(seed=seed, mode=mode,
                classification_nll_delta=new["classification_nll"]-old_mode["classification_nll"],
                accuracy_delta=new["accuracy"]-old_mode["accuracy"],
                mean_margin_delta=new["mean_true_minus_best_other"]-old_mode["mean_true_minus_best_other"]))
            diagnostics[mode] = dict(bias_rms=float(np.sqrt(np.mean(bias[:, c]**2))),
                true_error_rms_before=float(np.sqrt(np.mean(offsets[:, c, c]**2))),
                true_error_rms_after=float(np.sqrt(np.mean(shifted[:, c, c]**2))))
        records.append(dict(values[3], bias_fit=metadata, mean_diagnostics=diagnostics, conditions=conditions))
        print(f"[FOLD] seed{seed}: corrected NLL={stats['classification_nll']:.8f}, accuracy={stats['accuracy']:.6f}", flush=True)
    aggregate = {c: dict(raw_weight=weights.get(c, weight), **prior.compact(v84.summarize(
        *(np.concatenate([v[i] for v in terms]) for i in range(3))))) for c, terms in collected.items()}
    result = dict(revision=REVISION, source_revision=v84.REVISION, modes=list(v84.MODES), smoke=args.smoke,
        raw_weight=weight, bias_strength=1., bias_formula="equal-neighbor mean of nested whole-seed target-minus-prediction errors",
        conditions=aggregate, folds=records, fold_mode_changes_vs_v83=changes,
        queries=512*len(records), true_mode_labels=2560*len(records), candidate_scores_per_condition=12800*len(records),
        verification=dict(direct_mean_queries=checked, all_five_baseline_mode_means_match=True,
            fold_control_comparisons=3*len(records), all_fold_control_statistics_match_v84=True),
        weighting="equal seed / true mode / row", prior="uniform, instantaneous posterior",
        mean_diagnostic_units="normalized target coordinates", baseline_mean_rule_changed=False,
        conditional_means_changed=True, nested_error_fits_reconstructed=True,
        covariance_components_refitted=False, conditional_covariances_changed=False, weight_refitted=False,
        global_forward_refitted=False, scaling_refitted=False, temporal_rules_changed=False,
        bias_uses_query_target=False, bias_uses_true_mode=False,
        evaluation_paths_loaded=False, normal_null_loaded=False, readout_controls_actor=False,
        environment_steps=0, optimizer_updates=0, independent_confirmation=False,
        decision="reference-only-mean-bias-pilot-pending-analysis",
        limitations=["These reference folds also supported earlier weight selection and diagnostics; no independent confirmation.",
            "Inner cross-seed errors arise from smaller fitting banks than outer-fold prediction or deployment.",
            "Covariances remain the frozen pre-correction second moments for this isolated mean intervention.",
            "Instantaneous reference classification is not recursive query accuracy or RL return evidence."])
    args.result_dir.mkdir(parents=True, exist_ok=True)
    path = args.result_dir/("native_smoke.json" if args.smoke else "mean_bias.json")
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] {REVISION}: {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("v84-dir", "v83-dir", "v76-dir", "checkpoint-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
