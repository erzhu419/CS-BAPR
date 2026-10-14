"""Reference-only pilot of mode-shared conditional covariance under frozen means."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_general_ood_v84_reference_discrimination as v84

REVISION = "general_ood_v87_shared_covariance"
CONDITION = "shared_mix"


def shared_covariance(mixed):
    return np.broadcast_to(mixed.mean(axis=1, keepdims=True), mixed.shape)


def verify_control(stats, expected):
    keys = ("accuracy", "classification_nll", "true_density_nll_without_constant",
            "mean_true_minus_best_other", "nonpositive_margin_fraction")
    for current, old in [(stats, expected)]+[(stats["per_mode"][m], expected["per_mode"][m]) for m in v84.MODES]:
        np.testing.assert_allclose([current[k] for k in keys], [old[k] for k in keys], rtol=1e-10, atol=1e-9)
    for mode in v84.MODES:
        np.testing.assert_array_equal(list(stats["per_mode"][mode]["predicted_mode_fractions"].values()),
            [expected["per_mode"][mode]["predicted_mode_fractions"][m] for m in v84.MODES])


def compact(stats):
    result = {k: v for k, v in stats.items() if k != "per_mode"}
    result["per_mode"] = {m: {k: v for k, v in s.items() if k != "pairs"} for m, s in stats["per_mode"].items()}
    return result


def run(args):
    sys.path.insert(0, str(args.v83_dir))
    from scripts import v83_cross_seed_covariance as v83

    torch.set_num_threads(2)
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
    assert len(selection["selection"]["folds"]) == len(previous["folds"]) == 6
    records, collected, changes = [], [], []
    max_volume_difference = 0.
    for expected, old in zip(selection["selection"]["folds"], previous["folds"]):
        seed = expected["seed"]
        cached = torch.load(source/f"validation/outer_seed{seed}.pt", map_location="cpu", weights_only=False)
        assert cached["revision"] == v83.REVISION and cached["ridge_alpha"] == v83.ALPHA
        values = cached["values"]
        assert values[3]["seed"] == old["seed"] == seed and values[3]["validation_rows"] == 512
        assert values[3]["outer_seed_excluded_from_inner_fits"] and values[3]["inner_seed_excluded_from_its_error_fit"]
        _, _, offsets, cross, raw = v84.reconstruct_fold(bank, groups, values)
        for name, w in (("raw", 1.), ("cross_seed", 0.), ("selected_mix", weight)):
            verify_control(v84.summarize(*v84.scores(offsets, w*raw+(1-w)*cross)), old["conditions"][name])
        ll, energy, logdet = v84.scores(offsets, shared_covariance(weight*raw+(1-weight)*cross))
        volume_difference = float(np.max(np.abs(logdet-logdet[:, :1])))
        assert volume_difference == 0.
        max_volume_difference = max(max_volume_difference, volume_difference)
        stats = v84.summarize(ll, energy, logdet)
        collected.append((ll, energy, logdet))
        conditions = {c: compact(s) for c, s in old["conditions"].items()}
        conditions[CONDITION] = compact(stats)
        records.append(dict(values[3], conditions=conditions))
        for mode in v84.MODES:
            new, baseline = stats["per_mode"][mode], old["conditions"]["selected_mix"]["per_mode"][mode]
            changes.append(dict(seed=seed, mode=mode,
                classification_nll_delta=new["classification_nll"]-baseline["classification_nll"],
                accuracy_delta=new["accuracy"]-baseline["accuracy"],
                mean_margin_delta=new["mean_true_minus_best_other"]-baseline["mean_true_minus_best_other"]))
        print(f"[FOLD] seed{seed}: shared NLL={stats['classification_nll']:.8f}, accuracy={stats['accuracy']:.6f}; three controls match", flush=True)
    aggregate = v84.summarize(*(np.concatenate([v[i] for v in collected]) for i in range(3)))
    conditions = {c: compact(s) for c, s in previous["conditions"].items()}
    conditions[CONDITION] = dict(raw_weight=weight, **compact(aggregate))
    result = dict(revision=REVISION, source_revision=v84.REVISION, modes=list(v84.MODES),
        raw_weight=weight, covariance_formula="equal mean of five frozen v83 mixed covariances at each input",
        conditions=conditions, folds=records, fold_mode_changes_vs_v83=changes,
        queries=3072, true_mode_labels=15360, candidate_scores_per_condition=76800,
        verification=dict(fold_control_comparisons=18, all_fold_control_statistics_match_v84=True,
            max_candidate_logdet_difference=max_volume_difference),
        weighting="equal seed / true mode / row", prior="uniform, instantaneous posterior",
        covariance_uses_query_target=False, covariance_uses_true_mode=False,
        covariance_components_refitted=False, conditional_means_refitted=False, weight_refitted=False,
        global_forward_refitted=False, scaling_refitted=False, temporal_rules_changed=False,
        evaluation_paths_loaded=False, normal_null_loaded=False, readout_controls_actor=False,
        environment_steps=0, optimizer_updates=0, independent_confirmation=False,
        decision="reference-only-pilot-pending-analysis",
        limitations=["Same reference folds as the earlier weight selection and diagnostics, not independent confirmation.",
            "Shared covariance changes the precision metric as well as removing determinant differences.",
            "Instantaneous reference classification does not establish recursive query accuracy or RL returns."])
    args.result_dir.mkdir(parents=True, exist_ok=True)
    path = args.result_dir/"shared_covariance.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] {REVISION}: {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("v84-dir", "v83-dir", "v76-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    run(parser.parse_args())
