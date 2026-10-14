"""Matched-count prefix/suffix mean coverage with frozen v83 covariances."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_general_ood_v87_shared_covariance as prior

v84 = prior.v84
REVISION = "general_ood_v89_reference_prefix"
SEEDS = (135170, 135171)
KEYS = ("states", "executed_actions", "next_states")
CONDITIONS = ("prefix_mean_mix", "suffix_mean_mix")
ROWS_PER_SEED = 1100
SWITCH = 3800
STEPS = 6000


def source_indices(condition):
    if condition == CONDITIONS[0]:
        return np.linspace(0, SWITCH-1, ROWS_PER_SEED, dtype=int)
    assert condition == CONDITIONS[1]
    return np.arange(SWITCH+1, STEPS, 2)


def extensions(traces, condition):
    assert tuple(traces) == SEEDS
    indices = source_indices(condition)
    return {mode: {key: np.concatenate([t["oracle_observations"][indices, c] if key == "next_states"
        else t[key][indices] for t in traces.values()]) for key in KEYS} for c, mode in enumerate(v84.MODES)}


def verify_sources(bank, blocks, traces):
    old_indices = np.arange(SWITCH, STEPS, 2)
    for seed, trace in traces.items():
        assert len(trace["states"]) == len(trace["oracle_observations"]) == STEPS
        block, = [b for b in blocks if b["origin"] == "v76" and b["seed"] == seed and b["source_mode"] == "normal"]
        assert block["updating"] and block["source_steps"] == [SWITCH, STEPS] and block["stride"] == 2
        start, stop = block["bank_row_range"]
        assert stop-start == ROWS_PER_SEED
        payload = {k: trace[k][old_indices] for k in KEYS}
        inputs = bank.features.inputs(payload)
        design = bank.features.features(inputs)
        np.testing.assert_allclose(bank.tree.data[start:stop], (inputs-bank.features.input_mean)/bank.features.input_scale,
                                   rtol=1e-10, atol=1e-10)
        for c, mode in enumerate(v84.MODES):
            payload["next_states"] = trace["oracle_observations"][old_indices, c]
            expected = bank.targets(payload)-design@bank.coefficients[mode]
            np.testing.assert_allclose(bank.residuals[mode][start:stop], expected, rtol=1e-10, atol=1e-10)


def expand_banks(bank, groups, traces):
    from scripts.v74_density_coverage import DensityCoverageBank

    banks = {c: DensityCoverageBank.extend(bank, extensions(traces, c), c) for c in CONDITIONS}
    expanded_groups = np.concatenate((groups, np.repeat(SEEDS, ROWS_PER_SEED)))
    assert all(b.tree.n == len(expanded_groups) == bank.tree.n+2*ROWS_PER_SEED for b in banks.values())
    return banks, expanded_groups


def predict_heldout(bank, groups, seed, inputs, base_rows):
    from scripts import v83_cross_seed_covariance as v83

    reduced = v83.subset_bank(bank, groups != seed)
    centers = v83.LocalLinearMean(reduced).predict(inputs)
    design = bank.features.features(inputs)
    predicted = np.stack([design@bank.coefficients[m]+centers[m] for m in v84.MODES], axis=1)
    base_training_rows = int((groups[:base_rows] != seed).sum())
    _, indices = reduced.tree.query((inputs-bank.features.input_mean)/bank.features.input_scale, k=64)
    return predicted, dict(training_rows=reduced.density_transitions, base_training_rows=base_training_rows,
        added_training_rows=reduced.density_transitions-base_training_rows,
        excluded_rows=int((groups == seed).sum()), outer_seed_excluded_from_all_mean_rows=True,
        mean_added_neighbor_fraction=float(np.mean(indices >= base_training_rows)))


def coverage_offsets(offsets, baseline, expanded):
    return offsets-(expanded-baseline)[:, None, :, :]


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
    assert tuple(bank.residuals) == v84.MODES and bank.tree.n == 23000
    groups = v83.groups_from_blocks(selection["reference_blocks"], bank.tree.n)
    weight = selection["selection"]["raw_weight"]
    assert weight == previous["conditions"]["selected_mix"]["raw_weight"]
    stored = args.v76_dir/"run_checkpoints/formal/traces"
    traces = {}
    for seed in SEEDS:
        with np.load(stored/f"density_extension_seed{seed}_updating_normal.npz") as saved:
            traces[seed] = {k: saved[k] for k in (*KEYS, "oracle_observations")}
    verify_sources(bank, selection["reference_blocks"], traces)
    banks, expanded_groups = expand_banks(bank, groups, traces)
    print("[SOURCE] two reference traces reproduce 2200 base rows/all five labels; both additions have 2200 unused rows", flush=True)
    expected_folds = selection["selection"]["folds"]
    assert len(expected_folds) == len(previous["folds"]) == 6
    if args.smoke:
        expected_folds = [f for f in expected_folds if f["seed"] == SEEDS[0]]
        assert len(expected_folds) == 1
    old_by_seed = {f["seed"]: f for f in previous["folds"]}
    weights = dict(raw=1., cross_seed=0., selected_mix=weight)
    collected = {c: [] for c in (*weights, *CONDITIONS)}
    records, changes = [], []
    checked = 0
    for expected in expected_folds:
        seed = expected["seed"]
        old = old_by_seed[seed]
        cached = torch.load(source/f"validation/outer_seed{seed}.pt", map_location="cpu", weights_only=False)
        assert cached["revision"] == v83.REVISION and cached["ridge_alpha"] == v83.ALPHA
        values = cached["values"]
        assert values[3]["seed"] == seed and values[3]["validation_rows"] == 512
        assert values[3]["outer_seed_excluded_from_inner_fits"] and values[3]["inner_seed_excluded_from_its_error_fit"]
        inputs, baseline, offsets, cross, raw = v84.reconstruct_fold(bank, groups, values)
        rows = np.linspace(0, len(inputs)-1, 8, dtype=int)
        direct, _ = predict_heldout(bank, groups, seed, inputs[rows], bank.tree.n)
        np.testing.assert_allclose(direct, baseline[rows], rtol=1e-10, atol=1e-10)
        checked += len(rows)
        conditions, fitting, diagnostics = {}, {}, {}
        for name, w in weights.items():
            terms = v84.scores(offsets, w*raw+(1-w)*cross)
            stats = v84.summarize(*terms)
            prior.verify_control(stats, old["conditions"][name])
            collected[name].append(terms)
            conditions[name] = prior.compact(stats)
        covariance = weight*raw+(1-weight)*cross
        for condition, expanded in banks.items():
            predicted, metadata = predict_heldout(expanded, expanded_groups, seed, inputs, bank.tree.n)
            expected_added = 2*ROWS_PER_SEED-(ROWS_PER_SEED if seed in SEEDS else 0)
            assert metadata["added_training_rows"] == expected_added
            fitting[condition] = metadata
            shifted = coverage_offsets(offsets, baseline, predicted)
            terms = v84.scores(shifted, covariance)
            stats = v84.summarize(*terms)
            collected[condition].append(terms)
            conditions[condition] = prior.compact(stats)
            diagnostics[condition] = {}
            for c, mode in enumerate(v84.MODES):
                new, old_mode = stats["per_mode"][mode], old["conditions"]["selected_mix"]["per_mode"][mode]
                changes.append(dict(seed=seed, mode=mode, condition=condition,
                    classification_nll_delta=new["classification_nll"]-old_mode["classification_nll"],
                    accuracy_delta=new["accuracy"]-old_mode["accuracy"],
                    mean_margin_delta=new["mean_true_minus_best_other"]-old_mode["mean_true_minus_best_other"]))
                diagnostics[condition][mode] = dict(mean_movement_rms=float(np.sqrt(np.mean((predicted[:, c]-baseline[:, c])**2))),
                    true_error_rms_before=float(np.sqrt(np.mean(offsets[:, c, c]**2))),
                    true_error_rms_after=float(np.sqrt(np.mean(shifted[:, c, c]**2))))
            print(f"[FOLD] seed{seed} {condition}: NLL={stats['classification_nll']:.8f}, accuracy={stats['accuracy']:.6f}", flush=True)
        records.append(dict(values[3], mean_fit=fitting, mean_diagnostics=diagnostics, conditions=conditions))
    aggregate = {c: dict(raw_weight=weights.get(c, weight), **prior.compact(v84.summarize(
        *(np.concatenate([v[i] for v in terms]) for i in range(3))))) for c, terms in collected.items()}
    result = dict(revision=REVISION, source_revision=v84.REVISION, smoke=args.smoke, modes=list(v84.MODES),
        conditions=aggregate, folds=records, fold_mode_changes_vs_v83=changes, raw_weight=weight,
        base_reference_rows=bank.tree.n, expanded_mean_rows=bank.tree.n+2*ROWS_PER_SEED,
        additional_inputs_per_condition=2*ROWS_PER_SEED, additional_labels_per_condition=2*ROWS_PER_SEED*len(v84.MODES),
        reference_sources=[dict(seed=s, role="density_extension", updating=True, source_mode="normal",
            prefix_rows=ROWS_PER_SEED, suffix_rows=ROWS_PER_SEED) for s in SEEDS],
        sampling=dict(prefix="1100 integer linspace indices from 0 through 3799 per seed",
            suffix="odd indices from 3801 through 5999 per seed; stride 2"),
        queries=512*len(records), true_mode_labels=2560*len(records), candidate_scores_per_condition=12800*len(records),
        verification=dict(source_base_rows_checked=2200, all_five_source_labels_match=True,
            direct_mean_queries=checked, all_five_baseline_mode_means_match=True,
            fold_control_comparisons=3*len(records), all_fold_control_statistics_match_v84=True,
            whole_seed_exclusion_includes_added_rows=True, equal_added_training_counts=True),
        mean_diagnostic_units="normalized target coordinates", mean_fitting_support_changed=True,
        conditional_covariances_changed=False, weight_refitted=False, global_forward_refitted=False, scaling_refitted=False,
        temporal_rules_changed=False, evaluation_paths_loaded=False, normal_null_loaded=False, readout_controls_actor=False,
        environment_steps=0, optimizer_updates=0, additional_counterfactual_steps=0, independent_confirmation=False,
        decision="reference-only-matched-coverage-pilot-pending-analysis",
        limitations=["Previously inspected reference folds; no independent confirmation or RL return evidence.",
            "Mean support gains 2200 inputs/11000 cached labels; the suffix condition matches this added budget.",
            "Covariance remains on the original reference support for this isolated mean intervention.",
            "Prefix versus suffix changes the sampled state/action distribution, not only clock time."])
    args.result_dir.mkdir(parents=True, exist_ok=True)
    path = args.result_dir/("native_smoke.json" if args.smoke else "reference_prefix.json")
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] {REVISION}: {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("v84-dir", "v83-dir", "v76-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
