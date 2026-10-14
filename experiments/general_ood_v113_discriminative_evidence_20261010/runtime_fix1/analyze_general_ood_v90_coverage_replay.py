"""Replay fixed prefix/suffix mean additions with the original v83 covariance."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import multivariate_normal
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_general_ood_v89_reference_prefix as v89

REVISION = "general_ood_v90_coverage_replay"
CONDITIONS = v89.CONDITIONS
CONTROLS = ("linear_mean_updating_cov", "linear_mean_cross_seed_error_cov", "linear_mean_cross_seed_mix_cov")


def make_densities(bank, expanded, errors, weight):
    from scripts.v78_local_linear_mean import LocalLinearMean
    from scripts.v83_cross_seed_covariance import CrossSeedDensityBank

    densities = {}
    for condition in CONDITIONS:
        density = CrossSeedDensityBank(bank, errors, weight)
        density.correction = LocalLinearMean(expanded[condition])
        densities[condition] = density
    return densities


def verify_baseline(current, expected):
    for key in ("mean_true_prediction_error_rms", "mean_true_minus_best_other_log_density",
                "mean_energy_advantage", "mean_covariance_volume_advantage", "raw_mode_accuracy"):
        np.testing.assert_allclose(current[key], expected[key], rtol=1e-9, atol=1e-8)
    for mode, pair in current["fixed_competitor"].items():
        for key, value in pair.items():
            np.testing.assert_allclose(value, expected["fixed_competitor"][mode][key], rtol=1e-9, atol=1e-8)


def verify_frozen_volume(current, expected):
    for window, values in current.items():
        for mode, pair in values["fixed_competitor"].items():
            np.testing.assert_allclose(pair["mean_covariance_volume_advantage"],
                expected[window]["fixed_competitor"][mode]["mean_covariance_volume_advantage"], rtol=1e-9, atol=1e-8)


def freeze_references(args, densities, source):
    from scripts.v73_recursive_mode_readout import RecursiveModeReference

    path = args.checkpoint_dir/"coverage_phases.pt"
    weights = {c: d.raw_weight for c, d in densities.items()}
    if path.is_file():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        assert saved["revision"] == REVISION and saved["raw_weights"] == weights
        assert saved["spec"] == source.v76.SPEC and saved["normal_null_transitions"] == 12000
        assert set(saved["phases"]) == set(CONDITIONS)
        references = {}
        for c, phase in saved["phases"].items():
            ref = RecursiveModeReference()
            ref.bank, ref.phase, ref.normal_null_calibration_transitions = densities[c], phase, 12000
            references[c] = ref
        print("[REUSE] same-population phase calibration", flush=True)
    else:
        stored = args.v69_dir/"run_checkpoints/reference/traces"
        nulls = [source.v76.v68.load_trace(source.v76.v69.trace_path(stored, c))
            for c in source.v76.v69.reference_cells(source.v76.v69.SPEC) if c["role"] == "normal_null"]
        assert len(nulls) == 2 and sum(len(t["states"]) for t in nulls) == 12000
        references = {}
        for c, density in densities.items():
            references[c] = RecursiveModeReference.calibrate(density, nulls)
            assert references[c].normal_null_calibration_transitions == 12000
            print(f"[CALIBRATE] {c}: same 12000 normal-null transitions", flush=True)
        torch.save(dict(revision=REVISION, spec=source.v76.SPEC, raw_weights=weights,
            normal_null_transitions=12000, phases={c: r.phase for c, r in references.items()}), path)
    return references


def reference_metadata(reference):
    result = {k: v for k, v in reference.metadata().items() if k != "phase"}
    result["bank"].update(conditional_mean="v78 local linear mean on the fixed v89 expanded reference support",
        conditional_mean_source=reference.bank.correction.bank.coverage_condition,
        conditional_mean_changed=True, conditional_covariance_changed=False,
        mean_fitting_rows=reference.bank.correction.bank.tree.n,
        covariance_reference_rows=reference.bank.bank.tree.n, covariance_source="unchanged v83 reference support/components")
    return result


def native_smoke(inputs, bank, expanded, densities, errors, weight, scores, source):
    baseline = source.CrossSeedDensityBank(bank, errors, weight)
    selected = [(c, p) for c, p in inputs if
        (c["group"] == "v76_fresh_updating" and c["seed"] == 135180 and c["mode"] == "compound") or
        (c["group"] == "v61_development_replay" and c["mode"] in ("slippery", "heavy"))]
    assert len(selected) == 3
    checked, gaussians = 0, 0
    for cell, path in selected:
        trace = source.v76.v68.load_trace(path)
        old = scores[source.v78.v77.cell_key(source.v78.replay_cell(cell))]
        for window, start, stop in source.v78.v77.windows():
            query = {k: trace[k][start:stop] for k in v89.KEYS}
            if "oracle_observations" in trace:
                query["oracle_observations"] = trace["oracle_observations"][start:stop]
            inputs = bank.features.inputs(query)
            _, covariance, _ = baseline.geometry(inputs)
            old_terms = baseline.terms(query)[0]
            verify_baseline(source.v78.v77.attribution(query, old_terms, bank, cell["mode"], 0, len(inputs)),
                old["attribution"][source.MIX_CONDITION][window])
            for condition, density in densities.items():
                centers, actual_covariance, _ = density.geometry(inputs)
                direct = source.LocalLinearMean(expanded[condition]).predict(inputs)
                terms = density.terms(query)[0]
                np.testing.assert_array_equal(terms["logdet"], old_terms["logdet"])
                targets, design = bank.targets(query), bank.design(query)
                rows = np.linspace(0, len(inputs)-1, 5, dtype=int)
                for c, mode in enumerate(source.MODES):
                    np.testing.assert_array_equal(centers[mode], direct[mode])
                    np.testing.assert_array_equal(actual_covariance[mode], covariance[mode])
                    errors_physical = (targets-design@bank.coefficients[mode]-direct[mode])*bank.target_scale
                    for row in rows:
                        physical_covariance = covariance[mode][row]*bank.target_scale[:, None]*bank.target_scale[None, :]
                        expected = multivariate_normal.logpdf(errors_physical[row], cov=physical_covariance)
                        np.testing.assert_allclose(terms["likelihoods"][row, c], expected, rtol=1e-10, atol=1e-8)
                        gaussians += 1
            checked += len(inputs)
            print(f"[SMOKE] {cell['group']} {cell['mode']} {window}: covariance, means, Gaussian scores and baseline match", flush=True)
    return dict(revision=REVISION, passed=True, evaluation_paths_available=len(scores), smoke_paths=3,
        checked_transitions=checked, checked_windows=6, gaussian_checks=gaussians,
        both_expanded_means_match=True, covariance_arrays_match_v83=True,
        baseline_attribution_matches_v83=True, physical_gaussian_scores_match=True,
        environment_steps=0, optimizer_updates=0, readout_controls_actor=False)


def evaluate(args, inputs, references, densities, scores, source):
    from scripts.v73_recursive_mode_readout import observe_likelihoods

    weight = next(iter(densities.values())).raw_weight
    records = []
    if args.resume_from:
        saved = json.loads(args.resume_from.read_text())
        assert saved["revision"] == REVISION and saved["spec"] == source.v76.SPEC
        assert saved["raw_weight"] == weight and saved["conditions"] == list(CONDITIONS)
        records = saved["records"]
        assert [source.v78.v77.cell_key(r) for r in records] == [source.v78.v77.cell_key(source.v78.replay_cell(c))
            for c, _ in inputs[:len(records)]]
    for cell, path in inputs[len(records):]:
        trace = source.v76.v68.load_trace(path)
        assert len(trace["states"]) == source.v76.SPEC["steps"]
        previous = scores[source.v78.v77.cell_key(source.v78.replay_cell(cell))]
        record = source.v78.replay_cell(cell)
        for field in ("observers", "attribution", "change_minus_nochange"):
            record[field] = {c: deepcopy(previous[field][c]) for c in CONTROLS}
        for condition, density in densities.items():
            terms = density.terms(trace)[0]
            ll = terms["likelihoods"]
            windows = source.v76.v73.window_statistics(trace, ll,
                observe_likelihoods(ll, trace["dones"], references[condition], "recursive"), cell["mode"])
            record["observers"][condition] = {w: windows[w] for w in source.v78.v77.WINDOWS}
            details = {w: source.v78.v77.attribution(trace, terms, density.bank, cell["mode"], a, b)
                for w, a, b in source.v78.v77.windows()}
            verify_frozen_volume(details, previous["attribution"][source.MIX_CONDITION])
            record["attribution"][condition] = details
            normal = record if cell["mode"] == "normal" else next(r for r in records if r["group"] == record["group"]
                and r["seed"] == cell["seed"] and r["updating"] == cell["updating"] and r["mode"] == "normal")
            record["change_minus_nochange"][condition] = {w: record["observers"][condition][w]["mean_coefficient"]
                -normal["observers"][condition][w]["mean_coefficient"] for w in source.v76.v68.BASE.CHANGE_WINDOWS}
        records.append(record)
        (args.checkpoint_dir/"progress.json").write_text(json.dumps(dict(revision=REVISION, spec=source.v76.SPEC,
            raw_weight=weight, conditions=list(CONDITIONS), records=records), allow_nan=False)+"\n")
        print(f"[EVAL] {len(records)}/{len(inputs)} {record['group']} seed{cell['seed']} {cell['mode']} updating={cell['updating']}", flush=True)
    return records


def run(args):
    sys.path.insert(0, str(args.v83_dir))
    from scripts import analyze_general_ood_v83_cross_seed_covariance as source

    torch.set_num_threads(2)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    pilot = json.loads((args.v89_dir/"run_results/pilot/reference_prefix.json").read_text())
    scores = json.loads((args.v83_dir/"run_results/cross-seed-covariance/cross_seed_covariance.json").read_text())
    saved = torch.load(args.v76_dir/"run_checkpoints/formal/behavior_coverage_references.pt", map_location="cpu", weights_only=False)
    assert pilot["revision"] == v89.REVISION and not pilot["smoke"] and len(pilot["folds"]) == 6
    assert scores["revision"] == source.REVISION and saved["revision"] == source.v76.REVISION
    assert scores["spec"] == saved["spec"] == source.v76.SPEC
    bank = saved["references"]["updating_mixed"].bank
    selection = scores["reference_selection"]
    weight = selection["selection"]["raw_weight"]
    assert weight == pilot["raw_weight"] and bank.tree.n == 23000
    groups = source.groups_from_blocks(selection["reference_blocks"], bank.tree.n)
    stored = args.v76_dir/"run_checkpoints/formal/traces"
    traces = {}
    for seed in v89.SEEDS:
        with np.load(stored/f"density_extension_seed{seed}_updating_normal.npz") as data:
            traces[seed] = {k: data[k] for k in (*v89.KEYS, "oracle_observations")}
    v89.verify_sources(bank, selection["reference_blocks"], traces)
    expanded, _ = v89.expand_banks(bank, groups, traces)
    cached = torch.load(args.v83_dir/"run_checkpoints/cross-seed-covariance/cross_seed_errors.pt", map_location="cpu", weights_only=False)
    assert cached["revision"] == source.REVISION and cached["rows"] == bank.tree.n and cached["ridge_alpha"] == source.ALPHA
    densities = make_densities(bank, expanded, cached["errors"], weight)
    inputs = source.v76.evaluation_inputs(args, stored)
    index = {source.v78.v77.cell_key(r): r for r in scores["records"]}
    assert len(inputs) == len(index) == 65
    assert {source.v78.v77.cell_key(source.v78.replay_cell(c)) for c, _ in inputs} == set(index)
    assert all(p.is_file() for _, p in inputs), "missing evaluation trace"
    null_dir = args.v69_dir/"run_checkpoints/reference/traces"
    null_paths = [source.v76.v69.trace_path(null_dir, c) for c in source.v76.v69.reference_cells(source.v76.v69.SPEC)
        if c["role"] == "normal_null"]
    assert len(null_paths) == 2 and all(p.is_file() for p in null_paths), "missing normal-null trace"
    if args.smoke:
        result = native_smoke(inputs, bank, expanded, densities, cached["errors"], weight, index, source)
        name = "native_smoke.json"
    else:
        references = freeze_references(args, densities, source)
        records = evaluate(args, inputs, references, densities, index, source)
        result = dict(revision=REVISION, source_revision=source.REVISION, spec=source.v76.SPEC,
            conditions=list(CONDITIONS), controls_reused=list(CONTROLS), raw_weight=weight,
            mean_fitting_rows=25200, covariance_reference_rows=23000,
            additional_inputs_per_condition=2200, additional_labels_per_condition=11000,
            environment_steps=0, optimizer_updates=0, additional_counterfactual_steps=0, readout_controls_actor=False,
            forward_models_refitted=False, scaling_refitted=False, weight_refitted=False,
            conditional_covariances_changed=False, mean_fitting_support_changed=True,
            normal_null_transitions_per_condition=12000, normal_thresholds_recalibrated=True, temporal_rules_changed=False,
            all_paths_development_replays=True, all_fixed_competitor_volume_terms_match_v83=True,
            reference_metadata={c: reference_metadata(r) for c, r in references.items()},
            records=records, summary=source.v78.v77.summarize(records, (*CONTROLS, *CONDITIONS)),
            decision="development-coverage-replay-not-adoption",
            limitations=["Previously inspected paths, one source training seed and mild known modes; no independent confirmation.",
                "Each addition uses 2200 more inputs/11000 cached labels, with an equal-count suffix control.",
                "Covariance remains fitted on the original support and pre-expansion error bank.",
                "This is an offline readout comparison, not actor-feedback or RL return evidence."])
        name = "coverage_replay.json"
    (args.result_dir/name).write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] {REVISION}: {args.result_dir/name}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("v89-dir", "v83-dir", "v76-dir", "v74-dir", "v73-dir", "v72-dir", "v71-dir", "v69-dir", "v70-dir", "v61-dir", "v62-dir", "checkpoint-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
