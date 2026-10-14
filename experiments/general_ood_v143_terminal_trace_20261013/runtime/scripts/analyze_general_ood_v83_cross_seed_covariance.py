"""Select a nested-cross-fitted error covariance before fixed development replays."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from scripts import analyze_general_ood_v82_reference_covariance_mix as v82
from scripts.v73_recursive_mode_readout import RecursiveModeReference, observe_likelihoods
from scripts.v77_mean_covariance import MODES, check_shared_forward
from scripts.v78_local_linear_mean import LocalLinearMean, correction_terms
from scripts.v81_prediction_error_covariance import ALPHA, error_covariance
from scripts.v82_reference_covariance_mix import VALIDATION_ROWS_PER_SEED
from scripts.v83_cross_seed_covariance import REVISION, ERROR_CONDITION, MIX_CONDITION, CrossSeedDensityBank, cross_seed_errors, fit_selection, groups_from_blocks

v78, v76, CONTROLS = v82.v78, v82.v76, v82.CONTROLS


def freeze_selection(args, bank):
    provenance = json.loads((args.v82_dir/"run_checkpoints/reference-covariance-mix/reference_selection.json").read_text())
    assert provenance["revision"] == v82.REVISION and provenance["spec"] == v76.SPEC and provenance["bank_alignment_verified"]
    groups = groups_from_blocks(provenance["reference_blocks"], bank.tree.n)
    assert len(groups) == bank.density_transitions == 23000 and len(np.unique(groups)) == 6
    path = args.checkpoint_dir/"nested_selection.json"
    if path.is_file():
        saved = json.loads(path.read_text())
        assert saved["revision"] == REVISION and saved["ridge_alpha"] == ALPHA and saved["spec"] == v76.SPEC
        assert saved["selection"]["rows_per_seed"] == VALIDATION_ROWS_PER_SEED
    else:
        selection = fit_selection(bank, groups, args.checkpoint_dir/"validation")
        saved = dict(revision=REVISION, ridge_alpha=ALPHA, spec=v76.SPEC, selection=selection,
            provenance_reused_from=v82.REVISION, reference_blocks=provenance["reference_blocks"])
        path.write_text(json.dumps(saved, indent=2, allow_nan=False)+"\n")
    print(f"[SELECT] raw_weight={saved['selection']['raw_weight']:.8f}", flush=True)
    return saved, groups


def freeze_errors(args, bank, groups):
    path = args.checkpoint_dir/"cross_seed_errors.pt"
    if path.is_file():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        assert saved["revision"] == REVISION and saved["ridge_alpha"] == ALPHA
        assert saved["rows"] == bank.density_transitions
        errors = saved["errors"]
    else:
        errors = cross_seed_errors(bank, groups)
        torch.save(dict(revision=REVISION, ridge_alpha=ALPHA, rows=bank.density_transitions, errors=errors), path)
    return errors


def freeze_references(args, densities):
    path = args.checkpoint_dir/"cross_seed_phases.pt"
    weights = {c: d.raw_weight for c, d in densities.items()}
    if path.is_file():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        assert saved["revision"] == REVISION and saved["raw_weights"] == weights
        references = {}
        for name, density in densities.items():
            ref = RecursiveModeReference()
            ref.bank, ref.phase, ref.normal_null_calibration_transitions = density, saved["phases"][name], 12000
            references[name] = ref
    else:
        stored = args.v69_dir/"run_checkpoints/reference/traces"
        nulls = [v76.v68.load_trace(v76.v69.trace_path(stored, c)) for c in v76.v69.reference_cells(v76.v69.SPEC)
                 if c["role"] == "normal_null"]
        references = {}
        for name, density in densities.items():
            references[name] = RecursiveModeReference.calibrate(density, nulls)
            assert references[name].normal_null_calibration_transitions == 12000
            print(f"[CALIBRATE] {name}: same 12000 normal-null transitions", flush=True)
        torch.save(dict(revision=REVISION, raw_weights=weights,
                        phases={c: r.phase for c, r in references.items()}), path)
    return references


def geometry_statistics(diagnostic, start, stop):
    return {m: {k: float(diagnostic[k][start:stop, i].mean()) for k in diagnostic}
            for i, m in enumerate(MODES)}


def evaluate(args, inputs, references, densities, scores):
    weights = {c: d.raw_weight for c, d in densities.items()}
    records = []
    if args.resume_from:
        saved = json.loads(args.resume_from.read_text())
        assert saved["revision"] == REVISION and saved["spec"] == v76.SPEC
        assert saved["raw_weights"] == weights and saved["ridge_alpha"] == ALPHA
        records = saved["records"]
        assert [v78.v77.cell_key(r) for r in records] == [v78.v77.cell_key(v78.replay_cell(c)) for c, _ in inputs[:len(records)]]
    for cell, path in inputs[len(records):]:
        trace = v76.v68.load_trace(path)
        assert len(trace["states"]) == v76.SPEC["steps"]
        previous = scores[v78.v77.cell_key(v78.replay_cell(cell))]
        record = v78.replay_cell(cell)
        record.update({k: deepcopy(previous[k]) for k in ("observers", "attribution", "change_minus_nochange")})
        record["cross_seed_geometry"] = {}
        for name, density in densities.items():
            terms, diagnostic = density.terms(trace)
            ll = terms["likelihoods"]
            windows = v76.v73.window_statistics(trace, ll,
                observe_likelihoods(ll, trace["dones"], references[name], "recursive"), cell["mode"])
            record["observers"][name] = {w: windows[w] for w in v78.v77.WINDOWS}
            record["attribution"][name] = {w: v78.v77.attribution(trace, terms, density.bank, cell["mode"], a, b)
                                            for w, a, b in v78.v77.windows()}
            v82.v81.verify_mean_statistics(record["attribution"][name], previous)
            record["cross_seed_geometry"][name] = {w: geometry_statistics(diagnostic, a, b) for w, a, b in v78.v77.windows()}
            normal = record if cell["mode"] == "normal" else next(r for r in records if r["group"] == record["group"]
                and r["seed"] == cell["seed"] and r["updating"] == cell["updating"] and r["mode"] == "normal")
            record["change_minus_nochange"][name] = {w: record["observers"][name][w]["mean_coefficient"]
                -normal["observers"][name][w]["mean_coefficient"] for w in v76.v68.BASE.CHANGE_WINDOWS}
        records.append(record)
        (args.checkpoint_dir/"progress.json").write_text(json.dumps(dict(revision=REVISION, spec=v76.SPEC,
            raw_weights=weights, ridge_alpha=ALPHA, records=records), allow_nan=False)+"\n")
        print(f"[EVAL] {len(records)}/{len(inputs)} {record['group']} seed{cell['seed']} {cell['mode']} updating={cell['updating']}", flush=True)
    return records


def native_smoke(inputs, density, banks, scores):
    selected = [(c, p) for c, p in inputs if (c["group"] == "v76_fresh_updating" and c["mode"] == "compound" and c["seed"] == 135180)
                or (c["group"] == "v61_development_replay" and c["mode"] == "heavy")]
    assert len(selected) == 2
    checked = 0
    for cell, path in selected:
        trace = v76.v68.load_trace(path)
        previous = scores[v78.v77.cell_key(v78.replay_cell(cell))]
        attribution = {}
        for w, a, b in v78.v77.windows():
            query = {k: trace[k][a:b] for k in v76.v74.KEYS}
            inputs = density.bank.features.inputs(query)
            means, _, _ = density.geometry(inputs)
            expected = LocalLinearMean(density.bank).predict(inputs)
            for mode in MODES:
                np.testing.assert_array_equal(means[mode], expected[mode])
            terms, _ = density.terms(query)
            np.testing.assert_allclose(terms["likelihoods"], density.log_likelihoods(query), rtol=1e-11, atol=1e-8)
            raw = correction_terms(query, banks, LocalLinearMean(density.bank))["linear_mean_updating_cov"]
            np.testing.assert_allclose(terms["mean_error_rms"], raw["mean_error_rms"], rtol=1e-11, atol=1e-11)
            endpoint = CrossSeedDensityBank(density.bank, density.errors, 1.)
            np.testing.assert_allclose(endpoint.log_likelihoods(query), raw["likelihoods"], rtol=1e-11, atol=1e-8)
            pure = CrossSeedDensityBank(density.bank, density.errors, 0.)
            _, covariance, _ = pure.geometry(inputs)
            bank = density.bank
            _, indices = bank.tree.query((inputs-bank.features.input_mean)/bank.features.input_scale, k=64)
            for mode in MODES:
                expected_cov = np.asarray([error_covariance(e)[0] for e in density.errors[mode][indices]])
                np.testing.assert_array_equal(covariance[mode], expected_cov)
            attribution[w] = v78.v77.attribution(query, terms, bank, cell["mode"], 0, b-a)
            checked += b-a
        v82.v81.verify_mean_statistics(attribution, previous)
    return dict(revision=REVISION, passed=True, evaluation_paths=2, checked_transitions=checked,
        means_match_v78=True, mean_errors_match_v78=True, raw_endpoint_matches_v78=True,
        error_endpoint_matches_reference_moments=True, density_batch_matches_readout=True,
        environment_steps=0, optimizer_updates=0)


def run(args):
    torch.set_num_threads(2)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    saved = torch.load(args.v76_dir/"run_checkpoints/formal/behavior_coverage_references.pt", map_location="cpu", weights_only=False)
    assert saved["revision"] == v76.REVISION and saved["spec"] == v76.SPEC
    banks = {"frozen": saved["references"]["frozen_mixed"].bank, "updating": saved["references"]["updating_mixed"].bank}
    check_shared_forward(banks)
    selection, groups = freeze_selection(args, banks["updating"])
    errors = freeze_errors(args, banks["updating"], groups)
    weight = selection["selection"]["raw_weight"]
    densities = {ERROR_CONDITION: CrossSeedDensityBank(banks["updating"], errors, 0., selection["selection"])}
    if weight > 0.:
        densities[MIX_CONDITION] = CrossSeedDensityBank(banks["updating"], errors, weight, selection["selection"])
    selected_condition = MIX_CONDITION if weight > 0. else ERROR_CONDITION
    # No query outcomes or normal-null labels enter the frozen nested selection.
    scores = json.loads((args.v81_dir/"run_results/prediction-error-covariance/prediction_error_covariance.json").read_text())
    original = json.loads((args.v78_dir/"run_results/mean-correction/local_linear_mean.json").read_text())
    assert scores["revision"] == v82.v81.REVISION and original["revision"] == v78.REVISION
    assert scores["spec"] == original["spec"] == v76.SPEC
    index = {v78.v77.cell_key(r): r for r in scores["records"]}
    for record in original["records"]:
        prior = index[v78.v77.cell_key(record)]
        for field in ("observers", "attribution", "change_minus_nochange"):
            assert {c: prior[field][c] for c in v82.V78_CONTROLS} == record[field]
    inputs = v76.evaluation_inputs(args, args.v76_dir/"run_checkpoints/formal/traces")
    assert len(inputs) == len(index) == 65 and {v78.v77.cell_key(v78.replay_cell(c)) for c, _ in inputs} == set(index)
    if args.smoke:
        result, name = native_smoke(inputs, densities[selected_condition], banks, index), "native_smoke.json"
    else:
        references = freeze_references(args, densities)
        records = evaluate(args, inputs, references, densities, index)
        result = dict(revision=REVISION, spec=v76.SPEC, new_conditions=list(densities), selected_condition=selected_condition,
            controls_reused=list(CONTROLS), environment_steps=0, optimizer_updates=0, readout_controls_actor=False,
            forward_models_refitted=False, scaling_refitted=False, temporal_rules_changed=False,
            conditional_mean_rule_changed=False, conditional_covariances_changed=True, mean_errors_reproduce_v78=True,
            ridge_alpha=ALPHA, all_paths_development_replays=True,
            reference_metadata={c: {k: v for k, v in r.metadata().items() if k != "phase"} for c, r in references.items()},
            records=records, summary=v78.v77.summarize(records, (*CONTROLS, *densities)),
            decision="development-nested-cross-seed-covariance-not-confirmation",
            limitations=["Known-mode reference selection and development replays, not new RL return or general OOD evidence.",
                "Cross-seed errors use smaller fitting banks than deployment; covariance can overstate full-bank prediction error.",
                "Six controls are reused statistics, not independent replications."])
        name = "cross_seed_covariance.json"
    result["reference_selection"] = selection
    result["full_error_bank_rms"] = {m: float(np.sqrt(np.mean(errors[m]**2))) for m in MODES}
    result["selected_condition"] = selected_condition
    path = args.result_dir/name
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] {REVISION}: {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("v82-dir", "v81-dir", "v78-dir", "v76-dir", "v74-dir", "v73-dir", "v72-dir", "v71-dir", "v69-dir", "v70-dir", "v61-dir", "v62-dir", "checkpoint-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
