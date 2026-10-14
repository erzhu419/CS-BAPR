"""Freeze a reference-only covariance mixture, then replay the fixed development set."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from scripts import analyze_general_ood_v81_prediction_error_covariance as v81
from scripts.v73_recursive_mode_readout import RecursiveModeReference, observe_likelihoods
from scripts.v77_mean_covariance import MODES, check_shared_forward
from scripts.v78_local_linear_mean import CONDITIONS as V78_CONTROLS, LocalLinearMean, correction_terms
from scripts.v81_prediction_error_covariance import ALPHA, CONDITION as LOO_CONTROL, PredictionErrorDensityBank
from scripts.v82_reference_covariance_mix import CONDITION, VALIDATION_ROWS_PER_SEED, ReferenceMixedDensityBank, fit_selection, reference_groups

REVISION = "general_ood_v82_reference_covariance_mix"
CONTROLS = (*V78_CONTROLS, LOO_CONTROL)
v78, v76 = v81.v78, v81.v76


def freeze_selection(args, bank):
    path = args.checkpoint_dir/"reference_selection.json"
    if path.is_file():
        saved = json.loads(path.read_text())
        assert saved["revision"] == REVISION and saved["ridge_alpha"] == ALPHA and saved["spec"] == v76.SPEC
        assert saved["selection"]["rows_per_seed"] == VALIDATION_ROWS_PER_SEED
    else:
        groups, blocks = reference_groups(args, bank)
        selection = fit_selection(bank, groups)
        saved = dict(revision=REVISION, ridge_alpha=ALPHA, spec=v76.SPEC, reference_blocks=blocks,
                     bank_alignment_verified=True, selection=selection)
        path.write_text(json.dumps(saved, indent=2, allow_nan=False)+"\n")
    print(f"[SELECT] raw_weight={saved['selection']['raw_weight']:.8f}", flush=True)
    return saved


def freeze_reference(args, density):
    cache = args.checkpoint_dir/"mixed_phase.pt"
    if cache.is_file():
        saved = torch.load(cache, map_location="cpu", weights_only=False)
        assert saved["revision"] == REVISION and saved["raw_weight"] == density.raw_weight
        ref = RecursiveModeReference()
        ref.bank, ref.phase, ref.normal_null_calibration_transitions = density, saved["phase"], saved["normal_null_transitions"]
    else:
        stored = args.v69_dir/"run_checkpoints/reference/traces"
        nulls = [v76.v68.load_trace(v76.v69.trace_path(stored, c)) for c in v76.v69.reference_cells(v76.v69.SPEC)
                 if c["role"] == "normal_null"]
        ref = RecursiveModeReference.calibrate(density, nulls)
        torch.save(dict(revision=REVISION, raw_weight=density.raw_weight,
            normal_null_transitions=ref.normal_null_calibration_transitions, phase=ref.phase), cache)
        print("[CALIBRATE] same 12000 normal-null transitions", flush=True)
    return ref


def geometry_statistics(diagnostic, start, stop):
    result = v81.geometry_statistics(diagnostic, start, stop)
    for i, mode in enumerate(MODES):
        result["modes"][mode].update({k: float(diagnostic[k][start:stop, i].mean())
            for k in ("raw_covariance_trace", "loo_covariance_trace")})
    return result


def evaluate(args, inputs, reference, density, scores):
    records = []
    if args.resume_from:
        saved = json.loads(args.resume_from.read_text())
        assert saved["revision"] == REVISION and saved["spec"] == v76.SPEC
        assert saved["raw_weight"] == density.raw_weight and saved["ridge_alpha"] == ALPHA
        records = saved["records"]
        assert [v78.v77.cell_key(r) for r in records] == [v78.v77.cell_key(v78.replay_cell(c)) for c, _ in inputs[:len(records)]]
    for cell, path in inputs[len(records):]:
        trace = v76.v68.load_trace(path)
        assert len(trace["states"]) == v76.SPEC["steps"]
        terms, diagnostic = density.terms(trace)
        ll = terms["likelihoods"]
        windows = v76.v73.window_statistics(trace, ll,
            observe_likelihoods(ll, trace["dones"], reference, "recursive"), cell["mode"])
        previous = scores[v78.v77.cell_key(v78.replay_cell(cell))]
        record = v78.replay_cell(cell)
        record.update(observers=deepcopy(previous["observers"]), attribution=deepcopy(previous["attribution"]),
                      change_minus_nochange=deepcopy(previous["change_minus_nochange"]))
        record["observers"][CONDITION] = {w: windows[w] for w in v78.v77.WINDOWS}
        record["attribution"][CONDITION] = {w: v78.v77.attribution(trace, terms, density.bank, cell["mode"], a, b)
                                            for w, a, b in v78.v77.windows()}
        v81.verify_mean_statistics(record["attribution"][CONDITION], previous)
        record["mixture_geometry"] = {w: geometry_statistics(diagnostic, a, b) for w, a, b in v78.v77.windows()}
        normal = record if cell["mode"] == "normal" else next(r for r in records if r["group"] == record["group"]
            and r["seed"] == cell["seed"] and r["updating"] == cell["updating"] and r["mode"] == "normal")
        record["change_minus_nochange"][CONDITION] = {w: record["observers"][CONDITION][w]["mean_coefficient"]
            -normal["observers"][CONDITION][w]["mean_coefficient"] for w in v76.v68.BASE.CHANGE_WINDOWS}
        records.append(record)
        (args.checkpoint_dir/"progress.json").write_text(json.dumps(dict(revision=REVISION, spec=v76.SPEC,
            raw_weight=density.raw_weight, ridge_alpha=ALPHA, records=records), allow_nan=False)+"\n")
        print(f"[EVAL] {len(records)}/{len(inputs)} {record['group']} seed{cell['seed']} {cell['mode']} updating={cell['updating']}", flush=True)
    return records


def native_smoke(inputs, density, banks, scores):
    result = v81.native_smoke(inputs, density, banks, scores)
    cell, path = next((c, p) for c, p in inputs if c["group"] == "v76_fresh_updating" and c["mode"] == "compound")
    trace = v76.v68.load_trace(path)
    query = {k: trace[k][3800:3850] for k in v76.v74.KEYS}
    loo = PredictionErrorDensityBank(banks["updating"]).log_likelihoods(query)
    raw = correction_terms(query, banks, LocalLinearMean(banks["updating"]))["linear_mean_updating_cov"]["likelihoods"]
    for weight, expected in ((0., loo), (1., raw)):
        actual = ReferenceMixedDensityBank(banks["updating"], weight).log_likelihoods(query)
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-8)
    result.update(revision=REVISION, covariance_endpoints_match_v78_v81=True)
    return result


def run(args):
    torch.set_num_threads(2)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    saved = torch.load(args.v76_dir/"run_checkpoints/formal/behavior_coverage_references.pt", map_location="cpu", weights_only=False)
    assert saved["revision"] == v76.REVISION and saved["spec"] == v76.SPEC
    banks = {"frozen": saved["references"]["frozen_mixed"].bank, "updating": saved["references"]["updating_mixed"].bank}
    check_shared_forward(banks)
    selection = freeze_selection(args, banks["updating"])
    density = ReferenceMixedDensityBank(banks["updating"], selection["selection"]["raw_weight"], selection["selection"])
    # Evaluation scores are not loaded until reference-only selection has been frozen.
    scores = json.loads((args.v81_dir/"run_results/prediction-error-covariance/prediction_error_covariance.json").read_text())
    original = json.loads((args.v78_dir/"run_results/mean-correction/local_linear_mean.json").read_text())
    assert scores["revision"] == v81.REVISION and original["revision"] == v78.REVISION
    assert scores["spec"] == original["spec"] == v76.SPEC
    index = {v78.v77.cell_key(r): r for r in scores["records"]}
    for record in original["records"]:
        prior = index[v78.v77.cell_key(record)]
        for field in ("observers", "attribution", "change_minus_nochange"):
            assert {c: prior[field][c] for c in V78_CONTROLS} == record[field]
    inputs = v76.evaluation_inputs(args, args.v76_dir/"run_checkpoints/formal/traces")
    assert len(inputs) == len(index) == 65 and {v78.v77.cell_key(v78.replay_cell(c)) for c, _ in inputs} == set(index)
    if args.smoke:
        result, name = native_smoke(inputs, density, banks, index), "native_smoke.json"
    else:
        reference = freeze_reference(args, density)
        assert reference.normal_null_calibration_transitions == 12000
        records = evaluate(args, inputs, reference, density, index)
        result = dict(revision=REVISION, spec=v76.SPEC, new_condition=CONDITION, controls_reused=list(CONTROLS),
            environment_steps=0, optimizer_updates=0, readout_controls_actor=False,
            forward_models_refitted=False, scaling_refitted=False, temporal_rules_changed=False,
            conditional_mean_rule_changed=False, conditional_covariances_changed=True, mean_errors_reproduce_v78=True,
            ridge_alpha=ALPHA, all_paths_development_replays=True,
            reference_metadata={k: v for k, v in reference.metadata().items() if k != "phase"},
            records=records, summary=v78.v77.summarize(records, (*CONTROLS, CONDITION)),
            decision="development-reference-selected-covariance-mixture-not-confirmation",
            limitations=["Development replays and one source training seed; no RL return or new-environment evidence.",
                "Six controls are reused statistics, not independent replications.",
                "Reference-seed validation covers known counterfactual modes, not arbitrary query extrapolation."])
        name = "reference_covariance_mix.json"
    result["reference_selection"] = selection
    path = args.result_dir/name
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] {REVISION}: {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("v81-dir", "v78-dir", "v76-dir", "v74-dir", "v73-dir", "v72-dir", "v71-dir", "v69-dir", "v70-dir", "v61-dir", "v62-dir", "checkpoint-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
