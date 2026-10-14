"""Isolate reference prediction-error covariance under fixed v78 means."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from scripts import analyze_general_ood_v78_local_linear_mean as v78
from scripts.v73_recursive_mode_readout import RecursiveModeReference, observe_likelihoods
from scripts.v77_mean_covariance import MODES, check_shared_forward
from scripts.v78_local_linear_mean import CONDITIONS as CONTROLS, LocalLinearMean, correction_terms
from scripts.v81_prediction_error_covariance import ALPHA, CONDITION, PredictionErrorDensityBank

REVISION = "general_ood_v81_prediction_error_covariance"
v76 = v78.v76


def geometry_statistics(diagnostic, start, stop):
    return dict(mean_query_leverage=float(diagnostic["query_leverage"][start:stop].mean()),
        modes={m: {k: float(diagnostic[k][start:stop, i].mean())
            for k in ("covariance_trace", "reference_loo_rms", "shrinkage")} for i, m in enumerate(MODES)})


def verify_mean_statistics(attribution, previous):
    for window, values in attribution.items():
        np.testing.assert_allclose(values["mean_true_prediction_error_rms"],
            previous["attribution"]["linear_mean_updating_cov"][window]["mean_true_prediction_error_rms"], rtol=1e-9, atol=1e-10)


def freeze_reference(args, density):
    cache = args.checkpoint_dir/"prediction_error_phase.pt"
    if cache.is_file():
        saved = torch.load(cache, map_location="cpu", weights_only=False)
        assert saved["revision"] == REVISION and saved["spec"] == v76.SPEC and saved["ridge_alpha"] == ALPHA
        ref = RecursiveModeReference()
        ref.bank, ref.phase, ref.normal_null_calibration_transitions = density, saved["phase"], saved["normal_null_transitions"]
    else:
        stored = args.v69_dir/"run_checkpoints/reference/traces"
        nulls = [v76.v68.load_trace(v76.v69.trace_path(stored, c)) for c in v76.v69.reference_cells(v76.v69.SPEC)
                 if c["role"] == "normal_null"]
        ref = RecursiveModeReference.calibrate(density, nulls)
        torch.save(dict(revision=REVISION, spec=v76.SPEC, ridge_alpha=ALPHA,
            normal_null_transitions=ref.normal_null_calibration_transitions, phase=ref.phase), cache)
        print(f"[CALIBRATE] {CONDITION}: same saved normal-null population", flush=True)
    return ref


def evaluate(args, inputs, reference, density, scores):
    records = []
    if args.resume_from:
        saved = json.loads(args.resume_from.read_text())
        assert saved["revision"] == REVISION and saved["spec"] == v76.SPEC and saved["ridge_alpha"] == ALPHA
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
        observers, attribution = deepcopy(previous["observers"]), deepcopy(previous["attribution"])
        observers[CONDITION] = {w: windows[w] for w in v78.v77.WINDOWS}
        attribution[CONDITION] = {w: v78.v77.attribution(trace, terms, density.bank, cell["mode"], a, b)
                                 for w, a, b in v78.v77.windows()}
        verify_mean_statistics(attribution[CONDITION], previous)
        record = dict(**v78.replay_cell(cell), observers=observers, attribution=attribution,
            prediction_error_geometry={w: geometry_statistics(diagnostic, a, b) for w, a, b in v78.v77.windows()})
        normal = record if cell["mode"] == "normal" else next(r for r in records if r["group"] == record["group"]
            and r["seed"] == cell["seed"] and r["updating"] == cell["updating"] and r["mode"] == "normal")
        record["change_minus_nochange"] = deepcopy(previous["change_minus_nochange"])
        record["change_minus_nochange"][CONDITION] = {w: observers[CONDITION][w]["mean_coefficient"]-normal["observers"][CONDITION][w]["mean_coefficient"]
            for w in v76.v68.BASE.CHANGE_WINDOWS}
        records.append(record)
        (args.checkpoint_dir/"progress.json").write_text(json.dumps(dict(revision=REVISION, spec=v76.SPEC, ridge_alpha=ALPHA, records=records), allow_nan=False)+"\n")
        print(f"[EVAL] {len(records)}/{len(inputs)} {record['group']} seed{cell['seed']} {cell['mode']} updating={cell['updating']}", flush=True)
    return records


def native_smoke(inputs, density, banks, scores):
    selected = [(c, p) for c, p in inputs if (c["group"] == "v76_fresh_updating" and c["mode"] == "compound" and c["seed"] == 135180)
                or (c["group"] == "v61_development_replay" and c["mode"] == "heavy")]
    assert len(selected) == 2
    checked, records = 0, []
    original = LocalLinearMean(banks["updating"])
    for cell, path in selected:
        trace = v76.v68.load_trace(path)
        previous = scores[v78.v77.cell_key(v78.replay_cell(cell))]
        attribution, details = {}, {}
        for w, a, b in v78.v77.windows():
            query = {k: trace[k][a:b] for k in v76.v74.KEYS}
            inputs = density.bank.features.inputs(query)
            means, _, _ = density.geometry(inputs)
            expected = original.predict(inputs)
            for mode in MODES:
                np.testing.assert_allclose(means[mode], expected[mode], rtol=1e-11, atol=1e-11)
            terms, diagnostic = density.terms(query)
            np.testing.assert_allclose(terms["likelihoods"], density.log_likelihoods(query), rtol=1e-11, atol=1e-9)
            baseline = correction_terms(query, banks, original)
            np.testing.assert_allclose(terms["mean_error_rms"], baseline["linear_mean_updating_cov"]["mean_error_rms"], rtol=1e-11, atol=1e-11)
            attribution[w] = v78.v77.attribution(query, terms, density.bank, cell["mode"], 0, b-a)
            details[w] = geometry_statistics(diagnostic, 0, b-a)
            checked += b-a
        verify_mean_statistics(attribution, previous)
        records.append(dict(**v78.replay_cell(cell), attribution=attribution, geometry=details))
    return dict(revision=REVISION, passed=True, evaluation_paths=2, checked_transitions=checked,
        means_match_v78=True, mean_errors_match_v78=True, density_batch_matches_readout=True,
        environment_steps=0, optimizer_updates=0, records=records)


def run(args):
    torch.set_num_threads(2)
    saved = torch.load(args.v76_dir/"run_checkpoints/formal/behavior_coverage_references.pt", map_location="cpu", weights_only=False)
    scores = json.loads((args.v78_dir/"run_results/mean-correction/local_linear_mean.json").read_text())
    assert saved["revision"] == v76.REVISION and scores["revision"] == v78.REVISION
    assert saved["spec"] == scores["spec"] == v76.SPEC
    banks = {"frozen": saved["references"]["frozen_mixed"].bank, "updating": saved["references"]["updating_mixed"].bank}
    check_shared_forward(banks)
    density = PredictionErrorDensityBank(banks["updating"])
    inputs = v76.evaluation_inputs(args, args.v76_dir/"run_checkpoints/formal/traces")
    index = {v78.v77.cell_key(r): r for r in scores["records"]}
    assert len(inputs) == len(index) == 65 and {v78.v77.cell_key(v78.replay_cell(c)) for c, _ in inputs} == set(index)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        result, name = native_smoke(inputs, density, banks, index), "native_smoke.json"
    else:
        reference = freeze_reference(args, density)
        assert reference.normal_null_calibration_transitions == scores["reference_metadata"]["linear_mean_updating_cov"]["normal_null_calibration_transitions"]
        records = evaluate(args, inputs, reference, density, index)
        result = dict(revision=REVISION, source_revision=v78.REVISION, spec=v76.SPEC,
            new_condition=CONDITION, controls_reused_from_v78=list(CONTROLS),
            environment_steps=0, optimizer_updates=0, readout_controls_actor=False,
            forward_models_refitted=False, scaling_refitted=False, temporal_rules_changed=False,
            conditional_covariances_changed=True, conditional_mean_rule_changed=False,
            mean_errors_reproduce_v78=True, query_residual_used_for_geometry=False,
            ridge_alpha=ALPHA, all_paths_development_replays=True,
            reference_metadata={k: v for k, v in reference.metadata().items() if k != "phase"},
            records=records, summary=v78.v77.summarize(records, (*CONTROLS, CONDITION)),
            decision="development-prediction-error-covariance-not-confirmation",
            limitations=["Development replays, one source training seed and mild known modes; not RL return evidence.",
                "Local reference leave-one-out errors do not measure query extrapolation error; no query-leverage inflation is applied.",
                "OAS estimates the zero-centered prediction-error second moment without bias recentering.",
                "The five baseline statistics are reused from v78, not independent replications."])
        name = "prediction_error_covariance.json"
    path = args.result_dir/name
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] {REVISION}: {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("v78-dir", "v76-dir", "v74-dir", "v73-dir", "v72-dir", "v71-dir", "v69-dir", "v70-dir", "v61-dir", "v62-dir", "checkpoint-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
