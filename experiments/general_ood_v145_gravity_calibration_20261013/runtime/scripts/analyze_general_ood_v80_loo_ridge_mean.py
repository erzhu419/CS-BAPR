"""Matched development replay of reference-only ridge selection."""
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
from scripts.v77_mean_covariance import MeanCovarianceBank, check_shared_forward
from scripts.v78_local_linear_mean import CONDITIONS as CONTROL_CONDITIONS, CONTROLS, correction_terms
from scripts.v80_loo_ridge_mean import ALPHAS, CONDITIONS, LooRidgeDensityBank, LooRidgeMean

REVISION = "general_ood_v80_loo_ridge_mean"
v76 = v78.v76


def selection_statistics(details, start, stop):
    return dict(alpha_fractions={str(a): float(np.mean(details["selected_alpha"][start:stop] == a)) for a in ALPHAS},
        mean_reference_loo_mse_by_alpha={str(a): float(details["loo_mse"][start:stop, i].mean()) for i, a in enumerate(ALPHAS)},
        mean_selected_query_leverage=float(details["ridge_query_leverage"][start:stop].mean()),
        mean_selected_prediction_weight_l1=float(details["prediction_weight_l1"][start:stop].mean()))


def freeze_references(args, banks, correction):
    cache = args.checkpoint_dir/"loo_phases.pt"
    references = {}
    if cache.is_file():
        saved = torch.load(cache, map_location="cpu", weights_only=False)
        assert saved["revision"] == REVISION and saved["spec"] == v76.SPEC and tuple(saved["alphas"]) == ALPHAS
        for condition, phase in saved["phases"].items():
            ref = RecursiveModeReference()
            ref.bank = LooRidgeDensityBank(banks, correction, CONDITIONS[condition])
            ref.phase, ref.normal_null_calibration_transitions = phase, saved["normal_null_transitions"]
            references[condition] = ref
    else:
        stored = args.v69_dir/"run_checkpoints/reference/traces"
        nulls = [v76.v68.load_trace(v76.v69.trace_path(stored, c)) for c in v76.v69.reference_cells(v76.v69.SPEC)
                 if c["role"] == "normal_null"]
        for condition, source in CONDITIONS.items():
            references[condition] = RecursiveModeReference.calibrate(LooRidgeDensityBank(banks, correction, source), nulls)
            print(f"[CALIBRATE] {condition}: same saved normal-null population", flush=True)
        torch.save(dict(revision=REVISION, spec=v76.SPEC, alphas=ALPHAS,
            normal_null_transitions=next(iter(references.values())).normal_null_calibration_transitions,
            phases={c: r.phase for c, r in references.items()}), cache)
    assert set(references) == set(CONDITIONS)
    return references


class SavedMeans:
    def __init__(self, centers):
        self.centers = centers

    def predict(self, inputs):
        assert len(inputs) == len(next(iter(self.centers.values())))
        return self.centers


def evaluate(args, inputs, references, banks, correction, scores):
    records = []
    if args.resume_from:
        saved = json.loads(args.resume_from.read_text())
        assert saved["revision"] == REVISION and saved["spec"] == v76.SPEC and tuple(saved["alphas"]) == ALPHAS
        records = saved["records"]
        assert [v78.v77.cell_key(r) for r in records] == [v78.v77.cell_key(v78.replay_cell(c)) for c, _ in inputs[:len(records)]]
    for cell, path in inputs[len(records):]:
        trace = v76.v68.load_trace(path)
        assert len(trace["states"]) == v76.SPEC["steps"]
        centers, details = correction.predict_details(banks["updating"].features.inputs(trace))
        terms = correction_terms(trace, banks, SavedMeans(centers))
        previous = scores[v78.v77.cell_key(v78.replay_cell(cell))]
        observers, attribution = deepcopy(previous["observers"]), deepcopy(previous["attribution"])
        for condition, source in CONDITIONS.items():
            values = terms["linear_mean_"+source+"_cov"]
            ll = values["likelihoods"]
            windows = v76.v73.window_statistics(trace, ll,
                observe_likelihoods(ll, trace["dones"], references[condition], "recursive"), cell["mode"])
            observers[condition] = {w: windows[w] for w in v78.v77.WINDOWS}
            attribution[condition] = {w: v78.v77.attribution(trace, values, banks["frozen"], cell["mode"], a, b)
                                     for w, a, b in v78.v77.windows()}
        record = dict(**v78.replay_cell(cell), observers=observers, attribution=attribution,
            reference_selection={w: selection_statistics(details, a, b) for w, a, b in v78.v77.windows()})
        normal = record if cell["mode"] == "normal" else next(r for r in records if r["group"] == record["group"]
            and r["seed"] == cell["seed"] and r["updating"] == cell["updating"] and r["mode"] == "normal")
        record["change_minus_nochange"] = deepcopy(previous["change_minus_nochange"])
        record["change_minus_nochange"].update({c: {w: observers[c][w]["mean_coefficient"]-normal["observers"][c][w]["mean_coefficient"]
            for w in v76.v68.BASE.CHANGE_WINDOWS} for c in CONDITIONS})
        records.append(record)
        (args.checkpoint_dir/"progress.json").write_text(json.dumps(dict(revision=REVISION, spec=v76.SPEC, alphas=ALPHAS, records=records), allow_nan=False)+"\n")
        print(f"[EVAL] {len(records)}/{len(inputs)} {record['group']} seed{cell['seed']} {cell['mode']} updating={cell['updating']}", flush=True)
    return records


def run(args):
    torch.set_num_threads(2)
    saved = torch.load(args.v76_dir/"run_checkpoints/formal/behavior_coverage_references.pt", map_location="cpu", weights_only=False)
    scores = json.loads((args.v78_dir/"run_results/mean-correction/local_linear_mean.json").read_text())
    assert saved["revision"] == v76.REVISION and scores["revision"] == v78.REVISION
    assert saved["spec"] == scores["spec"] == v76.SPEC
    banks = {"frozen": saved["references"]["frozen_mixed"].bank, "updating": saved["references"]["updating_mixed"].bank}
    check_shared_forward(banks)
    correction = LooRidgeMean(banks["updating"])
    inputs = v76.evaluation_inputs(args, args.v76_dir/"run_checkpoints/formal/traces")
    index = {v78.v77.cell_key(r): r for r in scores["records"]}
    assert len(inputs) == len(index) == 65 and {v78.v77.cell_key(v78.replay_cell(c)) for c, _ in inputs} == set(index)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        selected = [(c, p) for c, p in inputs if (c["group"] == "v76_fresh_updating" and c["mode"] == "compound" and c["seed"] == 135180)
                    or (c["group"] == "v61_development_replay" and c["mode"] == "heavy")]
        assert len(selected) == 2
        from sklearn.linear_model import Ridge
        checked = 0
        selection = []
        bank = banks["updating"]
        for cell, path in selected:
            trace = v76.v68.load_trace(path)
            for w, a, b in v78.v77.windows():
                query = {k: trace[k][a:b] for k in v76.v74.KEYS}
                x = bank.features.inputs(query)
                centers, detail = correction.predict_details(x)
                normalized = (x-bank.features.input_mean)/bank.features.input_scale
                indices = bank.tree.query(normalized, k=64)[1]
                for row, (q, neighbors) in enumerate(zip(normalized, indices)):
                    model = Ridge(alpha=detail["selected_alpha"][row], fit_intercept=True, solver="cholesky")
                    model.fit(bank.tree.data[neighbors]-q, correction.paired_residuals[neighbors])
                    actual = np.concatenate([centers[m][row] for m in v78.v77.MODES])
                    np.testing.assert_allclose(actual, model.intercept_, rtol=1e-9, atol=1e-10)
                terms = correction_terms(query, banks, SavedMeans(centers))
                for condition, source in CONTROLS.items():
                    np.testing.assert_allclose(terms[condition]["likelihoods"], MeanCovarianceBank(banks, source).log_likelihoods(query), rtol=1e-11, atol=1e-9)
                for condition, source in CONDITIONS.items():
                    expected = LooRidgeDensityBank(banks, correction, source).log_likelihoods(query)
                    np.testing.assert_allclose(terms["linear_mean_"+source+"_cov"]["likelihoods"], expected, rtol=1e-10, atol=1e-9)
                selection.append(dict(**v78.replay_cell(cell), window=w, **selection_statistics(detail, 0, b-a)))
                checked += b-a
        result = dict(revision=REVISION, passed=True, checked_transitions=checked, evaluation_paths=2,
            selected_predictions_match_sklearn=True, selected_densities_match_readout=True,
            constant_controls_match_v78_densities=True, selection=selection,
            environment_steps=0, optimizer_updates=0)
        name = "native_smoke.json"
    else:
        references = freeze_references(args, banks, correction)
        assert all(r.normal_null_calibration_transitions == scores["reference_metadata"]["linear_mean_frozen_cov"]["normal_null_calibration_transitions"] for r in references.values())
        records = evaluate(args, inputs, references, banks, correction, index)
        result = dict(revision=REVISION, source_revision=v78.REVISION, spec=v76.SPEC,
            new_conditions=CONDITIONS, controls_reused_from_v78=list(CONTROL_CONDITIONS),
            environment_steps=0, optimizer_updates=0, readout_controls_actor=False,
            forward_models_refitted=False, scaling_refitted=False, temporal_rules_changed=False,
            conditional_covariances_changed=False, conditional_mean_rule_changed=True,
            selection_uses_query_next_state=False, selection_uses_true_mode=False,
            shared_ridge_strength_across_modes=True, ridge_alphas=ALPHAS, all_paths_development_replays=True,
            reference_metadata={c: {k: v for k, v in r.metadata().items() if k != "phase"} for c, r in references.items()},
            records=records, summary=v78.v77.summarize(records, (*CONTROL_CONDITIONS, *CONDITIONS)),
            decision="development-reference-only-ridge-selection-not-confirmation",
            limitations=["Development replays, one source training seed and mild known modes; not RL return evidence.",
                "Local leave-one-out selection reuses query-selected reference neighbors and does not validate extrapolation to the query.",
                "Covariances still describe the saved raw residual banks, not refitted linear prediction errors.",
                "All five baseline statistics are reused from v78, not newly independent controls."])
        name = "loo_ridge_mean.json"
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
