"""Frozen v113 logit decomposition and simulator-center intervention."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_general_ood_v113_discriminative_evidence as v113

MODES = v113.MODES
REVISION = "general_ood_v114_classifier_attribution"
SMOKE_WINDOWS = {(135252, m, True, "first_post50") for m in ("normal", "slippery", "high_grav", "compound")}
SMOKE_WINDOWS |= {(135191, "slippery", False, "first_post50"),
                  (135192, "high_grav", True, "first_post50"),
                  (134020, "slippery", True, "last64")}


def window_key(record):
    return record["seed"], record["mode"], record["updating"], record["window"]


def margin_parts(fitted, z, logits, truth, competitor):
    model = fitted[1]
    competitor = np.broadcast_to(np.asarray(competitor, dtype=int), (len(z),))
    contributions = (model.coef_[truth]-model.coef_[competitor])*z
    intercept = model.intercept_[truth]-model.intercept_[competitor]
    linear, square = contributions[:, :55].sum(axis=1), contributions[:, 55:].sum(axis=1)
    margin = logits[:, truth]-logits[np.arange(len(z)), competitor]
    np.testing.assert_allclose(intercept+linear+square, margin, rtol=1e-10, atol=1e-8)
    return dict(mean_margin=float(margin.mean()), intercept=float(intercept.mean()),
        linear_residual=float(linear.mean()), squared_residual=float(square.mean()),
        positive_margin_fraction=float(np.mean(margin > 0)),
        competitor_fractions={m: float(np.mean(competitor == i)) for i, m in enumerate(MODES)},
        linear_by_reference_mode=dict(zip(MODES, contributions[:, :55].reshape(-1, 5, 11).sum(axis=2).mean(axis=0).tolist())),
        squared_by_reference_mode=dict(zip(MODES, contributions[:, 55:].reshape(-1, 5, 11).sum(axis=2).mean(axis=0).tolist())))


def center_diagnostics(fitted, observed, predicted, mode):
    truth = MODES.index(mode)
    z = fitted[0].transform(v113.features(observed, predicted))
    logits = fitted[1].decision_function(z)
    np.testing.assert_allclose(logits, z@fitted[1].coef_.T+fitted[1].intercept_, rtol=1e-12, atol=1e-10)
    other = logits.copy(); other[:, truth] = -np.inf
    evidence = v113.scores(fitted, observed, predicted)
    statistics = {a: v113.statistics(p, truth) for a, p in evidence.items()}
    parts = margin_parts(fitted, z, logits, truth, other.argmax(axis=1))
    np.testing.assert_allclose(parts["mean_margin"],
        statistics["quadratic_logistic"]["mean_true_minus_best_other_score"], rtol=1e-10, atol=1e-8)
    return dict(candidates=statistics, best_other=parts,
        versus_normal=margin_parts(fitted, z, logits, truth, 0) if truth else None,
        true_prediction_error_rms=float(np.sqrt(np.mean((observed-predicted[:, truth])**2, axis=1)).mean()),
        standardized_feature_magnitude={name: dict(mean_abs=float(np.abs(values).mean()), max_abs=float(np.abs(values).max()))
            for name, values in (("linear", z[:, :55]), ("square", z[:, 55:]))})


def oracle_predictions(query, bank, mode):
    centers = v113.v112.v111.oracle_centers(query, bank, mode)
    forward = np.stack([bank.design(query)@bank.coefficients[m] for m in MODES], axis=1)
    return centers+forward


def check_original(actual, previous):
    for arm in v113.ARMS:
        for field, expected in previous["candidates"][arm].items():
            values = actual["candidates"][arm][field]
            if isinstance(expected, dict):
                np.testing.assert_allclose([values[m] for m in MODES], [expected[m] for m in MODES], rtol=1e-9, atol=1e-8)
            else:
                np.testing.assert_allclose(values, expected, rtol=1e-9, atol=1e-8)


def cohorts(records):
    return dict(
        current_failed_updating=[r for r in records if r["origin"] == "v110" and r["updating"] and r["mode"] != "normal"],
        current_normal=[r for r in records if r["origin"] == "v110" and r["mode"] == "normal"],
        current_matched_frozen=[r for r in records if r["origin"] == "v110" and not r["updating"] and r["mode"] != "normal"],
        earlier_six_failures=[r for r in records if r["old_coverage_failure"]],
        all_normal=[r for r in records if r["mode"] == "normal"],
        all_frozen_changed=[r for r in records if not r["updating"] and r["mode"] != "normal"],
        all_updating_changed=[r for r in records if r["updating"] and r["mode"] != "normal"])


def summarize(records):
    fields = ("instantaneous_mode_accuracy", "mean_true_minus_best_other_score", "classification_nll")
    result = {}
    for name, selected in cohorts(records).items():
        if not selected:
            continue
        result[name] = {}
        for arm in ("fitted", "oracle"):
            rows = [r[arm] for r in selected if r[arm] is not None]
            if not rows:
                continue
            result[name][arm] = dict(windows=len(rows), candidates={a: {f: float(np.mean([r["candidates"][a][f] for r in rows]))
                for f in fields} for a in v113.ARMS},
                best_other={f: float(np.mean([r["best_other"][f] for r in rows]))
                    for f in ("mean_margin", "intercept", "linear_residual", "squared_residual")})
        # Paired changes exclude the two real-only historical windows.
        paired = [r for r in selected if r["oracle"] is not None]
        result[name]["paired_oracle_minus_fitted"] = dict(windows=len(paired), candidates={a: {f: float(np.mean([
            r["oracle"]["candidates"][a][f]-r["fitted"]["candidates"][a][f] for r in paired]))
            for f in fields} for a in v113.ARMS}) if paired else None
    return result


def run(args):
    sys.path.insert(0, str(args.v83_dir))
    from scripts import analyze_general_ood_v83_cross_seed_covariance as source

    torch.set_num_threads(2)
    baseline = json.loads((args.v113_dir/"run_results/formal/discriminative_evidence.json").read_text())
    assert baseline["revision"] == v113.REVISION and baseline["passed"] and not baseline["smoke"]
    saved = torch.load(args.v113_dir/"run_checkpoints/formal/classifier_models.pt", map_location="cpu", weights_only=False)
    assert saved["revision"] == v113.REVISION and not saved["smoke"] and saved["settings"] == v113.SETTINGS
    fitted = saved["final_model"]
    np.testing.assert_array_equal(fitted[1].classes_, np.arange(5))
    assert fitted[1].coef_.shape == (5, 110)
    selected = [r for r in baseline["records"] if not args.smoke or window_key(r) in SMOKE_WINDOWS]
    assert len(selected) == (7 if args.smoke else 76)
    family, _ = v113.v106.load_family(args, source)
    assert not set(family.mean_groups) & {r["seed"] for r in selected}
    paths = {v113.v112.v105.key(c): p for c, p, _ in v113.v98.evaluation_inputs(args, source)}
    ranges = {w: (a, b) for w, a, b in source.v78.v77.windows()}
    records = []
    for previous in selected:
        cell = v113.trace_cell(previous)
        directory = args.v110_dir/f"run_checkpoints/seed{cell['seed']}/traces"
        path = source.v76.trace_path(directory, cell) if previous["origin"] == "v110" else paths[v113.v112.v105.key(cell)]
        trace = source.v76.v68.load_trace(path)
        start, stop = ranges[previous["window"]]
        query = {k: trace[k][start:stop] for k in v113.v98.v97.KEYS}
        correction = family.correction.predict(family.bank.features.inputs(query))
        forward = family.bank.design(query)
        predicted = np.stack([forward@family.bank.coefficients[m]+correction[m] for m in MODES], axis=1)
        observed = family.bank.targets(query)
        actual = center_diagnostics(fitted, observed, predicted, cell["mode"])
        check_original(actual, previous)
        oracle = None
        if "oracle_observations" in trace:
            query["oracle_observations"] = trace["oracle_observations"][start:stop]
            oracle = center_diagnostics(fitted, observed, oracle_predictions(query, family.bank, cell["mode"]), cell["mode"])
            np.testing.assert_allclose(oracle["true_prediction_error_rms"], 0., atol=1e-10)
        else:
            assert previous["group"] == "v62_development_replay"
        records.append(dict(**cell, origin=previous["origin"], window=previous["window"], transitions=stop-start,
            old_coverage_failure=previous["old_coverage_failure"], fitted=actual, oracle=oracle))
        print(f"[ATTRIBUTION] {len(records)}/{len(selected)} {window_key(previous)}", flush=True)
    transitions = sum(r["transitions"] for r in records)
    oracle_rows = [r for r in records if r["oracle"] is not None]
    assert (transitions, len(oracle_rows)) == ((364, 6) if args.smoke else (4248, 74))
    result = dict(revision=REVISION, source_revision=v113.REVISION, passed=True, smoke=args.smoke,
        checked_windows=len(records), checked_transitions=transitions,
        oracle_windows=len(oracle_rows), oracle_transitions=sum(r["transitions"] for r in oracle_rows),
        original_score_windows_verified=len(records), original_scalar_statistics_verified=10*len(records),
        records=records, summary=summarize(records),
        classifier_model="v113 formal final_model, not validation_model or partial smoke model",
        margin_definition="true logit minus best other logit, competitor selected separately for each center arm",
        decomposition="standardized-feature intercept + linear residual contribution + squared residual contribution",
        oracle_role="exact simulator outcomes at the same observed state/action; diagnostic center intervention only",
        environment_steps=0, classifier_fits=0, reference_refitted=False, scaling_refitted=False,
        rl_optimizer_updates=0, posterior_replayed=False, normal_thresholds_recalibrated=False,
        readout_controls_actor=False, readout_adopted=False, decision="fixed-classifier-attribution-not-candidate-selection",
        limitations=["Simulator centers are unavailable at deployment; this is not a deployable candidate or identifiability ceiling.",
            "Correct-center scores intervene on classifier inputs without refitting; improvements do not isolate a pure causal effect of mean error.",
            "Standardized feature magnitude is not a training-support or extrapolation certificate; the intercept is not a mode prior.",
            "Previously inspected windows are development evidence; no full-history belief, actor or return evaluation."])
    args.result_dir.mkdir(parents=True, exist_ok=True)
    path = args.result_dir/("native_smoke.json" if args.smoke else "classifier_attribution.json")
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] {REVISION}: {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("v113-dir", "v110-dir", "v104-dir", "v98-dir", "v97-dir", "v95-dir", "v94-dir", "v83-dir",
                 "v78-dir", "v76-dir", "v74-dir", "v73-dir", "v72-dir", "v71-dir", "v69-dir", "v70-dir",
                 "v61-dir", "v62-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
