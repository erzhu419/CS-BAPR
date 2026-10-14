"""Expanded-mean held-out errors on fixed v98 covariance support."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_general_ood_v111_fresh_attribution as v111

v106, v98, v105 = v111.v106, v111.v98, v111.v106.v105
MODES, CONDITIONS = v98.v94.MODES, v98.v94.CONDITIONS
REVISION = "general_ood_v112_matched_errors"
ARMS = ("old", "new", "matched", "oracle_old_cov", "oracle_matched_cov")


def fit_errors(family, directory, smoke=False):
    mean, groups = family.mean, family.mean_groups
    support = family.incumbent.mean.tree.n
    np.testing.assert_array_equal(mean.tree.data[:support], family.incumbent.mean.tree.data)
    np.testing.assert_array_equal(groups[:support], family.groups)
    for m in MODES:
        np.testing.assert_array_equal(mean.residuals[m][:support], family.incumbent.mean.residuals[m])
    errors = {m: np.empty_like(family.incumbent.errors[m]) for m in MODES}
    fits = []
    directory.mkdir(parents=True, exist_ok=True)
    for seed in np.unique(family.groups):
        rows = np.flatnonzero(family.groups == seed)
        if smoke:
            rows = rows[[0, -1]]
        path = directory/f"errors_seed{seed}.pt"
        if path.is_file() and not smoke:
            saved = torch.load(path, map_location="cpu", weights_only=False)
            assert saved["revision"] == REVISION and saved["ridge_alpha"] == .064
            np.testing.assert_array_equal(saved["mean_groups"], groups)
            np.testing.assert_array_equal(saved["rows"], rows)
            values, fit = saved["errors"], saved["fit"]
        else:
            values, fit = v98.heldout_errors(mean, groups, seed, rows)
            if not smoke:
                torch.save(dict(revision=REVISION, ridge_alpha=.064, mean_groups=groups,
                    rows=rows, errors=values, fit=fit), path)
        assert fit["source_seed_excluded_from_all_mean_rows"]
        assert fit["mean_training_rows"] == int(np.sum(groups != seed))
        assert fit["excluded_mean_rows"] == int(np.sum(groups == seed))
        assert fit["error_rows"] == len(rows)
        for m in MODES:
            assert values[m].shape == errors[m][rows].shape and np.isfinite(values[m]).all()
            errors[m][rows] = values[m]
        fits.append(fit)
        print(f"[ERRORS] {len(fits)}/{len(np.unique(family.groups))} seed{seed}: "
              f"{len(rows)} rows, {fit['mean_training_rows']} training rows", flush=True)
    assert sum(f["error_rows"] for f in fits) == (2*len(fits) if smoke else support)
    # A smoke tests fold exclusion and the identity intervention, not partial fitted errors.
    return (family.incumbent.errors if smoke else errors), fits


def geometry(family, errors, inputs):
    from scripts.v83_cross_seed_covariance import error_covariance

    old_fixed, members, old_covariance, _ = family.incumbent.geometry(inputs)
    predicted = family.correction.predict(inputs)
    new_fixed = np.stack([predicted[m] for m in MODES], axis=1)
    support = family.incumbent.mean
    normalized = (inputs-support.features.input_mean)/support.features.input_scale
    _, neighbors = support.tree.query(normalized, k=64)
    base = np.stack([family.raw_weight*family.bank.geometry(inputs, m)[1]+(1-family.raw_weight)*
        np.asarray([error_covariance(e)[0] for e in errors[m][neighbors]]) for m in MODES], axis=1)
    moment = v98.v94.v93.displacement_second_moment(old_fixed, members)
    return old_fixed, new_fixed, old_covariance, v98.v94.covariance_variants(base, moment)


def statistics(terms, mode):
    ll = terms["likelihoods"]
    truth = MODES.index(mode)
    other = ll.copy(); other[:, truth] = -np.inf
    competitor, rows = other.argmax(axis=1), np.arange(len(ll))
    return dict(v111.likelihood_statistics(ll, mode),
        true_prediction_error_rms=float(terms["mean_error_rms"][:, truth].mean()),
        mean_energy_advantage=float(.5*np.mean(terms["energy"][rows, competitor]-terms["energy"][:, truth])),
        mean_volume_advantage=float(.5*np.mean(terms["logdet"][rows, competitor]-terms["logdet"][:, truth])))


def diagnose(query, family, errors, mode, identity=False):
    inputs = family.bank.features.inputs(query)
    old, new, old_cov, matched_cov = geometry(family, errors, inputs)
    if identity:
        for c in CONDITIONS:
            np.testing.assert_allclose(matched_cov[c], old_cov[c], rtol=1e-10, atol=1e-10)
    centers = dict(old=old, new=new, matched=new)
    if "oracle_observations" in query:
        oracle = v111.oracle_centers(query, family.bank, mode)
        centers.update(oracle_old_cov=oracle, oracle_matched_cov=oracle)
    terms = {a: v98.v94.terms_for_covariances(query, family.bank, fixed,
        matched_cov if a in ("matched", "oracle_matched_cov") else old_cov) for a, fixed in centers.items()}
    diagnostics = {c: {a: statistics(terms[a][c], mode) for a in centers} for c in CONDITIONS}
    covariance = {c: {a: dict(mean_trace_by_mode=dict(zip(MODES,
        np.trace(values, axis1=-2, axis2=-1).mean(axis=0).tolist())),
        mean_logdet_by_mode=dict(zip(MODES, np.linalg.slogdet(values)[1].mean(axis=0).tolist())))
        for a, values in (("old", old_cov[c]), ("matched", matched_cov[c]))} for c in CONDITIONS}
    return dict(densities=diagnostics, covariance=covariance), terms


def population(older, fresh, smoke=False):
    selected = [("v98", r, w) for r, w in v105.select_windows(older, smoke)]
    selected.extend(("v110", r, "first_post50") for r in v111.select_windows(fresh, smoke))
    assert len(selected) == (6 if smoke else 76)
    assert len({(origin, v105.key(r), w) for origin, r, w in selected}) == len(selected)
    return selected


def summarize(records):
    subsets = dict(
        current_failed_updating=[r for r in records if r["origin"] == "v110" and r["updating"] and r["mode"] != "normal"],
        current_matched_frozen=[r for r in records if r["origin"] == "v110" and not r["updating"] and r["mode"] != "normal"],
        current_normal=[r for r in records if r["origin"] == "v110" and r["mode"] == "normal"],
        earlier_six_failures=[r for r in records if r["old_coverage_failure"]],
        earlier_updating_changed=[r for r in records if r["origin"] == "v98" and r["group"].startswith("v95_") and r["updating"] and r["mode"] != "normal"],
        earlier_frozen_changed=[r for r in records if r["origin"] == "v98" and r["group"].startswith("v95_") and not r["updating"] and r["mode"] != "normal"],
        earlier_normal=[r for r in records if r["origin"] == "v98" and r["mode"] == "normal"],
        historical_slippery=[r for r in records if r["group"] == "v62_development_replay" and r["mode"] == "slippery"])
    fields = ("instantaneous_mode_accuracy", "mean_true_minus_best_other_log_density", "true_prediction_error_rms",
              "mean_energy_advantage", "mean_volume_advantage")
    output = {}
    for name, rows in subsets.items():
        if not rows:
            continue
        conditions = {}
        for c in CONDITIONS:
            conditions[c] = {}
            for a in ARMS:
                available = [r["diagnostics"]["densities"][c][a] for r in rows
                             if a in r["diagnostics"]["densities"][c]]
                if available:
                    conditions[c][a] = dict(windows=len(available), **{field: float(np.mean(
                        [r[field] for r in available])) for field in fields})
        output[name] = dict(windows=len(rows), conditions=conditions)
    return output


def run(args):
    sys.path.insert(0, str(args.v83_dir))
    from scripts import analyze_general_ood_v83_cross_seed_covariance as source

    torch.set_num_threads(2)
    family, _ = v106.load_family(args, source)
    old = json.loads((args.v98_dir/"run_results/replay_diverse_recipe/regime_replay.json").read_text())
    parts = [json.loads((args.v110_dir/f"run_results/seed{s}/fresh_seed.json").read_text()) for s in v111.v110.SEEDS]
    assert old["revision"] == v98.REVISION and old["passed"] and old["evaluation_paths"] == 95
    assert all(p["revision"] == v111.v110.REVISION and p["passed"] and not p["smoke"] for p in parts)
    for p in parts:
        for c in CONDITIONS:
            assert family.incumbent.metadata(c) == p["reference_metadata"]["old"][c]["bank"]
            assert family.metadata(c) == p["reference_metadata"]["new"][c]["bank"]
    selected = population(old["records"], [r for p in parts for r in p["records"]], args.smoke)
    assert not set(family.mean_groups) & {r["seed"] for _, r, _ in selected}
    errors, fits = fit_errors(family, args.checkpoint_dir, args.smoke)
    paths = {v105.key(c): path for c, path, _ in v98.evaluation_inputs(args, source)}
    ranges = {w: (a, b) for w, a, b in source.v78.v77.windows()}
    records, cached_checks, old_checks = [], 0, 0
    for origin, cell, window in selected:
        directory = args.v110_dir/f"run_checkpoints/seed{cell['seed']}"
        path = source.v76.trace_path(directory/"traces", cell) if origin == "v110" else paths[v105.key(cell)]
        trace = source.v76.v68.load_trace(path)
        start, stop = ranges[window]
        query = {k: trace[k][start:stop] for k in v98.v97.KEYS}
        if "oracle_observations" in trace:
            query["oracle_observations"] = trace["oracle_observations"][start:stop]
        diagnostic, terms = diagnose(query, family, errors, cell["mode"], args.smoke)
        if origin == "v110":
            saved = torch.load(directory/f"terms/{cell['mode']}_{cell['updating']}.pt", map_location="cpu", weights_only=False)
            assert saved["revision"] == v111.v110.REVISION and tuple(saved["key"]) == v111.summary.key(cell)
            for a in ("old", "new"):
                for c in CONDITIONS:
                    np.testing.assert_allclose(terms[a][c]["likelihoods"], saved["likelihoods"][a][c][start:stop], rtol=1e-10, atol=1e-8)
                    cached_checks += 1
        else:
            for c in CONDITIONS:
                actual = source.v78.v77.attribution(query, terms["old"][c], family.bank, cell["mode"], 0, stop-start)
                v98.v94.v90.verify_baseline(actual, cell["attribution"][c][window])
                old_checks += 1
        records.append(dict(**{k: cell[k] for k in ("group", "seed", "mode", "updating")},
            origin=origin, window=window, transitions=stop-start, diagnostics=diagnostic,
            oracle_available="oracle_observations" in query,
            old_coverage_failure=origin == "v98" and cell["updating"] and
                (cell["seed"], cell["mode"], window) in v105.FAILURES))
        print(f"[PILOT] {len(records)}/{len(selected)} {origin} {v105.key(cell)} {window}", flush=True)
    transitions = sum(r["transitions"] for r in records)
    assert transitions == (328 if args.smoke else 4248)
    result = dict(revision=REVISION, passed=True, smoke=args.smoke, checked_windows=len(records),
        checked_transitions=transitions, error_fits=fits, fitted_error_rows=sum(f["error_rows"] for f in fits),
        full_error_bank_refitted=not args.smoke, smoke_covariance_identity_only=args.smoke,
        raw_reference_rows=family.bank.tree.n, mean_reference_rows=family.mean.tree.n,
        error_reference_rows=family.incumbent.mean.tree.n, error_support_seed_count=len(fits),
        mean_source_seed_count=len(np.unique(family.mean_groups)), raw_weight=family.raw_weight,
        unchanged_old_attribution_comparisons=old_checks, unchanged_cached_likelihood_comparisons=cached_checks,
        arms=list(ARMS), records=records, summary=summarize(records),
        oracle_checked_windows=sum(r["oracle_available"] for r in records),
        error_fit="expanded mean with all rows of each error source seed excluded; old error support only",
        covariance_formula="fixed raw weight * old raw covariance + complementary weight * OAS second moment of matched held-out errors",
        moment="old mean center and old eight whole-seed-held-out members unchanged",
        trace_match="new support covariance rescaled to trace of new support covariance plus unchanged old moment",
        means_changed_vs_v106=False, raw_geometry_changed=False, covariance_support_changed=False,
        coefficient_tuned=False, scaling_refitted=False, global_forward_refitted=False,
        normal_thresholds_recalibrated=False, posterior_replayed=False, readout_controls_actor=False,
        readout_adopted=False, environment_steps=0, optimizer_updates=0,
        weighting="equal inspected window; instantaneous likelihood diagnostics, not full-history accuracy",
        decision="identity-smoke-only" if args.smoke else "development-matched-error-pilot-not-adoption",
        limitations=["Current failure-selected and earlier inspected windows are development data.",
            "Oracle centers are simulator-only diagnostics, not deployable readouts or performance bounds.",
            "Historical v62 traces lack counterfactual observations; their real candidate controls remain included.",
            "No normal-null calibration, full-history posterior, actor or return evaluation in this pilot."])
    args.result_dir.mkdir(parents=True, exist_ok=True)
    path = args.result_dir/("native_smoke.json" if args.smoke else "matched_errors.json")
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] {REVISION}: {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("v110-dir", "v104-dir", "v98-dir", "v97-dir", "v95-dir", "v94-dir", "v83-dir", "v78-dir",
                 "v76-dir", "v74-dir", "v73-dir", "v72-dir", "v71-dir", "v69-dir", "v70-dir", "v61-dir",
                 "v62-dir", "checkpoint-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
