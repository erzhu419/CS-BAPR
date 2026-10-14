"""Small native mode-gap and correct-center attribution of v110 failures."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_general_ood_v110_fresh_mixture as v110

v106, v98, summary = v110.v106, v110.v98, v110.summary
REVISION = "general_ood_v111_fresh_attribution"
FAILURES = {(135251, "slippery"), (135252, "slippery"), (135252, "high_grav"), (135252, "compound")}


def select_windows(records, smoke=False):
    primary = v110.v109.v108.PRIMARY
    failed = {(r["seed"], r["mode"]) for r in records if r["updating"] and r["mode"] != "normal"
        and r["arms"]["normal_density_mixture"][primary]["first_post50"]["mean_true_minus_best_other_score"] <= 0}
    assert failed == FAILURES
    selected = [r for r in records if r["mode"] == "normal" or (r["seed"], r["mode"]) in failed]
    assert len(selected) == len({summary.key(r) for r in selected}) == 14
    if smoke:
        selected = [r for r in selected if r["seed"] == 135252 and r["updating"] and r["mode"] in ("normal", "compound")]
        assert len(selected) == 2
    return selected


def oracle_centers(query, bank, mode):
    truth = v98.v94.MODES.index(mode)
    oracle = query["oracle_observations"]
    assert oracle.shape == (len(query["states"]), 5, 11) and np.isfinite(oracle).all()
    np.testing.assert_array_equal(oracle[:, truth], query["next_states"])
    design = bank.design(query)
    return np.stack([bank.targets(dict(query, next_states=oracle[:, i]))-design@bank.coefficients[m]
        for i, m in enumerate(v98.v94.MODES)], axis=1)


def likelihood_statistics(ll, mode):
    truth = v98.v94.MODES.index(mode)
    other = ll.copy(); other[:, truth] = -np.inf
    prediction = ll.argmax(axis=1)
    return dict(instantaneous_mode_accuracy=float(np.mean(prediction == truth)),
        mean_true_minus_best_other_log_density=float(np.mean(ll[:, truth]-other.max(axis=1))),
        mean_log_density=dict(zip(v98.v94.MODES, ll.mean(axis=0).tolist())),
        predicted_mode_fractions={m: float(np.mean(prediction == i)) for i, m in enumerate(v98.v94.MODES)})


def support(mean, groups, inputs, incumbent_rows):
    normalized = (inputs-mean.features.input_mean)/mean.features.input_scale
    distance, neighbors = mean.tree.query(normalized, k=64)
    return dict(mean_nearest_radius=float(distance[:, 0].mean()), mean_radius64=float(distance[:, -1].mean()),
        added_neighbor_fraction=float(np.mean(neighbors >= incumbent_rows)),
        seed_fractions={str(int(s)): float(np.mean(groups[neighbors] == s)) for s in np.unique(groups)})


def diagnose(query, family, mode):
    inputs = family.bank.features.inputs(query)
    old_fixed, _, covariance, _ = family.incumbent.geometry(inputs)
    prediction = family.correction.predict(inputs)
    new_fixed = np.stack([prediction[m] for m in v98.v94.MODES], axis=1)
    centers = dict(old=old_fixed, new=new_fixed, oracle=oracle_centers(query, family.bank, mode))
    terms = {a: v98.v94.terms_for_covariances(query, family.bank, fixed, covariance) for a, fixed in centers.items()}
    truth, rows = v98.v94.MODES.index(mode), np.arange(len(inputs))
    oracle = query["oracle_observations"]
    gaps = np.sqrt(np.mean(((oracle-oracle[:, truth:truth+1])/family.bank.target_scale)**2, axis=2))
    other_gaps = np.delete(gaps, truth, axis=1)
    mode_gaps = dict(mean_nearest_mode_gap_rms=float(other_gaps.min(axis=1).mean()),
        mean_gap_rms_by_mode={m: float(gaps[:, i].mean()) for i, m in enumerate(v98.v94.MODES) if i != truth})
    densities = {}
    for c in summary.CONDITIONS:
        densities[c] = {}
        for arm in centers:
            values = terms[arm][c]; ll = values["likelihoods"]
            other = ll.copy(); other[:, truth] = -np.inf
            competitor = other.argmax(axis=1)
            error = values["mean_error_rms"][:, truth]
            densities[c][arm] = dict(likelihood_statistics(ll, mode),
                true_prediction_error_rms=float(error.mean()),
                error_exceeds_nearest_oracle_gap_fraction=float(np.mean(error > other_gaps.min(axis=1))),
                mean_energy_advantage=float(.5*np.mean(values["energy"][rows, competitor]-values["energy"][:, truth])),
                mean_volume_advantage=float(.5*np.mean(values["logdet"][rows, competitor]-values["logdet"][:, truth])))
        old, new = terms["old"][c]["likelihoods"], terms["new"][c]["likelihoods"]
        mixture = v110.v109.compose(old, new)["normal_density_mixture"]
        densities[c]["normal_density_mixture"] = likelihood_statistics(mixture, mode)
        densities[c]["normal_mixture_attribution"] = dict(
            normal_log_density_change_vs_old=float((mixture[:, 0]-old[:, 0]).mean()),
            heavy_log_density_change_vs_old=float((new[:, 1]-old[:, 1]).mean()),
            normal_minus_heavy_margin_change=float(((mixture[:, 0]-new[:, 1])-(old[:, 0]-old[:, 1])).mean()),
            mean_old_normal_component_responsibility=float(np.exp(old[:, 0]-np.logaddexp(old[:, 0], new[:, 0])).mean()),
            mean_mixture_log_penalty_vs_best_normal=float((mixture[:, 0]-np.maximum(old[:, 0], new[:, 0])).mean()))
    return dict(mode_gaps=mode_gaps, densities=densities, support=dict(
        old=support(family.incumbent.mean, family.groups, inputs, family.incumbent.mean.tree.n),
        new=support(family.mean, family.mean_groups, inputs, family.incumbent.mean.tree.n))), terms


def run(args):
    sys.path.insert(0, str(args.v83_dir))
    from scripts import analyze_general_ood_v83_cross_seed_covariance as source

    torch.set_num_threads(2)
    parts = [json.loads((args.v110_dir/f"run_results/seed{s}/fresh_seed.json").read_text()) for s in v110.SEEDS]
    assert all(r["revision"] == v110.REVISION and r["passed"] and not r["smoke"] for r in parts)
    assert all(r["reference_metadata"] == parts[0]["reference_metadata"] for r in parts)
    population = select_windows([r for p in parts for r in p["records"]], args.smoke)
    family, _ = v106.load_family(args, source)
    for c in summary.CONDITIONS:
        assert family.incumbent.metadata(c) == parts[0]["reference_metadata"]["old"][c]["bank"]
        assert family.metadata(c) == parts[0]["reference_metadata"]["new"][c]["bank"]
    start, stop = next((a, b) for w, a, b in source.v78.v77.windows() if w == "first_post50")
    assert (start, stop) == (3800, 3850)
    records, comparisons = [], 0
    for cell in population:
        directory = args.v110_dir/f"run_checkpoints/seed{cell['seed']}"
        trace = source.v76.v68.load_trace(source.v76.trace_path(directory/"traces", cell))
        query = {k: trace[k][start:stop] for k in (*v98.v97.KEYS, "oracle_observations")}
        diagnostics, terms = diagnose(query, family, cell["mode"])
        cached = torch.load(directory/f"terms/{cell['mode']}_{cell['updating']}.pt", map_location="cpu", weights_only=False)
        assert cached["revision"] == v110.REVISION and cached["spec"] == v110.SPEC
        assert tuple(cached["key"]) == summary.key(cell)
        for c in summary.CONDITIONS:
            for arm in ("old", "new"):
                np.testing.assert_allclose(terms[arm][c]["likelihoods"], cached["likelihoods"][arm][c][start:stop], rtol=1e-10, atol=1e-8)
            for arm in v110.ARMS:
                expected = cell["arms"][arm][c]["first_post50"]
                actual = diagnostics["densities"][c][arm]
                for field in ("instantaneous_mode_accuracy", "instantaneous_mean_true_minus_best_other_log_density"):
                    key = "mean_true_minus_best_other_log_density" if field.startswith("instantaneous_mean") else field
                    np.testing.assert_allclose(actual[key], expected[field], rtol=1e-9, atol=1e-8)
                comparisons += 1
        records.append(dict(**{k: cell[k] for k in ("group", "seed", "mode", "updating")},
            role="normal_control" if cell["mode"] == "normal" else "failed_updating_with_frozen_pair",
            window="first_post50", transitions=stop-start, diagnostics=diagnostics,
            saved_recursive={a: cell["arms"][a][v110.v109.v108.PRIMARY]["first_post50"] for a in v110.ARMS}))
        print(f"[ATTRIBUTION] {len(records)}/{len(population)} {summary.key(cell)}", flush=True)
    result = dict(revision=REVISION, source_revision=v110.REVISION, passed=True, smoke=args.smoke,
        checked_windows=len(records), checked_transitions=50*len(records), cached_component_comparisons=6*len(records),
        v110_instantaneous_comparisons=comparisons, records=records,
        covariance_reference_unchanged=True, covariance_moment_recentered=False,
        oracle_role="exact simulator conditional centers at observed state/action; diagnostic only; frozen covariance",
        environment_steps=0, optimizer_updates=0, reference_refitted=False, normal_thresholds_recalibrated=False,
        posterior_replayed=False, coefficient_tuned=False, readout_controls_actor=False, readout_adopted=False,
        confirmation_outcomes_used_for_diagnosis=True, decision="failure-attribution-not-candidate-selection",
        limitations=["Failure-selected windows are now development evidence for any subsequent redesign.",
            "Simulator centers are unavailable to deployment and are not a calibrated candidate or Bayes-optimal ceiling.",
            "Only instantaneous densities are intervened on; original complete-history readouts are retained separately.",
            "One source policy and known Hopper modes; no actor or return evidence."])
    args.result_dir.mkdir(parents=True, exist_ok=True)
    path = args.result_dir/("native_smoke.json" if args.smoke else "fresh_attribution.json")
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] {REVISION}: {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("v110-dir", "v104-dir", "v98-dir", "v97-dir", "v95-dir", "v94-dir", "v83-dir", "v78-dir",
                 "v76-dir", "v74-dir", "v73-dir", "v72-dir", "v71-dir", "v69-dir", "v70-dir", "v61-dir",
                 "v62-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
