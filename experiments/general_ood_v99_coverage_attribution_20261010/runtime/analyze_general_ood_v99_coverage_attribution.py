"""Native support and crossed mean/covariance attribution of v98 outcomes."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_general_ood_v98_regime_replay as v98
import analyze_general_ood_v96_mean_support as v96

REVISION = "general_ood_v99_coverage_attribution"
RECIPES = ("old_recipe", "diverse_recipe")
CANDIDATE = "suffix_disagreement_mix"


def record_key(record):
    return tuple(record[k] for k in ("group", "seed", "mode", "updating"))


def select_windows(data):
    indices = {r: {record_key(c): c for c in data[r]["records"]} for r in RECIPES}
    assert indices[RECIPES[0]].keys() == indices[RECIPES[1]].keys()
    selections = []
    for seed, mode, window in v96.FAILURES:
        for updating in (False, True):
            group = "v95_development_updating" if updating else "v95_development_frozen"
            key = (group, seed, mode, updating)
            selections.append(("original_v95_failure_pair", window, {r: indices[r][key] for r in RECIPES}))
    tradeoffs = []
    for key, old in indices["old_recipe"].items():
        if old["mode"] == "normal":
            continue
        diverse = indices["diverse_recipe"][key]
        for window in ("first_post50", "last64"):
            if (old["observers"][CANDIDATE][window]["mean_true_minus_best_other_score"] > 0
                    >= diverse["observers"][CANDIDATE][window]["mean_true_minus_best_other_score"]):
                tradeoffs.append((key, window))
    assert tradeoffs == [(("v62_development_replay", 134020, "slippery", True), "last64")]
    for key, window in tradeoffs:
        normal = (key[0], key[1], "normal", key[3])
        for k, role in ((key, "matched_recipe_tradeoff"), (normal, "matched_tradeoff_normal")):
            selections.append((role, window, {r: indices[r][k] for r in RECIPES}))
    return selections


def neighbor_partition(neighbors, old_raw_rows, expanded_raw_rows):
    return dict(original_raw=float(np.mean(neighbors < old_raw_rows)),
        v97_extension=float(np.mean((neighbors >= old_raw_rows) & (neighbors < expanded_raw_rows))),
        v94_suffix=float(np.mean(neighbors >= expanded_raw_rows)))


def support(query, raw, mean, groups, old_raw_rows, mode, previous, window, worker):
    inputs = raw.features.inputs(query)
    prediction = worker.local_prediction(mean, inputs)
    truth = v98.v94.MODES.index(mode)
    fixed = np.stack([prediction["linear"][m] for m in v98.v94.MODES], axis=1)
    error = raw.design(query)@raw.coefficients[mode]+fixed[:, truth]-raw.targets(query)
    rms = np.sqrt(np.mean(error**2, axis=1)).mean()
    for c in v98.v94.CONDITIONS:
        np.testing.assert_allclose(rms, previous["attribution"][c][window]["mean_true_prediction_error_rms"],
                                  rtol=1e-9, atol=1e-8)
    scaled = (inputs-mean.features.input_mean)/mean.features.input_scale
    _, neighbors = mean.tree.query(scaled, k=64)
    partition = neighbor_partition(neighbors, old_raw_rows, raw.tree.n)
    oracle = query["oracle_observations"]
    np.testing.assert_array_equal(oracle[:, truth], query["next_states"])
    gaps = {m: (oracle[:, i]-oracle[:, truth])/raw.target_scale
            for i, m in enumerate(v98.v94.MODES) if i != truth}
    result = dict(support={k: worker.statistics(v) for k, v in prediction["support"].items()},
        neighbor_partition=partition, neighbor_seed_fractions={str(int(s)): float(np.mean(groups[neighbors] == s))
            for s in np.unique(groups)}, local_reference_loo_rms=worker.statistics(prediction["loo_rms"][:, truth]),
        prediction_error_rms=float(rms), coordinate_error_rms=np.sqrt(np.mean(error**2, axis=0)).tolist(),
        error_directions=v96.error_directions(error, gaps))
    return result, fixed


def crossed_scores(query, families, fixed, covariances, previous, window, source):
    output = {}
    mode = previous["old_recipe"]["mode"]
    for mean_recipe in RECIPES:
        for covariance_recipe in RECIPES:
            terms = v98.v94.terms_for_covariances(query, families[mean_recipe].bank,
                                                 fixed[mean_recipe], covariances[covariance_recipe])
            arm = mean_recipe+"_mean__"+covariance_recipe+"_covariance"
            output[arm] = {c: source.v78.v77.attribution(query, terms[c], families[mean_recipe].bank,
                mode, 0, len(query["states"])) for c in v98.v94.CONDITIONS}
            if mean_recipe == covariance_recipe:
                for c in v98.v94.CONDITIONS:
                    v98.v94.v90.verify_baseline(output[arm][c], previous[mean_recipe]["attribution"][c][window])
    return output


def run(args):
    sys.path.insert(0, str(args.v83_dir))
    from scripts import analyze_general_ood_v83_cross_seed_covariance as source
    from scripts import analyze_general_ood_v79_linear_support as worker

    torch.set_num_threads(2)
    data = {r: json.loads((args.v98_dir/f"run_results/replay_{r}/regime_replay.json").read_text()) for r in RECIPES}
    assert all(d["passed"] and d["revision"] == v98.REVISION and d["evaluation_paths"] == 95 for d in data.values())
    population = select_windows(data)
    if args.smoke:
        population = population[:2]
    original, _, _ = v98.v95.load_readouts(args, source)
    families = {}
    for recipe in RECIPES:
        fit_args = SimpleNamespace(**vars(args), recipe=recipe,
                                  checkpoint_dir=args.v98_dir/f"run_checkpoints/{recipe}")
        payload, added_groups, _ = v98.load_extensions(fit_args)
        raw, mean, groups = v98.expand(original, payload, added_groups, recipe)
        assert all((fit_args.checkpoint_dir/f"errors_seed{s}.pt").is_file() for s in np.unique(groups))
        errors, _ = v98.freeze_errors(fit_args, mean, groups, source)
        family = v98.CoverageDensity(raw, mean, groups, errors, original.raw_weight)
        family.recipe = recipe
        families[recipe] = family
    paths = {source.v78.v77.cell_key(c): p for c, p, _ in v98.evaluation_inputs(args, source)}
    records = []
    ranges = {w: (a, b) for w, a, b in source.v78.v77.windows()}
    for role, window, previous in population:
        cell = previous["old_recipe"]
        trace = source.v76.v68.load_trace(paths[source.v78.v77.cell_key(cell)])
        start, stop = ranges[window]
        query = {k: trace[k][start:stop] for k in (*v98.v97.KEYS, "oracle_observations")}
        diagnostics, fixed, covariances = {}, {}, {}
        for recipe, family in families.items():
            diagnostics[recipe], direct = support(query, family.bank, family.mean, family.groups,
                original.bank.tree.n, cell["mode"], previous[recipe], window, worker)
            fixed[recipe], _, covariances[recipe], _ = family.geometry(family.bank.features.inputs(query))
            np.testing.assert_allclose(fixed[recipe], direct, rtol=1e-10, atol=1e-10)
        record = dict(**{k: cell[k] for k in ("group", "seed", "mode", "updating")}, role=role, window=window,
            transitions=stop-start, diagnostics=diagnostics,
            saved_recursive={r: previous[r]["observers"][CANDIDATE][window] for r in RECIPES},
            crossed_scores=crossed_scores(query, families, fixed, covariances, previous, window, source))
        records.append(record)
        print(f"[ATTRIBUTION] {len(records)}/{len(population)} {cell['seed']} {cell['mode']} {window} updating={cell['updating']}", flush=True)
    rows = sum(r["transitions"] for r in records)
    assert (len(records), rows) == ((2, 100) if args.smoke else (20, 1084))
    result = dict(revision=REVISION, source_revision=v98.REVISION, passed=True, smoke=args.smoke,
        checked_windows=len(records), checked_transitions=rows, diagonal_attribution_comparisons=6*len(records),
        support_prediction_rms_comparisons=6*len(records), crossed_mean_covariance_arms=4,
        records=records, environment_steps=0, optimizer_updates=0, reference_refitted=False,
        normal_thresholds_recalibrated=False, coefficient_tuned=False, readout_controls_actor=False,
        decision="development-coverage-attribution-not-readout-adoption",
        limitations=["Failure-selected development windows, one source policy and known simulator modes.",
            "Crossed scores are instantaneous density attribution, not recalibrated recursive risk or RL return evidence.",
            "Neighbor support and local leave-one-out errors are diagnostics, not independent deployment errors."])
    args.result_dir.mkdir(parents=True, exist_ok=True)
    path = args.result_dir/("native_smoke.json" if args.smoke else "coverage_attribution.json")
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] {REVISION}: {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("v98-dir", "v97-dir", "v95-dir", "v94-dir", "v83-dir", "v78-dir", "v76-dir", "v74-dir",
                 "v73-dir", "v72-dir", "v71-dir", "v69-dir", "v70-dir", "v61-dir", "v62-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
