"""Cross density means, covariances and cached pre-window posterior histories."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_general_ood_v101_age_replay as v101

v98, v99 = v101.v98, v101.v100.v99
REVISION = "general_ood_v102_failure_attribution"
RECIPES = ("incumbent_suffix", "age_balanced")
CANDIDATE = "suffix_disagreement_mix"
FAILURES = ((135191, "high_grav", "first_post50"),
            (135192, "heavy", "first_post50"), (135192, "high_grav", "last64"))


def select_windows(data):
    indices = {r: {v99.record_key(c): c for c in data[r]["records"]} for r in RECIPES}
    assert indices[RECIPES[0]].keys() == indices[RECIPES[1]].keys()
    negative = []
    for key, old in indices[RECIPES[0]].items():
        if old["group"] != "v95_development_updating" or old["mode"] == "normal":
            continue
        new = indices[RECIPES[1]][key]
        for window in ("first_post50", "last64"):
            if old["observers"][CANDIDATE][window]["mean_true_minus_best_other_score"] > 0 >= new["observers"][CANDIDATE][window]["mean_true_minus_best_other_score"]:
                negative.append((old["seed"], old["mode"], window))
    assert tuple(negative) == FAILURES
    population = []
    for seed, mode, window in FAILURES:
        for observed, role in ((mode, "new_failure_pair"), ("normal", "matched_normal")):
            for updating in (True, False):
                group = "v95_development_updating" if updating else "v95_development_frozen"
                key = (group, seed, observed, updating)
                population.append((role, window, {r: indices[r][key] for r in RECIPES}))
    assert len({(v99.record_key(p[RECIPES[0]]), w) for _, w, p in population}) == 12
    return population


def posterior_statistics(scores, mode):
    truth = v98.v94.MODES.index(mode)
    other = scores.copy()
    other[:, truth] = -np.inf
    predicted = scores.argmax(axis=1)
    return dict(mode_accuracy=float(np.mean(predicted == truth)),
        mean_true_minus_best_other_score=float(np.mean(scores[:, truth]-other.max(axis=1))),
        predicted_mode_fractions={m: float(np.mean(predicted == i)) for i, m in enumerate(v98.v94.MODES)})


def posterior_from_prefix(likelihoods, prefix_log_posterior):
    from scripts.v73_recursive_mode_readout import RecursiveModeFilter

    sensor = RecursiveModeFilter()
    sensor.log_posterior = prefix_log_posterior.copy()
    return np.asarray([sensor.update(row) for row in likelihoods])


def crossed_scores(query, families, fixed, covariances, prefixes, previous, cached, window, source):
    instant, posterior = {}, {}
    mode = previous[RECIPES[0]]["mode"]
    for mean_recipe in RECIPES:
        for covariance_recipe in RECIPES:
            arm = mean_recipe+"_mean__"+covariance_recipe+"_covariance"
            terms = v98.v94.terms_for_covariances(query, families[mean_recipe].bank,
                fixed[mean_recipe], covariances[covariance_recipe])
            instant[arm] = {c: source.v78.v77.attribution(query, terms[c], families[mean_recipe].bank,
                mode, 0, len(query["states"])) for c in v98.v94.CONDITIONS}
            if mean_recipe == covariance_recipe:
                for c in v98.v94.CONDITIONS:
                    v98.v94.v90.verify_baseline(instant[arm][c], previous[mean_recipe]["attribution"][c][window])
                    np.testing.assert_allclose(terms[c]["likelihoods"], cached[mean_recipe][c], rtol=1e-10, atol=1e-8)
            for history_recipe in RECIPES:
                name = history_recipe+"_history__"+arm
                posterior[name] = {c: posterior_statistics(posterior_from_prefix(
                    terms[c]["likelihoods"], prefixes[history_recipe][c]), mode) for c in v98.v94.CONDITIONS}
                if history_recipe == mean_recipe == covariance_recipe:
                    for c in v98.v94.CONDITIONS:
                        expected = previous[mean_recipe]["observers"][c][window]
                        for field in ("mode_accuracy", "mean_true_minus_best_other_score"):
                            np.testing.assert_allclose(posterior[name][c][field], expected[field], rtol=1e-9, atol=1e-8)
                        assert posterior[name][c]["predicted_mode_fractions"] == expected["predicted_mode_fractions"]
    return instant, posterior


def load_families(args, original, source):
    families, directories = {}, {}
    for recipe in RECIPES:
        native_recipe = "diverse_recipe" if recipe == "incumbent_suffix" else "age_balanced"
        directory = args.v98_dir if recipe == "incumbent_suffix" else args.v101_dir
        fit_args = SimpleNamespace(**vars(args), recipe=native_recipe,
            checkpoint_dir=directory/f"run_checkpoints/{native_recipe}")
        if recipe == "incumbent_suffix":
            payload, added_groups, _ = v98.load_extensions(fit_args)
        else:
            payload, added_groups, _, _ = v101.load_extensions(args, source)
        raw, mean, groups = v98.expand(original, payload, added_groups, native_recipe)
        assert all((fit_args.checkpoint_dir/f"errors_seed{s}.pt").is_file() for s in np.unique(groups))
        errors, _ = v98.freeze_errors(fit_args, mean, groups, source)
        family = v98.CoverageDensity(raw, mean, groups, errors, original.raw_weight)
        family.recipe = native_recipe
        families[recipe], directories[recipe] = family, fit_args.checkpoint_dir
    return families, directories


def load_history(directory, native_recipe, index, key):
    from scripts.v73_recursive_mode_readout import temporal_scores

    path = directory/"terms"/f"eval_{index:02d}.pt"
    assert path.is_file(), path
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert saved["revision"] == v98.REVISION and saved["recipe"] == native_recipe and tuple(saved["key"]) == key
    assert (saved["raw_rows"], saved["mean_rows"]) == (27400, 29600)
    likelihoods = {c: saved["terms"][c]["likelihoods"] for c in v98.v94.CONDITIONS}
    assert all(a.shape == (6000, 5) for a in likelihoods.values())
    scores = {c: temporal_scores(ll, "recursive") for c, ll in likelihoods.items()}
    return likelihoods, scores


def run(args):
    sys.path.insert(0, str(args.v83_dir))
    from scripts import analyze_general_ood_v83_cross_seed_covariance as source
    from scripts import analyze_general_ood_v79_linear_support as support_worker

    torch.set_num_threads(2)
    data = dict(incumbent_suffix=json.loads((args.v98_dir/"run_results/replay_diverse_recipe/regime_replay.json").read_text()),
        age_balanced=json.loads((args.v101_dir/"run_results/replay/regime_replay.json").read_text()))
    assert all(d["passed"] and d["stage"] == "replay" and d["evaluation_paths"] == 95 for d in data.values())
    assert data["age_balanced"]["revision"] == v101.REVISION and data["age_balanced"]["calibration_protocol_matches_incumbent"]
    assert [v99.record_key(r) for r in data[RECIPES[0]]["records"]] == [v99.record_key(r) for r in data[RECIPES[1]]["records"]]
    population = select_windows(data)
    if args.smoke:
        population = population[:2]
    original, _, _ = v98.v95.load_readouts(args, source)
    families, directories = load_families(args, original, source)
    available = v98.evaluation_inputs(args, source)
    paths = {v99.record_key(c): (i, p) for i, (c, p, _) in enumerate(available)}
    assert [v99.record_key(c) for c, _, _ in available] == [v99.record_key(r) for r in data[RECIPES[0]]["records"]]
    ranges, histories, records = {w: (a, b) for w, a, b in source.v78.v77.windows()}, {}, []
    for role, window, previous in population:
        cell, (start, stop) = previous[RECIPES[0]], ranges[window]
        key = v99.record_key(cell)
        index, path = paths[key]
        trace = source.v76.v68.load_trace(path)
        query = {k: trace[k][start:stop] for k in (*v98.v97.KEYS, "oracle_observations")}
        diagnostics, fixed, covariances, prefixes, cached = {}, {}, {}, {}, {}
        for recipe, family in families.items():
            diagnostics[recipe], direct = v99.support(query, family.bank, family.mean, family.groups,
                original.bank.tree.n, cell["mode"], previous[recipe], window, support_worker)
            fixed[recipe], _, covariances[recipe], _ = family.geometry(family.bank.features.inputs(query))
            np.testing.assert_allclose(fixed[recipe], direct, rtol=1e-10, atol=1e-10)
            if (recipe, key) not in histories:
                histories[recipe, key] = load_history(directories[recipe], family.recipe, index, key)
            likelihoods, scores = histories[recipe, key]
            prefixes[recipe] = {c: scores[c][start-1] for c in v98.v94.CONDITIONS}
            cached[recipe] = {c: likelihoods[c][start:stop] for c in v98.v94.CONDITIONS}
        instant, posterior = crossed_scores(query, families, fixed, covariances, prefixes, previous, cached, window, source)
        records.append(dict(**{k: cell[k] for k in ("group", "seed", "mode", "updating")},
            role=role, window=window, window_start=start, transitions=stop-start, diagnostics=diagnostics,
            pre_window_log_posterior={r: {c: p.tolist() for c, p in values.items()} for r, values in prefixes.items()},
            crossed_scores=instant, posterior_scores=posterior))
        print(f"[CROSSED] {len(records)}/{len(population)} {cell['seed']} {cell['mode']} {window} updating={cell['updating']}", flush=True)
    rows = sum(r["transitions"] for r in records)
    assert (len(records), rows, len(histories)) == ((2, 100, 4) if args.smoke else (12, 656, 20))
    result = dict(revision=REVISION, passed=True, smoke=args.smoke, checked_windows=len(records),
        checked_transitions=rows, cached_history_paths_per_recipe=len(histories)//2,
        diagonal_density_comparisons=6*len(records), diagonal_likelihood_slice_comparisons=6*len(records),
        diagonal_posterior_comparisons=6*len(records), support_prediction_rms_comparisons=6*len(records),
        crossed_mean_covariance_arms=4, crossed_history_mean_covariance_arms=8, covariance_conditions=3,
        records=records, environment_steps=0, optimizer_updates=0, reference_refitted=False,
        normal_thresholds_recalibrated=False, temporal_rules_changed=False, coefficient_tuned=False,
        readout_controls_actor=False, readout_adopted=False, decision="development-local-attribution-not-adoption",
        limitations=["Three selected development failures and matched normal/frozen windows; one source policy and known modes.",
            "Prefix histories are reused exactly; only within-window density means/covariances are crossed.",
            "These substitutions are local sensitivities, not a full hybrid-policy replay, recalibrated risk or RL-return evidence."])
    args.result_dir.mkdir(parents=True, exist_ok=True)
    path = args.result_dir/("native_smoke.json" if args.smoke else "failure_attribution.json")
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] {REVISION}: {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("v101-dir", "v100-dir", "v98-dir", "v97-dir", "v95-dir", "v94-dir", "v83-dir", "v78-dir",
                 "v76-dir", "v74-dir", "v73-dir", "v72-dir", "v71-dir", "v69-dir", "v70-dir", "v61-dir", "v62-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
