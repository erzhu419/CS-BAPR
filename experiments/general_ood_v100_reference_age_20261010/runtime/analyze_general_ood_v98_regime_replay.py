"""Matched v97 reference expansion, source-excluded errors and development replay."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import Covariance, multivariate_normal
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_general_ood_v95_fresh_seed as v95
import probe_general_ood_v97_regime_coverage as v97

v94 = v95.v94
REVISION = "general_ood_v98_regime_replay"
CONTROLS = tuple("v94_unchanged__"+c for c in v94.CONDITIONS)


class CoverageDensity(v94.PairedSuffixDensity):
    def metadata(self, condition):
        result = super().metadata(condition)
        result.update(reference_revision=REVISION, recipe=self.recipe,
            conditional_mean="local-linear v94 mean support plus equal-budget v97 references",
            raw_support_expanded=True, full_error_bank_recomputed=True)
        return result


def load_extensions(args):
    combined, groups, sources = {}, [], []
    for seed in v97.SEEDS:
        directory = args.v97_dir/f"run_checkpoints/seed{seed}"
        bundle = torch.load(directory/"reference_inputs.pt", map_location="cpu", weights_only=False)
        scalar = json.loads((args.v97_dir/f"run_results/seed{seed}/reference_collection.json").read_text())
        assert bundle["revision"] == scalar["revision"] == v97.REVISION
        assert bundle["seed"] == scalar["seed"] == seed and bundle["spec"] == scalar["spec"] == v97.SPEC
        assert scalar["passed"] and not scalar["smoke"] and scalar["paired_prefixes"]
        assert scalar["equal_recipe_budgets"] and not scalar["evaluation_paths_used_for_reference"]
        assert scalar["per_recipe_reference_inputs"] == {c: 2200 for c in v97.RECIPES}
        data = bundle["extensions"][args.recipe]
        for mode in v94.MODES:
            for key, width in (("states", 11), ("executed_actions", 3), ("next_states", 11)):
                assert data[mode][key].shape == (2200, width) and np.isfinite(data[mode][key]).all()
                if key != "next_states":
                    np.testing.assert_array_equal(data[mode][key], data["normal"][key])
        np.testing.assert_array_equal(bundle["groups"][args.recipe], np.full(2200, seed))
        groups.append(bundle["groups"][args.recipe])
        sources.extend(bundle["sources"][args.recipe])
        combined[seed] = data
    payload = {m: {k: np.concatenate([combined[s][m][k] for s in v97.SEEDS]) for k in v97.KEYS}
               for m in v94.MODES}
    return payload, np.concatenate(groups), sources


def expand(original, payload, added_groups, recipe):
    from scripts.v74_density_coverage import DensityCoverageBank

    raw = DensityCoverageBank.extend(original.bank, payload, recipe+"_raw")
    mean = DensityCoverageBank.extend(original.mean, payload, recipe+"_mean")
    old_raw, old_mean = original.bank.tree.n, original.mean.tree.n
    # Keep the complete raw bank as the mean-bank prefix, then the old suffix-only rows.
    order = np.concatenate((np.arange(old_raw), np.arange(old_mean, mean.tree.n), np.arange(old_raw, old_mean)))
    mean.tree = cKDTree(mean.tree.data[order])
    mean.residuals = {m: mean.residuals[m][order] for m in v94.MODES}
    groups = np.concatenate((original.groups[:old_raw], added_groups, original.groups[old_raw:]))
    np.testing.assert_array_equal(mean.tree.data[:raw.tree.n], raw.tree.data)
    for m in v94.MODES:
        np.testing.assert_array_equal(mean.residuals[m][:raw.tree.n], raw.residuals[m])
        np.testing.assert_array_equal(raw.coefficients[m], original.bank.coefficients[m])
        np.testing.assert_array_equal(mean.coefficients[m], original.bank.coefficients[m])
    for b in (raw, mean):
        for k in ("input_mean", "input_scale"):
            np.testing.assert_array_equal(getattr(b.features, k), getattr(original.bank.features, k))
        np.testing.assert_array_equal(b.target_scale, original.bank.target_scale)
    return raw, mean, groups


def heldout_errors(mean, groups, seed, rows):
    from scripts.v83_cross_seed_covariance import subset_bank, LocalLinearMean

    assert np.all(groups[rows] == seed)
    reduced = subset_bank(mean, groups != seed)
    inputs = mean.tree.data[rows]*mean.features.input_scale+mean.features.input_mean
    prediction = LocalLinearMean(reduced).predict(inputs)
    errors = {m: mean.residuals[m][rows]-prediction[m] for m in v94.MODES}
    return errors, dict(seed=int(seed), error_rows=len(rows), mean_training_rows=reduced.tree.n,
        excluded_mean_rows=int(np.sum(groups == seed)), source_seed_excluded_from_all_mean_rows=True)


def freeze_errors(args, mean, groups, source):
    errors = {m: np.empty_like(mean.residuals[m]) for m in v94.MODES}
    fits = []
    for seed in np.unique(groups):
        rows = np.flatnonzero(groups == seed)
        path = args.checkpoint_dir/f"errors_seed{seed}.pt"
        if path.is_file():
            saved = torch.load(path, map_location="cpu", weights_only=False)
            assert saved["revision"] == REVISION and saved["recipe"] == args.recipe and saved["ridge_alpha"] == .064
            np.testing.assert_array_equal(saved["groups"], groups)
            np.testing.assert_array_equal(saved["rows"], rows)
            values, fit = saved["errors"], saved["fit"]
        else:
            values, fit = heldout_errors(mean, groups, seed, rows)
            torch.save(dict(revision=REVISION, recipe=args.recipe, ridge_alpha=.064,
                groups=groups, rows=rows, errors=values, fit=fit), path)
        for m in v94.MODES:
            assert values[m].shape == errors[m][rows].shape and np.isfinite(values[m]).all()
            errors[m][rows] = values[m]
        fits.append(fit)
        print(f"[ERRORS] {args.recipe} seed{seed}: {len(rows)} rows from {fit['mean_training_rows']} other-seed rows", flush=True)
    assert sum(f["error_rows"] for f in fits) == mean.tree.n
    return errors, fits


def cached_terms(args, family, trace, key, name):
    path = args.checkpoint_dir/"terms"/f"{name}.pt"
    if path.is_file():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        assert saved["revision"] == REVISION and saved["recipe"] == args.recipe and tuple(saved["key"]) == tuple(key)
        assert saved["raw_rows"] == family.bank.tree.n and saved["mean_rows"] == family.mean.tree.n
        return saved["terms"], saved["geometry"]
    terms, geometry = family.terms(trace)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(revision=REVISION, recipe=args.recipe, key=key, raw_rows=family.bank.tree.n,
        mean_rows=family.mean.tree.n, terms=terms, geometry=geometry), path)
    return terms, geometry


def phases(args, family, source):
    from scripts.v73_recursive_mode_readout import RecursiveModeReference

    path = args.checkpoint_dir/"phases.pt"
    if path.is_file():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        assert saved["revision"] == REVISION and saved["recipe"] == args.recipe
        assert saved["normal_null_transitions"] == 12000 and saved["raw_weight"] == family.raw_weight
        np.testing.assert_array_equal(saved["groups"], family.groups)
        references = {}
        for c in v94.CONDITIONS:
            ref = RecursiveModeReference()
            ref.bank = SimpleNamespace(metadata=lambda c=c: family.metadata(c))
            ref.phase, ref.normal_null_calibration_transitions = saved["phases"][c], 12000
            references[c] = ref
        return references
    null_dir = args.v69_dir/"run_checkpoints/reference/traces"
    nulls = [source.v76.v68.load_trace(source.v76.v69.trace_path(null_dir, c))
             for c in source.v76.v69.reference_cells(source.v76.v69.SPEC) if c["role"] == "normal_null"]
    assert len(nulls) == 2 and sum(len(t["states"]) for t in nulls) == 12000
    terms = [cached_terms(args, family, t, ("normal_null", i), f"normal_null_{i}")[0] for i, t in enumerate(nulls)]
    references = v94.calibrate_paired(family, nulls, terms)
    torch.save(dict(revision=REVISION, recipe=args.recipe, raw_weight=family.raw_weight, groups=family.groups,
        normal_null_transitions=12000, phases={c: r.phase for c, r in references.items()}), path)
    print(f"[CALIBRATE] {args.recipe}: all three controls, same 12000 normal-null transitions", flush=True)
    return references


def smoke(args, original, raw, mean, groups, source):
    from scripts.v83_cross_seed_covariance import LocalLinearMean

    checks = 0
    for seed in np.unique(groups):
        rows = np.flatnonzero(groups == seed)[[0, -1]]
        errors, fit = heldout_errors(mean, groups, seed, rows)
        assert fit["source_seed_excluded_from_all_mean_rows"] and all(np.isfinite(a).all() for a in errors.values())
        checks += len(rows)
    seed = v95.SEEDS[0]
    saved = json.loads((args.v95_dir/f"run_results/seed{seed}/fresh_seed.json").read_text())
    previous = next(r for r in saved["records"] if r["updating"] and r["mode"] == "high_grav")
    trace = source.v76.v68.load_trace(source.v76.trace_path(args.v95_dir/f"run_checkpoints/seed{seed}/traces", previous))
    start, stop = next((a, b) for w, a, b in source.v78.v77.windows() if w == "first_post50")
    query = {k: trace[k][start:stop] for k in (*v97.KEYS, "oracle_observations")}
    terms, _ = original.terms(query)
    for c in v94.CONDITIONS:
        actual = source.v78.v77.attribution(query, terms[c], original.bank, "high_grav", 0, 50)
        v94.v90.verify_baseline(actual, previous["attribution"][c]["first_post50"])
    prediction = LocalLinearMean(mean).predict(raw.features.inputs(query))
    assert all(a.shape == (50, 11) and np.isfinite(a).all() for a in prediction.values())
    available = evaluation_inputs(args, source)
    return dict(passed=True, partial_heldout_error_rows_checked=checks, unchanged_v94_attribution_comparisons=3,
        evaluation_paths_available=len(available),
        raw_mean_prefix_and_all_labels_aligned=True, forward_and_scales_unchanged=True,
        partial_checks_only=True, full_error_bank_fitted=False, normal_thresholds_recalibrated=False)


def gaussian_check(family, source):
    rows = [0, family.bank.base_density_transitions, family.bank.tree.n-1,
            family.bank.tree.n, family.mean.tree.n-1]
    inputs = family.mean.tree.data[rows]*family.mean.features.input_scale+family.mean.features.input_mean
    fixed, members, covariance, _ = family.geometry(inputs)
    direct = source.LocalLinearMean(family.mean).predict(inputs)
    np.testing.assert_allclose(fixed, np.stack([direct[m] for m in v94.MODES], axis=1), rtol=1e-10, atol=1e-10)
    member_checks = v94.v93.verify_members(family.mean, family.groups, inputs, family.bank.tree.n, members, family.member_fits)
    targets = np.zeros((5, 11))
    checked = 0
    from scripts.v77_mean_covariance import density_terms
    for c in v94.CONDITIONS:
        for i in range(5):
            error = targets-fixed[:, i]
            ll = density_terms(error, covariance[c][:, i], family.bank.target_scale)[2]
            for row in range(5):
                scale = family.bank.target_scale
                physical = covariance[c][row, i]*scale[:, None]*scale[None, :]
                expected = multivariate_normal.logpdf(error[row]*scale, cov=Covariance.from_cholesky(np.linalg.cholesky(physical)))
                np.testing.assert_allclose(ll[row], expected, rtol=1e-10, atol=1e-8)
                checked += 1
    return dict(physical_gaussian_checks=checked, direct_member_checks=member_checks)


def evaluation_inputs(args, source):
    old = json.loads((args.v94_dir/"run_results/replay/suffix_replay.json").read_text())
    index = {source.v78.v77.cell_key(r): r for r in old["records"]}
    population = []
    for cell, path in source.v76.evaluation_inputs(args, args.v76_dir/"run_checkpoints/formal/traces"):
        normalized = source.v78.replay_cell(cell)
        population.append((normalized, path, index[source.v78.v77.cell_key(normalized)]))
    assert len(population) == len(index) == 65
    for seed in v95.SEEDS:
        saved = json.loads((args.v95_dir/f"run_results/seed{seed}/fresh_seed.json").read_text())
        assert saved["revision"] == v95.REVISION and len(saved["records"]) == 10
        for previous in saved["records"]:
            cell = {k: previous[k] for k in ("seed", "mode", "updating")}
            cell["group"] = "v95_development_updating" if cell["updating"] else "v95_development_frozen"
            path = source.v76.trace_path(args.v95_dir/f"run_checkpoints/seed{seed}/traces", previous)
            population.append((cell, path, previous))
    assert len(population) == 95 and all(p.is_file() for _, p, _ in population)
    assert not set(v97.SEEDS) & {c["seed"] for c, _, _ in population}
    return population


def replay(args, family, references, source):
    from scripts.v73_recursive_mode_readout import observe_likelihoods

    population = evaluation_inputs(args, source)
    path = args.resume_from or args.checkpoint_dir/"progress.json"
    records = []
    if path.is_file():
        saved = json.loads(path.read_text())
        assert saved["revision"] == REVISION and saved["recipe"] == args.recipe
        records = saved["records"]
        assert [source.v78.v77.cell_key(r) for r in records] == [source.v78.v77.cell_key(c) for c, _, _ in population[:len(records)]]
    for i, (cell, trace_path, previous) in enumerate(population[len(records):], start=len(records)):
        trace = source.v76.v68.load_trace(trace_path)
        assert len(trace["states"]) == source.v76.SPEC["steps"]
        terms, geometry = cached_terms(args, family, trace, source.v78.v77.cell_key(cell), f"eval_{i:02d}")
        record = dict(**cell, geometry=geometry, observers={}, attribution={}, change_minus_nochange={})
        for field in ("observers", "attribution", "change_minus_nochange"):
            record[field] = {control: deepcopy(previous[field][c]) for control, c in zip(CONTROLS, v94.CONDITIONS)}
        for c in v94.CONDITIONS:
            ll = terms[c]["likelihoods"]
            windows = source.v76.v73.window_statistics(trace, ll, observe_likelihoods(ll, trace["dones"], references[c], "recursive"), cell["mode"])
            record["observers"][c] = {w: windows[w] for w in source.v78.v77.WINDOWS}
            record["attribution"][c] = {w: source.v78.v77.attribution(trace, terms[c], family.bank, cell["mode"], a, b)
                                      for w, a, b in source.v78.v77.windows()}
            normal = record if cell["mode"] == "normal" else next(r for r in records
                if r["group"] == cell["group"] and r["seed"] == cell["seed"] and r["updating"] == cell["updating"] and r["mode"] == "normal")
            record["change_minus_nochange"][c] = {w: record["observers"][c][w]["mean_coefficient"]
                -normal["observers"][c][w]["mean_coefficient"] for w in source.v76.v68.BASE.CHANGE_WINDOWS}
        records.append(record)
        path.write_text(json.dumps(dict(revision=REVISION, recipe=args.recipe, records=records), allow_nan=False)+"\n")
        print(f"[REPLAY] {args.recipe} {i+1}/95 {source.v78.v77.cell_key(cell)}", flush=True)
    return dict(records=records, summary=source.v78.v77.summarize(records, (*CONTROLS, *v94.CONDITIONS)),
                evaluation_paths=len(records), evaluated_transitions=len(records)*source.v76.SPEC["steps"], all_paths_development=True)


def run(args):
    sys.path.insert(0, str(args.v83_dir))
    from scripts import analyze_general_ood_v83_cross_seed_covariance as source

    torch.set_num_threads(2)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    original, _, _ = v95.load_readouts(args, source)
    payload, added_groups, sources = load_extensions(args)
    raw, mean, groups = expand(original, payload, added_groups, args.recipe)
    assert (raw.tree.n, mean.tree.n, len(np.unique(groups))) == (27400, 29600, 8)
    if args.stage == "smoke":
        result = smoke(args, original, raw, mean, groups, source)
    else:
        errors, fits = freeze_errors(args, mean, groups, source)
        family = CoverageDensity(raw, mean, groups, errors, original.raw_weight)
        family.recipe = args.recipe
        checked = v94.v92.verify_error_rows(mean, groups, errors, raw.tree.n)
        references = phases(args, family, source)
        result = dict(passed=True, full_error_bank_fitted=True, inner_fits=fits, direct_error_rows_checked=checked,
            normal_thresholds_recalibrated=True, normal_null_transitions_per_condition=12000,
            reference_metadata={c: r.metadata() for c, r in references.items()}, **gaussian_check(family, source))
        if args.stage == "replay":
            result.update(replay(args, family, references, source))
    result.update(revision=REVISION, stage=args.stage, recipe=args.recipe, reference_sources=sources,
        raw_rows=raw.tree.n, mean_rows=mean.tree.n, source_seeds=8, additional_reference_inputs=4400,
        raw_weight=original.raw_weight, disagreement_coefficient=1., conditions=list(v94.CONDITIONS),
        forward_models_refitted=False, scaling_refitted=False, weight_refitted=False, coefficient_tuned=False,
        temporal_rules_changed=False, environment_steps=0, optimizer_updates=0,
        readout_controls_actor=False, readout_adopted=False, independent_confirmation=False,
        decision="development-reference-coverage-pilot-not-actor-adoption",
        limitations=["Known-mode simulator references, one source policy; no independent training or return evidence.",
            "Both recipes add the same reference budget; the unchanged control retains fewer rows and six members.",
            "Normal-null calibration population and procedure are reused, not independent calibration confirmation.",
            "All v94/v95 evaluation paths are now development evidence; fresh confirmation is still required."])
    name = dict(smoke="native_smoke.json", fit="reference_fit.json", replay="regime_replay.json")[args.stage]
    (args.result_dir/name).write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] {REVISION} {args.recipe} {args.stage}: {args.result_dir/name}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("v97-dir", "v95-dir", "v94-dir", "v83-dir", "v78-dir", "v76-dir", "v74-dir", "v73-dir",
                 "v72-dir", "v71-dir", "v69-dir", "v70-dir", "v61-dir", "v62-dir", "checkpoint-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--recipe", choices=tuple(v97.RECIPES), required=True)
    parser.add_argument("--stage", choices=("smoke", "fit", "replay"), required=True)
    parser.add_argument("--resume-from", type=Path)
    run(parser.parse_args())
