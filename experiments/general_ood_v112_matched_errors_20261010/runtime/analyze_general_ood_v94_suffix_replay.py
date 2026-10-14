"""Paired suffix uncertainty replay on the unchanged 65 development paths."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from scipy.stats import Covariance, multivariate_normal
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_general_ood_v93_mean_disagreement as v93
import analyze_general_ood_v90_coverage_replay as v90

v92, v91, v89 = v93.v92, v93.v91, v93.v89
MODES = v93.v84.MODES
REVISION = "general_ood_v94_suffix_replay"
CONDITIONS = ("suffix_support_mix", "suffix_disagreement_mix", "suffix_trace_match_mix")
CONTROLS = (*v90.CONTROLS, *v89.CONDITIONS)


def full_errors(raw, groups, mean, mean_groups, directory):
    from scripts.v83_cross_seed_covariance import ALPHA

    path = directory/"full_suffix_errors.pt"
    if path.is_file():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        assert saved["revision"] == REVISION and saved["ridge_alpha"] == ALPHA
        assert saved["raw_rows"] == raw.tree.n and saved["mean_rows"] == mean.tree.n
        np.testing.assert_array_equal(saved["raw_groups"], groups)
        np.testing.assert_array_equal(saved["mean_groups"], mean_groups)
        errors, fits = saved["errors"], saved["inner_fits"]
        print("[REUSE] full-bank suffix cross-seed errors", flush=True)
    else:
        old, old_fits = v91.aligned_errors(raw, groups, mean, mean_groups)
        extra, extra_fits = v92.added_errors(mean, mean_groups, raw.tree.n)
        errors = {m: np.concatenate((old[m], extra[m])) for m in MODES}
        fits = old_fits+extra_fits
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(dict(revision=REVISION, ridge_alpha=ALPHA, raw_rows=raw.tree.n, mean_rows=mean.tree.n,
            raw_groups=groups, mean_groups=mean_groups, errors=errors, inner_fits=fits), path)
    assert all(errors[m].shape == mean.residuals[m].shape and np.isfinite(errors[m]).all() for m in MODES)
    assert sum(f["error_rows"] for f in fits) == mean.tree.n
    assert all(f["source_seed_excluded_from_all_mean_rows"] for f in fits)
    return errors, fits


def covariance_variants(base, moment):
    base_trace = np.trace(base, axis1=-2, axis2=-1)
    assert np.all(base_trace > 0)
    multiplier = 1+np.trace(moment, axis1=-2, axis2=-1)/base_trace
    return {CONDITIONS[0]: base, CONDITIONS[1]: base+moment,
            CONDITIONS[2]: base*multiplier[..., None, None]}


def terms_for_covariances(trace, bank, centers, covariances):
    from scripts.v77_mean_covariance import density_terms

    targets, design = bank.targets(trace), bank.design(trace)
    output = {c: {k: [] for k in ("energy", "logdet", "likelihoods", "mean_error_rms")} for c in CONDITIONS}
    for i, mode in enumerate(MODES):
        errors = targets-design@bank.coefficients[mode]-centers[:, i]
        for condition in CONDITIONS:
            energy, logdet, likelihood = density_terms(errors, covariances[condition][:, i], bank.target_scale)
            for key, value in (("energy", energy), ("logdet", logdet), ("likelihoods", likelihood),
                               ("mean_error_rms", np.sqrt(np.mean(errors**2, axis=1)))):
                output[condition][key].append(value)
    return {c: {k: np.column_stack(v) for k, v in values.items()} for c, values in output.items()}


class PairedSuffixDensity:
    def __init__(self, raw, mean, groups, errors, weight):
        from scripts.v83_cross_seed_covariance import LocalLinearMean, subset_bank

        self.bank, self.mean, self.groups = raw, mean, groups
        self.errors, self.raw_weight = errors, weight
        self.members, self.member_fits = [], []
        for seed in np.unique(groups):
            keep = groups != seed
            reduced = subset_bank(mean, keep)
            self.members.append(LocalLinearMean(reduced))
            original = int(keep[:raw.tree.n].sum())
            self.member_fits.append(dict(seed=int(seed), training_rows=reduced.tree.n,
                base_training_rows=original, added_training_rows=reduced.tree.n-original,
                excluded_rows=int((~keep).sum()), source_seed_excluded_from_original_and_added_rows=True))

    def geometry(self, inputs):
        centers, covariance, diagnostic = v92.support_geometry(self.bank, self.mean, self.errors, inputs, self.raw_weight)
        fixed = np.stack([centers[m] for m in MODES], axis=1)
        members = np.asarray([np.stack([p[m] for m in MODES], axis=1)
                              for p in (model.predict(inputs) for model in self.members)])
        base = np.stack([covariance[m] for m in MODES], axis=1)
        moment = v93.displacement_second_moment(fixed, members)
        variants = covariance_variants(base, moment)
        directional = np.trace(variants[CONDITIONS[1]], axis1=-2, axis2=-1)
        scaled = np.trace(variants[CONDITIONS[2]], axis1=-2, axis2=-1)
        np.testing.assert_allclose(directional, scaled, rtol=1e-10, atol=1e-10)
        diagnostic.update(maximum_trace_match_difference=float(np.max(np.abs(directional-scaled))),
            per_mode_traces={m: dict(base=float(np.trace(base[:, i], axis1=-2, axis2=-1).mean()),
                added=float(np.trace(moment[:, i], axis1=-2, axis2=-1).mean()), final=float(directional[:, i].mean()))
                for i, m in enumerate(MODES)})
        return fixed, members, variants, diagnostic

    def terms(self, trace):
        fixed, _, covariance, diagnostic = self.geometry(self.bank.features.inputs(trace))
        return terms_for_covariances(trace, self.bank, fixed, covariance), diagnostic

    def metadata(self, condition):
        return dict(modes=list(MODES), condition=condition, mean_fitting_rows=self.mean.tree.n,
            raw_reference_rows=self.bank.tree.n, error_reference_rows=self.mean.tree.n,
            conditional_mean="unchanged full-bank v89 suffix local-linear mean", neighbors=64, ridge_alpha=.064,
            covariance="v92 full-bank error mix" if condition == CONDITIONS[0] else
                ("v92 mix + source-seed mean-displacement second moment" if condition == CONDITIONS[1] else "trace-matched rescaling of v92 mix"),
            members=len(self.members), member_fits=self.member_fits, disagreement_coefficient=1.,
            raw_weight=self.raw_weight, query_targets_used_for_geometry=False,
            forward_refitted=False, scaling_refitted=False,
            likelihood_units="physical delta-state Gaussian density including log determinant")


def cached_terms(path, family, trace, key):
    if path.is_file():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        assert saved["revision"] == REVISION and saved["raw_weight"] == family.raw_weight
        assert saved["key"] == key and saved["conditions"] == list(CONDITIONS)
        terms, diagnostic = saved["terms"], saved["diagnostic"]
        assert all(a.shape == (len(trace["states"]), len(MODES)) and np.isfinite(a).all()
                   for values in terms.values() for a in values.values())
        print(f"[REUSE] paired terms {key}", flush=True)
    else:
        terms, diagnostic = family.terms(trace)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(revision=REVISION, raw_weight=family.raw_weight, conditions=list(CONDITIONS),
            key=key, terms=terms, diagnostic=diagnostic), path)
    return terms, diagnostic


def calibrate_paired(family, nulls, terms):
    from scripts.v73_recursive_mode_readout import RecursiveModeReference

    references = {}
    for condition in CONDITIONS:
        # Feed shared, precomputed scores into the unchanged calibration algorithm.
        density = SimpleNamespace(metadata=lambda c=condition: family.metadata(c),
            log_likelihoods=lambda trace, c=condition: next(values[c]["likelihoods"]
                for original, values in zip(nulls, terms) if trace is original))
        references[condition] = RecursiveModeReference.calibrate(density, nulls)
    return references


def freeze_references(args, family, source, null_paths):
    from scripts.v73_recursive_mode_readout import RecursiveModeReference

    path = args.checkpoint_dir/"paired_phases.pt"
    if path.is_file():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        assert saved["revision"] == REVISION and saved["raw_weight"] == family.raw_weight
        assert saved["spec"] == source.v76.SPEC and saved["normal_null_transitions"] == 12000
        assert set(saved["phases"]) == set(CONDITIONS)
        references = {}
        for c, phase in saved["phases"].items():
            ref = RecursiveModeReference()
            ref.bank = SimpleNamespace(metadata=lambda condition=c: family.metadata(condition))
            ref.phase, ref.normal_null_calibration_transitions = phase, 12000
            references[c] = ref
        print("[REUSE] same-population paired phase calibration", flush=True)
    else:
        nulls = [source.v76.v68.load_trace(p) for p in null_paths]
        assert len(nulls) == 2 and sum(len(t["states"]) for t in nulls) == 12000
        terms = [cached_terms(args.checkpoint_dir/f"terms/normal_null_{i}.pt", family, t, ("normal_null", i))[0]
                 for i, t in enumerate(nulls)]
        references = calibrate_paired(family, nulls, terms)
        torch.save(dict(revision=REVISION, raw_weight=family.raw_weight, spec=source.v76.SPEC,
            normal_null_transitions=12000, phases={c: r.phase for c, r in references.items()}), path)
        print("[CALIBRATE] all three conditions: same 12000 normal-null transitions", flush=True)
    return references


def native_smoke(inputs, family, expanded, original_errors, scores, source):
    baseline = source.CrossSeedDensityBank(family.bank, original_errors, family.raw_weight)
    old_densities = v90.make_densities(family.bank, expanded, original_errors, family.raw_weight)
    selected = [(c, p) for c, p in inputs if
        (c["group"] == "v76_fresh_updating" and c["seed"] == 135180 and c["mode"] == "compound") or
        (c["group"] == "v61_development_replay" and c["mode"] in ("slippery", "heavy"))]
    assert len(selected) == 3
    checked, gaussians, controls, member_checks = 0, 0, 0, 0
    maximum_difference, maximum_trace_difference = 0., 0.
    for cell, path in selected:
        trace = source.v76.v68.load_trace(path)
        old = scores[source.v78.v77.cell_key(source.v78.replay_cell(cell))]
        for window, start, stop in source.v78.v77.windows():
            query = {k: trace[k][start:stop] for k in v89.KEYS}
            if "oracle_observations" in trace:
                query["oracle_observations"] = trace["oracle_observations"][start:stop]
            for c, density in ((source.MIX_CONDITION, baseline), *old_densities.items()):
                terms = density.terms(query)[0]
                actual = source.v78.v77.attribution(query, terms, family.bank, cell["mode"], 0, len(query["states"]))
                v90.verify_baseline(actual, old["attribution"][c][window])
                controls += 1
            inputs = family.bank.features.inputs(query)
            fixed, members, covariance, diagnostic = family.geometry(inputs)
            direct = source.LocalLinearMean(family.mean).predict(inputs)
            np.testing.assert_allclose(fixed, np.stack([direct[m] for m in MODES], axis=1), rtol=1e-10, atol=1e-10)
            member_checks += v93.verify_members(family.mean, family.groups, inputs, family.bank.tree.n,
                                                members, family.member_fits)
            moment = v93.displacement_second_moment(fixed, members)
            np.testing.assert_allclose(covariance[CONDITIONS[1]], covariance[CONDITIONS[0]]+moment, rtol=1e-10, atol=1e-10)
            terms = terms_for_covariances(query, family.bank, fixed, covariance)
            rows = np.linspace(0, len(inputs)-1, 5, dtype=int)
            targets, design = family.bank.targets(query), family.bank.design(query)
            for condition in CONDITIONS:
                for c, mode in enumerate(MODES):
                    offsets = (targets-design@family.bank.coefficients[mode]-fixed[:, c])*family.bank.target_scale
                    for row in rows:
                        physical = covariance[condition][row, c]*family.bank.target_scale[:, None]*family.bank.target_scale[None, :]
                        distribution = Covariance.from_cholesky(np.linalg.cholesky(physical))
                        expected = multivariate_normal.logpdf(offsets[row], cov=distribution)
                        actual = terms[condition]["likelihoods"][row, c]
                        maximum_difference = max(maximum_difference, abs(float(actual-expected)))
                        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-8)
                        gaussians += 1
            maximum_trace_difference = max(maximum_trace_difference, diagnostic["maximum_trace_match_difference"])
            checked += len(inputs)
            print(f"[SMOKE] {cell['group']} {cell['mode']} {window}: paired geometry, members, physical scores and old controls match", flush=True)
    return dict(revision=REVISION, passed=True, smoke_paths=3, checked_windows=6,
        checked_transitions=checked, gaussian_checks=gaussians, old_attribution_comparisons=controls,
        direct_member_prediction_checks=member_checks, conditional_means_unchanged=True,
        covariance_formula_matches=True, trace_matched_control_matches=True,
        maximum_trace_match_difference=maximum_trace_difference, maximum_physical_score_difference=maximum_difference)


def evaluate(args, inputs, references, family, scores, source):
    from scripts.v73_recursive_mode_readout import observe_likelihoods

    records = []
    if args.resume_from:
        saved = json.loads(args.resume_from.read_text())
        assert saved["revision"] == REVISION and saved["spec"] == source.v76.SPEC
        assert saved["raw_weight"] == family.raw_weight and saved["conditions"] == list(CONDITIONS)
        records = saved["records"]
        assert [source.v78.v77.cell_key(r) for r in records] == [source.v78.v77.cell_key(source.v78.replay_cell(c))
            for c, _ in inputs[:len(records)]]
    for i, (cell, path) in enumerate(inputs[len(records):], start=len(records)):
        trace = source.v76.v68.load_trace(path)
        assert len(trace["states"]) == source.v76.SPEC["steps"]
        key = source.v78.v77.cell_key(source.v78.replay_cell(cell))
        previous = scores[key]
        record = source.v78.replay_cell(cell)
        for field in ("observers", "attribution", "change_minus_nochange"):
            record[field] = {c: deepcopy(previous[field][c]) for c in CONTROLS}
        terms, record["geometry"] = cached_terms(args.checkpoint_dir/f"terms/eval_{i:02d}.pt", family, trace, key)
        for condition in CONDITIONS:
            ll = terms[condition]["likelihoods"]
            windows = source.v76.v73.window_statistics(trace, ll,
                observe_likelihoods(ll, trace["dones"], references[condition], "recursive"), cell["mode"])
            record["observers"][condition] = {w: windows[w] for w in source.v78.v77.WINDOWS}
            record["attribution"][condition] = {w: source.v78.v77.attribution(trace, terms[condition], family.bank,
                cell["mode"], a, b) for w, a, b in source.v78.v77.windows()}
            normal = record if cell["mode"] == "normal" else next(r for r in records if r["group"] == record["group"]
                and r["seed"] == cell["seed"] and r["updating"] == cell["updating"] and r["mode"] == "normal")
            record["change_minus_nochange"][condition] = {w: record["observers"][condition][w]["mean_coefficient"]
                -normal["observers"][condition][w]["mean_coefficient"] for w in source.v76.v68.BASE.CHANGE_WINDOWS}
        records.append(record)
        (args.checkpoint_dir/"progress.json").write_text(json.dumps(dict(revision=REVISION, spec=source.v76.SPEC,
            raw_weight=family.raw_weight, conditions=list(CONDITIONS), records=records), allow_nan=False)+"\n")
        print(f"[EVAL] {len(records)}/{len(inputs)} {key}", flush=True)
    return records


def run(args):
    sys.path.insert(0, str(args.v83_dir))
    from scripts import analyze_general_ood_v83_cross_seed_covariance as source

    torch.set_num_threads(2)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    pilot = json.loads((args.v93_dir/"run_results/pilot/mean_disagreement.json").read_text())
    scores = json.loads((args.v90_dir/"run_results/replay/coverage_replay.json").read_text())
    assert pilot["revision"] == v93.REVISION and not pilot["smoke"] and pilot["passed"]
    assert scores["revision"] == v90.REVISION and len(scores["records"]) == 65
    saved = torch.load(args.v76_dir/"run_checkpoints/formal/behavior_coverage_references.pt", map_location="cpu", weights_only=False)
    assert saved["revision"] == source.v76.REVISION and scores["spec"] == saved["spec"] == source.v76.SPEC
    raw = saved["references"]["updating_mixed"].bank
    selection = json.loads((args.v83_dir/"run_checkpoints/cross-seed-covariance/nested_selection.json").read_text())
    assert selection["revision"] == source.REVISION and raw.tree.n == 23000
    groups = source.groups_from_blocks(selection["reference_blocks"], raw.tree.n)
    weight = selection["selection"]["raw_weight"]
    assert weight == scores["raw_weight"] == pilot["raw_weight"]
    stored = args.v76_dir/"run_checkpoints/formal/traces"
    traces = {}
    for seed in v89.SEEDS:
        with np.load(stored/f"density_extension_seed{seed}_updating_normal.npz") as data:
            traces[seed] = {k: data[k] for k in (*v89.KEYS, "oracle_observations")}
    v89.verify_sources(raw, selection["reference_blocks"], traces)
    expanded, mean_groups = v89.expand_banks(raw, groups, traces)
    mean = expanded["suffix_mean_mix"]
    errors, fits = full_errors(raw, groups, mean, mean_groups, args.checkpoint_dir)
    checked_errors = v92.verify_error_rows(mean, mean_groups, errors, raw.tree.n)
    family = PairedSuffixDensity(raw, mean, mean_groups, errors, weight)
    assert len(family.members) == 6 and mean.tree.n == 25200
    inputs = source.v76.evaluation_inputs(args, stored)
    index = {source.v78.v77.cell_key(r): r for r in scores["records"]}
    assert len(inputs) == len(index) == 65
    assert {source.v78.v77.cell_key(source.v78.replay_cell(c)) for c, _ in inputs} == set(index)
    assert all(p.is_file() for _, p in inputs), "missing evaluation trace"
    null_dir = args.v69_dir/"run_checkpoints/reference/traces"
    null_paths = [source.v76.v69.trace_path(null_dir, c) for c in source.v76.v69.reference_cells(source.v76.v69.SPEC)
                  if c["role"] == "normal_null"]
    assert len(null_paths) == 2 and all(p.is_file() for p in null_paths), "missing normal-null trace"
    if args.smoke:
        cached = torch.load(args.v83_dir/"run_checkpoints/cross-seed-covariance/cross_seed_errors.pt", map_location="cpu", weights_only=False)
        assert cached["revision"] == source.REVISION and cached["rows"] == raw.tree.n and cached["ridge_alpha"] == source.ALPHA
        result = native_smoke(inputs, family, expanded, cached["errors"], index, source)
        name = "native_smoke.json"
    else:
        references = freeze_references(args, family, source, null_paths)
        records = evaluate(args, inputs, references, family, index, source)
        summary = source.v78.v77.summarize(records, (*CONTROLS, *CONDITIONS))
        for group in scores["summary"]:
            for condition in CONTROLS:
                assert summary[group][condition] == scores["summary"][group][condition]
        result = dict(revision=REVISION, source_revision=v93.REVISION, spec=source.v76.SPEC,
            records=records, summary=summary, reference_metadata={c: r.metadata() for c, r in references.items()},
            normal_null_transitions_per_condition=12000, normal_thresholds_recalibrated=True,
            all_five_reused_control_summaries_match_v90=True, all_paths_development_replays=True,
            decision="development-suffix-uncertainty-replay-not-adoption")
        name = "suffix_replay.json"
    result.update(conditions=list(CONDITIONS), controls_reused=list(CONTROLS), raw_weight=weight,
        evaluation_paths_available=65, mean_fitting_rows=25200, raw_reference_rows=23000, error_reference_rows=25200,
        members=6, full_bank_error_inner_fits=fits, direct_error_rows_checked=checked_errors,
        error_seed_exclusion_includes_original_and_added_rows=True, query_targets_used_for_geometry=False,
        conditional_means_changed_vs_v89=False, disagreement_coefficient=1.,
        paired_geometry_shared=True, forward_models_refitted=False, scaling_refitted=False, weight_refitted=False,
        temporal_rules_changed=False, environment_steps=0, optimizer_updates=0, additional_counterfactual_steps=0,
        readout_controls_actor=False, independent_confirmation=False,
        limitations=["Previously inspected development paths and reference folds, one source training seed and known mild modes; no independent confirmation.",
            "The full-bank replay has six source-deletion members rather than five after reference outer exclusion.",
            "Equal traces do not equalize determinants or precision matrices; the rescaling control tests additional directional shape.",
            "Deletion sensitivity is not a calibrated variance; the fixed added second moment can double-count error.",
            "Offline recursive readout evidence is not actor adaptation or RL return evidence."])
    (args.result_dir/name).write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] {REVISION}: {args.result_dir/name}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("v93-dir", "v90-dir", "v83-dir", "v76-dir", "v74-dir", "v73-dir", "v72-dir", "v71-dir",
                 "v69-dir", "v70-dir", "v61-dir", "v62-dir", "checkpoint-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
