"""Full-history age-balanced means with the incumbent covariance geometry."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import Covariance, multivariate_normal
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_general_ood_v102_failure_attribution as v102

v101, v98 = v102.v101, v102.v98
REVISION = "general_ood_v103_frozen_covariance"
RECIPE = "age_mean_incumbent_covariance"


class FrozenCovarianceDensity:
    def __init__(self, incumbent, candidate):
        from scripts.v83_cross_seed_covariance import LocalLinearMean

        self.incumbent = incumbent
        self.bank, self.mean = incumbent.bank, candidate.mean
        self.groups, self.raw_weight = incumbent.groups, incumbent.raw_weight
        self.recipe = RECIPE
        self.correction = LocalLinearMean(self.mean)
        np.testing.assert_array_equal(self.groups, candidate.groups)
        for field in ("input_mean", "input_scale"):
            np.testing.assert_array_equal(getattr(self.bank.features, field), getattr(candidate.bank.features, field))
        np.testing.assert_array_equal(self.bank.target_scale, candidate.bank.target_scale)
        for mode in v98.v94.MODES:
            np.testing.assert_array_equal(self.bank.coefficients[mode], candidate.bank.coefficients[mode])

    def geometry(self, inputs):
        # Keep the displacement moment centered on the incumbent mean, not the new mean.
        _, members, covariance, diagnostic = self.incumbent.geometry(inputs)
        prediction = self.correction.predict(inputs)
        fixed = np.stack([prediction[m] for m in v98.v94.MODES], axis=1)
        return fixed, members, covariance, diagnostic

    def terms(self, trace):
        fixed, _, covariance, diagnostic = self.geometry(self.bank.features.inputs(trace))
        return v98.v94.terms_for_covariances(trace, self.bank, fixed, covariance), diagnostic

    def metadata(self, condition):
        result = self.incumbent.metadata(condition)
        result.update(reference_revision=REVISION, recipe=RECIPE,
            conditional_mean="v101 age-balanced local-linear mean",
            mean_fitting_rows=self.mean.tree.n, mean_reference_revision=v101.REVISION,
            covariance_reference_revision=v98.REVISION, covariance_reference_recipe="diverse_recipe",
            displacement_moment_center="incumbent v98 mean; not recentered on the new mean",
            full_error_bank_recomputed=False, complete_covariance_geometry_unchanged=True)
        return result


def physical_checks(query, family, fixed, covariance):
    terms = v98.v94.terms_for_covariances(query, family.bank, fixed, covariance)
    targets, design, scale = family.bank.targets(query), family.bank.design(query), family.bank.target_scale
    checked = 0
    for c in v98.v94.CONDITIONS:
        for i, mode in enumerate(v98.v94.MODES):
            errors = (targets-design@family.bank.coefficients[mode]-fixed[:, i])*scale
            for row in (0, len(errors)//2, len(errors)-1):
                physical = covariance[c][row, i]*scale[:, None]*scale[None, :]
                expected = multivariate_normal.logpdf(errors[row],
                    cov=Covariance.from_cholesky(np.linalg.cholesky(physical)))
                np.testing.assert_allclose(terms[c]["likelihoods"][row, i], expected, rtol=1e-10, atol=1e-8)
                checked += 1
    return terms, checked


def smoke(args, family, incumbent, data, source):
    previous = json.loads((args.v102_dir/"run_results/formal/failure_attribution.json").read_text())
    assert previous["passed"] and not previous["smoke"] and previous["checked_windows"] == 12
    paths = {v102.v99.record_key(c): p for c, p, _ in v98.evaluation_inputs(args, source)}
    ranges = {w: (a, b) for w, a, b in source.v78.v77.windows()}
    checks, gaussians, rows = 0, 0, 0
    arm = "age_balanced_mean__incumbent_suffix_covariance"
    for record in previous["records"]:
        trace = source.v76.v68.load_trace(paths[v102.v99.record_key(record)])
        a, b = ranges[record["window"]]
        query = {k: trace[k][a:b] for k in (*v98.v97.KEYS, "oracle_observations")}
        inputs = family.bank.features.inputs(query)
        fixed, members, cov, _ = family.geometry(inputs)
        old_fixed, old_members, old_cov, _ = incumbent.geometry(inputs)
        np.testing.assert_array_equal(members, old_members)
        direct = family.correction.predict(inputs)
        np.testing.assert_allclose(fixed, np.stack([direct[m] for m in v98.v94.MODES], axis=1), rtol=1e-10, atol=1e-10)
        for c in v98.v94.CONDITIONS:
            np.testing.assert_array_equal(cov[c], old_cov[c])
        np.testing.assert_allclose(cov[v98.v94.CONDITIONS[1]], cov[v98.v94.CONDITIONS[0]]
            +v98.v94.v93.displacement_second_moment(old_fixed, old_members), rtol=1e-10, atol=1e-10)
        terms, n = physical_checks(query, family, fixed, cov)
        gaussians += n
        for c in v98.v94.CONDITIONS:
            actual = source.v78.v77.attribution(query, terms[c], family.bank, record["mode"], 0, b-a)
            v98.v94.v90.verify_baseline(actual, record["crossed_scores"][arm][c])
            checks += 1
        rows += b-a
    assert (checks, rows) == (36, 656)
    assert [v102.v99.record_key(c) for c, _, _ in v98.evaluation_inputs(args, source)] == [
        v102.v99.record_key(r) for r in data["records"]]
    return dict(passed=True, checked_windows=12, checked_transitions=rows,
        complete_covariance_comparisons=checks, v102_crossed_density_comparisons=checks,
        physical_gaussian_checks=gaussians, evaluation_paths_available=95,
        normal_thresholds_recalibrated=False, partial_checks_only=True)


def run(args):
    sys.path.insert(0, str(args.v83_dir))
    from scripts import analyze_general_ood_v83_cross_seed_covariance as source

    torch.set_num_threads(2)
    args.recipe = RECIPE
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    incumbent_data = json.loads((args.v98_dir/"run_results/replay_diverse_recipe/regime_replay.json").read_text())
    candidate_data = json.loads((args.v101_dir/"run_results/replay/regime_replay.json").read_text())
    assert all(d["passed"] and d["evaluation_paths"] == 95 for d in (incumbent_data, candidate_data))
    assert [v102.v99.record_key(r) for r in incumbent_data["records"]] == [
        v102.v99.record_key(r) for r in candidate_data["records"]]
    original, _, _ = v98.v95.load_readouts(args, source)
    reference_args = argparse.Namespace(**{k: v for k, v in vars(args).items()
        if k not in ("recipe", "checkpoint_dir")})
    families, _ = v102.load_families(reference_args, original, source)
    family = FrozenCovarianceDensity(families["incumbent_suffix"], families["age_balanced"])
    if args.stage == "smoke":
        result = smoke(args, family, families["incumbent_suffix"], incumbent_data, source)
    else:
        references = v98.phases(args, family, source)
        metadata = {c: r.metadata() for c, r in references.items()}
        result = dict(passed=True, reference_metadata=metadata, normal_thresholds_recalibrated=True,
            normal_null_transitions_per_condition=12000,
            calibration_protocol_matches_incumbent=v101.verify_calibration_protocol(metadata, incumbent_data))
        if args.stage == "replay":
            result.update(v98.replay(args, family, references, source))
    result.update(revision=REVISION, stage=args.stage, recipe=RECIPE,
        cache_implementation_revision=v98.REVISION, raw_rows=family.bank.tree.n,
        mean_rows=family.mean.tree.n, source_seeds=8, additional_reference_inputs=4400,
        raw_weight=family.raw_weight, disagreement_coefficient=1., conditions=list(v98.v94.CONDITIONS),
        inner_fits=incumbent_data["inner_fits"], error_bank_refitted=False,
        covariance_reference_recipe="v98_diverse_recipe", mean_reference_recipe="v101_age_balanced",
        complete_covariance_geometry_unchanged=True, covariance_moment_recentered=False,
        geometry_diagnostics_source="incumbent covariance reference, not candidate mean support",
        forward_models_refitted=False, scaling_refitted=False, coefficient_tuned=False,
        temporal_rules_changed=False, environment_steps=0, optimizer_updates=0,
        readout_controls_actor=False, readout_adopted=False, independent_confirmation=False,
        decision="development-full-history-hybrid-replay-not-actor-adoption",
        limitations=["Known simulator modes, one source policy and inspected development trajectories.",
            "Incumbent covariance is a frozen intervention, not a refitted error model for the new mean.",
            "Normal calibration is reused in population and procedure; mean risk is not a false-positive rate.",
            "No new independent confirmation, actor feedback or RL-return evidence."])
    name = dict(smoke="native_smoke.json", calibrate="normal_calibration.json", replay="regime_replay.json")[args.stage]
    path = args.result_dir/name
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] {REVISION} {args.stage}: {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("v102-dir", "v101-dir", "v100-dir", "v98-dir", "v97-dir", "v95-dir", "v94-dir",
                 "v83-dir", "v78-dir", "v76-dir", "v74-dir", "v73-dir", "v72-dir", "v71-dir",
                 "v69-dir", "v70-dir", "v61-dir", "v62-dir", "checkpoint-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--stage", choices=("smoke", "calibrate", "replay"), required=True)
    parser.add_argument("--resume-from", type=Path)
    run(parser.parse_args())
