"""Expanded standard means with complete incumbent covariance and full history."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_general_ood_v103_frozen_covariance as v103
import analyze_general_ood_v105_behavior_support as v105

v98 = v105.v98
REVISION = "general_ood_v106_standard_mean_replay"
RECIPE = "standard_mean_incumbent_covariance"


class ExpandedMeanDensity(v103.FrozenCovarianceDensity):
    def __init__(self, incumbent, mean, mean_groups):
        from scripts.v83_cross_seed_covariance import LocalLinearMean

        self.incumbent = incumbent
        self.bank, self.mean, self.raw_weight = incumbent.bank, mean, incumbent.raw_weight
        # Cache groups describe the frozen covariance; mean_groups cover the larger mean bank.
        self.groups, self.mean_groups = incumbent.groups, mean_groups
        self.recipe, self.correction = RECIPE, LocalLinearMean(mean)
        assert len(mean_groups) == mean.tree.n and mean.tree.n > incumbent.mean.tree.n
        np.testing.assert_array_equal(mean_groups[:len(self.groups)], self.groups)
        for field in ("input_mean", "input_scale"):
            np.testing.assert_array_equal(getattr(self.bank.features, field), getattr(mean.features, field))
        np.testing.assert_array_equal(self.bank.target_scale, mean.target_scale)
        for mode in v98.v94.MODES:
            np.testing.assert_array_equal(self.bank.coefficients[mode], mean.coefficients[mode])

    def metadata(self, condition):
        result = self.incumbent.metadata(condition)
        result.update(reference_revision=REVISION, recipe=RECIPE,
            conditional_mean="v104 standard reference expansion of the v98 local-linear mean",
            mean_fitting_rows=self.mean.tree.n, mean_reference_revision=v105.REVISION,
            mean_source_seeds=len(np.unique(self.mean_groups)),
            covariance_source_seeds=len(np.unique(self.groups)),
            covariance_mean_reference_rows=self.incumbent.mean.tree.n,
            covariance_reference_revision=v98.REVISION, covariance_reference_recipe="diverse_recipe",
            displacement_moment_center="incumbent v98 mean; not recentered on the expanded mean",
            full_error_bank_recomputed=False, complete_covariance_geometry_unchanged=True)
        return result


def verify_calibration_protocol(metadata, incumbent):
    for c in v98.v94.CONDITIONS:
        a, b = metadata[c], incumbent["reference_metadata"][c]
        for field in ("window_steps", "switch_hazard", "transition_prior", "actual_change_probability",
                      "mode_weights", "initial_prior", "history_reset", "normal_null_calibration_transitions", "settings"):
            assert a[field] == b[field], (c, field)
        assert a["phase"].keys() == b["phase"].keys() and a["phase"]["selection"] == b["phase"]["selection"]
        for phase in ("initial", "after_reset"):
            for field in ("ages", "counts", "normal_start_count", "settings"):
                assert a["phase"][phase][field] == b["phase"][phase][field], (c, phase, field)
        for field in ("modes", "raw_reference_rows", "error_reference_rows", "neighbors", "ridge_alpha",
                      "members", "disagreement_coefficient", "raw_weight"):
            assert a["bank"][field] == b["bank"][field], (c, field)
        assert (a["bank"]["mean_fitting_rows"], b["bank"]["mean_fitting_rows"]) == (34000, 29600)
        assert a["bank"]["covariance_mean_reference_rows"] == 29600
    return True


def load_family(args, source):
    original, _, _ = v98.v95.load_readouts(args, source)
    payload, added_groups, _ = v98.load_extensions(SimpleNamespace(v97_dir=args.v97_dir, recipe="diverse_recipe"))
    raw, mean, groups = v98.expand(original, payload, added_groups, "diverse_recipe")
    old_args = SimpleNamespace(recipe="diverse_recipe", checkpoint_dir=args.v98_dir/"run_checkpoints/diverse_recipe")
    assert all((old_args.checkpoint_dir/f"errors_seed{s}.pt").is_file() for s in np.unique(groups))
    errors, _ = v98.freeze_errors(old_args, mean, groups, source)
    incumbent = v98.CoverageDensity(raw, mean, groups, errors, original.raw_weight)
    incumbent.recipe = "diverse_recipe"
    payloads, added_groups, sources = v105.load_extensions(args.v104_dir)
    means, mean_groups = v105.extend_means(mean, payloads, groups, added_groups)
    family = ExpandedMeanDensity(incumbent, means["standard"], mean_groups["standard"])
    assert (raw.tree.n, mean.tree.n, family.mean.tree.n) == (27400, 29600, 34000)
    assert (len(np.unique(family.groups)), len(np.unique(family.mean_groups))) == (8, 10)
    return family, sources["standard"]


def smoke(args, family, pilot, incumbent_data, source):
    paths = {v105.key(c): p for c, p, _ in v98.evaluation_inputs(args, source)}
    selected = v105.select_windows(incumbent_data["records"], True)
    previous = {(v105.key(r), r["window"]): r for r in pilot["records"]}
    ranges = {w: (a, b) for w, a, b in source.v78.v77.windows()}
    checks, gaussians, rows = 0, 0, 0
    for cell, window in selected:
        trace = source.v76.v68.load_trace(paths[v105.key(cell)])
        a, b = ranges[window]
        query = {k: trace[k][a:b] for k in v105.v104.KEYS}
        inputs = family.bank.features.inputs(query)
        fixed, members, cov, _ = family.geometry(inputs)
        old_fixed, old_members, old_cov, _ = family.incumbent.geometry(inputs)
        np.testing.assert_array_equal(members, old_members)
        for c in v98.v94.CONDITIONS:
            np.testing.assert_array_equal(cov[c], old_cov[c])
        np.testing.assert_allclose(cov[v98.v94.CONDITIONS[1]], cov[v98.v94.CONDITIONS[0]]+
            v98.v94.v93.displacement_second_moment(old_fixed, old_members), rtol=1e-10, atol=1e-10)
        terms, n = v103.physical_checks(query, family, fixed, cov)
        gaussians += n
        for c in v98.v94.CONDITIONS:
            actual = source.v78.v77.attribution(query, terms[c], family.bank, cell["mode"], 0, b-a)
            np.testing.assert_allclose(actual["mean_true_prediction_error_rms"],
                previous[v105.key(cell), window]["diagnostics"]["standard"]["prediction_error_rms"], rtol=1e-9, atol=1e-8)
            checks += 1
        rows += b-a
    assert (checks, rows) == (12, 228)
    return dict(passed=True, checked_windows=4, checked_transitions=rows,
        complete_covariance_comparisons=checks, v105_prediction_rms_comparisons=checks,
        physical_gaussian_checks=gaussians, partial_checks_only=True)


def verify_replay_means(result, pilot):
    index = {v105.key(r): r for r in result["records"]}
    checked = 0
    for row in pilot["records"]:
        actual = index[v105.key(row)]
        for c in v98.v94.CONDITIONS:
            np.testing.assert_allclose(actual["attribution"][c][row["window"]]["mean_true_prediction_error_rms"],
                row["diagnostics"]["standard"]["prediction_error_rms"], rtol=1e-9, atol=1e-8)
            checked += 1
    assert checked == 186
    return checked


def run(args):
    sys.path.insert(0, str(args.v83_dir))
    from scripts import analyze_general_ood_v83_cross_seed_covariance as source

    torch.set_num_threads(2)
    args.recipe = RECIPE
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    pilot = json.loads((args.v105_dir/"run_results/formal/behavior_support.json").read_text())
    assert pilot["revision"] == v105.REVISION and pilot["passed"] and not pilot["smoke"]
    assert (pilot["checked_windows"], pilot["checked_transitions"]) == (62, 3548)
    data = json.loads((args.v98_dir/"run_results/replay_diverse_recipe/regime_replay.json").read_text())
    assert data["passed"] and data["revision"] == v98.REVISION and data["evaluation_paths"] == 95
    family, sources = load_family(args, source)
    assert sources == pilot["reference_sources"]["standard"]
    if args.stage == "smoke":
        result = smoke(args, family, pilot, data, source)
    else:
        references = v98.phases(args, family, source)
        metadata = {c: r.metadata() for c, r in references.items()}
        result = dict(passed=True, reference_metadata=metadata, normal_thresholds_recalibrated=True,
            normal_null_transitions_per_condition=12000,
            calibration_protocol_matches_incumbent=verify_calibration_protocol(metadata, data))
        if args.stage == "replay":
            result.update(v98.replay(args, family, references, source))
            assert [v105.key(r) for r in result["records"]] == [v105.key(r) for r in data["records"]]
            for actual, previous in zip(result["records"], data["records"]):
                for c in v98.v94.CONDITIONS:
                    for field in ("observers", "attribution", "change_minus_nochange"):
                        assert actual[field]["v94_unchanged__"+c] == previous[field]["v94_unchanged__"+c]
            result["v105_prediction_rms_comparisons"] = verify_replay_means(result, pilot)
            result["ordered_paths_and_unchanged_controls_verified"] = True
    result.update(revision=REVISION, stage=args.stage, recipe=RECIPE,
        cache_implementation_revision=v98.REVISION, mean_reference_revision=v105.REVISION,
        raw_rows=family.bank.tree.n, mean_rows=family.mean.tree.n,
        covariance_mean_reference_rows=family.incumbent.mean.tree.n,
        covariance_source_seeds=8, mean_source_seeds=10, reference_sources=sources,
        extension_union_source_index_rows=8800, incumbent_extension_source_index_rows=4400,
        baseline_reference_budget_matched=False, raw_weight=family.raw_weight,
        complete_covariance_geometry_unchanged=True, covariance_moment_recentered=False,
        error_bank_refitted=False, forward_models_refitted=False, scaling_refitted=False,
        temporal_rules_changed=False, coefficient_tuned=False, environment_steps=0, optimizer_updates=0,
        readout_controls_actor=False, readout_adopted=False, independent_confirmation=False,
        decision="development-expanded-standard-mean-replay-not-actor-adoption",
        limitations=["Known-mode simulator labels, one source policy and inspected development paths.",
            "Expanded means use additional reference data; the incumbent is not a budget-matched baseline.",
            "The frozen covariance is an intervention, not an error refit for the expanded mean.",
            "Recognition and normal-risk readouts alone do not establish RL-return gains."])
    name = dict(smoke="native_smoke.json", calibrate="normal_calibration.json", replay="regime_replay.json")[args.stage]
    path = args.result_dir/name
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] {REVISION} {args.stage}: {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("v105-dir", "v104-dir", "v98-dir", "v97-dir", "v95-dir", "v94-dir", "v83-dir", "v78-dir",
                 "v76-dir", "v74-dir", "v73-dir", "v72-dir", "v71-dir", "v69-dir", "v70-dir", "v61-dir",
                 "v62-dir", "checkpoint-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--stage", choices=("smoke", "calibrate", "replay"), required=True)
    parser.add_argument("--resume-from", type=Path)
    run(parser.parse_args())
