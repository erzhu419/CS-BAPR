"""Source-excluded errors, matched calibration and full age-balanced replay."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_general_ood_v100_reference_age as v100

v98 = v100.v98
REVISION = "general_ood_v101_age_replay"
RECIPE = "age_balanced"


class AgeBalancedDensity(v98.CoverageDensity):
    def metadata(self, condition):
        result = super().metadata(condition)
        result.update(reference_revision=REVISION, reference_payload_revision=v100.REVISION,
            conditional_mean="local-linear v94 support plus v100 age-balanced references")
        return result


def load_extensions(args, source):
    traces = {seed: {mode: source.v76.v68.load_trace(source.v76.trace_path(
        args.v97_dir/f"run_checkpoints/seed{seed}/traces",
        next(c for c in v98.v97.cells(seed) if c["mode"] == mode)))
        for mode in ("normal", "high_grav", "compound")} for seed in v98.v97.SEEDS}
    payloads, groups, sources = v100.extensions(traces, v98.v97.SPEC)
    pilot = json.loads((args.v100_dir/"run_results/formal/reference_age.json").read_text())
    assert pilot["revision"] == v100.REVISION and pilot["passed"] and not pilot["smoke"]
    assert (pilot["checked_windows"], pilot["checked_transitions"]) == (62, 3548)
    assert sources[RECIPE] == pilot["reference_sources"][RECIPE]
    return payloads[RECIPE], groups[RECIPE], sources[RECIPE], pilot


def verify_pilot_means(args, raw, mean, pilot, source):
    from scripts.v83_cross_seed_covariance import LocalLinearMean

    paths = {source.v78.v77.cell_key(c): p for c, p, _ in v98.evaluation_inputs(args, source)}
    start, stop = next((a, b) for w, a, b in source.v78.v77.windows() if w == "first_post50")
    selected = [r for r in pilot["records"] if r["seed"] == v98.v95.SEEDS[0]
        and r["mode"] in ("slippery", "high_grav") and r["window"] == "first_post50"]
    assert len(selected) == 4
    for r in selected:
        trace = source.v76.v68.load_trace(paths[source.v78.v77.cell_key(r)])
        query = {k: trace[k][start:stop] for k in v98.v97.KEYS}
        prediction = LocalLinearMean(mean).predict(raw.features.inputs(query))[r["mode"]]
        error = raw.design(query)@raw.coefficients[r["mode"]]+prediction-raw.targets(query)
        actual = float(np.sqrt(np.mean(error**2, axis=1)).mean())
        np.testing.assert_allclose(actual, r["diagnostics"][RECIPE]["prediction_error_rms"], rtol=1e-9, atol=1e-8)
    return len(selected)


def verify_calibration_protocol(metadata, incumbent):
    for c in v98.v94.CONDITIONS:
        a, b = metadata[c], incumbent["reference_metadata"][c]
        for field in ("window_steps", "switch_hazard", "transition_prior", "actual_change_probability",
                      "mode_weights", "initial_prior", "history_reset", "normal_null_calibration_transitions", "settings"):
            assert a[field] == b[field], (c, field)
        assert a["phase"].keys() == b["phase"].keys()
        assert a["phase"]["selection"] == b["phase"]["selection"]
        for phase in ("initial", "after_reset"):
            for field in ("ages", "counts", "normal_start_count", "settings"):
                assert a["phase"][phase][field] == b["phase"][phase][field], (c, phase, field)
        for field in ("modes", "mean_fitting_rows", "raw_reference_rows", "error_reference_rows",
                      "neighbors", "ridge_alpha", "members", "disagreement_coefficient", "raw_weight"):
            assert a["bank"][field] == b["bank"][field], (c, field)
    return True


def run(args):
    sys.path.insert(0, str(args.v83_dir))
    from scripts import analyze_general_ood_v83_cross_seed_covariance as source

    torch.set_num_threads(2)
    args.recipe = RECIPE
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    incumbent = json.loads((args.v98_dir/"run_results/replay_diverse_recipe/regime_replay.json").read_text())
    assert incumbent["revision"] == v98.REVISION and incumbent["recipe"] == "diverse_recipe"
    assert incumbent["passed"] and incumbent["stage"] == "replay"
    assert (incumbent["evaluation_paths"], incumbent["evaluated_transitions"]) == (95, 570000)
    original, _, _ = v98.v95.load_readouts(args, source)
    payload, added_groups, sources, pilot = load_extensions(args, source)
    raw, mean, groups = v98.expand(original, payload, added_groups, RECIPE)
    assert (raw.tree.n, mean.tree.n, len(np.unique(groups))) == (27400, 29600, 8)
    assert incumbent["raw_weight"] == original.raw_weight
    if args.stage == "smoke":
        result = v98.smoke(args, original, raw, mean, groups, source)
        result["v100_mean_comparisons"] = verify_pilot_means(args, raw, mean, pilot, source)
    else:
        errors, fits = v98.freeze_errors(args, mean, groups, source)
        family = AgeBalancedDensity(raw, mean, groups, errors, original.raw_weight)
        family.recipe = RECIPE
        checked = v98.v94.v92.verify_error_rows(mean, groups, errors, raw.tree.n)
        references = v98.phases(args, family, source)
        metadata = {c: r.metadata() for c, r in references.items()}
        matched = verify_calibration_protocol(metadata, incumbent)
        result = dict(passed=True, full_error_bank_fitted=True, inner_fits=fits,
            direct_error_rows_checked=checked, normal_thresholds_recalibrated=True,
            normal_null_transitions_per_condition=12000, calibration_protocol_matches_incumbent=matched,
            reference_metadata=metadata, **v98.gaussian_check(family, source))
        if args.stage == "replay":
            result.update(v98.replay(args, family, references, source))
    result.update(revision=REVISION, source_revision=v100.REVISION, cache_implementation_revision=v98.REVISION,
        stage=args.stage, recipe=RECIPE, reference_sources=sources,
        raw_rows=raw.tree.n, mean_rows=mean.tree.n, source_seeds=8, additional_reference_inputs=4400,
        raw_weight=original.raw_weight, disagreement_coefficient=1., conditions=list(v98.v94.CONDITIONS),
        incumbent_revision=v98.REVISION, incumbent_recipe="diverse_recipe", incumbent_replay_reused=True,
        forward_models_refitted=False, scaling_refitted=False, weight_refitted=False,
        coefficient_tuned=False, temporal_rules_changed=False, environment_steps=0, optimizer_updates=0,
        readout_controls_actor=False, readout_adopted=False, independent_confirmation=False,
        decision="development-age-balanced-replay-not-actor-adoption",
        limitations=["Known-mode simulator labels and one source policy; all evaluation paths are development evidence.",
            "Reused v98 suffix comparator has identical budgets and calibration inputs, not a new independent replication.",
            "Normal-window mean risk is not a false-positive rate; no RL-return or actor adaptation evidence."])
    name = dict(smoke="native_smoke.json", fit="reference_fit.json", replay="regime_replay.json")[args.stage]
    path = args.result_dir/name
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] {REVISION} {args.stage}: {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("v100-dir", "v98-dir", "v97-dir", "v95-dir", "v94-dir", "v83-dir", "v78-dir",
                 "v76-dir", "v74-dir", "v73-dir", "v72-dir", "v71-dir", "v69-dir", "v70-dir",
                 "v61-dir", "v62-dir", "checkpoint-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--stage", choices=("smoke", "fit", "replay"), required=True)
    parser.add_argument("--resume-from", type=Path)
    run(parser.parse_args())
