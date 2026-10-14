"""Mean/support pilot of matched standard/exploratory reference collections."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_general_ood_v98_regime_replay as v98
import probe_general_ood_v104_behavior_exploration as v104

REVISION = "general_ood_v105_behavior_support"
ARMS = ("incumbent", "standard", "exploratory")
FAILURES = ((135191, "slippery", "first_post50"), (135192, "slippery", "first_post50"),
            (135192, "slippery", "last64"), (135192, "high_grav", "first_post50"),
            (135192, "compound", "first_post50"), (135192, "compound", "last64"))


def key(record):
    return tuple(record[k] for k in ("group", "seed", "mode", "updating"))


def select_windows(records, smoke=False):
    population = [(r, w) for r in records if r["group"].startswith("v95_development_")
                  for w in ("first_post50", "last64")]
    population.extend((r, "last64") for r in records if r["group"] == "v62_development_replay"
                      and r["seed"] == 134020 and r["updating"] and r["mode"] in ("normal", "slippery"))
    assert len(population) == 62 and len({(key(r), w) for r, w in population}) == 62
    failures = {(r["seed"], r["mode"], w) for r, w in population if r["updating"]
                and (r["seed"], r["mode"], w) in FAILURES}
    assert failures == set(FAILURES)
    if smoke:
        wanted = {("v95_development_updating", 135191, "slippery", "first_post50"),
                  ("v95_development_updating", 135192, "compound", "last64"),
                  ("v95_development_frozen", 135192, "normal", "first_post50"),
                  ("v62_development_replay", 134020, "slippery", "last64")}
        population = [(r, w) for r, w in population if (r["group"], r["seed"], r["mode"], w) in wanted]
        assert len(population) == 4
    return population


def load_extensions(directory):
    combined = {b: [] for b in v104.BEHAVIORS}
    groups, sources = [], {b: [] for b in v104.BEHAVIORS}
    for seed in v104.SEEDS:
        bundle = torch.load(directory/f"run_checkpoints/seed{seed}/reference_inputs.pt",
                            map_location="cpu", weights_only=False)
        scalar = json.loads((directory/f"run_results/seed{seed}/reference_collection.json").read_text())
        assert bundle["revision"] == scalar["revision"] == v104.REVISION
        assert bundle["seed"] == scalar["seed"] == seed and bundle["spec"] == scalar["spec"] == v104.SPEC
        assert scalar["passed"] and not scalar["smoke"] and scalar["paired_prefixes"]
        assert scalar["equal_behavior_budgets"] and scalar["exact_true_mode_replays_verified"]
        assert scalar["actual_actions_enter_replay_and_context"]
        assert not scalar["evaluation_paths_used_for_reference"] and not scalar["counterfactuals_added_to_RL_replay"]
        assert scalar["per_behavior_reference_inputs"] == {b: 2200 for b in v104.BEHAVIORS}
        assert scalar["per_behavior_budgets"]["standard"] == scalar["per_behavior_budgets"]["exploratory"]
        for behavior in v104.BEHAVIORS:
            data = bundle["extensions"][behavior]
            for mode in v98.v94.MODES:
                for field, width in (("states", 11), ("executed_actions", 3), ("next_states", 11)):
                    assert data[mode][field].shape == (2200, width) and np.isfinite(data[mode][field]).all()
                    if field != "next_states":
                        np.testing.assert_array_equal(data[mode][field], data["normal"][field])
            np.testing.assert_array_equal(bundle["groups"][behavior], np.full(2200, seed))
            combined[behavior].append(data)
            sources[behavior].extend(bundle["sources"][behavior])
        groups.append(bundle["groups"]["standard"])
    payload = {b: {m: {k: np.concatenate([data[m][k] for data in combined[b]]) for k in v104.KEYS}
                   for m in v98.v94.MODES} for b in v104.BEHAVIORS}
    return payload, np.concatenate(groups), sources


def extend_means(incumbent, payloads, old_groups, added_groups):
    from scripts.v74_density_coverage import DensityCoverageBank

    assert not set(added_groups) & set(old_groups)
    means, groups = dict(incumbent=incumbent), dict(incumbent=old_groups)
    for behavior in v104.BEHAVIORS:
        means[behavior] = DensityCoverageBank.extend(incumbent, payloads[behavior], behavior+"_mean_only")
        groups[behavior] = np.concatenate((old_groups, added_groups))
        assert means[behavior].tree.n == len(groups[behavior]) == incumbent.tree.n+len(added_groups)
        np.testing.assert_array_equal(means[behavior].tree.data[:incumbent.tree.n], incumbent.tree.data)
        for mode in v98.v94.MODES:
            np.testing.assert_array_equal(means[behavior].residuals[mode][:incumbent.tree.n], incumbent.residuals[mode])
            np.testing.assert_array_equal(means[behavior].coefficients[mode], incumbent.coefficients[mode])
    return means, groups


def diagnose(query, raw, mean, groups, mode, old_mean_rows, worker, verify_production=False):
    inputs = raw.features.inputs(query)
    prediction = worker.local_prediction(mean, inputs)
    if verify_production:
        from scripts.v83_cross_seed_covariance import LocalLinearMean
        direct = LocalLinearMean(mean).predict(inputs)
        for m in v98.v94.MODES:
            np.testing.assert_allclose(prediction["linear"][m], direct[m], rtol=1e-10, atol=1e-10)
    truth = v98.v94.MODES.index(mode)
    error = raw.design(query)@raw.coefficients[mode]+prediction["linear"][mode]-raw.targets(query)
    scaled = (inputs-mean.features.input_mean)/mean.features.input_scale
    _, neighbors = mean.tree.query(scaled, k=64)
    result = dict(prediction_error_rms=float(np.sqrt(np.mean(error**2, axis=1)).mean()),
        support={k: worker.statistics(v) for k, v in prediction["support"].items()},
        new_reference_neighbor_fraction=float(np.mean(neighbors >= old_mean_rows)),
        neighbor_seed_fractions={str(int(s)): float(np.mean(groups[neighbors] == s)) for s in np.unique(groups)},
        local_reference_loo_rms=worker.statistics(prediction["loo_rms"][:, truth]),
        coordinate_error_rms=np.sqrt(np.mean(error**2, axis=0)).tolist())
    return result, prediction["linear"]


def summarize(records):
    subsets = dict(
        six_old_failures=[r for r in records if r["old_coverage_failure"]],
        v95_updating_changed=[r for r in records if r["group"] == "v95_development_updating" and r["mode"] != "normal"],
        v95_frozen_changed=[r for r in records if r["group"] == "v95_development_frozen" and r["mode"] != "normal"],
        v95_normal=[r for r in records if r["group"].startswith("v95_") and r["mode"] == "normal"],
        v62_slippery=[r for r in records if r["group"] == "v62_development_replay" and r["mode"] == "slippery"],
        v62_normal=[r for r in records if r["group"] == "v62_development_replay" and r["mode"] == "normal"])
    output = {}
    for name, rows in subsets.items():
        if not rows:
            continue
        arms = {arm: dict(
            prediction_error_rms=float(np.mean([r["diagnostics"][arm]["prediction_error_rms"] for r in rows])),
            nearest_radius=float(np.mean([r["diagnostics"][arm]["support"]["nearest_radius"]["mean"] for r in rows])),
            radius64=float(np.mean([r["diagnostics"][arm]["support"]["radius64"]["mean"] for r in rows])),
            new_reference_neighbor_fraction=float(np.mean([r["diagnostics"][arm]["new_reference_neighbor_fraction"] for r in rows])))
            for arm in ARMS}
        comparisons = {}
        for left, right in (("standard", "incumbent"), ("exploratory", "incumbent"), ("exploratory", "standard")):
            a, b = arms[left]["prediction_error_rms"], arms[right]["prediction_error_rms"]
            comparisons[left+"_vs_"+right] = dict(prediction_rms_change_percent=100*(a/b-1),
                improved_windows=sum(r["diagnostics"][left]["prediction_error_rms"] <
                                     r["diagnostics"][right]["prediction_error_rms"] for r in rows))
        output[name] = dict(windows=len(rows), arms=arms, comparisons=comparisons)
    return output


def run(args):
    sys.path.insert(0, str(args.v83_dir))
    from scripts import analyze_general_ood_v83_cross_seed_covariance as source
    from scripts import analyze_general_ood_v79_linear_support as worker

    torch.set_num_threads(2)
    data = json.loads((args.v98_dir/"run_results/replay_diverse_recipe/regime_replay.json").read_text())
    assert data["passed"] and data["revision"] == v98.REVISION and data["evaluation_paths"] == 95
    population = select_windows(data["records"], args.smoke)
    original, _, _ = v98.v95.load_readouts(args, source)
    old_payload, old_groups, _ = v98.load_extensions(SimpleNamespace(v97_dir=args.v97_dir, recipe="diverse_recipe"))
    raw, incumbent, incumbent_groups = v98.expand(original, old_payload, old_groups, "diverse_recipe")
    assert (raw.tree.n, incumbent.tree.n) == (27400, 29600)
    payload, added_groups, sources = load_extensions(args.v104_dir)
    assert not set(added_groups) & {r["seed"] for r in data["records"]}
    means, groups = extend_means(incumbent, payload, incumbent_groups, added_groups)
    paths = {key(c): p for c, p, _ in v98.evaluation_inputs(args, source)}
    ranges = {w: (a, b) for w, a, b in source.v78.v77.windows()}
    records = []
    for previous, window in population:
        trace = source.v76.v68.load_trace(paths[key(previous)])
        start, stop = ranges[window]
        query = {k: trace[k][start:stop] for k in v104.KEYS}
        diagnostics = {arm: diagnose(query, raw, mean, groups[arm], previous["mode"],
            incumbent.tree.n, worker, args.smoke)[0] for arm, mean in means.items()}
        for condition in v98.v94.CONDITIONS:
            np.testing.assert_allclose(diagnostics["incumbent"]["prediction_error_rms"],
                previous["attribution"][condition][window]["mean_true_prediction_error_rms"], rtol=1e-9, atol=1e-8)
        records.append(dict(**{k: previous[k] for k in ("group", "seed", "mode", "updating")},
            window=window, transitions=stop-start, diagnostics=diagnostics,
            old_coverage_failure=previous["updating"] and (previous["seed"], previous["mode"], window) in FAILURES))
        print(f"[SUPPORT] {len(records)}/{len(population)} seed{previous['seed']} {previous['mode']} {window} updating={previous['updating']}", flush=True)
    rows = sum(r["transitions"] for r in records)
    assert (len(records), rows) == ((4, 228) if args.smoke else (62, 3548))
    result = dict(revision=REVISION, passed=True, smoke=args.smoke, checked_windows=len(records),
        checked_transitions=rows, arms=list(ARMS), mean_rows={k: m.tree.n for k, m in means.items()},
        raw_rows=raw.tree.n, reference_sources=sources, new_reference_inputs_per_behavior=4400,
        extension_union_source_index_rows=dict(incumbent=4400, standard=8800, exploratory=8800),
        primary_equal_budget_comparison="exploratory_vs_standard", baseline_budget_matched=False,
        incumbent_prediction_rms_comparisons=3*len(records),
        production_prediction_checks=3*len(records) if args.smoke else 0,
        prediction_error_units="v72 target-scale-normalized observation delta; mean pointwise coordinate RMS",
        summary=summarize(records), records=records, environment_steps=0, optimizer_updates=0,
        covariance_computed=False, covariance_reference_modified=False, error_bank_refitted=False,
        forward_models_refitted=False, scaling_refitted=False, normal_thresholds_recalibrated=False,
        posterior_replayed=False, coefficient_tuned=False, readout_controls_actor=False, readout_adopted=False,
        decision="development-mean-support-pilot-not-adoption",
        limitations=["Inspected development windows and known simulator modes from one trained source policy.",
            "Support and local leave-one-out diagnostics do not establish recognition or return improvements.",
            "No covariance intervention, normal recalibration or complete belief replay in this pilot."])
    args.result_dir.mkdir(parents=True, exist_ok=True)
    path = args.result_dir/("native_smoke.json" if args.smoke else "behavior_support.json")
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] {REVISION}: {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("v104-dir", "v98-dir", "v97-dir", "v95-dir", "v94-dir", "v83-dir", "v78-dir", "v76-dir",
                 "v74-dir", "v73-dir", "v72-dir", "v71-dir", "v69-dir", "v70-dir", "v61-dir", "v62-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
