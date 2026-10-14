"""Frozen neural-mean scores, normal-only calibration and causal history replay."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_general_ood_v116_matched_neural_classifier as v116

v115, v113, v98 = v116.v115, v116.v113, v116.v113.v98
MODES, ARMS = v116.MODES, v116.ARMS
REVISION = "general_ood_v117_temporal_classifier_replay"
PRIMARY = "suffix_disagreement_mix"
SMOKE_KEYS = {(135252, m, u) for m in ("normal", "compound", "high_grav") for u in (False, True)}
SMOKE_KEYS |= {(s, m, True) for s, changed in ((135190, "compound"), (135192, "high_grav"))
               for m in ("normal", changed)}


def key(cell):
    return tuple(cell[k] for k in ("group", "seed", "mode", "updating"))


def load_models(args, family):
    means, saved = v116.load_frozen_means(args.v115_dir/"run_checkpoints/formal/mean_models.pt",
                                         len(family.bank.features.input_mean))
    assert saved["final_training"]["source_seeds"] == np.unique(family.mean_groups).astype(int).tolist()
    assert saved["final_training"]["reference_inputs"] == len(family.mean_groups)
    models = {}
    for arm, directory, revision in (("matched_classifier", args.v116_dir, v116.REVISION),
                                     ("original_classifier", args.v113_dir, v113.REVISION)):
        saved = torch.load(directory/"run_checkpoints/formal/classifier_models.pt", map_location="cpu", weights_only=False)
        assert saved["revision"] == revision and not saved["smoke"] and saved["settings"] == v113.SETTINGS
        if arm == "matched_classifier":
            assert saved["mean_source_revision"] == v115.REVISION
        models[arm] = saved["final_model"]
    return means["final_model"], models


def transition_scores(bank, mean, classifiers, trace):
    # Ignore counterfactual observations: only the executed transition is scored.
    query = {k: trace[k] for k in v98.v97.KEYS}
    inputs, design = bank.features.inputs(query), bank.design(query)
    prediction = np.stack([design@bank.coefficients[m] for m in MODES], axis=1)
    prediction += v115.residual_prediction(mean, bank, inputs)
    return v116.score_arms(classifiers["matched_classifier"], classifiers["original_classifier"],
                          bank.targets(query), prediction)


def calibrate_scores(nulls, values):
    from scripts.v73_recursive_mode_readout import RecursiveModeReference

    references = {}
    for arm in ARMS:
        bank = SimpleNamespace(metadata=lambda a=arm: dict(arm=a, mean_revision=v115.REVISION,
            classifier_revision=v116.REVISION if a == "matched_classifier" else v113.REVISION if a == "original_classifier" else None,
            score_units="normalized empirical log scores, not physical transition densities"),
            log_likelihoods=lambda trace, a=arm: next(v[a] for t, v in zip(nulls, values) if t is trace))
        references[arm] = RecursiveModeReference.calibrate(bank, nulls)
    return references


def reference_metadata(references):
    metadata = {a: deepcopy(r.metadata()) for a, r in references.items()}
    for m in metadata.values():
        m["mode_weights"] = "recursive empirical score weights; not a calibrated Bayesian environment posterior"
    return metadata


def verify_temporal_protocol(metadata, incumbent):
    for arm, actual in metadata.items():
        for field in ("window_steps", "switch_hazard", "transition_prior", "actual_change_probability",
                      "initial_prior", "history_reset", "settings", "normal_null_calibration_transitions"):
            assert actual[field] == incumbent[field], (arm, field)
        assert actual["phase"]["selection"] == incumbent["phase"]["selection"]
        for phase in ("initial", "after_reset"):
            for field in ("ages", "counts", "normal_start_count", "settings"):
                assert actual["phase"][phase][field] == incumbent["phase"][phase][field], (arm, phase, field)


def evaluation_population(args, source, incumbent):
    import probe_general_ood_v110_fresh_mixture as v110

    index = {key(r): r for r in incumbent["records"]}
    population = [(c, p, {a: index[key(c)]["observers"][a] for a in v98.v94.CONDITIONS})
                  for c, p, _ in v98.evaluation_inputs(args, source)]
    for seed in v110.SEEDS:
        previous = json.loads((args.v110_dir/f"run_results/seed{seed}/fresh_seed.json").read_text())
        assert previous["revision"] == v110.REVISION and previous["passed"] and not previous["smoke"]
        assert [key(r) for r in previous["records"]] == [key(c) for c in v110.cells(seed)]
        for c, old in zip(v110.cells(seed), previous["records"]):
            path = source.v76.trace_path(args.v110_dir/f"run_checkpoints/seed{seed}/traces", c)
            population.append((c, path, old["arms"]["old"]))
    assert len(population) == len({key(c) for c, _, _ in population}) == 125
    assert all(p.is_file() for _, p, _ in population)
    if args.smoke:
        population = [(c, p, old) for c, p, old in population if (c["seed"], c["mode"], c["updating"]) in SMOKE_KEYS]
        assert len(population) == 10
    return population


def window_statistics(trace, scores, series, mode, source):
    stats = source.v76.v73.window_statistics(trace, scores, series, mode)
    ranges = source.v76.v73.v68.v66.v65.v64.window_ranges(len(scores), 3800)
    for window, row in stats.items():
        a, b = ranges[window]
        row["instantaneous_mean_true_minus_best_other_score"] = row.pop("instantaneous_mean_true_minus_best_other_log_density")
        row["above_threshold_fraction"] = float(np.mean(series["signal"][a:b] > series["threshold"][a:b]))
    return stats


def verify_pilot(scores, cell, pilot, source):
    ranges = {w: (a, b) for w, a, b in source.v78.v77.windows()}
    checked = []
    for row in pilot:
        if key(row) != key(cell):
            continue
        a, b = ranges[row["window"]]
        for arm in ARMS:
            actual = v113.statistics(scores[arm][a:b], MODES.index(cell["mode"]))
            for field, expected in row["arms"][arm].items():
                if isinstance(expected, dict):
                    np.testing.assert_array_equal([actual[field][m] for m in MODES], [expected[m] for m in MODES])
                else:
                    # Whole-history float32 GEMMs can round differently from 50/64-row pilot GEMMs.
                    np.testing.assert_allclose(actual[field], expected, rtol=1e-5, atol=1e-5)
        checked.append(v116.v114.window_key(row))
    return checked


def summarize(records):
    subsets = dict(all=lambda r: True, updating=lambda r: r["updating"], frozen=lambda r: not r["updating"])
    for group in dict.fromkeys(r["group"] for r in records):
        subsets[group] = lambda r, g=group: r["group"] == g
    output = {}
    for name, select in subsets.items():
        selected = [r for r in records if select(r)]
        normal = [r for r in selected if r["mode"] == "normal"]
        changed = [r for r in selected if r["mode"] != "normal"]
        if not normal or not changed:
            continue
        index = {key(r): r for r in normal}
        arms = {}
        for arm in (*ARMS, "incumbent_v98"):
            def windows(r):
                return r["incumbent"][PRIMARY] if arm == "incumbent_v98" else r["arms"][arm]
            arms[arm] = dict(changed_paths=len(changed), normal_paths=len(normal),
                early_accuracy=float(np.mean([windows(r)["first_post50"]["mode_accuracy"] for r in changed])),
                tail_accuracy=float(np.mean([windows(r)["last64"]["mode_accuracy"] for r in changed])),
                normal_full_accuracy=float(np.mean([windows(r)["full"]["mode_accuracy"] for r in normal])),
                maximum_normal_mean_risk=max(windows(r)[w]["mean_coefficient"] for r in normal for w in ("first800", "full", "last1000")),
                nonpositive_changed_margin_windows=sum(windows(r)[w]["mean_true_minus_best_other_score"] <= 0
                    for r in changed for w in ("first_post50", "last64")),
                positive_departure_contrasts=sum(windows(r)[w]["mean_coefficient"] > windows(
                    index[r["group"], r["seed"], "normal", r["updating"]])[w]["mean_coefficient"]
                    for r in changed for w in ("first_post50", "last64")))
            if arm in ARMS:
                arms[arm].update(normal_full_above_threshold_fraction=float(np.mean([
                    windows(r)["full"]["above_threshold_fraction"] for r in normal])),
                    worst_normal_episode_mean_risk=max(e["mean_risk"] for r in normal for e in r["normal_episodes"][arm]))
        output[name] = arms
    return output


def run(args):
    sys.path.insert(0, str(args.v83_dir))
    from scripts import analyze_general_ood_v83_cross_seed_covariance as source
    from scripts.v73_recursive_mode_readout import observe_likelihoods
    import analyze_general_ood_v107_normal_attribution as v107

    torch.set_num_threads(2)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    family, _ = v113.v106.load_family(args, source)
    mean, classifiers = load_models(args, family)
    pilot = json.loads((args.v116_dir/"run_results/formal/matched_neural_classifier.json").read_text())
    assert pilot["revision"] == v116.REVISION and pilot["passed"] and not pilot["smoke"] and len(pilot["records"]) == 76
    incumbent = json.loads((args.v98_dir/"run_results/replay_diverse_recipe/regime_replay.json").read_text())
    assert incumbent["revision"] == v98.REVISION and incumbent["passed"] and len(incumbent["records"]) == 95
    null_cells = [c for c in source.v76.v69.reference_cells(source.v76.v69.SPEC) if c["role"] == "normal_null"]
    nulls = [source.v76.v68.load_trace(source.v76.v69.trace_path(args.v69_dir/"run_checkpoints/reference/traces", c)) for c in null_cells]
    assert len(nulls) == 2 and sum(len(t["states"]) for t in nulls) == 12000
    values = [transition_scores(family.bank, mean, classifiers, t) for t in nulls]
    references = calibrate_scores(nulls, values)
    metadata = reference_metadata(references)
    verify_temporal_protocol(metadata, incumbent["reference_metadata"][PRIMARY])
    torch.save(dict(revision=REVISION, phases={a: r.phase for a, r in references.items()},
        normal_null_cells=null_cells, reference_metadata=metadata), args.checkpoint_dir/"phases.pt")
    print("[CALIBRATE] three frozen score arms; same 12000 normal-only transitions and temporal rules", flush=True)
    population = evaluation_population(args, source, incumbent)
    assert not set(family.mean_groups) & {c["seed"] for c, _, _ in population}
    assert not {c["seed"] for c in null_cells} & {c["seed"] for c, _, _ in population}
    records, comparisons = [], []
    for i, (cell, path, old) in enumerate(population):
        trace = source.v76.v68.load_trace(path)
        assert len(trace["dones"]) == 6000
        scores = transition_scores(family.bank, mean, classifiers, trace)
        assert all(p.shape == (6000, 5) and np.isfinite(p).all() for p in scores.values())
        comparisons.extend(verify_pilot(scores, cell, pilot["records"], source))
        arms, episodes = {}, {}
        for arm in ARMS:
            series = observe_likelihoods(scores[arm], trace["dones"], references[arm], "recursive")
            arms[arm] = window_statistics(trace, scores[arm], series, cell["mode"], source)
            if cell["mode"] == "normal":
                episodes[arm] = v107.episode_diagnostics(trace, series)
            if args.smoke:
                prefix = observe_likelihoods(scores[arm][:3850], trace["dones"][:3850], references[arm], "recursive")
                for field in series:
                    np.testing.assert_array_equal(series[field][:3850], prefix[field])
        records.append(dict(**cell, arms=arms, incumbent=old, normal_episodes=episodes or None))
        print(f"[REPLAY] {i+1}/{len(population)} {key(cell)}", flush=True)
    assert len(comparisons) == len(set(comparisons))
    expected = [r for r in pilot["records"] if key(r) in {key(c) for c, _, _ in population}]
    assert set(comparisons) == {v116.v114.window_key(r) for r in expected}
    assert args.smoke or len(comparisons) == 76
    assert metadata == reference_metadata(references)
    result = dict(revision=REVISION, passed=True, smoke=args.smoke, records=records, summary=summarize(records),
        checked_paths=len(records), checked_transitions=6000*len(records), unchanged_v116_score_windows_verified=len(comparisons),
        arms=list(ARMS), incumbent_condition=PRIMARY, reference_metadata=metadata, normal_null_cells=null_cells,
        normal_null_transitions_per_arm=12000, complete_history_preserved=True, temporal_rules_changed=False,
        prefix_causality_checked=args.smoke, normal_thresholds_recalibrated=True, calibration_uses_evaluation_queries=False,
        score_units="empirical log scores, not transition densities or calibrated Bayesian evidence",
        averaging="equal paths; early and tail windows reported separately", all_paths_development=True,
        model_fits=0, environment_steps=0, optimizer_updates=0, readout_controls_actor=False, readout_adopted=False,
        decision="development-temporal-score-replay-not-adoption",
        limitations=["Previously inspected Hopper paths, known modes and one source training seed.",
            "Recursive score multiplication is an empirical filter, not a physical likelihood model.",
            "Normal-null calibration data are reused; exceedance fractions are measured, not guaranteed false-alarm rates.",
            "Incumbent scalars are copied from verified earlier replays; reference budgets and model families differ.",
            "No actor integration, return improvement or cross-environment claim."])
    path = args.result_dir/("native_smoke.json" if args.smoke else "temporal_classifier_replay.json")
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] {REVISION}: {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("v116-dir", "v115-dir", "v113-dir", "v110-dir", "v104-dir", "v98-dir", "v97-dir", "v95-dir",
                 "v94-dir", "v83-dir", "v78-dir", "v76-dir", "v74-dir", "v73-dir", "v72-dir", "v71-dir",
                 "v69-dir", "v70-dir", "v61-dir", "v62-dir", "checkpoint-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
