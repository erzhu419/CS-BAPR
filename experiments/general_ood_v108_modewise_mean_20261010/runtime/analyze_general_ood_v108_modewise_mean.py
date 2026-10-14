"""Mode-wise mean ablation using cached normalized densities and matched nulls."""
import argparse
import json
from pathlib import Path
from statistics import mean
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_general_ood_v107_normal_attribution as v107
import summarize_general_ood_v98 as summary

v98 = v107.v98
REVISION = "general_ood_v108_modewise_mean"
ARMS = ("old", "new", "old_normal_new_ood", "new_normal_old_ood")
PRIMARY = "suffix_disagreement_mix"
MODES = tuple(v98.v94.MODES)
assert MODES == ("normal", "heavy", "slippery", "high_grav", "compound")
MEAN_SOURCES = {
    "old": dict.fromkeys(MODES, "old"),
    "new": dict.fromkeys(MODES, "new"),
    "old_normal_new_ood": dict.fromkeys(MODES, "new") | {"normal": "old"},
    "new_normal_old_ood": dict.fromkeys(MODES, "old") | {"normal": "new"},
}


def compose(old, new):
    assert old.shape == new.shape and old.ndim == 2 and old.shape[1] == len(MODES)
    return {a: np.column_stack([(old if sources[m] == "old" else new)[:, i]
        for i, m in enumerate(MODES)]) for a, sources in MEAN_SOURCES.items()}


def calibrate(nulls, likelihoods, bank_metadata):
    references = {}
    for arm in ARMS:
        family = SimpleNamespace(metadata=lambda c, a=arm: bank_metadata[a][c])
        terms = [{c: dict(likelihoods=compose(row["old"][c], row["new"][c])[arm])
                  for c in summary.CONDITIONS} for row in likelihoods]
        references[arm] = v98.v94.calibrate_paired(family, nulls, terms)
    return references


def likelihood_shift(old, new):
    return dict(old_mean_log_density=dict(zip(MODES, old.mean(axis=0).tolist())),
        new_mean_log_density=dict(zip(MODES, new.mean(axis=0).tolist())),
        mean_log_density_change=dict(zip(MODES, (new-old).mean(axis=0).tolist())),
        compound_minus_normal_change=float(((new[:, 4]-new[:, 0])-(old[:, 4]-old[:, 0])).mean()))


def summarize(records):
    index = {summary.key(r): r for r in records}
    views = {}
    for arm in ARMS:
        views[arm] = []
        for r in records:
            normal = index[r["group"], r["seed"], "normal", r["updating"]]
            contrasts = {c: {w: r["arms"][arm][c][w]["mean_coefficient"]
                - normal["arms"][arm][c][w]["mean_coefficient"] for w in summary.WINDOWS}
                for c in summary.CONDITIONS}
            views[arm].append(dict(r, observers=r["arms"][arm], change_minus_nochange=contrasts))
    subsets = dict(all=lambda r: True, updating=lambda r: r["updating"], frozen=lambda r: not r["updating"],
        v95_updating=lambda r: r["group"] == "v95_development_updating",
        v95_frozen=lambda r: r["group"] == "v95_development_frozen")
    for seed in sorted({r["seed"] for r in records if r["group"] == "v95_development_updating"}):
        subsets[f"v95_updating_seed{seed}"] = lambda r, s=seed: r["group"] == "v95_development_updating" and r["seed"] == s
    populations = {name: {a: {c: summary.aggregate([r for r in rows if select(r)], c)
        for c in summary.CONDITIONS} for a, rows in views.items()} for name, select in subsets.items()}
    failures = {}
    for arm in ARMS:
        rows = [index["v95_development_updating", s, m, True]["arms"][arm][PRIMARY][w]
            for s, m, w in v107.v105.FAILURES if ("v95_development_updating", s, m, True) in index]
        failures[arm] = dict(windows=len(rows), **{k: mean(r[k] for r in rows)
            for k in ("mode_accuracy", "mean_true_minus_best_other_score")})
    historical = index.get(("v62_development_replay", 134020, "slippery", True))
    return dict(populations=populations, old_failure_windows=failures,
        historical_slippery_tail={a: historical["arms"][a][PRIMARY]["last64"] for a in ARMS} if historical else None,
        averaging="equal changed path/window; first-post50 and last64 have equal weight",
        normal_metric="maximum window mean risk, not a false-positive rate")


def run(args):
    sys.path.insert(0, str(args.v83_dir))
    from scripts import analyze_general_ood_v83_cross_seed_covariance as source

    torch.set_num_threads(2)
    data = dict(old=json.loads((args.v98_dir/"run_results/replay_diverse_recipe/regime_replay.json").read_text()),
        new=json.loads((args.v106_dir/"run_results/replay/regime_replay.json").read_text()))
    assert all(d["passed"] and d["stage"] == "replay" and d["evaluation_paths"] == 95 for d in data.values())
    assert data["old"]["revision"] == v98.REVISION and data["new"]["revision"] == "general_ood_v106_standard_mean_replay"
    assert data["new"]["complete_covariance_geometry_unchanged"] and not data["new"]["covariance_moment_recentered"]
    assert data["new"]["calibration_protocol_matches_incumbent"] and data["old"]["raw_weight"] == data["new"]["raw_weight"]
    assert [summary.key(r) for r in data["old"]["records"]] == [summary.key(r) for r in data["new"]["records"]]
    population = v107.select_paths(data["new"]["records"], True) if args.smoke else data["new"]["records"]
    paths = {summary.key(c): (i, p) for i, (c, p, _) in enumerate(v98.evaluation_inputs(args, source))}
    directories = dict(old=args.v98_dir/"run_checkpoints/diverse_recipe", new=args.v106_dir/"run_checkpoints/standard")
    saved_refs, groups = {}, {}
    for arm in v107.RECIPES:
        saved_refs[arm], groups[arm] = v107.load_references(directories[arm], v107.RECIPES[arm], data[arm])
    np.testing.assert_array_equal(groups["old"], groups["new"])
    bank_metadata = {a: {c: (data[a]["reference_metadata"][c]["bank"] if a in data else
        dict(revision=REVISION, mode_mean_sources=MEAN_SOURCES[a], covariance_reference="v98 diverse_recipe",
             raw_weight=data["old"]["raw_weight"], normalized_density_columns_reused=True))
        for c in summary.CONDITIONS} for a in ARMS}
    null_cells = [c for c in source.v76.v69.reference_cells(source.v76.v69.SPEC) if c["role"] == "normal_null"]
    nulls = [source.v76.v68.load_trace(source.v76.v69.trace_path(args.v69_dir/"run_checkpoints/reference/traces", c))
             for c in null_cells]
    assert len(nulls) == 2 and all(len(t["dones"]) == 6000 for t in nulls)
    assert {c["updating"] for c in null_cells} == {False, True}
    null_ll = [{a: v107.load_likelihoods(directories[a]/"terms"/f"normal_null_{i}.pt",
        v107.RECIPES[a], ("normal_null", i), 29600 if a == "old" else 34000) for a in v107.RECIPES}
        for i in range(2)]
    references = calibrate(nulls, null_ll, bank_metadata)
    for arm in v107.RECIPES:
        for c in summary.CONDITIONS:
            assert references[arm][c].metadata() == saved_refs[arm][c].metadata()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(dict(revision=REVISION, mode_mean_sources=MEAN_SOURCES, normal_null_cells=null_cells,
        normal_null_transitions=12000, phases={a: {c: r.phase for c, r in refs.items()}
        for a, refs in references.items()}), args.checkpoint_dir/"phases.pt")
    print("[CALIBRATE] four densities, same 12000 normal-null transitions; six original calibrations reproduced", flush=True)
    records, comparisons = [], 0
    for cell in population:
        index, path = paths[summary.key(cell)]
        trace = source.v76.v68.load_trace(path)
        assert len(trace["dones"]) == 6000
        ll = {a: v107.load_likelihoods(directories[a]/"terms"/f"eval_{index:02d}.pt",
            v107.RECIPES[a], summary.key(cell), 29600 if a == "old" else 34000) for a in v107.RECIPES}
        arms, episodes, shifts = {a: {} for a in ARMS}, {a: {} for a in ARMS}, {}
        for c in summary.CONDITIONS:
            assembled = compose(ll["old"][c], ll["new"][c])
            for arm, values in assembled.items():
                readout = source.v76.v73.observe_likelihoods(values, trace["dones"], references[arm][c], "recursive")
                arms[arm][c] = source.v76.v73.window_statistics(trace, values, readout, cell["mode"])
                if cell["mode"] == "normal":
                    episodes[arm][c] = v107.episode_diagnostics(trace, readout)
            for arm in v107.RECIPES:
                v107.verify_diagonal(arms[arm][c], data[arm]["records"][index]["observers"][c])
                comparisons += 1
            ranges = [(w, a, b) for w, a, b in source.v78.v77.windows()]
            if cell["mode"] == "normal":
                ranges = [("full", 0, 6000)]+[(f"episode{e['episode']}", e["start"], e["stop"])
                    for e in episodes["old"][c]]
            shifts[c] = {w: likelihood_shift(ll["old"][c][a:b], ll["new"][c][a:b]) for w, a, b in ranges}
        records.append(dict(**{k: cell[k] for k in ("group", "seed", "mode", "updating")},
            arms=arms, normal_episodes=episodes if cell["mode"] == "normal" else None, likelihood_shifts=shifts))
        print(f"[REPLAY] {len(records)}/{len(population)} {summary.key(cell)}", flush=True)
    assert len(records) == (4 if args.smoke else 95)
    result = dict(revision=REVISION, passed=True, smoke=args.smoke, checked_paths=len(records),
        checked_transitions=6000*len(records), homogeneous_path_condition_comparisons=comparisons,
        homogeneous_calibration_comparisons=6, normal_null_transitions_per_arm_condition=12000,
        mode_mean_sources=MEAN_SOURCES, summary=summarize(records), records=records,
        reference_metadata={a: {c: r.metadata() for c, r in refs.items()} for a, refs in references.items()},
        environment_steps=0, optimizer_updates=0, likelihoods_recomputed=False, density_refitted=False,
        geometry_refitted=False, calibration_refitted=True, covariance_reference_unchanged=True,
        temporal_rules_changed=False, complete_history_preserved=True, labels_used_for_mean_selection=False,
        readout_controls_actor=False, readout_adopted=False, decision="development-modewise-mean-ablation-not-adoption",
        limitations=["All paths are development evidence; one source policy and known simulator modes.",
            "Mode-specific densities reuse existing mean budgets; this is not a matched total-reference-budget comparison.",
            "Normal calibration reuses the original seed135060 frozen/updating population, not independent confirmation.",
            "No online model implementation or actor/return evidence; window mean risk is not a false-positive rate."])
    args.result_dir.mkdir(parents=True, exist_ok=True)
    path = args.result_dir/("native_smoke.json" if args.smoke else "modewise_mean.json")
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] {REVISION}: {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("v106-dir", "v98-dir", "v97-dir", "v95-dir", "v94-dir", "v83-dir", "v78-dir", "v76-dir",
                 "v74-dir", "v73-dir", "v72-dir", "v71-dir", "v69-dir", "v70-dir", "v61-dir", "v62-dir",
                 "checkpoint-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
