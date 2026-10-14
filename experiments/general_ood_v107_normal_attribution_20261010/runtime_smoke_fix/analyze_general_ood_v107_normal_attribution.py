"""Cross cached density likelihoods and calibration without new geometry fits."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_general_ood_v105_behavior_support as v105

v98 = v105.v98
REVISION = "general_ood_v107_normal_attribution"
RECIPES = dict(old="diverse_recipe", new="standard_mean_incumbent_covariance")
ARMS = tuple(f"{m}_mean__{c}_calibration" for m in RECIPES for c in RECIPES)
FAILED_MODES = {(s, m) for s, m, _ in v105.FAILURES} | {(135190, "compound")}


def select_paths(records, smoke=False):
    selected = [r for r in records if
        (r["group"].startswith("v95_development_") and
         (r["mode"] == "normal" or (r["seed"], r["mode"]) in FAILED_MODES)) or
        (r["group"] == "v62_development_replay" and r["seed"] == 134020 and r["updating"]
         and r["mode"] in ("normal", "slippery"))]
    assert len(selected) == len({v105.key(r) for r in selected}) == 18
    if smoke:
        selected = [r for r in selected if r["seed"] == 135192 and r["mode"] in ("normal", "compound")]
        assert len(selected) == 4
    return selected


def load_likelihoods(path, recipe, key, mean_rows):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert saved["revision"] == v98.REVISION and saved["recipe"] == recipe and tuple(saved["key"]) == key
    assert (saved["raw_rows"], saved["mean_rows"]) == (27400, mean_rows)
    values = {c: saved["terms"][c]["likelihoods"] for c in v98.v94.CONDITIONS}
    assert all(v.shape == (6000, 5) and np.isfinite(v).all() for v in values.values())
    return values


def load_references(directory, recipe, data):
    from scripts.v73_recursive_mode_readout import RecursiveModeReference

    saved = torch.load(directory/"phases.pt", map_location="cpu", weights_only=False)
    assert saved["revision"] == v98.REVISION and saved["recipe"] == recipe
    assert saved["normal_null_transitions"] == 12000 and saved["raw_weight"] == data["raw_weight"]
    references = {}
    for c in v98.v94.CONDITIONS:
        ref = RecursiveModeReference()
        ref.bank = SimpleNamespace(metadata=lambda c=c: data["reference_metadata"][c]["bank"])
        ref.phase, ref.normal_null_calibration_transitions = saved["phases"][c], 12000
        assert ref.phase.metadata() == data["reference_metadata"][c]["phase"]
        references[c] = ref
    return references, saved["groups"]


def crossed_readouts(likelihoods, references, dones):
    from scripts.v73_recursive_mode_readout import observe_likelihoods

    results = {f"{m}_mean__{c}_calibration": observe_likelihoods(ll, dones, references[c], "recursive")
               for m, ll in likelihoods.items() for c in RECIPES}
    for m in RECIPES:
        a, b = (results[f"{m}_mean__{c}_calibration"] for c in RECIPES)
        np.testing.assert_array_equal(a["mode_scores"], b["mode_scores"])
        np.testing.assert_array_equal(a["signal"], b["signal"])
    return results


def episode_diagnostics(trace, readout):
    ends = np.flatnonzero(trace["dones"])+1
    if len(ends) == 0 or ends[-1] != len(trace["dones"]):
        ends = np.r_[ends, len(trace["dones"])]
    starts = np.r_[0, ends[:-1]]
    scores = readout["mode_scores"]
    # Slice the one complete history; do not reset the posterior at episode boundaries.
    return [dict(episode=i, start=int(a), stop=int(b),
        mean_risk=float(readout["risk"][a:b].mean()), maximum_risk=float(readout["risk"][a:b].max()),
        normal_mode_accuracy=float(np.mean(scores[a:b].argmax(axis=1) == 0)),
        mean_signal=float(readout["signal"][a:b].mean()),
        mean_threshold=float(readout["threshold"][a:b].mean())) for i, (a, b) in enumerate(zip(starts, ends))]


def verify_diagonal(actual, expected):
    for w, values in expected.items():
        for k in ("mean_coefficient", "mode_accuracy", "mean_true_minus_best_other_score",
                  "instantaneous_mode_accuracy", "instantaneous_mean_true_minus_best_other_log_density"):
            np.testing.assert_allclose(actual[w][k], values[k], rtol=1e-9, atol=1e-8)
        assert actual[w]["predicted_mode_fractions"] == values["predicted_mode_fractions"]


def summarize(records):
    from statistics import mean

    result = {}
    for group in dict.fromkeys(r["group"] for r in records):
        rows = [r for r in records if r["group"] == group]
        normal, changed = ([r for r in rows if (r["mode"] == "normal") == flag] for flag in (True, False))
        result[group] = {arm: {c: dict(
            normal_paths=len(normal), changed_paths=len(changed),
            maximum_normal_mean_risk=max(r["arms"][arm][c][w]["mean_coefficient"]
                for r in normal for w in ("first800", "full", "last1000")),
            normal_full_accuracy=mean(r["arms"][arm][c]["full"]["mode_accuracy"] for r in normal),
            changed_accuracy=mean(r["arms"][arm][c][w]["mode_accuracy"] for r in changed
                for w in ("first_post50", "last64")),
            minimum_changed_mean_risk=min(r["arms"][arm][c][w]["mean_coefficient"] for r in changed
                for w in ("first_post50", "last64"))) for c in v98.v94.CONDITIONS} for arm in ARMS}
    return result


def run(args):
    sys.path.insert(0, str(args.v83_dir))
    from scripts import analyze_general_ood_v83_cross_seed_covariance as source

    torch.set_num_threads(2)
    data = dict(old=json.loads((args.v98_dir/"run_results/replay_diverse_recipe/regime_replay.json").read_text()),
        new=json.loads((args.v106_dir/"run_results/replay/regime_replay.json").read_text()))
    assert all(d["passed"] and d["stage"] == "replay" and d["evaluation_paths"] == 95 for d in data.values())
    assert data["new"]["revision"] == "general_ood_v106_standard_mean_replay"
    assert data["new"]["complete_covariance_geometry_unchanged"] and data["new"]["calibration_protocol_matches_incumbent"]
    assert [v105.key(r) for r in data["old"]["records"]] == [v105.key(r) for r in data["new"]["records"]]
    population = select_paths(data["new"]["records"], args.smoke)
    paths = {v105.key(c): (i, p) for i, (c, p, _) in enumerate(v98.evaluation_inputs(args, source))}
    directories = dict(old=args.v98_dir/"run_checkpoints/diverse_recipe", new=args.v106_dir/"run_checkpoints/standard")
    references, groups = {}, {}
    for m in RECIPES:
        references[m], groups[m] = load_references(directories[m], RECIPES[m], data[m])
    np.testing.assert_array_equal(groups["old"], groups["new"])
    records, comparisons = [], 0
    for cell in population:
        index, path = paths[v105.key(cell)]
        assert v105.key(data["old"]["records"][index]) == v105.key(cell)
        trace = source.v76.v68.load_trace(path)
        assert len(trace["dones"]) == 6000
        ll = {m: load_likelihoods(directories[m]/"terms"/f"eval_{index:02d}.pt", RECIPES[m],
            v105.key(cell), 29600 if m == "old" else 34000) for m in RECIPES}
        arms, episodes = {a: {} for a in ARMS}, {a: {} for a in ARMS}
        for c in v98.v94.CONDITIONS:
            readouts = crossed_readouts({m: ll[m][c] for m in RECIPES}, {m: references[m][c] for m in RECIPES}, trace["dones"])
            for arm, readout in readouts.items():
                density = arm.split("_mean__")[0]
                arms[arm][c] = source.v76.v73.window_statistics(trace, ll[density][c], readout, cell["mode"])
                if cell["mode"] == "normal":
                    episodes[arm][c] = episode_diagnostics(trace, readout)
            for m in RECIPES:
                verify_diagonal(arms[f"{m}_mean__{m}_calibration"][c], data[m]["records"][index]["observers"][c])
                comparisons += 1
        records.append(dict(**{k: cell[k] for k in ("group", "seed", "mode", "updating")},
            arms=arms, normal_episodes=episodes if cell["mode"] == "normal" else None))
        print(f"[CROSS] {len(records)}/{len(population)} {v105.key(cell)}", flush=True)
    result = dict(revision=REVISION, passed=True, smoke=args.smoke, checked_paths=len(records),
        checked_transitions=6000*len(records), diagonal_path_condition_comparisons=comparisons,
        arms=list(ARMS), summary=summarize(records), records=records,
        environment_steps=0, optimizer_updates=0, likelihoods_recomputed=False,
        geometry_refitted=False, calibration_refitted=False, coefficient_tuned=False,
        posterior_independent_of_calibration_verified=True, complete_history_preserved=True,
        readout_controls_actor=False, readout_adopted=False,
        decision="development-normal-risk-attribution-not-adoption",
        limitations=["Selected development paths, one source policy and known modes.",
            "Crossed old calibration is a sensitivity intervention, not a newly validated calibration.",
            "Changing calibration cannot fix posterior mode errors; risk means are not false-positive rates."])
    args.result_dir.mkdir(parents=True, exist_ok=True)
    path = args.result_dir/("native_smoke.json" if args.smoke else "normal_attribution.json")
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] {REVISION}: {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("v106-dir", "v98-dir", "v97-dir", "v95-dir", "v94-dir", "v83-dir", "v78-dir", "v76-dir",
                 "v74-dir", "v73-dir", "v72-dir", "v71-dir", "v69-dir", "v70-dir", "v61-dir", "v62-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
