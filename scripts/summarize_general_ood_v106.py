"""Matched development comparison of expanded means and frozen covariance."""
import argparse
import json
from pathlib import Path
from statistics import mean
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize_general_ood_v98 as base
import analyze_general_ood_v106_standard_mean_replay as worker


def summarize(candidate, incumbent, pilot):
    assert (candidate["revision"], candidate["recipe"]) == (worker.REVISION, worker.RECIPE)
    assert (incumbent["revision"], incumbent["recipe"]) == (worker.v98.REVISION, "diverse_recipe")
    for data in (candidate, incumbent):
        assert data["passed"] and data["stage"] == "replay" and not data["readout_controls_actor"]
        assert data["evaluation_paths"] == len(data["records"]) == 95 and data["evaluated_transitions"] == 570000
        if data is incumbent:
            assert data["conditions"] == list(base.CONDITIONS)
        else:
            assert data["cache_implementation_revision"] == worker.v98.REVISION
        assert all(set(base.CONDITIONS) <= r["observers"].keys() for r in data["records"])
        assert len({base.key(r) for r in data["records"]}) == 95
    assert [base.key(r) for r in candidate["records"]] == [base.key(r) for r in incumbent["records"]]
    assert candidate["calibration_protocol_matches_incumbent"] and candidate["complete_covariance_geometry_unchanged"]
    assert not candidate["covariance_moment_recentered"] and not candidate["error_bank_refitted"]
    assert (candidate["raw_rows"], candidate["mean_rows"], candidate["covariance_mean_reference_rows"]) == (27400, 34000, 29600)
    assert candidate["raw_weight"] == incumbent["raw_weight"]
    for actual, old in zip(candidate["records"], incumbent["records"]):
        for field in ("observers", "attribution", "change_minus_nochange"):
            for c in base.CONDITIONS:
                assert actual[field]["v94_unchanged__"+c] == old[field]["v94_unchanged__"+c]
    assert worker.verify_replay_means(candidate, pilot) == 186
    arms = dict(incumbent=incumbent, expanded_standard=candidate)
    subsets = dict(updating=lambda r: r["updating"], frozen=lambda r: not r["updating"],
        v95_updating=lambda r: r["group"] == "v95_development_updating",
        v95_frozen=lambda r: r["group"] == "v95_development_frozen")
    for seed in (135190, 135191, 135192):
        subsets[f"v95_updating_seed{seed}"] = lambda r, s=seed: r["group"] == "v95_development_updating" and r["seed"] == s
    populations = {name: {a: {c: base.aggregate([r for r in data["records"] if select(r)], c)
        for c in base.CONDITIONS} for a, data in arms.items()} for name, select in subsets.items()}
    failures, historical, crossings, normal = {}, {}, [], []
    for a, data in arms.items():
        index = {base.key(r): r for r in data["records"]}
        rows = [index["v95_development_updating", s, m, True]["observers"][base.CONDITIONS[1]][w]
                for s, m, w in worker.v105.FAILURES]
        failures[a] = {k: mean(r[k] for r in rows) for k in ("mode_accuracy", "mean_true_minus_best_other_score")}
        historical[a] = index["v62_development_replay", 134020, "slippery", True]["observers"][base.CONDITIONS[1]]["last64"]
    for i, (old, new) in enumerate(zip(incumbent["records"], candidate["records"])):
        c = base.CONDITIONS[1]
        if new["mode"] == "normal":
            for w in base.NORMAL_WINDOWS:
                normal.append(dict(**{k: new[k] for k in ("group", "seed", "updating")}, window=w,
                    arms={a: data["records"][i]["observers"][c][w] for a, data in arms.items()}))
        else:
            for w in base.WINDOWS:
                before, after = (r["observers"][c][w]["mean_true_minus_best_other_score"] for r in (old, new))
                if (before > 0) != (after > 0):
                    crossings.append(dict(**{k: new[k] for k in ("group", "seed", "mode", "updating")},
                        window=w, old_margin=before, new_margin=after))
    return dict(revision=worker.REVISION, passed=True, populations=populations,
        six_old_failures=failures, historical_slippery=historical, primary_margin_crossings=crossings,
        normal_windows=normal, trajectory_keys_controls_and_pilot_means_verified=True,
        reference_budget=dict(incumbent=4400, expanded_standard=8800, unit="extension source-index union rows", matched=False),
        averaging="equal changed path/window; early and tail windows have equal weight",
        normal_metric="window mean risk, not a false-positive rate", all_paths_development=True,
        readout_adopted=False, readout_controls_actor=False, decision="HOLD-normal-regression-attribution-next")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("result-dir", "incumbent", "pilot"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(json.loads((args.result_dir/"regime_replay.json").read_text()),
        json.loads(args.incumbent.read_text()), json.loads(args.pilot.read_text()))
    path = args.result_dir/"summary.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(path)
