"""Summarize matched development replays without promoting the readout."""
import argparse
import json
from pathlib import Path
from statistics import mean

CONDITIONS = ("suffix_support_mix", "suffix_disagreement_mix", "suffix_trace_match_mix")
WINDOWS = ("first_post50", "last64")
NORMAL_WINDOWS = ("first800", "full", "last1000")


def key(record):
    return tuple(record[k] for k in ("group", "seed", "mode", "updating"))


def aggregate(records, condition):
    normal = [r for r in records if r["mode"] == "normal"]
    changed = [r for r in records if r["mode"] != "normal"]
    return dict(changed_paths=len(changed), changed_windows=2*len(changed), normal_paths=len(normal),
        accuracy=mean(r["observers"][condition][w]["mode_accuracy"] for r in changed for w in WINDOWS),
        nonpositive_margin_windows=sum(r["observers"][condition][w]["mean_true_minus_best_other_score"] <= 0
            for r in changed for w in WINDOWS),
        positive_departure_contrasts=sum(r["change_minus_nochange"][condition][w] > 0
            for r in changed for w in WINDOWS),
        early_accuracy=mean(r["observers"][condition][WINDOWS[0]]["mode_accuracy"] for r in changed),
        tail_accuracy=mean(r["observers"][condition][WINDOWS[1]]["mode_accuracy"] for r in changed),
        maximum_normal_mean_risk=max(r["observers"][condition][w]["mean_coefficient"]
            for r in normal for w in NORMAL_WINDOWS))


def summarize(data):
    old, diverse = (data[k] for k in ("old_recipe", "diverse_recipe"))
    for recipe, d in data.items():
        assert d["recipe"] == recipe and d["passed"] and d["stage"] == "replay"
        assert d["evaluation_paths"] == len(d["records"]) == 95 and d["evaluated_transitions"] == 570000
        assert len({key(r) for r in d["records"]}) == 95
        assert d["conditions"] == list(CONDITIONS)
    assert [key(r) for r in old["records"]] == [key(r) for r in diverse["records"]]
    for a, b in zip(old["records"], diverse["records"]):
        for field in ("observers", "attribution", "change_minus_nochange"):
            for c in CONDITIONS:
                assert a[field]["v94_unchanged__"+c] == b[field]["v94_unchanged__"+c]
    subsets = dict(all=lambda r: True, updating=lambda r: r["updating"], frozen=lambda r: not r["updating"],
        v95_updating=lambda r: r["group"] == "v95_development_updating",
        v95_frozen=lambda r: r["group"] == "v95_development_frozen")
    arms = dict(v94_unchanged=(old, "v94_unchanged__"), old_recipe=(old, ""), diverse_recipe=(diverse, ""))
    output = {s: {arm: {c: aggregate([r for r in d["records"] if select(r)], prefix+c)
        for c in CONDITIONS} for arm, (d, prefix) in arms.items()} for s, select in subsets.items()}
    return dict(revision="general_ood_v98_regime_replay", passed=True, trajectory_keys_and_controls_match=True,
        averaging="equal changed path/window; first-post50 and last64 receive equal weight", populations=output,
        normal_metric="window mean risk, not a false-positive rate", all_paths_development=True,
        readout_adopted=False, readout_controls_actor=False, decision="HOLD")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    data = {r: json.loads((args.result_dir/f"replay_{r}/regime_replay.json").read_text())
            for r in ("old_recipe", "diverse_recipe")}
    path = args.result_dir/"summary.json"
    path.write_text(json.dumps(summarize(data), indent=2, allow_nan=False)+"\n")
    print(path)
