"""Compare complete age-balanced and incumbent suffix development replays."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize_general_ood_v98 as v98


def summarize(candidate, incumbent):
    assert candidate["revision"] == "general_ood_v101_age_replay" and candidate["recipe"] == "age_balanced"
    assert incumbent["revision"] == "general_ood_v98_regime_replay" and incumbent["recipe"] == "diverse_recipe"
    assert candidate["calibration_protocol_matches_incumbent"]
    for data in (candidate, incumbent):
        assert data["passed"] and data["stage"] == "replay"
        assert data["evaluation_paths"] == len(data["records"]) == 95
        assert data["evaluated_transitions"] == 570000 and not data["readout_controls_actor"]
        assert data["conditions"] == list(v98.CONDITIONS)
        assert len({v98.key(r) for r in data["records"]}) == 95
    for field in ("raw_rows", "mean_rows", "source_seeds", "additional_reference_inputs", "raw_weight",
                  "disagreement_coefficient", "normal_null_transitions_per_condition", "inner_fits"):
        assert candidate[field] == incumbent[field], field
    pairs = list(zip(incumbent["records"], candidate["records"]))
    assert [v98.key(a) for a, _ in pairs] == [v98.key(b) for _, b in pairs]
    for a, b in pairs:
        for field in ("observers", "attribution", "change_minus_nochange"):
            for c in v98.CONDITIONS:
                assert a[field]["v94_unchanged__"+c] == b[field]["v94_unchanged__"+c]
    subsets = dict(all=lambda r: True, updating=lambda r: r["updating"], frozen=lambda r: not r["updating"],
        v95_updating=lambda r: r["group"] == "v95_development_updating",
        v95_frozen=lambda r: r["group"] == "v95_development_frozen")
    for seed in (135190, 135191, 135192):
        subsets[f"v95_updating_seed{seed}"] = lambda r, s=seed: r["group"] == "v95_development_updating" and r["seed"] == s
    arms = dict(v94_unchanged=(incumbent, "v94_unchanged__"),
        incumbent_suffix=(incumbent, ""), age_balanced=(candidate, ""))
    populations = {name: {arm: {c: v98.aggregate([r for r in data["records"] if select(r)], prefix+c)
        for c in v98.CONDITIONS} for arm, (data, prefix) in arms.items()} for name, select in subsets.items()}
    windows, controls = [], []
    condition = "suffix_disagreement_mix"
    for a, b in pairs:
        historical = a["group"] == "v62_development_replay" and a["seed"] == 134020 and a["updating"] and a["mode"] in ("normal", "slippery")
        changed = a["group"] == "v95_development_updating" and a["mode"] != "normal"
        for w in (("last64",) if historical else v98.WINDOWS if changed else ()):
            record = dict(**{k: a[k] for k in ("group", "seed", "mode", "updating")}, window=w,
                arms={arm: dict(mode_accuracy=r["observers"][condition][w]["mode_accuracy"],
                    mean_true_minus_best_other_score=r["observers"][condition][w]["mean_true_minus_best_other_score"],
                    prediction_error_rms=r["attribution"][condition][w]["mean_true_prediction_error_rms"])
                    for arm, r in (("incumbent_suffix", a), ("age_balanced", b))})
            (controls if historical else windows).append(record)
    assert len(windows) == 24 and len(controls) == 2
    return dict(revision=candidate["revision"], passed=True, trajectory_keys_and_controls_match=True,
        populations=populations, v95_updating_changed_windows=windows, v62_tail_controls=controls,
        averaging="equal changed path/window; first-post50 and last64 receive equal weight",
        normal_metric="window mean risk, not a false-positive rate", all_paths_development=True,
        readout_adopted=False, readout_controls_actor=False, decision="HOLD")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--incumbent", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(json.loads((args.result_dir/"regime_replay.json").read_text()),
        json.loads(args.incumbent.read_text()))
    path = args.result_dir/"summary.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(path)
