"""Matched full-history comparison of the frozen-covariance intervention."""
import argparse
import json
from pathlib import Path
from statistics import mean
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize_general_ood_v98 as v98


def summarize(candidate, incumbent, age_balanced):
    arms = dict(incumbent_suffix=incumbent, age_balanced=age_balanced, frozen_covariance=candidate)
    expected = (("general_ood_v98_regime_replay", "diverse_recipe"),
        ("general_ood_v101_age_replay", "age_balanced"),
        ("general_ood_v103_frozen_covariance", "age_mean_incumbent_covariance"))
    for data, (revision, recipe) in zip(arms.values(), expected):
        assert (data["revision"], data["recipe"]) == (revision, recipe)
        assert data["passed"] and data["stage"] == "replay" and not data["readout_controls_actor"]
        assert data["evaluation_paths"] == len(data["records"]) == 95 and data["evaluated_transitions"] == 570000
        assert data["conditions"] == list(v98.CONDITIONS)
        assert len({v98.key(r) for r in data["records"]}) == 95
        assert [v98.key(r) for r in data["records"]] == [v98.key(r) for r in incumbent["records"]]
        for field in ("raw_rows", "mean_rows", "source_seeds", "additional_reference_inputs", "raw_weight",
                      "disagreement_coefficient", "normal_null_transitions_per_condition", "inner_fits"):
            assert data[field] == incumbent[field], field
        for old, new in zip(incumbent["records"], data["records"]):
            for field in ("observers", "attribution", "change_minus_nochange"):
                for c in v98.CONDITIONS:
                    assert old[field]["v94_unchanged__"+c] == new[field]["v94_unchanged__"+c]
    assert candidate["calibration_protocol_matches_incumbent"] and age_balanced["calibration_protocol_matches_incumbent"]
    assert candidate["complete_covariance_geometry_unchanged"] and not candidate["covariance_moment_recentered"]
    old_inputs = {(s["seed"], s["source_mode"], i) for s in incumbent["reference_sources"]
        for i in range(*s["source_steps"], s["stride"])}
    age_inputs = {(s["seed"], s["source_mode"], i) for s in age_balanced["reference_sources"]
        for i in range(s["start"], s["stop"], s["stride"])}
    reference_budget = dict(incumbent_distinct_extension_inputs=len(old_inputs),
        age_balanced_distinct_extension_inputs=len(age_inputs), overlap=len(old_inputs & age_inputs),
        hybrid_distinct_extension_inputs=len(old_inputs | age_inputs),
        equal_used_reference_input_budget=False, additional_environment_steps=0)
    for new, age in zip(candidate["records"], age_balanced["records"]):
        for c in v98.CONDITIONS:
            for w in v98.WINDOWS:
                assert abs(new["attribution"][c][w]["mean_true_prediction_error_rms"]
                    -age["attribution"][c][w]["mean_true_prediction_error_rms"]) < 1e-8
    subsets = dict(all=lambda r: True, updating=lambda r: r["updating"], frozen=lambda r: not r["updating"],
        v95_updating=lambda r: r["group"] == "v95_development_updating",
        v95_frozen=lambda r: r["group"] == "v95_development_frozen")
    for seed in (135190, 135191, 135192):
        subsets[f"v95_updating_seed{seed}"] = lambda r, s=seed: r["group"] == "v95_development_updating" and r["seed"] == s
    populations = {}
    for name, select in subsets.items():
        populations[name] = {}
        for arm, data in arms.items():
            records = [r for r in data["records"] if select(r)]
            populations[name][arm] = {c: dict(v98.aggregate(records, c),
                mean_true_prediction_error_rms=mean(r["attribution"][c][w]["mean_true_prediction_error_rms"]
                    for r in records if r["mode"] != "normal" for w in v98.WINDOWS)) for c in v98.CONDITIONS}
    return dict(revision=candidate["revision"], passed=True, trajectory_keys_and_controls_match=True,
        candidate_means_match_age_balanced=True, reference_budget=reference_budget, populations=populations,
        window_records=[dict(**{k: r[k] for k in ("group", "seed", "mode", "updating")},
            arms={a: {field: data["records"][i][field] for field in ("observers", "attribution", "change_minus_nochange")}
                  for a, data in arms.items()}) for i, r in enumerate(incumbent["records"])
            if r["group"].startswith("v95_development_") or r["group"] == "v62_development_replay"],
        averaging="equal changed path/window; first-post50 and last64 receive equal weight",
        normal_metric="window mean risk, not a false-positive rate", all_paths_development=True,
        readout_adopted=False, readout_controls_actor=False, decision="HOLD")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("result-dir", "incumbent", "age-balanced"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(json.loads((args.result_dir/"regime_replay.json").read_text()),
        json.loads(args.incumbent.read_text()), json.loads(args.age_balanced.read_text()))
    path = args.result_dir/"summary.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(path)
