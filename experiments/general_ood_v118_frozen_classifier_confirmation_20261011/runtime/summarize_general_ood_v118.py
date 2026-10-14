"""Merge the prespecified three frozen-classifier environment-seed shards."""
import argparse
import json
from pathlib import Path
from statistics import mean, stdev
import sys

from scipy.stats import t

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_general_ood_v118_frozen_classifier_confirmation as v118


def paired_interval(deltas):
    center = mean(deltas)
    half = float(t.ppf(.975, len(deltas)-1)*stdev(deltas)/len(deltas)**.5)
    return dict(deltas=deltas, mean_delta=center, confidence_interval_95=[center-half, center+half],
                method="paired environment-seed Student t interval, n=3; not source-training replication")


def summarize(parts):
    assert [p["seed"] for p in parts] == list(v118.SEEDS)
    records = []
    for p in parts:
        assert p["revision"] == v118.REVISION and p["passed"] and not p["smoke"]
        assert p["spec"] == v118.SPEC and p["arms"] == list(v118.ARMS)
        assert p["incumbent_conditions"] == list(v118.CONDITIONS) and p["checked_path_readouts"] == 60
        assert p["fresh_paths"] == len(p["records"]) == 10 and p["changed_windows"] == 16
        assert p["reference_metadata_unchanged"] and p["references_frozen_before_collection"]
        assert not any(p[f] for f in ("reference_refitted", "normal_thresholds_recalibrated", "temporal_rules_changed",
            "classifier_settings_changed", "mean_settings_changed", "candidate_selected", "readout_controls_actor", "readout_adopted"))
        assert p["reference_metadata"] == parts[0]["reference_metadata"]
        assert [v118.v117.key(r) for r in p["records"]] == [v118.v117.key(c) for c in v118.cells(p["seed"])]
        assert p["summary"] == v118.summarize(p["records"])
        records.extend(p["records"])
    totals = {k: sum(p[k] for p in parts) for k in v118.v95.COUNTERS}
    assert totals == dict(actual_transitions=180000, counterfactual_step_calls=900000,
                         environment_step_calls=1080000, exact_true_mode_replays=180000, optimizer_updates=90000)
    contrasts = {}
    for metric in ("early_accuracy", "tail_accuracy", "normal_full_mean_risk", "normal_full_above_threshold_fraction"):
        contrasts[metric] = paired_interval([p["summary"]["updating"]["matched_classifier"][metric]
            -p["summary"]["updating"]["incumbent_v98"][metric] for p in parts])
    return dict(revision=v118.REVISION, passed=True, paths=30, changed_windows=48, **totals,
        populations=v118.summarize(records), per_seed={str(p["seed"]): p["summary"] for p in parts},
        paired_updating_matched_minus_v98=contrasts, frozen_metadata_matches_all_shards=True,
        averaging="equal paths; early and tail windows separate; uncertainty clustered by environment seed",
        readout_adopted=False, readout_controls_actor=False, candidate_selected=False,
        decision="completed-frozen-confirmation-requires-joint-result-assessment",
        limitations=parts[0]["limitations"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    parts = [json.loads((args.result_dir/f"seed{s}/fresh_seed.json").read_text()) for s in v118.SEEDS]
    path = args.result_dir/"summary.json"
    path.write_text(json.dumps(summarize(parts), indent=2, allow_nan=False)+"\n")
    print(path)
