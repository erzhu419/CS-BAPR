"""Merge the three frozen-candidate confirmation shards without retuning."""
import argparse
import json
from pathlib import Path
from statistics import mean, stdev
import sys

from scipy.stats import t

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_general_ood_v110_fresh_mixture as v110


def summarize(parts):
    assert [r["seed"] for r in parts] == list(v110.SEEDS)
    records = []
    for result in parts:
        assert result["revision"] == v110.REVISION and result["passed"] and not result["smoke"]
        assert result["spec"] == v110.SPEC and result["arms"] == list(v110.ARMS)
        assert result["conditions"] == list(v110.summary.CONDITIONS)
        assert result["fresh_paths"] == len(result["records"]) == 10 and result["changed_windows"] == 16
        assert result["reference_metadata_unchanged"] and result["complete_history_preserved"]
        assert not any(result[k] for k in ("reference_refitted", "normal_thresholds_recalibrated",
            "mixture_weights_fitted", "temporal_rules_changed", "readout_controls_actor", "readout_adopted"))
        assert result["mixture_weights"] == [.5, .5]
        assert result["reference_metadata"] == parts[0]["reference_metadata"]
        assert [v110.summary.key(r) for r in result["records"]] == [
            v110.summary.key(r) for r in v110.cells(result["seed"])]
        assert result["summary"] == v110.summarize(result["records"])
        records.extend(result["records"])
    totals = {k: sum(r[k] for r in parts) for k in v110.v95.COUNTERS}
    assert totals == dict(actual_transitions=180000, counterfactual_step_calls=900000,
        environment_step_calls=1080000, exact_true_mode_replays=180000, optimizer_updates=90000)
    primary = v110.v109.v108.PRIMARY
    deltas = [r["summary"]["updating"]["normal_density_mixture"][primary]["accuracy"]
        -r["summary"]["updating"]["old"][primary]["accuracy"] for r in parts]
    average, sem = mean(deltas), stdev(deltas)/len(deltas)**.5
    half_width = float(t.ppf(.975, len(deltas)-1)*sem)
    return dict(revision=v110.REVISION, passed=True, paths=len(records), changed_windows=48, **totals,
        populations=v110.summarize(records), primary_condition=primary,
        paired_updating_accuracy=dict(seeds=list(v110.SEEDS), deltas=deltas, mean_delta=average,
            confidence_interval_95=[average-half_width, average+half_width],
            method="paired environment-seed Student t interval; n=3, df=2, not independent training replication"),
        reference_metadata_matches_all_shards=True, readout_adopted=False, readout_controls_actor=False,
        decision="HOLD-confirmation-insufficient-gain-with-normal-regression",
        averaging="equal paths and equal first_post50/last64 windows; uncertainty clustered by environment seed",
        limitations=["Three environment seeds conditional on one source policy and known Hopper modes.",
            "Mean risk is not a false-positive rate; no actor-return evidence.",
            "Any redesign informed by these outcomes makes them development data for that redesign."])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    parts = [json.loads((args.result_dir/f"seed{s}/fresh_seed.json").read_text()) for s in v110.SEEDS]
    result = summarize(parts)
    path = args.result_dir/"summary.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(path)
