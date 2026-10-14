"""Summarize independent source training seeds, not evaluation pseudo-replicates."""
import argparse
import json
from pathlib import Path

import numpy as np


def summarize(records):
    assert len(records) == 3 and sorted(r["training_seed"] for r in records) == [0, 1, 2]
    per_seed = []
    for row in sorted(records, key=lambda r: r["training_seed"]):
        assert row["revision"] == "general_ood_v121_locomotion_source" and not row["smoke"]
        assert row["trained_from_scratch"] and row["training_mode"] == "normal-only"
        assert row["spec"] == records[0]["spec"] and row["config"] == records[0]["config"]
        assert row["training"]["interactions"] == 100000 and row["training"]["optimizer_updates"] == 99745
        assert row["checkpoint_selection"] == "final-not-best"
        evaluation = row["final_evaluation"]
        per_seed.append(dict(training_seed=row["training_seed"],
            mean_forward_velocity=evaluation["mean_episode_velocity"],
            mean_episode_length=evaluation["mean_episode_length"], mean_return=evaluation["mean_return"],
            physical_terminations=evaluation["physical_terminations"],
            mean_fraction_abs_velocity_below_point1=float(np.mean([
                e["fraction_abs_velocity_below_point1"] for e in evaluation["episodes"]])),
            checkpoint=row["checkpoint"], periodic_evaluations=row["training"]["periodic_evaluations"],
            final_evaluation=evaluation, runtime=row["runtime"]))
    metrics = ("mean_forward_velocity", "mean_episode_length", "mean_return",
               "mean_fraction_abs_velocity_below_point1")
    return dict(revision="general_ood_v121_locomotion_source", independent_training_seeds=3,
        training_interactions=300000, optimizer_updates=299235, spec=records[0]["spec"],
        per_training_seed=per_seed, aggregates={k: dict(mean=float(np.mean([r[k] for r in per_seed])),
            minimum=float(min(r[k] for r in per_seed)), maximum=float(max(r[k] for r in per_seed))) for k in metrics},
        source_readout_refitted=False, checkpoint_selection="final-not-best")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads((args.result_dir/f"seed{seed}/source.json").read_text()) for seed in (0, 1, 2)]
    summary = summarize(records)
    path = args.result_dir/"summary.json"
    path.write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print(json.dumps(summary["aggregates"], indent=2))


if __name__ == "__main__":
    main()
