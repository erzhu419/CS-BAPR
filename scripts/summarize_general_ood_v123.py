"""Report longer source training without treating its reused prefixes as new data."""
import argparse
import json
from pathlib import Path

import numpy as np


def summarize(sources, extensions):
    assert len(sources) == len(extensions) == 3
    assert sorted(r["training_seed"] for r in sources) == sorted(r["training_seed"] for r in extensions) == [0, 1, 2]
    sources = {r["training_seed"]: r for r in sources}
    records = []
    keys = ("mean_episode_velocity", "mean_episode_length", "mean_return", "physical_terminations")
    for row in sorted(extensions, key=lambda r: r["training_seed"]):
        source = sources[row["training_seed"]]
        assert source["revision"] == "general_ood_v122_target_l1" and not source["smoke"]
        assert row["revision"] == "general_ood_v123_source_extension" and not row["smoke"]
        assert source["config"] == row["config"] and source["runtime"] == row["runtime"]
        assert row["actual_new_training_interactions"] == row["actual_new_optimizer_updates"] == 200000
        assert row["source_interactions"] == source["training"]["interactions"] == 100000
        assert row["training"]["interactions"] == 300000 and row["training"]["optimizer_updates"] == 299745
        before, after = source["final_evaluation"], row["final_evaluation"]
        assert [e["seed"] for e in before["episodes"]] == [e["seed"] for e in after["episodes"]]
        assert row["checkpoint_selection"] == source["checkpoint_selection"] == "final-not-best"
        records.append(dict(training_seed=row["training_seed"], at100k={k: before[k] for k in keys},
            at300k={k: after[k] for k in keys}, change={k: after[k]-before[k] for k in keys},
            periodic_evaluations=row["training"]["periodic_evaluations"],
            final_evaluation=after, final_target_scale=row["final_target_scale"]))
    return dict(revision="general_ood_v123_source_extension", independent_training_seeds=3,
        actual_new_training_interactions=600000, reused_source_interactions=300000,
        per_training_seed=records, mean_at300k={k: float(np.mean([r["at300k"][k] for r in records])) for k in keys},
        mean_change={k: float(np.mean([r["change"][k] for r in records])) for k in keys},
        source_readout_refitted=False, interpretation="training-budget continuation, not a matched-budget L1 ablation")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    read = lambda directory: [json.loads((directory/f"seed{seed}/source.json").read_text()) for seed in (0, 1, 2)]
    summary = summarize(read(args.source_dir), read(args.result_dir))
    (args.result_dir/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print(json.dumps(summary["mean_at300k"], indent=2))


if __name__ == "__main__":
    main()
