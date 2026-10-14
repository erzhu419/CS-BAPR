"""Keep independent training replication separate from repeated evaluation noise."""
import argparse
import json
from pathlib import Path

import numpy as np

import analyze_general_ood_v130_independent_replication as worker
from summarize_general_ood_v129 import aggregate, differences


def summarize(rows):
    assert len(rows) == len(worker.SEEDS) and {r["training_seed"] for r in rows} == set(worker.SEEDS)
    assert all(r["runtime"] == rows[0]["runtime"] for r in rows)
    records, noise_records, steps = [], [], 0
    for row in sorted(rows, key=lambda r: r["training_seed"]):
        seed, spec = row["training_seed"], worker.stage_spec(row["training_seed"])
        assert row["revision"] == worker.REVISION and row["spec"] == spec and row["evaluation_device"] == "cpu"
        assert row["optimizer_updates"] == row["new_training_interactions"] == 0
        assert not row["source_readout_refitted"] and row["reward_or_physics_changes"] == []
        assert row["checkpoint_selection"] == "predeclared-final-only"
        sources = row["sources"]
        assert [s["model"] for s in sources] == list(worker.MODELS)
        assert [s["age"] for s in sources] == [300000, 350000, 350000]
        assert [s["revision"] for s in sources] == [worker.v128.v123.REVISION, worker.v128.REVISION, worker.v128.REVISION]
        assert [s["arm"] for s in sources] == [None, *worker.v128.ARMS]
        assert all(s["config"] == sources[0]["config"] for s in sources)
        assert sources[0]["config"] == rows[0]["sources"][0]["config"]
        expected, evaluations = worker.v129.cells(spec), row["evaluations"]
        assert [{k: e[k] for k in expected[0]} for e in evaluations] == expected
        for e in evaluations:
            assert e["noise_scale"] == (0. if e["mode"] == "deterministic" else 1.)
            assert 0 < e["length"] <= 1000 and not e["diagnostic_cutoff"]
            assert e["physical_termination"] or e["time_limit_truncation"]
            assert all(np.isfinite(e[k]) for k in ("episode_return", "mean_velocity"))
        interactions = sum(e["length"] for e in evaluations)
        assert interactions == row["evaluation_interactions"]
        steps += interactions
        modes = {}
        for mode in ("deterministic", "sampled"):
            models = {m: aggregate([e for e in evaluations if e["model"] == m and e["mode"] == mode])
                      for m in worker.MODELS}
            modes[mode] = dict(models=models, **differences(models))
        records.append(dict(training_seed=seed, independent_replication=seed >= 3, modes=modes))
        for noise in spec["noise_seeds"]:
            models = {m: aggregate([e for e in evaluations if e["model"] == m and e["noise_seed"] == noise])
                      for m in worker.MODELS}
            noise_records.append(dict(training_seed=seed, noise_seed=noise, models=models, **differences(models)))
    return dict(revision=worker.REVISION, independent_training_seeds=5, new_training_seeds=[3, 4],
        frozen_models=15, evaluation_episodes=1575, evaluation_interactions=steps,
        evaluation_optimizer_updates=0, per_training_seed=records, per_noise=noise_records,
        interpretation="Seeds3/4 independently replicate a fixed intervention; seeds0/1/2 are prior study seeds. Noise/reset repeats are not training seeds. Normal Hopper only.")


def training_cost(root):
    records = []
    for seed in (3, 4):
        prefix = json.loads((root/f"prefix_seed{seed}/source.json").read_text())
        source = json.loads((root/f"source_seed{seed}/source.json").read_text())
        assert (prefix["revision"], prefix["training_seed"], prefix["smoke"]) == (worker.v128.v123.v122.REVISION, seed, False)
        assert prefix["training"]["interactions"] == 100000 and prefix["spec"] == worker.v128.v123.v122.stage_spec()
        assert (source["revision"], source["training_seed"], source["smoke"]) == (worker.v128.v123.REVISION, seed, False)
        assert source["spec"] == worker.v128.v123.stage_spec() and source["training"]["interactions"] == 300000
        assert source["actual_new_training_interactions"] == source["actual_new_optimizer_updates"] == 200000
        assert prefix["config"] == source["config"] and source["config"]["weight_reg"] == 0.
        assert prefix["training"]["optimizer_updates"] == 100000-prefix["config"]["batch_size"]+1
        assert source["training"]["optimizer_updates"] == 300000-source["config"]["batch_size"]+1
        updates = source["training"]["optimizer_updates"]
        for arm in worker.v128.ARMS:
            row = json.loads((root/f"seed{seed}_{arm}/mean_retention.json").read_text())
            assert (row["revision"], row["training_seed"], row["smoke"], row["arm"]) == (worker.v128.REVISION, seed, False, arm)
            assert row["spec"] == worker.v128.stage_spec(arm) and row["config"] == source["config"]
            assert row["reference_checkpoint"] == source["checkpoint"] and row["checkpoint_selection"] == "final-not-best"
            assert row["actual_new_training_interactions"] == row["actual_new_optimizer_updates"] == 50000
            assert row["training"]["interactions"] == 350000
            assert row["training"]["optimizer_updates"] == source["training"]["optimizer_updates"]+50000
            updates += row["actual_new_optimizer_updates"]
        records.append(dict(training_seed=seed, source_interactions=300000,
            continuation_interactions=100000, new_training_interactions=400000, new_optimizer_updates=updates))
    return dict(new_training_interactions=sum(r["new_training_interactions"] for r in records),
        new_optimizer_updates=sum(r["new_optimizer_updates"] for r in records), per_training_seed=records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads((args.result_dir/f"seed{seed}_cpu/replication.json").read_text()) for seed in worker.SEEDS]
    result = summarize(rows)
    result["training_cost"] = training_cost(args.result_dir)
    (args.result_dir/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result["per_training_seed"], indent=2))


if __name__ == "__main__":
    main()
