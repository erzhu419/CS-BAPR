"""Per-training-seed and per-noise frozen comparisons; never pool them as seeds."""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze_general_ood_v129_heldout_robustness import MODELS, REVISION, cells, stage_spec


def aggregate(episodes):
    return dict(episodes=len(episodes),
        mean_length=float(np.mean([e["length"] for e in episodes])),
        mean_return=float(np.mean([e["episode_return"] for e in episodes])),
        mean_velocity=float(np.mean([e["mean_velocity"] for e in episodes])),
        falls=sum(e["physical_termination"] for e in episodes),
        time_limits=sum(e["time_limit_truncation"] for e in episodes))


def differences(groups):
    keys = ("mean_length", "mean_return", "mean_velocity", "falls")
    return {name: {k: groups[MODELS[2]][k]-groups[model][k] for k in keys}
            for name, model in (("retention_minus_control", MODELS[1]), ("retention_minus_reference", MODELS[0]))}


def summarize(rows):
    assert len(rows) == 3 and {r["training_seed"] for r in rows} == {0, 1, 2}
    records, noise_records, evaluation_steps = [], [], 0
    for row in sorted(rows, key=lambda r: r["training_seed"]):
        seed, spec = row["training_seed"], stage_spec(row["training_seed"])
        assert row["revision"] == REVISION and row["spec"] == spec and row["evaluation_device"] == "cpu"
        assert row["optimizer_updates"] == row["new_training_interactions"] == 0
        assert not row["source_readout_refitted"] and row["reward_or_physics_changes"] == []
        assert row["checkpoint_selection"] == "predeclared-final-only"
        sources = row["sources"]
        assert [s["model"] for s in sources] == list(MODELS)
        assert [s["age"] for s in sources] == [300000, 350000, 350000]
        assert [s["revision"] for s in sources] == ["general_ood_v123_source_extension",
            "general_ood_v128_mean_retention", "general_ood_v128_mean_retention"]
        assert [s["arm"] for s in sources] == [None, "unchanged-control", "mean-only-source-regularization"]
        assert all(s["config"] == sources[0]["config"] for s in sources)
        evaluations, expected = row["evaluations"], cells(spec)
        assert [{k: e[k] for k in expected[0]} for e in evaluations] == expected
        assert row["deterministic_baseline_matches"] == 15
        for e in evaluations:
            assert e["baseline_match"] is (True if e["mode"] == "deterministic" else None)
            assert e["noise_scale"] == (0. if e["mode"] == "deterministic" else 1.)
            assert 0 < e["length"] <= spec["max_steps"] and not e["diagnostic_cutoff"]
            assert e["physical_termination"] or e["time_limit_truncation"]
            assert all(np.isfinite(e[k]) for k in ("episode_return", "mean_velocity"))
        steps = sum(e["length"] for e in evaluations)
        assert steps == row["evaluation_interactions"]
        evaluation_steps += steps
        modes = {}
        for mode in ("deterministic", "sampled"):
            models = {m: aggregate([e for e in evaluations if e["model"] == m and e["mode"] == mode])
                      for m in MODELS}
            modes[mode] = dict(models=models, **differences(models))
        records.append(dict(training_seed=seed, modes=modes))
        for noise in spec["noise_seeds"]:
            models = {m: aggregate([e for e in evaluations if e["model"] == m and e["noise_seed"] == noise])
                      for m in MODELS}
            noise_records.append(dict(training_seed=seed, noise_seed=noise, models=models, **differences(models)))
    return dict(revision=REVISION, independent_training_seeds=3, frozen_models=9,
        evaluation_episodes=945, evaluation_interactions=evaluation_steps, deterministic_baseline_matches=45,
        optimizer_updates=0, new_training_interactions=0, per_training_seed=records,
        per_noise=noise_records,
        interpretation="Twenty paired new noise schedules per training seed; resets/noise are not independent training seeds. Normal Hopper only.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads((args.result_dir/f"seed{seed}/heldout.json").read_text()) for seed in (0, 1, 2)]
    result = summarize(rows)
    (args.result_dir/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result["per_training_seed"], indent=2))


if __name__ == "__main__":
    main()
