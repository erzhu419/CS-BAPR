"""Separate anchor, new reset and action-noise outcomes for each trained policy."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_general_ood_v134_frozen_robustness as worker
from summarize_general_ood_v129 import aggregate

GROUPS = (("anchor", "deterministic"), ("new-reset", "deterministic"), ("new-reset", "sampled"))


def summarize(rows, sources):
    cells = [(seed, arm) for seed in worker.v133.v132.SEEDS for arm in worker.v133.v132.ARMS]
    assert len(rows) == len(sources) == len(cells)
    indexed = {(r["training_seed"], r["arm"]): r for r in rows}
    originals = {(r["training_seed"], r["arm"]): r for r in sources}
    assert set(indexed) == set(originals) == set(cells)
    records, total_steps = [], 0
    for seed, arm in cells:
        row, original = indexed[seed, arm], originals[seed, arm]
        json.dumps(row, allow_nan=False)
        assert row["revision"] == worker.REVISION and row["spec"] == worker.stage_spec(seed, arm) and not row["smoke"]
        assert row["evaluation_device"] == "cpu" and not row["cuda_initialized"] and not row["replay_loaded"]
        assert row["optimizer_updates"] == row["new_training_interactions"] == 0
        assert row["checkpoint_selection"] == "predeclared-final-only" and not row["source_readout_refitted"]
        assert row["reward_or_physics_changes"] == [] and row["cpu_anchor_requires_gpu_trajectory_parity"] is False
        source = row["source"]
        assert original["revision"] == source["revision"] == worker.v133.REVISION and not original["smoke"]
        assert source["checkpoint"] == original["checkpoint"] and original["checkpoint_selection"] == "final-not-best"
        assert original["training"]["interactions"] == source["context_count"] == 300000
        assert original["training"]["optimizer_updates"] == source["training_steps"] == 299745
        assert source["initial_context_normalizer"]["count"] == 300000
        assert source["config"] == dict(original["config"], device="cpu")
        assert source["runtime"] == original["runtime"] and source["gpu_anchor_evaluation"] == original["final_evaluation"]
        assert all(row["runtime"][k] == original["runtime"][k] for k in row["runtime"])
        episodes, expected = row["evaluations"], worker.cells(row["spec"])
        assert [{k: e[k] for k in expected[0]} for e in episodes] == expected
        for e in episodes:
            assert e["noise_scale"] == (0. if e["mode"] == "deterministic" else 1.)
            assert 0 < e["length"] <= 1000 and not e["diagnostic_cutoff"]
            assert e["physical_termination"] or e["time_limit_truncation"]
        steps = sum(e["length"] for e in episodes)
        assert row["evaluation_interactions"] == steps
        total_steps += steps
        groups = {}
        for group, mode in GROUPS:
            subset = [e for e in episodes if (e["group"], e["mode"]) == (group, mode)]
            groups[f"{group}/{mode}"] = dict(aggregate(subset), minimum_length=min(e["length"] for e in subset),
                mean_policy_std=sum(e["mean_policy_std"] for e in subset)/len(subset),
                mean_action_noise_l2=sum(e["mean_action_noise_l2"] for e in subset)/len(subset))
        noise_groups = {str(noise): aggregate([e for e in episodes if e["noise_seed"] == noise])
                        for noise in row["spec"]["action_noise_seeds"]}
        anchor, gpu = groups["anchor/deterministic"], original["final_evaluation"]
        records.append(dict(training_seed=seed, arm=arm, checkpoint=source["checkpoint"], seconds=row["seconds"],
            groups=groups, per_noise=noise_groups, evaluation_interactions=steps,
            cpu_minus_gpu_anchor=dict(mean_length=anchor["mean_length"]-gpu["mean_episode_length"],
                mean_return=anchor["mean_return"]-gpu["mean_return"], falls=anchor["falls"]-gpu["physical_terminations"])))
    paired = []
    for seed in worker.v133.v132.SEEDS:
        control, pilot = [next(r for r in records if (r["training_seed"], r["arm"]) == (seed, arm))
                          for arm in worker.v133.v132.ARMS]
        paired.append(dict(training_seed=seed, huber_minus_mse={group: {key: pilot["groups"][group][key]-control["groups"][group][key]
            for key in ("mean_length", "mean_return", "mean_velocity", "falls")} for group in control["groups"]}))
    return dict(revision=worker.REVISION, independent_training_seeds=2, frozen_models=4,
        evaluation_episodes=340, evaluation_interactions=total_steps, optimizer_updates=0, new_training_interactions=0,
        records=records, paired_differences=paired,
        scope="Normal Hopper reset/action-noise robustness; only two training seeds. No OOD or baseline superiority claim. CPU anchors retain rather than reject numerical trajectory differences from GPU anchors.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    cells = [(seed, arm) for seed in worker.v133.v132.SEEDS for arm in worker.v133.v132.ARMS]
    rows = [json.loads((args.result_dir/f"seed{seed}_{arm}/robustness.json").read_text()) for seed, arm in cells]
    sources = [json.loads((args.source_dir/f"seed{seed}_{arm}/source.json").read_text()) for seed, arm in cells]
    result = summarize(rows, sources)
    (args.result_dir/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    for row in result["records"]:
        sampled = row["groups"]["new-reset/sampled"]
        print(f"seed{row['training_seed']} {row['arm']}: sampled_length={sampled['mean_length']:.2f} "
              f"return={sampled['mean_return']:.2f} falls={sampled['falls']}/{sampled['episodes']}")


if __name__ == "__main__":
    main()
