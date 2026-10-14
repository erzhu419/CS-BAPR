"""Validate full-state source lineage and compare both budget-extended arms."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_general_ood_v133_source_extension as worker
import summarize_general_ood_v132 as source_summary

OUTCOMES = ("mean_return", "mean_episode_velocity", "mean_episode_length", "physical_terminations")


def summarize(sources, extensions):
    source_summary.summarize(sources)
    cells = [(seed, arm) for seed in worker.v132.SEEDS for arm in worker.v132.ARMS]
    assert len(extensions) == len(cells)
    originals = {(r["training_seed"], r["arm"]): r for r in sources}
    indexed = {(r["training_seed"], r["arm"]): r for r in extensions}
    assert set(indexed) == set(cells)
    records, pairs = [], []
    for seed, arm in cells:
        row, source = indexed[seed, arm], originals[seed, arm]
        json.dumps(row, allow_nan=False)
        assert row["revision"] == worker.REVISION and not row["smoke"]
        assert row["spec"] == worker.stage_spec(arm) and row["training_driver_revision"] == worker.v132.REVISION
        assert row["source_revision"] == source["revision"] and row["source_checkpoint"] == source["resumable_checkpoint"]
        assert row["source_runtime"] == source["runtime"] and row["config"] == source["config"]
        assert all(row["runtime"][k] == source["runtime"][k] for k in source["runtime"] if k != "gpu")
        assert row["reward_settings"] == source["reward_settings"] and row["effective_beta"] == -2.
        assert not row["trained_from_scratch"] and row["training_mode"] == "normal-only"
        assert row["checkpoint_selection"] == "final-not-best"
        assert not row["periodic_evaluation_selects_checkpoint"] and not row["source_readout_refitted"]
        assert row["optimizer_updates_during_evaluation"] == 0
        assert row["source_interactions"] == 100000
        assert row["actual_new_training_interactions"] == row["actual_new_optimizer_updates"] == 200000
        training, final = row["training"], row["final_evaluation"]
        assert training["interactions"] == 300000 and training["optimizer_updates"] == 299745
        blocks = training["diagnostic_blocks"]
        assert [b["step"] for b in blocks] == list(range(5000, 300001, 5000))
        assert sum(b["interactions"] for b in blocks) == 300000 and sum(b["updates"] for b in blocks) == 299745
        assert blocks[:20] == source["training"]["diagnostic_blocks"]
        evaluations = training["periodic_evaluations"]
        assert [e["step"] for e in evaluations] == [0, 25000, 50000, 75000, *range(125000, 300000, 25000)]
        assert evaluations[:4] == source["training"]["periodic_evaluations"]
        episodes = final["episodes"]
        assert [e["seed"] for e in episodes] == list(range(150000, 150005))
        assert all(0 < e["length"] <= 1000 and
                   (e["physical_termination"] or e["time_limit_truncation"]) for e in episodes)
        for field, episode_field in (("mean_return", "episode_return"), ("mean_episode_velocity", "mean_forward_velocity"),
                                     ("mean_episode_length", "length")):
            assert abs(final[field]-sum(e[episode_field] for e in episodes)/len(episodes)) < 1e-9
        assert final["physical_terminations"] == sum(e["physical_termination"] for e in episodes)
        assert final["environment_step_calls"] == sum(e["length"] for e in episodes)
        new_blocks = blocks[20:]
        assert sum(b["interactions"] for b in new_blocks) == sum(b["updates"] for b in new_blocks) == 200000
        before = source["final_evaluation"]
        records.append(dict(training_seed=seed, arm=arm, source_checkpoint=row["source_checkpoint"],
            checkpoint=row["checkpoint"], resumable_checkpoint=row["resumable_checkpoint"],
            runtime=row["runtime"], source_runtime=source["runtime"], seconds=row["seconds"],
            at100k={k: before[k] for k in OUTCOMES}, at300k={k: final[k] for k in OUTCOMES},
            budget_change={k: final[k]-before[k] for k in OUTCOMES}, final_evaluation=final,
            new_training_time_limits=sum(b["time_limit_truncations"] for b in new_blocks),
            last5k_metrics=blocks[-1]["metrics"],
            peak_logged_new_block_mean_td_mse=max(b["metrics"]["td_mse"] for b in new_blocks),
            periodic_evaluations=evaluations))
    for seed in worker.v132.SEEDS:
        control, pilot = (indexed[seed, arm]["final_evaluation"] for arm in worker.v132.ARMS)
        pairs.append(dict(training_seed=seed, huber_minus_mse={k: pilot[k]-control[k] for k in OUTCOMES}))
    return dict(revision=worker.REVISION, independent_training_seeds=2, final_models=4,
        actual_new_training_interactions=800000, actual_new_optimizer_updates=800000,
        reused_source_interactions=400000, total_lineage_interactions=1200000, total_lineage_optimizer_updates=1198980,
        final_evaluation_episodes=20, source_readout_refitted=False,
        records=records, paired_differences=pairs,
        scope="Two training-seed pairs in normal Hopper; five reset repeats are not training seeds. Final-only models, not selected from periodic peaks. Block metrics are update means, not global peak certificates.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    def read(directory):
        return [json.loads((directory/f"seed{seed}_{arm}/source.json").read_text())
                for seed in worker.v132.SEEDS for arm in worker.v132.ARMS]
    result = summarize(read(args.source_dir), read(args.result_dir))
    (args.result_dir/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    for row in result["records"]:
        final = row["at300k"]
        print(f"seed{row['training_seed']} {row['arm']}: return={final['mean_return']:.2f} "
              f"length={final['mean_episode_length']:.1f} falls={final['physical_terminations']}/5 "
              f"td_mse={row['last5k_metrics']['td_mse']:.4f}")


if __name__ == "__main__":
    main()
