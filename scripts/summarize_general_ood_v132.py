"""Paired source endpoints; training seeds are not evaluation reset repeats."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_general_ood_v132_source_stability as worker


def summarize(rows):
    cells = [(seed, arm) for seed in worker.SEEDS for arm in worker.ARMS]
    assert len(rows) == len(cells)
    indexed = {(r["training_seed"], r["arm"]): r for r in rows}
    assert set(indexed) == set(cells)
    records, pairs = [], []
    for seed, arm in cells:
        row = indexed[seed, arm]
        json.dumps(row, allow_nan=False)
        assert row["revision"] == worker.REVISION and not row["smoke"]
        assert row["spec"] == worker.stage_spec(arm)
        assert row["config"] == rows[0]["config"] and row["runtime"] == rows[0]["runtime"]
        assert row["reward_settings"] == rows[0]["reward_settings"]
        assert row["trained_from_scratch"] and row["training_mode"] == "normal-only"
        assert row["effective_beta"] == -2. and row["config"]["weight_reg"] == 0.
        assert row["checkpoint_selection"] == "final-not-best"
        assert not row["periodic_evaluation_selects_checkpoint"] and not row["source_readout_refitted"]
        assert row["optimizer_updates_during_evaluation"] == 0
        training, final = row["training"], row["final_evaluation"]
        assert training["interactions"] == 100000 and training["optimizer_updates"] == 99745
        blocks = training["diagnostic_blocks"]
        assert [b["step"] for b in blocks] == list(range(5000, 100001, 5000))
        assert sum(b["interactions"] for b in blocks) == training["interactions"]
        assert sum(b["updates"] for b in blocks) == training["optimizer_updates"]
        assert [e["step"] for e in training["periodic_evaluations"]] == [0, 25000, 50000, 75000]
        episodes = final["episodes"]
        assert [e["seed"] for e in episodes] == list(range(150000, 150005))
        assert all(0 < e["length"] <= 1000 and
                   (e["physical_termination"] or e["time_limit_truncation"]) for e in episodes)
        for field, episode_field in (("mean_return", "episode_return"),
                                     ("mean_episode_velocity", "mean_forward_velocity"),
                                     ("mean_episode_length", "length")):
            assert abs(final[field]-sum(e[episode_field] for e in episodes)/len(episodes)) < 1e-9
        assert final["physical_terminations"] == sum(e["physical_termination"] for e in episodes)
        assert final["environment_step_calls"] == sum(e["length"] for e in episodes)
        records.append(dict(training_seed=seed, arm=arm, checkpoint=row["checkpoint"],
            resumable_checkpoint=row["resumable_checkpoint"], seconds=row["seconds"],
            final_evaluation=final, training_time_limits=sum(b["time_limit_truncations"] for b in blocks),
            last5k_block=blocks[-1], peak_logged_block_mean_td_mse=max(b["metrics"]["td_mse"] for b in blocks)))
    for seed in worker.SEEDS:
        control, pilot = (indexed[seed, arm]["final_evaluation"] for arm in worker.ARMS)
        pairs.append(dict(training_seed=seed, huber_minus_mse={key: pilot[key]-control[key]
            for key in ("mean_return", "mean_episode_velocity", "mean_episode_length", "physical_terminations")}))
    return dict(revision=worker.REVISION, independent_training_seeds=2, final_models=4,
        interactions=sum(r["training"]["interactions"] for r in rows),
        optimizer_updates=sum(r["training"]["optimizer_updates"] for r in rows),
        final_evaluation_episodes=sum(len(r["final_evaluation"]["episodes"]) for r in rows),
        training_time_limits=sum(r["training_time_limits"] for r in records),
        final_physical_terminations=sum(r["final_evaluation"]["physical_terminations"] for r in rows),
        runtime=rows[0]["runtime"], config=rows[0]["config"], records=records, paired_differences=pairs,
        scope="Two training-seed pairs, normal Hopper only; no significance or OOD claim. Block metrics are means over updates, including fields named abs_max.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads((args.result_dir/f"seed{seed}_{arm}/source.json").read_text())
            for seed in worker.SEEDS for arm in worker.ARMS]
    result = summarize(rows)
    (args.result_dir/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    for row in result["records"]:
        final = row["final_evaluation"]
        print(f"seed{row['training_seed']} {row['arm']}: return={final['mean_return']:.2f} "
              f"velocity={final['mean_episode_velocity']:.4f} length={final['mean_episode_length']:.1f} "
              f"falls={final['physical_terminations']}/5 td_mse={row['last5k_block']['metrics']['td_mse']:.4f}")


if __name__ == "__main__":
    main()
