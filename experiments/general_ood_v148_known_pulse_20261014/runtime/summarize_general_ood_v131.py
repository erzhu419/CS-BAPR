"""Merge the six complete frozen source diagnostics, retaining each source."""
import argparse
import json
from pathlib import Path

SOURCE_CELLS = ((3, 100000), (4, 100000), (1, 300000), (2, 300000), (3, 300000), (4, 300000))
NOISE_SEEDS = [13100, 13101, 13102]


def summarize(result_dir):
    records, runtime = [], None
    for seed, age in SOURCE_CELLS:
        path = result_dir/f"seed{seed}_age{age}"/"source_diagnostics.json"
        row = json.loads(path.read_text())
        assert row["revision"] == "general_ood_v131_source_diagnostics" and not row["smoke"]
        assert (row["training_seed"], row["source_steps"]) == (seed, age)
        assert row["optimizer_updates"] == row["environment_step_calls"] == row["new_training_interactions"] == 0
        assert row["gradient_evaluation_only"] and not row["source_readout_refitted"]
        assert row["checkpoint_selection"] == "predeclared-final-full-state"
        assert row["noise_seeds"] == NOISE_SEEDS
        assert row["runtime"]["device"] == "cpu" and row["runtime"]["cuda_initialized"] is False
        if runtime is None:
            runtime = row["runtime"]
        assert row["runtime"] == runtime
        assert set(row["pools"]) == {"uniform_training", "last10k"}
        pools = {}
        for name, cells in row["pools"].items():
            assert [cell["noise_seed"] for cell in cells] == NOISE_SEEDS
            for cell in cells:
                assert cell["samples"] == cell["residuals"]["transitions"] == 4096
                assert cell["batches"] == 16 and len(cell["per_critic"]) == 5
                assert cell["native_critic_loss_reconstructed"]
                assert cell["replay_done"]["transitions"]+cell["replay_not_done"]["transitions"] == 4096
            mean = lambda values: sum(values)/len(values)
            metrics = dict(mse=mean([c["residuals"]["mse"] for c in cells]),
                abs_td_p50=mean([c["residuals"]["abs_td"]["p50"] for c in cells]),
                abs_td_p99=mean([c["residuals"]["abs_td"]["p99"] for c in cells]),
                top1pct_loss_share=mean([c["residuals"]["top1pct_loss_share"] for c in cells]),
                top10pct_loss_share=mean([c["residuals"]["top10pct_loss_share"] for c in cells]),
                replay_done_fraction=cells[0]["replay_done"]["transitions"]/4096,
                replay_done_loss_share=mean([c["replay_done"].get("mse", 0.)*c["replay_done"]["transitions"]/
                    (c["residuals"]["mse"]*4096) if c["residuals"]["mse"] else 0. for c in cells]))
            for key in ("predicted_q", "target", "discounted_bootstrap_q", "discounted_entropy",
                        "normalized_reward", "policy_log_prob", "policy_std_mean"):
                metrics[key+"_mean"] = mean([c["scales"][key]["mean"] for c in cells])
                metrics[key+"_p99"] = mean([c["scales"][key]["p99"] for c in cells])
            metrics["batch_metrics"] = {key: dict(mean=mean([c["batch_metrics"][key]["mean"] for c in cells]),
                max=max(c["batch_metrics"][key]["max"] for c in cells)) for key in cells[0]["batch_metrics"]}
            pools[name] = metrics
        records.append(dict(training_seed=seed, source_steps=age, alpha=row["alpha"],
                            checkpoint=row["checkpoint"], pools=pools))
    return dict(revision="general_ood_v131_source_diagnostics", source_states=6,
        independent_training_seeds=4, diagnostic_cells=36, replay_rows_with_noise_repeats=147456,
        optimizer_updates=0, environment_step_calls=0, new_training_interactions=0,
        runtime=runtime, records=records,
        scope="Frozen endpoint diagnosis, not training causality or matched cross-source effects.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.result_dir)
    (args.result_dir/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    for row in result["records"]:
        pool = row["pools"]["uniform_training"]
        print(f"seed{row['training_seed']} age{row['source_steps']}: mse={pool['mse']:.4g} "
              f"top1pct={pool['top1pct_loss_share']:.3f} done_loss={pool['replay_done_loss_share']:.3f} "
              f"Q={pool['predicted_q_mean']:.3f}")


if __name__ == "__main__":
    main()
