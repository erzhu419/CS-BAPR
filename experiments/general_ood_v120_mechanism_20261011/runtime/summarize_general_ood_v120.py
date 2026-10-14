"""Compact mechanism diagnostics; noise draws are not extra environment seeds."""
import argparse
import json
from pathlib import Path

import numpy as np

import analyze_general_ood_v120_mechanism as worker


def run(directory):
    shards = [json.loads((directory/f"seed{s}/mechanism.json").read_text()) for s in worker.v119.SEEDS]
    for seed, row in zip(worker.v119.SEEDS, shards):
        assert row["passed"] and row["revision"] == worker.REVISION and row["spec"]["seed"] == seed
        assert not row["spec"]["smoke"] and row["optimizer_updates"] == 0
        assert row["existing_trace_paths"] == 20 and len(row["source_evaluation"]) == 5
        assert row["reward_settings"] == shards[0]["reward_settings"]
    fields = ("mean_forward_velocity", "mean_control_cost", "mean_reward", "mean_risk", "fraction_abs_velocity_below_point1")
    trace_summary = {}
    for arm in worker.v119.ARMS:
        trace_summary[arm] = {}
        for group in ("normal", "changed"):
            rows = [r["windows"]["post"] for s in shards for r in s["trace_records"]
                    if r["arm"] == arm and (r["mode"] == "normal") == (group == "normal")]
            trace_summary[arm][group] = {f: float(np.mean([r[f] for r in rows])) for f in fields}
    source = {str(s["spec"]["seed"]): {r["mode"]: {k: r[k] for k in ("mean_return", "mean_episode_velocity")}
        for r in s["source_evaluation"]} for s in shards}
    gradients = {}
    for label in ("source", *worker.v119.ARMS):
        gradients[label] = {}
        for pool in ("prechange", "post_compound"):
            gradients[label][pool] = {field: [float(np.mean([r[field] for r in s["gradients"][label][pool]]))
                for s in shards] for field in ("mean_q_std", "alpha", "unit_beta_increment_over_base_gradient",
                    "beta_minus2_vs_minus3_gradient_cosine", "reward_normalization_std")}
    output = dict(revision=worker.REVISION, passed=True, existing_trace_paths=60, optimizer_updates=0,
        evaluation_interactions=sum(s["evaluation_interactions"] for s in shards), trace_summary=trace_summary,
        source_evaluation=source, gradient_noise_means_by_environment_seed=gradients, readout_adopted=False,
        limitations=shards[0]["limitations"], decision="mechanism-diagnosis-not-new-algorithm-performance")
    path = directory/"summary.json"
    path.write_text(json.dumps(output, indent=2, allow_nan=False)+"\n")
    print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    run(parser.parse_args().result_dir)
