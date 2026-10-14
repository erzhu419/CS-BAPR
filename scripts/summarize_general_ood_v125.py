"""Paired final-only actor-beta continuation comparisons."""
import argparse
import json
from pathlib import Path

import numpy as np

from run_general_ood_v125_actor_beta import ARMS, REVISION, stage_spec, v124


def summarize(rows):
    assert len(rows) == 6 and {(r["training_seed"], r["arm"]) for r in rows} == {
        (seed, arm) for seed in (0, 1, 2) for arm in ARMS}
    records = []
    keys = ("mean_length", "mean_return", "mean_velocity", "falls")
    for seed in (0, 1, 2):
        pair = {r["arm"]: r for r in rows if r["training_seed"] == seed}
        control, zero = pair["beta-minus2"], pair["beta-zero"]
        assert control["config"] == zero["config"] and control["runtime"] == zero["runtime"]
        assert control["source_checkpoint"] == zero["source_checkpoint"]
        for arm, row in pair.items():
            assert row["revision"] == REVISION and not row["smoke"]
            assert row["spec"] == stage_spec(arm) and row["effective_beta"] == ARMS[arm]
            assert row["source_revision"] == "general_ood_v123_source_extension"
            assert row["source_interactions"] == 300000
            assert row["actual_new_training_interactions"] == row["actual_new_optimizer_updates"] == 50000
            assert row["training"]["interactions"] == 350000 and row["training"]["optimizer_updates"] == 349745
            assert row["training"]["diagnostic_blocks"][-1]["metrics"]["effective_beta"] == ARMS[arm]
            assert row["checkpoint_selection"] == "final-not-best" and not row["periodic_evaluation_selects_checkpoint"]
            assert not row["source_readout_refitted"] and row["final_target_scale"]["mean_discounted_l1_uplift"] == 0.
            evaluation = row["final_evaluation"]
            assert evaluation["protocol"] == v124.stage_spec() and evaluation["device"] == "cpu"
            assert evaluation["optimizer_updates"] == 0
            for mode, noises in (("deterministic", (None,)), ("sampled", v124.NOISE_SEEDS)):
                assert [(e["reset_seed"], e["noise_seed"]) for e in evaluation["modes"][mode]["episodes"]] == [
                    (reset, noise) for reset in range(150000, 150005) for noise in noises]
        arms = {arm: {mode: {k: r[k] for k in keys} for mode, r in row["final_evaluation"]["modes"].items()}
                for arm, row in pair.items()}
        change = {mode: {k: arms["beta-zero"][mode][k]-arms["beta-minus2"][mode][k] for k in keys}
                  for mode in ("deterministic", "sampled")}
        records.append(dict(training_seed=seed, arms=arms, beta_zero_minus_control=change))
    return dict(revision=REVISION, independent_training_seeds=3, formal_paths=6,
        actual_new_training_interactions=300000, actual_new_optimizer_updates=300000,
        reused_unique_source_interactions=900000, checkpoint_selection="final-not-best",
        per_training_seed=records, mean_change={mode: {k: float(np.mean([
            r["beta_zero_minus_control"][mode][k] for r in records])) for k in keys}
            for mode in ("deterministic", "sampled")}, source_readout_refitted=False,
        interpretation="matched 50k continuation; actor beta only, not a vanilla-SAC or OOD comparison")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads((args.result_dir/f"seed{seed}_{arm}/actor_beta.json").read_text())
            for seed in (0, 1, 2) for arm in ARMS]
    result = summarize(rows)
    (args.result_dir/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result["per_training_seed"], indent=2))


if __name__ == "__main__":
    main()
