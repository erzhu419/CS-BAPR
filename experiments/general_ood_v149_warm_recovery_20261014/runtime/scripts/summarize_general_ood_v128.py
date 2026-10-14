"""Final-only paired mean-retention comparisons with separate noise groups."""
import argparse
import json
from pathlib import Path

import numpy as np

from run_general_ood_v128_mean_retention import ARMS, REVISION, evaluation_spec, stage_spec


def summarize(rows):
    assert len(rows) == 6 and {(r["training_seed"], r["arm"]) for r in rows} == {
        (seed, arm) for seed in (0, 1, 2) for arm in ARMS}
    records, evaluation_interactions = [], 0
    keys = ("mean_length", "mean_return", "mean_velocity", "falls")
    protocol = evaluation_spec()
    groups = {"deterministic": [None], **protocol["noise_groups"]}
    for seed in (0, 1, 2):
        pair = {r["arm"]: r for r in rows if r["training_seed"] == seed}
        control, retained = [pair[a] for a in ARMS]
        for key in ("config", "runtime", "source_checkpoint", "reference_checkpoint", "reward_settings"):
            assert control[key] == retained[key]
        arms = {}
        for arm, row in pair.items():
            assert row["revision"] == REVISION and not row["smoke"] and row["spec"] == stage_spec(arm)
            assert row["mean_retention_coefficient"] == ARMS[arm] and row["effective_beta"] == -2.
            assert row["source_revision"] == "general_ood_v123_source_extension" and row["source_interactions"] == 300000
            assert row["actual_new_training_interactions"] == row["actual_new_optimizer_updates"] == 50000
            assert row["training"]["interactions"] == 350000 and row["training"]["optimizer_updates"] == 349745
            assert row["config"]["weight_reg"] == 0. and row["final_target_scale"]["mean_discounted_l1_uplift"] == 0.
            assert row["checkpoint_selection"] == "final-not-best" and not row["periodic_evaluation_selects_checkpoint"]
            assert row["reference_frozen"] and not row["source_readout_refitted"]
            assert not row["current_std_directly_regularized"] and not row["initial_normalizer_swapped"]
            assert row["reward_or_physics_changes"] == []
            last = row["training"]["diagnostic_blocks"][-1]["metrics"]
            assert last["effective_beta"] == -2. and all(np.isfinite(last[k]) for k in (
                "mean_shift_kl", "mean_retention_loss", "reference_action_mse"))
            assert last["mean_retention_loss"] >= 0.
            if not ARMS[arm]:
                assert last["mean_retention_loss"] == 0.
            evaluation = row["final_evaluation"]
            assert evaluation["protocol"] == protocol and evaluation["device"] == "cpu"
            assert evaluation["optimizer_updates"] == 0 and set(evaluation["groups"]) == set(groups)
            for group, noises in groups.items():
                episodes = evaluation["groups"][group]["episodes"]
                assert [(e["reset_seed"], e["noise_seed"]) for e in episodes] == [
                    (reset, noise) for reset in protocol["reset_seeds"] for noise in noises]
                assert all(e["noise_scale"] == (0. if group == "deterministic" else 1.) for e in episodes)
                assert all(0 < e["length"] <= 1000 and not e["diagnostic_cutoff"] for e in episodes)
            steps = sum(e["length"] for g in evaluation["groups"].values() for e in g["episodes"])
            assert steps == evaluation["evaluation_interactions"]
            evaluation_interactions += steps
            arms[arm] = {group: {k: r[k] for k in keys} for group, r in evaluation["groups"].items()}
        change = {g: {k: arms["mean-only-source-regularization"][g][k]-arms["unchanged-control"][g][k]
                      for k in keys} for g in groups}
        records.append(dict(training_seed=seed, arms=arms, retention_minus_control=change))
    return dict(revision=REVISION, independent_training_seeds=3, formal_paths=6,
        actual_new_training_interactions=300000, actual_new_optimizer_updates=300000,
        reused_unique_source_interactions=900000, evaluation_episodes=300,
        evaluation_interactions=evaluation_interactions, optimizer_updates_during_evaluation=0,
        checkpoint_selection="final-not-best", per_training_seed=records,
        mean_change={g: {k: float(np.mean([r["retention_minus_control"][g][k] for r in records]))
                        for k in keys} for g in groups},
        interpretation="Matched normal-only continuation; mean-retention intervention, not an OOD comparison.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads((args.result_dir/f"seed{seed}_{arm}/mean_retention.json").read_text())
            for seed in (0, 1, 2) for arm in ARMS]
    result = summarize(rows)
    (args.result_dir/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result["per_training_seed"], indent=2))


if __name__ == "__main__":
    main()
