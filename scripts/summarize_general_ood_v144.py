"""Matched-budget standard-SAC learnability evidence, retaining seed separation."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_general_ood_v144_fixed_profile_sac as worker
from summarize_general_ood_v129 import aggregate


def validate(row, references):
    json.dumps(row, allow_nan=False)
    seed, profile = row["training_seed"], row["training_profile"]
    assert seed in (3, 4) and profile in worker.PROFILES
    spec = worker.stage_spec(profile)
    assert row["revision"] == worker.REVISION and not row["smoke"] and row["spec"] == spec
    assert row["config"] == worker.SETTINGS and row["training_device"] == "cuda" and row["runtime"]["stable_baselines3"] == "2.7.0"
    assert row["cumulative_training_interactions"] == 600000 and row["optimizer_updates"] == 599745
    assert row["observation_dim"] == 11 and not any(row[k] for k in ("use_context", "use_symbolic_constraints", "reward_normalization", "policy_receives_profile_labels"))
    assert row["reward_changes"] == row["healthy_termination_changes"] == []
    assert row["bootstrap_mask"] == "terminated-only" and row["replay_handle_timeout_termination"]
    assert row["optimizer_updates_during_evaluation"] == 0 and row["checkpoint_selection"] == "predeclared300k600k-not-best"
    training = row["training"]
    assert training["step"] == 600000 and training["snapshots"] == row["snapshots"]
    assert [b["step"] for b in training["blocks"]] == list(range(5000, 600001, 5000))
    assert sum(b["interactions"] for b in training["blocks"]) == 600000
    assert sum(b["updates"] for b in training["blocks"]) == 599745
    assert [e["step"] for e in training["periodic_evaluations"]] == list(range(0, 600000, 25000))
    assert [s["step"] for s in row["snapshots"]] == [300000, 600000]
    for snapshot in row["snapshots"]:
        steps = snapshot["step"]
        reference = references[steps, seed, profile]
        assert (reference["training_seed"], reference["training_profile"]) == (seed, profile)
        assert reference["revision"] == ("general_ood_v141_actor_rate" if steps == 300000 else "general_ood_v142_budget_extension")
        assert reference["training"]["interactions"] == steps and reference["training"]["optimizer_updates"] == snapshot["optimizer_updates"] == steps-255
        assert row["physics"] == reference["physics"] and row["reward_settings"] == reference["reward_settings"]
        assert Path(snapshot["checkpoint"]).name == f"step{steps}.zip"
        final = snapshot["final_evaluation"]
        protocol = dict(reference["final_evaluation"]["protocol"], context_mode="not-applicable")
        assert final["protocol"] == protocol and final["profile"] == profile and final["device"] == "cpu"
        assert final["optimizer_updates"] == 0 and not final["replay_loaded"]
        expected = [{k: e[k] for k in ("condition", "mode", "reset_seed", "noise_seed")} for e in reference["final_evaluation"]["episodes"]]
        episodes = final["episodes"]
        assert [{k: e[k] for k in expected[0]} for e in episodes] == expected
        for e in episodes:
            assert 0 < e["length"] <= 1000 and not e["diagnostic_cutoff"]
            assert e["physical_termination"] or e["time_limit_truncation"]
            assert e["native_reward_verified"] and e["context_updates"] == 0 and e["context_mode"] == "not-applicable"
            assert e["noise_scale"] == (0. if e["mode"] == "deterministic" else 1.)
            assert e["profile_counts"] == {profile: e["length"]} and e["switches"] == 0
            assert e["profile_events"][0]["actual_multipliers"] == row["physics"]["actual_multipliers"]
        assert final["evaluation_interactions"] == sum(e["length"] for e in episodes)
    return row["snapshots"]


def summarize(rows, references):
    expected = {(s, p) for s in (3, 4) for p in worker.PROFILES}
    assert len(rows) == 6 and {(r["training_seed"], r["training_profile"]) for r in rows} == expected
    records = []
    for row in sorted(rows, key=lambda r: (r["training_seed"], worker.PROFILES.index(r["training_profile"]))):
        seed, profile = row["training_seed"], row["training_profile"]
        snapshots = []
        for snapshot in validate(row, references):
            episodes = snapshot["final_evaluation"]["episodes"]
            reference = references[snapshot["step"], seed, profile]
            modes = {mode: dict(sac=aggregate([e for e in episodes if e["mode"] == mode]),
                prior_learner=aggregate([e for e in reference["final_evaluation"]["episodes"] if e["mode"] == mode]))
                for mode in ("deterministic", "sampled")}
            snapshots.append(dict(step=snapshot["step"], checkpoint=snapshot["checkpoint"], modes=modes,
                terminal_reason_sets=dict(Counter("+".join(e["final_physical_state"]["unhealthy_conditions"]) for e in episodes if e["physical_termination"]))))
        records.append(dict(seed=seed, profile=profile, seconds=row["seconds"], snapshots=snapshots,
            last_block_metrics=row["training"]["blocks"][-1]["metrics"]))
    return dict(revision=worker.REVISION, independent_training_seeds=2, training_interactions=3600000,
        optimizer_updates=3598470, evaluation_episodes=240,
        evaluation_interactions=sum(s["final_evaluation"]["evaluation_interactions"] for r in rows for s in r["snapshots"]),
        records=records, scope="Standard learner positive control with matched physics/seeds/budgets; implementation differences preclude algorithm-only attribution. Reset/noise repeats are not training seeds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=Path("results/general_ood_v144_fixed_profile_sac"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    rows = [json.loads((args.result_dir/f"seed{s}_{p}/source.json").read_text()) for s in (3, 4) for p in worker.PROFILES]
    references = {(steps, s, p): json.loads((root/f"results/{name}/seed{s}_{p}/source.json").read_text())
        for steps, name in ((300000, "general_ood_v141_actor_rate"), (600000, "general_ood_v142_budget_extension"))
        for s in (3, 4) for p in worker.PROFILES}
    result = summarize(rows, references)
    result["formal_json_bytes"] = sum((args.result_dir/f"seed{s}_{p}/source.json").stat().st_size for s in (3, 4) for p in worker.PROFILES)
    (args.result_dir/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    for r in result["records"]:
        print(r["seed"], r["profile"], [(s["step"], s["modes"]["sampled"]) for s in r["snapshots"]])
