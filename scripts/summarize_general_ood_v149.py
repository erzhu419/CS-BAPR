"""Full-state matched continuations, preserving checkpoint and seed separation."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_general_ood_v149_warm_recovery as worker
from summarize_general_ood_v147 import metrics, validate_final

CASES = tuple(f"seed{s}_{a}" for s in (5, 6) for a in worker.ARMS)


def validate(row, source):
    json.dumps(row, allow_nan=False)
    seed, arm = row["training_seed"], row["training_profile"]
    assert seed in (5, 6) and arm in worker.ARMS and row["revision"] == worker.REVISION and not row["smoke"]
    assert row["spec"] == worker.stage_spec(arm) and row["config"] == worker.backend.SETTINGS
    assert row["source"] == worker.validate_source(source, seed)
    assert row["training_device"] == "cuda" and row["runtime"]["stable_baselines3"] == "2.7.0"
    assert row["cumulative_training_interactions"] == 700000 and row["optimizer_updates"] == 699745
    assert row["continuation_training_interactions"] == row["continuation_optimizer_updates"] == 100000
    assert 0 <= row["actual_new_training_interactions"] <= 100000
    assert row["optimizer_updates_during_evaluation"] == row["state_resets_at_switch"] == 0
    assert row["reward_changes"] == row["healthy_termination_changes"] == row["motor_limit_changes"] == []
    assert row["reward_settings"] == source["reward_settings"]
    assert row["observation_dim"] == 11 and not any(row[k] for k in ("use_context", "use_symbolic_constraints", "reward_normalization", "policy_receives_profile_labels", "policy_receives_onset_clock"))
    assert row["bootstrap_mask"] == "terminated-only" and row["replay_handle_timeout_termination"]
    receipt = row["initialization"]
    assert receipt["step"] == receipt["replay_size"] == receipt["replay_position"] == 600000
    assert receipt["optimizer_updates"] == 599745 and receipt["replay_capacity"] == 1000000 and not receipt["replay_full"]
    assert receipt["optimizer_steps"] == dict(actor=[599745], critic=[599745], entropy=[599745])
    assert receipt["source_coverage"] == source["recovery_coverage"] and receipt["onset"] is None and receipt["active_profile"] == "id"
    t = row["training"]
    assert t["step"] == 700000 and t["initialization"] == receipt and t["source"] == row["source"] and t["snapshots"] == row["snapshots"]
    assert [b["step"] for b in t["blocks"]] == list(range(605000, 700001, 5000))
    assert sum(b["interactions"] for b in t["blocks"]) == sum(b["updates"] for b in t["blocks"]) == 100000
    assert [e["step"] for e in t["periodic_evaluations"]] == [600000, 625000, 650000, 675000]
    assert row["checkpoint_selection"] == "predeclared650k700k-not-best"
    c = row["recovery_coverage"]
    assert c["baseline_steps"]+c.get("pulse_steps", 0) == 100000 and t["blocks"][-1]["recovery_coverage"] == c
    if arm == "no-pulse":
        assert c.get("pulse_steps", 0) == c["pulse_assigned_episodes"] == 0
    assert [s["step"] for s in row["snapshots"]] == [650000, 700000]
    for snapshot in row["snapshots"]:
        assert snapshot["optimizer_updates"] == snapshot["step"]-255 and Path(snapshot["checkpoint"]).name == f"step{snapshot['step']}.zip"
        validate_final(snapshot["final_evaluation"])


def summarize(rows, sources):
    assert len(rows) == 4 and {f"seed{r['training_seed']}_{r['training_profile']}" for r in rows} == set(CASES)
    records, matched = [], []
    for row in sorted(rows, key=lambda r: (r["training_seed"], r["training_profile"])):
        validate(row, sources[row["training_seed"]])
        snapshots = []
        for snapshot in row["snapshots"]:
            episodes = snapshot["final_evaluation"]["episodes"]
            baseline = {(e["reset_seed"], e["noise_seed"]): e for e in episodes if e["condition"] == "no-pulse"}
            conditions = [dict(schedule=s, modes={mode: metrics([e for e in episodes if e["condition"] == s["name"] and e["mode"] == mode], baseline)
                for mode in ("deterministic", "sampled")}) for s in worker.training.evaluation.schedules()]
            snapshots.append(dict(step=snapshot["step"], checkpoint=snapshot["checkpoint"], conditions=conditions))
        records.append(dict(case=f"seed{row['training_seed']}_{row['training_profile']}", seed=row["training_seed"], arm=row["training_profile"],
            source=row["source"], initialization=row["initialization"], seconds=row["seconds"], training_coverage=row["recovery_coverage"],
            final_segment_interactions=row["actual_new_training_interactions"], snapshots=snapshots))
    for seed in (5, 6):
        left, right = (next(r for r in records if r["seed"] == seed and r["arm"] == a) for a in worker.ARMS)
        assert left["source"] == right["source"] and left["initialization"] == right["initialization"]
        for ls, rs in zip(left["snapshots"], right["snapshots"]):
            for lc, rc in zip(ls["conditions"], rs["conditions"]):
                matched.append(dict(seed=seed, step=ls["step"], condition=lc["schedule"]["name"],
                    modes={mode: {k: rc["modes"][mode][k]-lc["modes"][mode][k] for k in
                        ("mean_length", "mean_return", "healthy_full_survival", "recovered", "delayed_falls_after_recovery")}
                        for mode in ("deterministic", "sampled")}))
    return dict(revision=worker.REVISION, paired_source_seeds=2, source_training_interactions_once_per_seed=1200000,
        new_training_interactions=400000, new_optimizer_updates=400000, evaluation_episodes=1440, prefix_pairs_verified=1280,
        evaluation_interactions=sum(s["final_evaluation"]["evaluation_interactions"] for r in rows for s in r["snapshots"]),
        records=records, matched_randomized_minus_control=matched,
        scope="Complete matched warm continuation; source prefixes are shared, not four independent seeds. Retain both predefined snapshots and stochastic/deterministic modes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=Path("results/general_ood_v149_warm_recovery"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    rows = [json.loads((args.result_dir/case/"source.json").read_text()) for case in CASES]
    sources = {seed: json.loads((root/f"results/general_ood_v147_randomized_recovery/seed{seed}_no-pulse/source.json").read_text()) for seed in (5, 6)}
    summary = summarize(rows, sources)
    summary["formal_json_bytes"] = sum((args.result_dir/case/"source.json").stat().st_size for case in CASES)
    (args.result_dir/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    for r in summary["records"]:
        print(r["case"],[(s["step"], [(c["schedule"]["name"],round(c["modes"]["sampled"]["mean_length"],2),c["modes"]["sampled"]["healthy_full_survival"])
            for c in s["conditions"]]) for s in r["snapshots"]])
