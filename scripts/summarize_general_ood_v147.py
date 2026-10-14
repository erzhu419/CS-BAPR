"""Matched recovery-training evidence, retaining seed-specific regressions."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_general_ood_v147_randomized_recovery as worker
from summarize_general_ood_v146 import group_metrics

CASES = tuple(f"seed{s}_{a}" for s in (5, 6) for a in worker.ARMS)


def validate_episode(e, schedule, peak):
    assert e["dt"] == .008 and 0 < e["length"] <= 1000 and not e["diagnostic_cutoff"]
    assert e["physical_termination"] or e["time_limit_truncation"]
    assert e["native_reward_verified"] and e["context_mode"] == "not-applicable" and e["context_updates"] == 0
    assert e["noise_scale"] == (0. if e["mode"] == "deterministic" else 1.)
    onset, duration = schedule["onset"], schedule["duration"]
    steps = 0 if onset is None else max(e["length"]-onset, 0)
    if duration is not None:
        steps = min(steps, duration)
    counts = {"id": e["length"]-steps, "gravity-only": steps}
    assert e["profile_counts"] == {k: v for k, v in counts.items() if v}
    assert e["non_id_steps"] == steps
    events = [(1, "id", 1.)]
    if steps:
        events.append((onset+1, "gravity-only", peak))
    if onset is not None and duration is not None and e["length"] > onset+duration:
        events.append((onset+duration+1, "id", 1.))
    assert e["profile_events"] == [dict(step=t, profile=p, actual_multipliers=dict(body_mass=1., gravity=g, friction=1.)) for t, p, g in events]
    assert e["switches"] == len(events)-1
    assert e["recovery"] == worker.evaluation.recovery(e, schedule)


def validate_prefix(e, baseline, onset):
    left = next((m for m in e["prefix_markers"] if m["completed_steps"] == onset), None)
    right = next((m for m in baseline["prefix_markers"] if m["completed_steps"] == onset), None)
    assert left == right
    if left is None:
        assert e["length"] == baseline["length"] and not e["recovery"]["exposed"]


def validate_final(final):
    protocol = dict(schedules=worker.evaluation.schedules(), max_steps=1000,
        reset_seeds=list(range(154000, 154005)), action_noise_seeds=[137000, 137001, 137002],
        peak_gravity=worker.evaluation.PEAK, recovery_steps=100, device="cpu")
    assert final["protocol"] == protocol and final["device"] == "cpu"
    assert final["optimizer_updates"] == 0 and not final["replay_loaded"] and final["prefix_pairs_verified"] == 160
    expected = [dict(condition=s["name"], mode="deterministic" if n is None else "sampled", reset_seed=r, noise_seed=n)
        for s in protocol["schedules"] for r in protocol["reset_seeds"] for n in (None, *protocol["action_noise_seeds"])]
    episodes = final["episodes"]
    assert [{k: e[k] for k in expected[0]} for e in episodes] == expected
    baseline = {(e["reset_seed"], e["noise_seed"]): e for e in episodes if e["condition"] == "no-pulse"}
    by_name = {s["name"]: s for s in protocol["schedules"]}
    for e in episodes:
        schedule = by_name[e["condition"]]
        validate_episode(e, schedule, protocol["peak_gravity"])
        if schedule["onset"] is not None:
            validate_prefix(e, baseline[e["reset_seed"], e["noise_seed"]], schedule["onset"])
    assert final["evaluation_interactions"] == sum(e["length"] for e in episodes)


def validate(row):
    json.dumps(row, allow_nan=False)
    seed, arm = row["training_seed"], row["training_profile"]
    assert seed in (5, 6) and arm in worker.ARMS and row["revision"] == worker.REVISION and not row["smoke"]
    assert row["spec"] == worker.stage_spec(arm) and row["config"] == worker.backend.SETTINGS
    assert row["training_device"] == "cuda" and row["runtime"]["stable_baselines3"] == "2.7.0"
    assert row["cumulative_training_interactions"] == row["actual_new_training_interactions"] == 600000
    assert row["optimizer_updates"] == 599745 and row["optimizer_updates_during_evaluation"] == row["state_resets_at_switch"] == 0
    assert row["reward_changes"] == row["healthy_termination_changes"] == row["motor_limit_changes"] == []
    assert row["reward_settings"] == dict(healthy_reward=1., ctrl_cost_weight=.001, forward_reward_weight=1.)
    assert row["observation_dim"] == 11 and not any(row[k] for k in ("use_context", "use_symbolic_constraints", "reward_normalization", "policy_receives_profile_labels", "policy_receives_onset_clock"))
    assert row["bootstrap_mask"] == "terminated-only" and row["replay_handle_timeout_termination"]
    t = row["training"]
    assert t["step"] == 600000 and t["snapshots"] == row["snapshots"]
    assert [b["step"] for b in t["blocks"]] == list(range(5000, 600001, 5000))
    assert sum(b["interactions"] for b in t["blocks"]) == 600000 and sum(b["updates"] for b in t["blocks"]) == 599745
    assert [e["step"] for e in t["periodic_evaluations"]] == list(range(0, 600000, 25000))
    assert row["checkpoint_selection"] == "predeclared300k600k-not-best"
    c = row["recovery_coverage"]
    assert c["baseline_steps"]+c.get("pulse_steps", 0) == 600000
    assert t["blocks"][-1]["recovery_coverage"] == c
    if arm == "no-pulse":
        assert c.get("pulse_steps", 0) == c["pulse_assigned_episodes"] == 0
    assert [s["step"] for s in row["snapshots"]] == [300000, 600000]
    for s in row["snapshots"]:
        assert s["optimizer_updates"] == s["step"]-255 and Path(s["checkpoint"]).name == f"step{s['step']}.zip"
        validate_final(s["final_evaluation"])


def metrics(episodes, baseline):
    values = group_metrics(episodes, baseline)
    values["healthy_full_survival"] = sum(e["time_limit_truncation"] and not e["physical_termination"] for e in episodes)
    return values


def summarize(rows):
    assert len(rows) == 4 and {f"seed{r['training_seed']}_{r['training_profile']}" for r in rows} == set(CASES)
    records, matched = [], []
    for row in sorted(rows, key=lambda r: (r["training_seed"], r["training_profile"])):
        validate(row)
        snapshots = []
        for snapshot in row["snapshots"]:
            episodes = snapshot["final_evaluation"]["episodes"]
            baseline = {(e["reset_seed"], e["noise_seed"]): e for e in episodes if e["condition"] == "no-pulse"}
            conditions = [dict(schedule=s, modes={mode: metrics([e for e in episodes if e["condition"] == s["name"] and e["mode"] == mode], baseline)
                for mode in ("deterministic", "sampled")}) for s in worker.evaluation.schedules()]
            snapshots.append(dict(step=snapshot["step"], checkpoint=snapshot["checkpoint"], conditions=conditions))
        records.append(dict(case=f"seed{row['training_seed']}_{row['training_profile']}", seed=row["training_seed"], arm=row["training_profile"],
            seconds=row["seconds"], training_coverage=row["recovery_coverage"], snapshots=snapshots))
    for seed in (5, 6):
        control, randomized = (next(r for r in records if r["seed"] == seed and r["arm"] == a) for a in worker.ARMS)
        for left, right in zip(control["snapshots"], randomized["snapshots"]):
            for lc, rc in zip(left["conditions"], right["conditions"]):
                matched.append(dict(seed=seed, step=left["step"], condition=lc["schedule"]["name"],
                    modes={mode: {key: rc["modes"][mode][key]-lc["modes"][mode][key] for key in
                        ("mean_length", "mean_return", "healthy_full_survival", "recovered", "delayed_falls_after_recovery")}
                        for mode in ("deterministic", "sampled")}))
    return dict(revision=worker.REVISION, independent_training_seeds=2, cumulative_training_interactions=2400000,
        optimizer_updates=2398980, evaluation_episodes=1440, prefix_pairs_verified=1280,
        evaluation_interactions=sum(s["final_evaluation"]["evaluation_interactions"] for r in rows for s in r["snapshots"]),
        records=records, matched_randomized_minus_control=matched,
        scope="Recovery-state coverage only; retain seed-specific ID regressions and delayed falls. Evaluation repeats are not training seeds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=Path("results/general_ood_v147_randomized_recovery"))
    args = parser.parse_args()
    rows = [json.loads((args.result_dir/case/"source.json").read_text()) for case in CASES]
    summary = summarize(rows)
    summary["formal_json_bytes"] = sum((args.result_dir/case/"source.json").stat().st_size for case in CASES)
    (args.result_dir/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    for r in summary["records"]:
        print(r["case"], [(c["schedule"]["name"], round(c["modes"]["sampled"]["mean_length"], 2), c["modes"]["sampled"]["healthy_full_survival"])
            for c in r["snapshots"][-1]["conditions"]])
