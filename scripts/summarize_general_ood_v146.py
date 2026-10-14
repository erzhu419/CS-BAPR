"""Paired frozen pulse evidence, distinguishing recovery from delayed failure."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_general_ood_v146_pulse_recovery as worker
from summarize_general_ood_v129 import aggregate


def group_metrics(episodes, baseline):
    values = aggregate(episodes)
    values.update(exposed=sum(e["recovery"]["exposed"] for e in episodes),
        recovery_evaluable=sum(e["recovery"]["recovery_evaluable"] for e in episodes),
        recovered=sum(e["recovery"]["recovered"] for e in episodes),
        delayed_falls_after_recovery=sum(e["recovery"]["recovered"] and e["physical_termination"] for e in episodes),
        mean_paired_length_delta=sum(e["length"]-baseline[e["reset_seed"], e["noise_seed"]]["length"] for e in episodes)/len(episodes))
    return values


def validate(row):
    json.dumps(row, allow_nan=False)
    case = row["case"]
    assert case in worker.CASES and row["revision"] == worker.REVISION and not row["smoke"]
    assert row["spec"] == dict(case=case, schedules=worker.schedules(), smoke=False, max_steps=1000,
        reset_seeds=list(range(154000, 154005)), action_noise_seeds=[137000, 137001, 137002],
        peak_gravity=worker.PEAK, recovery_steps=100, device="cpu")
    assert row["evaluation_device"] == "cpu" and not row["cuda_initialized"] and not row["replay_loaded"]
    assert row["optimizer_updates"] == row["new_training_interactions"] == row["state_resets_at_switch"] == 0
    assert row["reward_changes"] == row["healthy_termination_changes"] == row["motor_limit_changes"] == []
    assert row["prefix_pairs_verified"] == 160
    source = row["source"]
    assert source["training_interactions"] == 600000 and source["optimizer_updates"] == 599745
    assert source["config"] == worker.backend.SETTINGS and Path(source["checkpoint"]).name == "step600000.zip"
    assert case == f"seed{source['seed']}_{source['profile']}"
    expected = [dict(condition=s["name"], mode="deterministic" if n is None else "sampled", reset_seed=r, noise_seed=n)
        for s in worker.schedules() for r in row["spec"]["reset_seeds"] for n in (None, *row["spec"]["action_noise_seeds"])]
    episodes = row["episodes"]
    assert [{k: e[k] for k in expected[0]} for e in episodes] == expected
    by_name = {s["name"]: s for s in worker.schedules()}
    base_g = 1. if source["profile"] == "id" else 1.5
    for e in episodes:
        assert e["dt"] == .008 and 0 < e["length"] <= 1000 and not e["diagnostic_cutoff"]
        assert e["native_reward_verified"] and (e["physical_termination"] or e["time_limit_truncation"])
        assert e["context_updates"] == 0 and e["context_mode"] == "not-applicable"
        assert e["noise_scale"] == (0. if e["mode"] == "deterministic" else 1.)
        s = by_name[e["condition"]]
        onset, duration = s["onset"], s["duration"]
        peak_steps = 0 if onset is None else max(e["length"]-onset, 0)
        if duration is not None: peak_steps = min(peak_steps, duration)
        counts = {source["profile"]: e["length"]-peak_steps, "gravity-only": peak_steps}
        assert e["profile_counts"] == {k: v for k, v in counts.items() if v}
        events = [(1, source["profile"], base_g)]
        if peak_steps: events.append((onset+1, "gravity-only", worker.PEAK))
        if onset is not None and duration is not None and e["length"] > onset+duration:
            events.append((onset+duration+1, source["profile"], base_g))
        assert e["profile_events"] == [dict(step=t, profile=p, actual_multipliers=dict(body_mass=1., gravity=g, friction=1.)) for t, p, g in events]
        assert e["recovery"] == worker.recovery(e, s)
    assert row["evaluation_interactions"] == sum(e["length"] for e in episodes)


def summarize(rows):
    assert len(rows) == 4 and {r["case"] for r in rows} == set(worker.CASES)
    records = []
    for row in sorted(rows, key=lambda r: worker.CASES.index(r["case"])):
        validate(row)
        baseline = {(e["reset_seed"], e["noise_seed"]): e for e in row["episodes"] if e["condition"] == "no-pulse"}
        conditions = []
        for s in worker.schedules():
            conditions.append(dict(schedule=s, modes={m: group_metrics([e for e in row["episodes"] if e["condition"] == s["name"] and e["mode"] == m], baseline) for m in ("deterministic", "sampled")}))
        records.append(dict(case=row["case"], source=row["source"], seconds=row["seconds"], conditions=conditions))
    return dict(revision=worker.REVISION, evaluation_episodes=720,
        evaluation_interactions=sum(r["evaluation_interactions"] for r in rows), prefix_pairs_verified=640,
        optimizer_updates=0, new_training_interactions=0, records=records,
        scope="Paired frozen recovery; 100-step recovery is not1000-step survival. Retain all unexposed episodes and per-policy noise modes; no adaptation/superiority claim.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=Path("results/general_ood_v146_pulse_recovery"))
    args = parser.parse_args()
    rows = [json.loads((args.result_dir/case/"pulse_recovery.json").read_text()) for case in worker.CASES]
    result = summarize(rows)
    result["formal_json_bytes"] = sum((args.result_dir/case/"pulse_recovery.json").stat().st_size for case in worker.CASES)
    (args.result_dir/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    for r in result["records"]:
        print(r["case"],[(s["schedule"]["name"], s["modes"]["sampled"]["mean_length"], s["modes"]["sampled"]["recovered"], s["modes"]["sampled"]["delayed_falls_after_recovery"]) for s in r["conditions"]])
