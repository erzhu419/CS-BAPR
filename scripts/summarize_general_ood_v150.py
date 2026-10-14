"""Same-run physical/noise controls, retaining every final700k policy."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_general_ood_v150_noise_timing as worker


def control_name(schedule):
    program = schedule["program"]
    return f"{program}_no-pulse" if program in worker.PROGRAMS[:2] else schedule["name"].replace("pulse10", "no-pulse")


def validate(row, source):
    json.dumps(row, allow_nan=False)
    spec = worker.stage_spec(row["case"], row["smoke"])
    assert row["case"] in worker.CASES and row["revision"] == worker.REVISION and row["spec"] == spec
    assert row["source"] == worker.source_metadata(source, row["case"])
    assert row["evaluation_device"] == "cpu" and not row["cuda_initialized"] and not row["replay_loaded"]
    assert row["optimizer_updates"] == row["new_training_interactions"] == row["state_resets_at_switch"] == row["reused_reference_episodes"] == 0
    assert row["reward_changes"] == row["healthy_termination_changes"] == row["motor_limit_changes"] == []
    expected = worker.cells(spec)
    assert [{k: e[k] for k in expected[0]} for e in row["episodes"]] == expected
    assert (row["prefix_pairs_verified"], row["noise_boundary_pairs_verified"]) == worker.validate_pairs(row["episodes"], spec)
    by_name = {s["name"]: s for s in spec["schedules"]}
    for e in row["episodes"]:
        s = by_name[e["condition"]]
        assert e["dt"] == .008 and 0 < e["length"] <= 1000 and not e["diagnostic_cutoff"]
        assert e["physical_termination"] or e["time_limit_truncation"]
        assert e["native_reward_verified"] and e["context_mode"] == "not-applicable" and e["context_updates"] == 0
        assert e["noise_program"] == s["program"]
        assert e["noise_scale"] == (0. if s["program"] == "global-quiet" else (1. if s["program"] == "native" else None))
        assert e["noise_scale_counts"] == dict(Counter(str(worker.noise_scale(s["program"], t, s["clock_onset"])) for t in range(e["length"])))
        assert e["mean_effective_policy_std"] >= 0 and e["mean_policy_std"] > 0 and e["mean_action_noise_l2"] >= 0
        if s["program"] == "global-quiet":
            assert e["mean_effective_policy_std"] == e["mean_action_noise_l2"] == 0
        onset = s["onset"]
        peak_steps = 0 if onset is None else min(max(e["length"]-onset, 0), 10)
        assert e["profile_counts"] == {k: v for k, v in {"id": e["length"]-peak_steps, "gravity-only": peak_steps}.items() if v}
        assert e["non_id_steps"] == peak_steps
        events = [(1, "id", 1.)]
        if peak_steps:
            events.append((onset+1, "gravity-only", spec["peak_gravity"]))
        if onset is not None and e["length"] > onset+10:
            events.append((onset+11, "id", 1.))
        assert e["profile_events"] == [dict(step=t, profile=p, actual_multipliers=dict(body_mass=1., gravity=g, friction=1.)) for t, p, g in events]
        assert e["switches"] == len(events)-1 and e["recovery"] == worker.evaluation.recovery(e, s)
    assert row["evaluation_interactions"] == sum(e["length"] for e in row["episodes"])


def metrics(episodes):
    return dict(episodes=len(episodes), mean_length=sum(e["length"] for e in episodes)/len(episodes),
        mean_return=sum(e["episode_return"] for e in episodes)/len(episodes),
        healthy_full_survival=sum(e["time_limit_truncation"] and not e["physical_termination"] for e in episodes),
        exposed=sum(e["recovery"]["exposed"] for e in episodes), recovered=sum(e["recovery"]["recovered"] for e in episodes),
        delayed_falls_after_recovery=sum(e["recovery"]["recovered"] and e["physical_termination"] for e in episodes))


def summarize(rows, sources):
    assert len(rows) == 4 and {r["case"] for r in rows} == set(worker.CASES) and not any(r["smoke"] for r in rows)
    records = []
    for row in sorted(rows, key=lambda r: worker.CASES.index(r["case"])):
        validate(row, sources[row["case"]])
        by_condition = {s["name"]: [e for e in row["episodes"] if e["condition"] == s["name"]] for s in row["spec"]["schedules"]}
        conditions = []
        for s in row["spec"]["schedules"]:
            if s["onset"] is None:
                continue
            shock, control = metrics(by_condition[s["name"]]), metrics(by_condition[control_name(s)])
            native = dict(name=f"native_after{s['onset']}_pulse10", program="native")
            native_shock, native_control = metrics(by_condition[native["name"]]), metrics(by_condition[control_name(native)])
            conditions.append(dict(schedule=s, shock=shock, no_shock=control,
                shock_specific_survival_delta=(shock["healthy_full_survival"]/shock["episodes"]-control["healthy_full_survival"]/control["episodes"])
                    -(native_shock["healthy_full_survival"]/native_shock["episodes"]-native_control["healthy_full_survival"]/native_control["episodes"]),
                shock_specific_length_delta=(shock["mean_length"]-control["mean_length"])-(native_shock["mean_length"]-native_control["mean_length"])))
        records.append(dict(case=row["case"], source=row["source"], seconds=row["seconds"], conditions=conditions,
            no_shock_controls={s["name"]: metrics(by_condition[s["name"]]) for s in row["spec"]["schedules"] if s["onset"] is None}))
    return dict(revision=worker.REVISION, new_evaluation_episodes=1200, new_training_interactions=0, optimizer_updates=0,
        evaluation_interactions=sum(r["evaluation_interactions"] for r in rows), prefix_pairs_verified=640,
        noise_boundary_pairs_verified=960, records=records,
        scope="Oracle timing attribution on fresh reset/noise repeats of two paired source seeds; not a learned controller or CS-BAPR mechanism evidence.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=Path("results/general_ood_v150_noise_timing"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    rows = [json.loads((args.result_dir/case/"noise_timing.json").read_text()) for case in worker.CASES]
    sources = {case: json.loads((root/f"results/general_ood_v149_warm_recovery/{case}/source.json").read_text()) for case in worker.CASES}
    summary = summarize(rows, sources)
    summary["formal_json_bytes"] = sum((args.result_dir/case/"noise_timing.json").stat().st_size for case in worker.CASES)
    (args.result_dir/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    for r in summary["records"]:
        print(r["case"], [(c["schedule"]["name"], round(c["shock"]["mean_length"], 2), c["shock"]["healthy_full_survival"],
            c["shock"]["episodes"], round(c["shock_specific_survival_delta"], 3)) for c in r["conditions"]])
