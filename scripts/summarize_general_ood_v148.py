"""Same-run paired within-support and extreme-amplitude recovery."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_general_ood_v148_known_pulse as worker
from summarize_general_ood_v147 import metrics, validate_episode, validate_prefix


def validate(row, source):
    json.dumps(row, allow_nan=False)
    case, spec = row["case"], worker.stage_spec(row["case"], row["smoke"])
    assert case in worker.CASES and row["revision"] == worker.REVISION and row["spec"] == spec
    assert row["evaluation_device"] == "cpu" and not row["cuda_initialized"] and not row["replay_loaded"]
    assert row["optimizer_updates"] == row["new_training_interactions"] == row["state_resets_at_switch"] == 0
    assert row["reward_changes"] == row["healthy_termination_changes"] == row["motor_limit_changes"] == []
    snapshot = next(s for s in source["snapshots"] if s["step"] == 600000)
    assert row["source"] == dict(revision=worker.training.REVISION, seed=source["training_seed"], arm=source["training_profile"],
        checkpoint=snapshot["checkpoint"], training_interactions=600000, optimizer_updates=599745, config=worker.backend.SETTINGS)
    assert case == f"seed{source['training_seed']}_{source['training_profile']}"
    assert row["reused_reference_episodes"] == 0
    expected = [dict(condition=s["name"], mode="deterministic" if n is None else "sampled", reset_seed=r, noise_seed=n)
        for s in spec["schedules"] for r in spec["reset_seeds"] for n in (None, *spec["action_noise_seeds"])]
    assert [{k: e[k] for k in expected[0]} for e in row["episodes"]] == expected
    baseline = {(e["reset_seed"], e["noise_seed"]): e for e in row["episodes"] if e["condition"] == "no-pulse"}
    assert row["prefix_pairs_verified"] == len(expected)-len(baseline)
    by_name = {s["name"]: s for s in spec["schedules"]}
    for e in row["episodes"]:
        schedule = by_name[e["condition"]]
        validate_episode(e, schedule, schedule["gravity"])
        if schedule["onset"] is not None:
            validate_prefix(e, baseline[e["reset_seed"], e["noise_seed"]], schedule["onset"])
    assert row["evaluation_interactions"] == sum(e["length"] for e in row["episodes"])


def summarize(rows, sources):
    assert len(rows) == 4 and {r["case"] for r in rows} == set(worker.CASES) and not any(r["smoke"] for r in rows)
    records = []
    for row in sorted(rows, key=lambda r: worker.CASES.index(r["case"])):
        validate(row, sources[row["case"]])
        episodes = row["episodes"]
        baseline = {(e["reset_seed"], e["noise_seed"]): e for e in episodes if e["condition"] == "no-pulse"}
        conditions = []
        for onset in (100, 300):
            modes = {}
            for mode in ("deterministic", "sampled"):
                known = metrics([e for e in episodes if e["condition"] == f"g2p0_after{onset}_pulse10" and e["mode"] == mode], baseline)
                extreme = metrics([e for e in episodes if e["condition"] == f"g2p659_after{onset}_pulse10" and e["mode"] == mode], baseline)
                modes[mode] = dict(known2p0=known, reference2p659=extreme)
            conditions.append(dict(onset=onset, duration=10, modes=modes))
        records.append(dict(case=row["case"], source=row["source"], seconds=row["seconds"], conditions=conditions,
            baseline={mode: metrics([e for e in episodes if e["condition"] == "no-pulse" and e["mode"] == mode], baseline) for mode in ("deterministic", "sampled")}))
    return dict(revision=worker.REVISION, new_evaluation_episodes=400, reused_reference_episodes=0,
        new_evaluation_interactions=sum(r["evaluation_interactions"] for r in rows), prefix_pairs_verified=320,
        optimizer_updates=0, new_training_interactions=0, records=records,
        scope="Frozen within-support versus extreme amplitude; exact same-run pre-pulse prefixes. Old cross-host prefixes differed, so native controls were recomputed. Not new training or mechanism evidence.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=Path("results/general_ood_v148_known_pulse"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    rows = [json.loads((args.result_dir/case/"known_pulse.json").read_text()) for case in worker.CASES]
    sources = {case: json.loads((root/f"results/general_ood_v147_randomized_recovery/{case}/source.json").read_text()) for case in worker.CASES}
    summary = summarize(rows, sources)
    summary["formal_json_bytes"] = sum((args.result_dir/case/"known_pulse.json").stat().st_size for case in worker.CASES)
    (args.result_dir/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    for r in summary["records"]:
        print(r["case"], 'baseline', r["baseline"]["sampled"]["mean_length"], [(c["onset"], {k:(round(v["mean_length"],2),v["healthy_full_survival"],v["recovered"])
            for k,v in c["modes"]["sampled"].items()}) for c in r["conditions"]])
