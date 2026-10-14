"""Budget-extension evidence, retaining exact prefixes and terminal morphology."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_general_ood_v142_budget_extension as worker
import summarize_general_ood_v137 as previous


def summarize(sources, rows):
    expected = {(seed, profile) for seed in (3, 4) for profile in worker.PROFILES}
    old = {(r["training_seed"], r["training_profile"]): r for r in sources}
    new = {(r["training_seed"], r["training_profile"]): r for r in rows}
    assert len(sources) == len(rows) == 6 and set(old) == set(new) == expected
    records = []
    for seed in (3, 4):
        for profile in worker.PROFILES:
            source, row = old[seed, profile], new[seed, profile]
            previous.validate(source, study=worker.source_worker)
            json.dumps(row, allow_nan=False)
            assert row["revision"] == worker.REVISION and not row["smoke"]
            assert row["spec"] == worker.stage_spec(profile) and row["training_driver_revision"] == worker.v132.REVISION
            assert row["source_revision"] == source["revision"] and row["source_checkpoint"] == source["resumable_checkpoint"]
            assert row["source_runtime"] == source["runtime"] and row["config"] == source["config"]
            assert row["physics"] == source["physics"] and row["reward_settings"] == source["reward_settings"]
            assert all(row["runtime"][k] == source["runtime"][k] for k in source["runtime"] if k != "gpu")
            assert not row["trained_from_scratch"] and row["training_mode"] == "fixed-profile-continuation"
            assert row["source_interactions"] == 300000 and row["source_optimizer_updates"] == 299745
            assert row["actual_new_training_interactions"] == row["actual_new_optimizer_updates"] == 300000
            assert row["checkpoint_selection"] == "final-not-best" and Path(row["checkpoint"]).name == "final.pt"
            assert not row["periodic_evaluation_selects_checkpoint"] and not row["source_readout_refitted"]
            assert row["optimizer_updates_during_evaluation"] == 0 and row["reward_changes"] == []
            assert row["effective_beta"] == -2. and not row["policy_receives_profile_labels"]
            training = row["training"]
            assert training["interactions"] == row["final_context_count"] == 600000
            assert training["optimizer_updates"] == 599745
            blocks = training["diagnostic_blocks"]
            assert [b["step"] for b in blocks] == list(range(5000, 600001, 5000))
            assert blocks[:60] == source["training"]["diagnostic_blocks"]
            assert sum(b["interactions"] for b in blocks) == 600000 and sum(b["updates"] for b in blocks) == 599745
            assert sum(b["updates"] for b in blocks[60:]) == 300000
            periodic = training["periodic_evaluations"]
            assert [e["step"] for e in periodic] == list(range(0, 600000, 25000))
            assert periodic[:12] == source["training"]["periodic_evaluations"]
            for e in periodic:
                assert e["profile"] == profile and e["actual_multipliers"] == row["physics"]["actual_multipliers"]
                assert [p["seed"] for p in e["episodes"]] == list(range(149000, 149003))
            episodes = previous.validate_final(row, study=worker.source_worker)
            failures = [e for e in episodes if e["physical_termination"]]
            reason_sets = Counter("+".join(e["final_physical_state"]["unhealthy_conditions"]) for e in failures)
            assert all(e["final_physical_state"]["unhealthy_conditions"] for e in failures)
            records.append(dict(training_seed=seed, training_profile=profile, source_checkpoint=row["source_checkpoint"],
                checkpoint=row["checkpoint"], seconds=row["seconds"],
                at300k={mode: previous.aggregate([e for e in source["final_evaluation"]["episodes"] if e["mode"] == mode]) for mode in ("deterministic", "sampled")},
                at600k={mode: previous.aggregate([e for e in episodes if e["mode"] == mode]) for mode in ("deterministic", "sampled")},
                last5k_metrics=blocks[-1]["metrics"],
                peak_new_block_mean_td_mse=max(b["metrics"]["td_mse"] for b in blocks[60:]),
                terminal_reason_sets=dict(reason_sets),
                terminal_height_range=[min(e["final_physical_state"]["height"] for e in episodes), max(e["final_physical_state"]["height"] for e in episodes)],
                terminal_max_abs_qvel=max(e["final_physical_state"]["max_abs_qvel"] for e in episodes)))
    return dict(revision=worker.REVISION, independent_training_seeds=2, final_models=6,
        actual_new_training_interactions=1800000, actual_new_optimizer_updates=1800000,
        reused_training_interactions=1800000, cumulative_training_interactions=3600000,
        cumulative_optimizer_updates=3598470, final_evaluation_episodes=120,
        final_evaluation_interactions=sum(r["final_evaluation"]["evaluation_interactions"] for r in rows), records=records,
        scope="Within-trajectory budget extension, not matched600k superiority over original300k controls")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=Path("results/general_ood_v142_budget_extension"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    rows = [json.loads((args.result_dir/f"seed{s}_{p}/source.json").read_text()) for s in (3, 4) for p in worker.PROFILES]
    sources = [json.loads((root/f"results/general_ood_v141_actor_rate/seed{s}_{p}/source.json").read_text()) for s in (3, 4) for p in worker.PROFILES]
    result = summarize(sources, rows)
    result["formal_json_bytes"] = sum((args.result_dir/f"seed{s}_{p}/source.json").stat().st_size for s in (3, 4) for p in worker.PROFILES)
    (args.result_dir/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    for r in result["records"]:
        print(r["training_seed"], r["training_profile"], r["at600k"]["sampled"], r["terminal_reason_sets"])
