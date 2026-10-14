"""Component outcomes with matching v137 ID/compound endpoint controls."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_general_ood_v138_physics_components as worker
import summarize_general_ood_v137 as previous


def validate(row, smoke=False):
    return previous.validate(row, smoke, study=worker)


def summarize(rows, controls):
    result = previous.summarize(rows, study=worker)
    expected = {(seed, profile) for seed in (3, 4) for profile in ("id", "mega_event")}
    assert len(controls) == 4 and {(r["training_seed"], r["training_profile"]) for r in controls} == expected
    records = []
    for row in sorted(controls, key=lambda r: (r["training_seed"], r["training_profile"])):
        assert row["revision"] == worker.v137.REVISION and not row["smoke"]
        assert row["spec"] == worker.v137.stage_spec(row["training_profile"])
        assert row["config"] == rows[0]["config"] and row["reward_settings"] == rows[0]["reward_settings"]
        assert all(row["runtime"][k] == rows[0]["runtime"][k] for k in row["runtime"] if k != "gpu")
        assert row["training"]["interactions"] == 300000 and row["training"]["optimizer_updates"] == 299745
        assert row["trained_from_scratch"] and row["checkpoint_selection"] == "final-not-best"
        assert row["final_evaluation"]["protocol"] == worker.evaluation_spec()
        records.append(dict(training_seed=row["training_seed"], training_profile=row["training_profile"],
            checkpoint=row["checkpoint"], revision=row["revision"],
            modes={mode: previous.aggregate([e for e in row["final_evaluation"]["episodes"] if e["mode"] == mode])
                   for mode in ("deterministic", "sampled")}))
    result.update(reused_control_models=4, reused_control_training_interactions=1200000,
        control_records=records, scope="Single-factor known-regime learnability, using matched v137 ID/compound controls. Component controls change task physics and do not establish unknown-regime adaptation or physical impossibility.")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--control-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads((args.result_dir/f"seed{seed}_{profile}/source.json").read_text())
            for seed in (3, 4) for profile in worker.PROFILES]
    controls = [json.loads((args.control_dir/f"seed{seed}_{profile}/source.json").read_text())
                for seed in (3, 4) for profile in ("id", "mega_event")]
    result = summarize(rows, controls)
    (args.result_dir/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    for row in result["records"]:
        print(row["training_seed"], row["training_profile"], row["modes"])


if __name__ == "__main__":
    main()
