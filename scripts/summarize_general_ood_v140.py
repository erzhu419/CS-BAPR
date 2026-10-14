"""Paired original-rate controls; preserve all seed/profile failures."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_general_ood_v140_critic_rate as worker
import summarize_general_ood_v137 as previous


def summarize(rows, controls, *, study=worker, config_change=None):
    config_change = {"lr_critic": worker.CRITIC_LR} if config_change is None else config_change
    result = previous.summarize(rows, study=study)
    indexed = {(r["training_seed"], r["training_profile"]): r for r in controls}
    assert len(controls) == 6 and set(indexed) == {(r["training_seed"], r["training_profile"]) for r in rows}
    for row, record in zip(sorted(rows, key=lambda r: (r["training_seed"], study.PROFILES.index(r["training_profile"]))), result["records"]):
        seed, profile = row["training_seed"], row["training_profile"]
        control = indexed[seed, profile]
        baseline = study.v137 if profile == "id" else study.v138
        previous.validate(control, study=baseline)
        assert row["config"] == dict(control["config"], **config_change)
        assert row["physics"] == control["physics"] and row["reward_settings"] == control["reward_settings"]
        assert all(row["runtime"][k] == control["runtime"][k] for k in row["runtime"] if k != "gpu")
        episodes = control["final_evaluation"]["episodes"]
        record["control"] = dict(revision=control["revision"], checkpoint=control["checkpoint"],
            modes={mode: previous.aggregate([e for e in episodes if e["mode"] == mode]) for mode in ("deterministic", "sampled")},
            last5k_block=control["training"]["diagnostic_blocks"][-1])
        record["peak_block_means"] = {k: max(b["metrics"][k] for b in row["training"]["diagnostic_blocks"])
                                     for k in ("td_mse", "q_mean", "alpha")}
        record["sampled_length_delta"] = record["modes"]["sampled"]["mean_length"]-record["control"]["modes"]["sampled"]["mean_length"]
    result.update(reused_control_training_interactions=1800000,
                  scope="Paired learning-rate intervention against original v137/v138 controls; reset/noise repeats are not training seeds")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=Path("results/general_ood_v140_critic_rate"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    rows, controls = [], []
    for seed in (3, 4):
        for profile in worker.PROFILES:
            rows.append(json.loads((args.result_dir/f"seed{seed}_{profile}/source.json").read_text()))
            study = "general_ood_v137_fixed_profile" if profile == "id" else "general_ood_v138_physics_components"
            controls.append(json.loads((root/"results"/study/f"seed{seed}_{profile}/source.json").read_text()))
    result = summarize(rows, controls)
    result["formal_json_bytes"] = sum((args.result_dir/f"seed{seed}_{profile}/source.json").stat().st_size for seed in (3, 4) for profile in worker.PROFILES)
    (args.result_dir/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    for row in result["records"]:
        print(row["training_seed"], row["training_profile"], row["modes"]["sampled"], row["last5k_block"]["metrics"]["td_mse"])


if __name__ == "__main__":
    main()
