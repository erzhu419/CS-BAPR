"""Slow-actor endpoints against matched original-rate controls."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_general_ood_v141_actor_rate as worker
import summarize_general_ood_v140 as previous


def summarize(rows, controls):
    return previous.summarize(rows, controls, study=worker, config_change=dict(lr_actor=worker.ACTOR_LR))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=Path("results/general_ood_v141_actor_rate"))
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
