"""Summarize frozen paired source diagnostics, not new training evidence."""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze_general_ood_v124_source_stability import REVISION, stage_spec


def summarize(rows):
    assert sorted(r["training_seed"] for r in rows) == [0, 1, 2]
    records = []
    for source in sorted(rows, key=lambda r: r["training_seed"]):
        spec = dict(stage_spec(), training_seed=source["training_seed"])
        assert source["revision"] == REVISION and source["spec"] == spec
        assert source["optimizer_updates"] == 0 and source["evaluation_device"] == "cpu"
        assert [r["age"] for r in source["models"]] == spec["ages"]
        for model in source["models"]:
            episodes = model["evaluations"]
            expected = [(reset, noise) for reset in spec["reset_seeds"] for noise in (None, *spec["noise_seeds"])]
            assert [(e["reset_seed"], e["noise_seed"]) for e in episodes] == expected
            modes = {}
            for mode in ("deterministic", "sampled"):
                selected = [e for e in episodes if e["action_mode"] == mode]
                modes[mode] = dict(episodes=len(selected), mean_length=float(np.mean([e["length"] for e in selected])),
                    mean_return=float(np.mean([e["episode_return"] for e in selected])),
                    mean_velocity=float(np.mean([e["mean_velocity"] for e in selected])),
                    falls=sum(e["physical_termination"] for e in selected),
                    time_limits=sum(e["time_limit_truncation"] for e in selected),
                    unhealthy_conditions={k: sum(k in e["final_physical_state"]["unhealthy_conditions"]
                        for e in selected if e["physical_termination"]) for k in ("height", "angle", "state")})
            paired = []
            for reset in spec["reset_seeds"]:
                subset = [e for e in episodes if e["reset_seed"] == reset]
                deterministic = next(e for e in subset if e["action_mode"] == "deterministic")
                sampled_length = float(np.mean([e["length"] for e in subset if e["action_mode"] == "sampled"]))
                paired.append(dict(reset_seed=reset, deterministic_length=deterministic["length"],
                                   sampled_mean_length=sampled_length, difference=sampled_length-deterministic["length"]))
            records.append(dict(training_seed=source["training_seed"], age=model["age"], modes=modes,
                                paired_by_reset=paired, gradients=model["gradients"]))
        assert source["evaluation_interactions"] == sum(e["length"] for m in source["models"] for e in m["evaluations"])
    return dict(revision=REVISION, independent_training_seeds=3, models=6, evaluation_episodes=120,
                optimizer_updates=0, new_training_interactions=0, source_readout_refitted=False,
                evaluation_interactions=sum(r["evaluation_interactions"] for r in rows), records=records,
                interpretation="paired CPU-inference diagnostic; policy-noise repeats are not independent training seeds")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads((args.result_dir/f"seed{seed}/stability.json").read_text()) for seed in (0, 1, 2)]
    result = summarize(rows)
    (args.result_dir/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps([dict(training_seed=r["training_seed"], age=r["age"], modes=r["modes"])
                      for r in result["records"]], indent=2))


if __name__ == "__main__":
    main()
