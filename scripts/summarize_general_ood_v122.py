"""Pair a single-factor source pilot with its unchanged v121 seed controls."""
import argparse
import json
from pathlib import Path

import numpy as np


def summarize(controls, pilots):
    assert len(controls) == len(pilots) == 3
    assert sorted(r["training_seed"] for r in controls) == sorted(r["training_seed"] for r in pilots) == [0, 1, 2]
    controls = {r["training_seed"]: r for r in controls}
    records = []
    for row in sorted(pilots, key=lambda r: r["training_seed"]):
        control = controls[row["training_seed"]]
        assert control["revision"] == "general_ood_v121_locomotion_source"
        assert row["revision"] == "general_ood_v122_target_l1" and not row["smoke"] and not control["smoke"]
        differences = {k: (control["config"].get(k), row["config"].get(k))
                       for k in control["config"].keys() | row["config"].keys()
                       if control["config"].get(k) != row["config"].get(k)}
        assert differences == {"weight_reg": (.01, 0.)}
        assert {k: v for k, v in row["spec"].items() if k not in ("study_revision", "intervention")} == control["spec"]
        assert row["runtime"] == control["runtime"]
        for item in (control, row):
            assert item["training"]["interactions"] == 100000 and item["training"]["optimizer_updates"] == 99745
            assert item["checkpoint_selection"] == "final-not-best"
        before, after = control["final_evaluation"], row["final_evaluation"]
        assert [e["seed"] for e in before["episodes"]] == [e["seed"] for e in after["episodes"]]
        keys = ("mean_episode_velocity", "mean_episode_length", "mean_return", "physical_terminations")
        records.append(dict(training_seed=row["training_seed"], control={k: before[k] for k in keys},
            pilot={k: after[k] for k in keys}, pilot_minus_control={k: after[k]-before[k] for k in keys},
            periodic_evaluations=row["training"]["periodic_evaluations"], final_target_scale=row["final_target_scale"]))
    return dict(revision="general_ood_v122_target_l1", independent_training_seeds=3,
        actual_new_training_interactions=300000, reused_control_interactions=300000,
        per_training_seed=records, mean_pilot_minus_control={k: float(np.mean([r["pilot_minus_control"][k] for r in records]))
            for k in records[0]["pilot_minus_control"]}, source_readout_refitted=False,
        intervention="remove target L1 uplift only; retain actor beta=-2 and remaining regularizers")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    read = lambda directory: [json.loads((directory/f"seed{seed}/source.json").read_text()) for seed in (0, 1, 2)]
    summary = summarize(read(args.control_dir), read(args.result_dir))
    (args.result_dir/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print(json.dumps(summary["mean_pilot_minus_control"], indent=2))


if __name__ == "__main__":
    main()
