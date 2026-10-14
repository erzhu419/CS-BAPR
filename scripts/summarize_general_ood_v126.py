"""Keep noise groups, source ages and inference devices separate."""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze_general_ood_v126_noise_sensitivity import REVISION, cells, stage_spec


def summarize(rows):
    assert len(rows) == 4 and {(r["training_seed"], r["evaluation_device"]) for r in rows} == {
        (0, "cpu"), (1, "cpu"), (2, "cpu"), (2, "cuda")}
    records = []
    for source in rows:
        spec = stage_spec(source["training_seed"], source["evaluation_device"])
        assert source["revision"] == REVISION and source["spec"] == spec
        assert source["optimizer_updates"] == source["new_training_interactions"] == 0
        assert not source["source_readout_refitted"] and not source["reward_or_physics_changes"]
        assert source["checkpoint_selection"] == "final-not-best"
        assert [m["age"] for m in source["models"]] == spec["ages"]
        for model in source["models"]:
            episodes = model["evaluations"]
            assert [(e["reset_seed"], e["noise_scale"], e["noise_seed"], e["noise_group"])
                    for e in episodes] == cells(spec)
            groups = []
            for scale in spec["noise_scales"]:
                for group in (["deterministic"] if scale == 0 else spec["noise_groups"]):
                    selected = [e for e in episodes if e["noise_scale"] == scale and e["noise_group"] == group]
                    groups.append(dict(noise_scale=scale, noise_group=group, episodes=len(selected),
                        **{key: float(np.mean([e[key] for e in selected])) for key in (
                            "length", "episode_return", "mean_velocity", "mean_policy_std",
                            "mean_action_noise_l2", "mean_context_l2")},
                        falls=sum(e["physical_termination"] for e in selected),
                        time_limits=sum(e["time_limit_truncation"] for e in selected)))
            records.append(dict(training_seed=source["training_seed"], device=source["evaluation_device"],
                age=model["age"], groups=groups,
                parity_maxima={key: max(e["same_input_parity"][key] for e in episodes)
                    for key in ("max_mean_abs_difference", "max_std_abs_difference", "max_action_abs_difference")}
                    if source["evaluation_device"] == "cuda" else None))
        assert source["evaluation_interactions"] == sum(e["length"] for m in source["models"] for e in m["evaluations"])
    cpu, gpu = [next(r for r in rows if r["training_seed"] == 2 and r["evaluation_device"] == d)
                for d in ("cpu", "cuda")]
    assert cpu["runtime"] == gpu["runtime"]
    for a, b in zip(cpu["models"], gpu["models"]):
        assert all(a[k] == b[k] for k in ("checkpoint", "config", "initial_context_normalizer", "alpha"))
    return dict(revision=REVISION, independent_training_seeds=3, evaluation_model_device_pairs=8,
        evaluation_episodes=sum(len(m["evaluations"]) for r in rows for m in r["models"]),
        evaluation_interactions=sum(r["evaluation_interactions"] for r in rows),
        optimizer_updates=0, new_training_interactions=0, records=records,
        interpretation="Frozen inference diagnosis; noise repeats and devices are not independent training seeds.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads((args.result_dir/f"seed{seed}_{device}/noise.json").read_text())
            for seed, device in ((0, "cpu"), (1, "cpu"), (2, "cpu"), (2, "cuda"))]
    result = summarize(rows)
    (args.result_dir/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result["records"], indent=2))


if __name__ == "__main__":
    main()
