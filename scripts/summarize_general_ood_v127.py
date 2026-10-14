"""Validate the complete factorial and report matched component contrasts."""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze_general_ood_v127_component_attribution import REVISION, AGES, cases, cells, coherent, stage_spec


def summarize(rows):
    assert len(rows) == 3 and sorted(r["training_seed"] for r in rows) == [0, 1, 2]
    records, contrasts = [], []
    for row in sorted(rows, key=lambda r: r["training_seed"]):
        spec = stage_spec(row["training_seed"])
        assert row["revision"] == REVISION and row["spec"] == spec and row["evaluation_device"] == "cpu"
        assert row["optimizer_updates"] == row["new_training_interactions"] == 0
        assert not row["source_readout_refitted"] and not row["reward_or_physics_changes"]
        assert row["checkpoint_selection"] == "final-not-best"
        assert [s["age"] for s in row["sources"]] == list(AGES)
        assert row["sources"][0]["config"] == row["sources"][1]["config"]
        expected = cells(spec)
        assert [{k: e[k] for k in expected[0]} for e in row["evaluations"]] == expected
        assert all(e["baseline_match"] is True if coherent(e) else e["baseline_match"] is None
                   for e in row["evaluations"])
        assert row["coherent_baseline_matches"] == sum(coherent(e) for e in expected) == 70
        assert row["evaluation_interactions"] == sum(e["length"] for e in row["evaluations"])
        per_seed = []
        for case in cases():
            for group in (["deterministic"] if case["mode"] == "deterministic" else spec["noise_groups"]):
                selected = [e for e in row["evaluations"] if all(e[k] == v for k, v in case.items())
                            and e["noise_group"] == group]
                record = dict(training_seed=row["training_seed"], **case, noise_group=group,
                    episodes=len(selected), **{key: float(np.mean([e[key] for e in selected])) for key in (
                        "length", "episode_return", "mean_velocity", "mean_policy_std",
                        "mean_action_noise_l2", "mean_context_l2")},
                    falls=sum(e["physical_termination"] for e in selected),
                    time_limits=sum(e["time_limit_truncation"] for e in selected))
                records.append(record)
                per_seed.append(record)
        for group in spec["noise_groups"]:
            base = next(r for r in per_seed if r["mode"] == "sampled" and r["noise_group"] == group
                        and all(r[k] == 350000 for k in ("mean_age", "std_age", "normalizer_age")))
            for factor in ("mean_age", "std_age", "normalizer_age"):
                target = {k: 300000 if k == factor else 350000 for k in ("mean_age", "std_age", "normalizer_age")}
                swapped = next(r for r in per_seed if r["mode"] == "sampled" and r["noise_group"] == group
                               and all(r[k] == v for k, v in target.items()))
                contrasts.append(dict(training_seed=row["training_seed"], noise_group=group, swapped_component=factor,
                    old_component_minus_350k={k: swapped[k]-base[k] for k in ("length", "episode_return", "falls")}))
    return dict(revision=REVISION, independent_training_seeds=3, evaluation_episodes=780,
        evaluation_interactions=sum(r["evaluation_interactions"] for r in rows), coherent_baseline_matches=210,
        new_training_interactions=0, optimizer_updates=0, records=records, single_component_contrasts=contrasts,
        interpretation="Frozen crossed functions and initial normalizers; hybrid policies are diagnostic, not an adopted method.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    result = summarize([json.loads((args.result_dir/f"seed{s}/components.json").read_text()) for s in (0, 1, 2)])
    (args.result_dir/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result["single_component_contrasts"], indent=2))


if __name__ == "__main__":
    main()
