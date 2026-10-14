"""Paired signal contrasts, not policy-return or training-seed statistics."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "experiments/general_ood_v41_frozen_signal_20261006"
DESTINATION = ROOT / "results/general_ood_v41_frozen_signal"
KEYS = ("surprise_raw", "surprise", "belief_reset_probability", "hypothetical_beta")
PHASES = {"first_post256": 4, "second_post256": 5}


def validate_run(data, manifest, stage):
    policy = data["source_policy"]
    seeds = (manifest["preflight_evaluation_seeds"] if stage == "preflight" else
             list(range(manifest["evaluation_seed_start"], manifest["evaluation_seed_start"] + manifest["evaluation_seeds"])))
    targets = manifest["preflight_target_modes"] if stage == "preflight" else manifest["target_modes"]
    assert policy in manifest["source_policies"]
    assert data["revision"] == manifest["revision"] and data["stage"] == stage
    assert data["evaluation_seeds"] == seeds and data["target_modes"] == targets
    for key in ("probe_interactions_per_rollout", "switch_step", "window", "probe_episode_time_limit",
                "reward_normalization_std", "hypothetical_beta_penalty_scale", "auxiliary_hazard_per_observation",
                "optimizer_updates_during_probe", "replay_insertions", "auxiliary_belief_controls_actions"):
        assert data[key] == manifest[key], (policy, key)
    assert data["actor_critic_target_frozen"] and data["restore_training_context_before_each_rollout"]
    assert data["loaded_training_update_steps"] == manifest["source_training_update_steps"]
    assert data["loaded_training_context_count"] == data["source_training_interactions"] == 100000
    source = json.loads((ROOT / f"results/general_ood_v40_strength/formal/Hopper-v4/seed1/general_ood_v40_strength_Hopper-v4_{policy}_seed1.json").read_text())
    assert data["source_checkpoint"] == source["checkpoint"]
    assert data["source_training_seed"] == source["training_seed"] == 1
    assert data["config"] == source["config"] and data["runtime"] == source["runtime"]
    records = data["records"]
    assert len(records) == len(seeds) * len(targets)
    assert data["total_probe_interactions"] == len(records) * 1536
    indexed = {(r["seed"], r["target_mode"]): r for r in records}
    assert set(indexed) == {(s, m) for s in seeds for m in targets}
    for row in records:
        control = indexed[row["seed"], "normal"]
        assert row["paired_prefix_equal"] and row["interactions"] == 1536
        assert row["actual_change"] == (row["target_mode"] != "normal")
        assert len(row["episode_endings_by_256"]) == len(row["reward_means_by_256"]) == 6
        assert row["reward_means_by_256"][:4] == control["reward_means_by_256"][:4]
        for name, count in manifest["observer_queries_per_rollout"].items():
            bins, baseline = row["observers"][name], control["observers"][name]
            assert [(b["start"], b["end"]) for b in bins] == [(i, i + 256) for i in range(0, 1536, 256)]
            assert sum(b["observations"] for b in bins) == count
            assert bins[:4] == baseline[:4]
            assert all(math.isfinite(v) for b in bins if b["means"] for v in b["means"].values())
            for phase, index in {"pre256": 3, **PHASES}.items():
                computed = {k: value - baseline[index]["means"][k] for k, value in bins[index]["means"].items()}
                assert computed == row["change_minus_nochange"][name][phase]
    return data


def contrast_statistics(actual, control):
    differences = [a - b for a, b in zip(actual, control)]
    return {"pairs": len(differences), "mean_actual": mean(actual), "mean_nochange": mean(control),
            "mean_paired_difference": mean(differences), "median_paired_difference": median(differences),
            "positive_pairs": sum(v > 0 for v in differences),
            "paired_difference_range": [min(differences), max(differences)]}


def ending_statistics(rows, controls, index):
    def timeout_impossible(group):
        return index < 5 or all(sum(r["episode_endings_by_256"][:index]) > 0 for r in group)

    return {"actual_total": sum(r["episode_endings_by_256"][index] for r in rows),
            "nochange_total": sum(r["episode_endings_by_256"][index] for r in controls),
            "actual_time_limit_impossible": timeout_impossible(rows),
            "nochange_time_limit_impossible": timeout_impossible(controls)}


def summarize(manifest, destination, stage="formal"):
    runs, artifacts, missing = {}, [], []
    for policy in manifest["source_policies"]:
        path = destination / stage / f"{manifest['revision']}_{policy}.json"
        if not path.exists():
            missing.append(policy)
            continue
        runs[policy] = validate_run(json.loads(path.read_text()), manifest, stage)
        artifacts.append({"path": str(path.relative_to(destination)), "bytes": path.stat().st_size})
    output = {"revision": manifest["revision"], "stage": stage, "missing_policies": missing,
              "matrix_complete": not missing, "artifacts": artifacts,
              "json_bytes": sum(a["bytes"] for a in artifacts), "decision": "pending-missing-policies"}
    if missing:
        return output
    methods = {}
    for policy, data in runs.items():
        indexed = {(r["seed"], r["target_mode"]): r for r in data["records"]}
        controls = [indexed[s, "normal"] for s in data["evaluation_seeds"]]
        profiles = {}
        for target in data["target_modes"]:
            if target == "normal":
                continue
            rows = [indexed[s, target] for s in data["evaluation_seeds"]]
            profiles[target] = {
                "observers": {name: {phase: {k: contrast_statistics(
                    [r["observers"][name][i]["means"][k] for r in rows],
                    [r["observers"][name][i]["means"][k] for r in controls]) for k in KEYS}
                    for phase, i in PHASES.items()} for name in manifest["observer_names"]},
                "episode_endings": {phase: ending_statistics(rows, controls, i)
                                    for phase, i in PHASES.items()},
            }
        methods[policy] = {"probe_seeds": data["evaluation_seeds"], "profiles": profiles,
                           "seconds": data["seconds"], "source_checkpoint": data["source_checkpoint"]}
    output.update(methods=methods, decision="diagnostic-complete-not-RL-confirmation",
                  optimizer_updates=0, new_training_seeds=0,
                  statistics_unit="paired probe seeds conditional on each of two fixed seed1 policies; no training-seed CI",
                  limitations=["Frozen weights and fixed reward scale exclude learning/normalizer drift.",
                               "Observers have different hazard/age time units and initialization histories.",
                               "Endings combine termination and truncation. The first post window is before TimeLimit; final-window endings without an earlier reset may be timeouts, not falls.",
                               "No-change controls are shared across profiles; profile rows are not independent controls.",
                               "Hypothetical beta never affects actions. This is not a 50x OOD return comparison."])
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("formal", "preflight"), default="formal")
    args = parser.parse_args()
    result = summarize(json.loads((STUDY / "dispatch_manifest.json").read_text()), DESTINATION, args.stage)
    path = DESTINATION / ("summary.json" if args.stage == "formal" else "preflight_summary.json")
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(result["decision"], "saved", path)


if __name__ == "__main__":
    main()
