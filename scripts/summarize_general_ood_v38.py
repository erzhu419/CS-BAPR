"""Summarize the frozen v38 matrix at the training-seed level."""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.summarize_general_ood_v37 import PROFILES, seed_statistics

REVISION = "general_ood_v38_belief"
STUDY = ROOT / "experiments/general_ood_v38_belief_20261005"
DESTINATION = ROOT / "results" / REVISION
CANDIDATE = "corrected-belief"
METRICS = (
    "surprise", "belief_reset_probability", "belief_effective_window",
    "weighted_lambda_effective", "effective_beta", "q_std", "alpha", "policy_loss", "q_loss",
)


def validate_run(data, manifest, env, variant, seed):
    budget = manifest["training_interactions"][env]
    expected = {
        "revision": REVISION, "stage": "formal", "environment": env,
        "variant": variant, "training_seed": seed,
        "checkpoint": f"{REVISION}_{env}_{variant}_seed{seed}_final.pt",
        "training_budget": budget, "loaded_training_context_count": budget,
        "restore_training_context_before_each_evaluation_block": True,
        "evaluation_only": True, "optimizer_updates_during_evaluation": 0,
        "ood_switch_steps": manifest["ood_switch_steps"][env],
        **{k: manifest[k] for k in (
            "checkpoint_selection", "training_switch_steps", "evaluation_seed_start",
            "evaluation_episodes", "fixed_profile_episodes", "ood_intensity",
            "ood_profile_family", "ood_schedule", "ood_switch_probability",
        )},
    }
    for key, value in expected.items():
        assert data[key] == value, (env, variant, seed, key)
    config = dict(manifest["common_configs"][env])
    if variant == "fixed-penalty":
        config["penalty_scale"] = 0.0
    assert data["config"] == config, (env, variant, seed, "config")
    training = data["training"]
    updates = budget - config["batch_size"] + 1
    assert training["interactions"] == budget
    assert training["optimizer_update_steps"] == updates
    blocks = training["diagnostic_blocks"]
    assert [b["step"] for b in blocks] == list(range(10000, budget + 1, 10000))
    assert [b["updates"] for b in blocks] == [10000 - config["batch_size"] + 1] + [10000] * (len(blocks) - 1)
    assert sum(b["updates"] for b in blocks) == updates
    assert all(math.isfinite(b[k]) for b in blocks for k in METRICS)
    assert [e["step"] for e in training["mode_events"]] == list(range(0, budget, manifest["training_switch_steps"]))
    assert set(data["results"]) == {"id", "markov_50x"}
    assert set(data["fixed_profile_results"]) == PROFILES[env]
    for results, count in ((data["results"], 20), (data["fixed_profile_results"], 5)):
        for name, block in results.items():
            assert len(block["rewards"]) == len(block["lengths"]) == count, name
            assert all(math.isfinite(x) for x in block["rewards"]), name
            assert all(0 < x <= config["max_steps_per_episode"] for x in block["lengths"]), name
            assert math.isfinite(block["std"]), name
            assert math.isclose(mean(block["rewards"]), block["mean"], rel_tol=1e-10, abs_tol=1e-10), name
            if name != "id":
                # The evaluator counts reset observations in addition to steps.
                assert sum(block["profile_counts"].values()) == sum(block["lengths"]) + count, name
    return data


def floor(data):
    return min(r["mean"] for r in data["fixed_profile_results"].values())


def decision_gate(contrasts):
    gates = {
        "ood_wins_both_seeds_against_both_controls": all(
            all(x > 0 for x in c["markov_50x"]["seed_values"]) for c in contrasts.values()),
        "no_all_seed_id_regression_against_either_control": all(
            not all(x < 0 for x in c["id"]["seed_values"]) for c in contrasts.values()),
        "no_worse_seedwise_fixed_profile_floor_against_either_control": all(
            all(x >= 0 for x in c["fixed_profile_floor"]["seed_values"]) for c in contrasts.values()),
    }
    return {**gates, "admit_further_confirmation": all(gates.values())}


def summarize(manifest, destination):
    runs, artifacts, missing = {}, [], []
    for env in manifest["environments"]:
        runs[env] = {variant: {} for variant in manifest["variants"]}
        for variant, cells in runs[env].items():
            for seed in manifest["training_seeds"]:
                path = destination / "formal" / env / f"seed{seed}" / f"{REVISION}_{env}_{variant}_seed{seed}.json"
                if not path.exists():
                    missing.append({"environment": env, "variant": variant, "seed": seed})
                    continue
                cells[seed] = validate_run(json.loads(path.read_text()), manifest, env, variant, seed)
                artifacts.append({"path": str(path.relative_to(destination)), "bytes": path.stat().st_size})
        for seed in manifest["training_seeds"]:
            paired = [cells[seed] for cells in runs[env].values() if seed in cells]
            if paired:
                assert all(d["runtime"] == paired[0]["runtime"] for d in paired), (env, seed, "runtime")
                assert all(d["training"]["mode_events"] == paired[0]["training"]["mode_events"] for d in paired), (env, seed, "mode_events")

    methods, contrasts, gates = {}, {}, {}
    for env, variants in runs.items():
        methods[env], contrasts[env] = {}, {}
        for variant, cells in variants.items():
            seeds = sorted(cells)
            diagnostics = []
            for seed in seeds:
                d = cells[seed]
                blocks = d["training"]["diagnostic_blocks"]
                count = sum(b["updates"] for b in blocks)
                ood = d["results"]["markov_50x"]
                diagnostics.append({
                    "training_seed": seed, "runtime": d["runtime"],
                    "training_seconds": d["training"]["seconds"],
                    "recent_completed_training_return": mean(d["training"]["recent_completed_episode_returns"]),
                    "update_weighted_means": {k: sum(b[k] * b["updates"] for b in blocks) / count for k in METRICS},
                    "first_block": blocks[0], "last_block": blocks[-1],
                    "ood_non_id_transition_count": sum(n for p, n in ood["profile_counts"].items() if p != "id"),
                    "ood_transition_count": sum(ood["lengths"]),
                })
            methods[env][variant] = {
                "training_seeds": seeds,
                "returns": {mode: seed_statistics([cells[s]["results"][mode]["mean"] for s in seeds]) for mode in ("id", "markov_50x")},
                "episode_lengths": {mode: seed_statistics([mean(cells[s]["results"][mode]["lengths"]) for s in seeds]) for mode in ("id", "markov_50x")},
                "fixed_profiles": {p: seed_statistics([cells[s]["fixed_profile_results"][p]["mean"] for s in seeds]) for p in sorted(PROFILES[env])},
                "fixed_profile_floor": seed_statistics([floor(cells[s]) for s in seeds]),
                "diagnostics": diagnostics,
            }
        for control in manifest["variants"]:
            if control == CANDIDATE:
                continue
            seeds = sorted(set(variants[CANDIDATE]) & set(variants[control]))
            candidate, comparator = variants[CANDIDATE], variants[control]
            contrasts[env][control] = {
                "training_seeds": seeds,
                **{mode: seed_statistics([candidate[s]["results"][mode]["mean"] - comparator[s]["results"][mode]["mean"] for s in seeds]) for mode in ("id", "markov_50x")},
                "fixed_profile_floor": seed_statistics([floor(candidate[s]) - floor(comparator[s]) for s in seeds]),
                "fixed_profiles": {p: seed_statistics([candidate[s]["fixed_profile_results"][p]["mean"] - comparator[s]["fixed_profile_results"][p]["mean"] for s in seeds]) for p in sorted(PROFILES[env])},
            }
        complete = all(len(cells) == len(manifest["training_seeds"]) for cells in variants.values())
        gates[env] = decision_gate(contrasts[env]) if complete else None
    eligible = [env for env, gate in gates.items() if gate and gate["admit_further_confirmation"]]
    return {
        "revision": REVISION, "completed_runs": len(artifacts), "expected_runs": manifest["logical_cells"],
        "matrix_complete": not missing, "missing_cells": missing,
        "statistics_unit": "training-seed means, not pooled evaluation episodes",
        "ci95_method": "descriptive Student t intervals across two training seeds; no multiplicity correction",
        "methods": methods, "corrected_minus_control": contrasts,
        "frozen_decision_gate": gates, "eligible_environments": eligible,
        "decision": "pending-missing-cells" if missing else ("eligible-for-confirmation" if eligible else "hold-mechanism-diagnosis"),
        "artifacts": artifacts, "evaluation_json_bytes": sum(a["bytes"] for a in artifacts),
        "limitations": [
            "Two training seeds and reduced Hopper budget; this is a development screen.",
            "Matched fixed-penalty is RE-SAC-style, not the v37 RE-SAC preset; do not pool rounds.",
            "No NAU, SINDy, Jacobian loss or evaluation-time belief update.",
            "Markov starts ID: Hopper corrected seed1 ends before its first OOD switch and has zero non-ID exposure.",
            "Combined surprise diagnostics do not identify which component caused saturation or establish causality for divergence.",
        ],
    }


def main():
    manifest = json.loads((STUDY / "dispatch_manifest.json").read_text())
    summary = summarize(manifest, DESTINATION)
    path = DESTINATION / "summary.json"
    path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"{summary['completed_runs']}/{summary['expected_runs']} valid results; {summary['decision']}; saved {path}")
    for env, methods in summary["methods"].items():
        for variant, data in methods.items():
            print(env, variant, {k: round(v["mean"], 2) if v else None for k, v in data["returns"].items()})
    print("frozen gates", summary["frozen_decision_gate"])


if __name__ == "__main__":
    main()
