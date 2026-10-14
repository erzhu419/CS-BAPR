"""Summarize one development seed, preserving original and repaired evaluation."""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "experiments/general_ood_v39_surprise_20261006"
DESTINATION = ROOT / "results/general_ood_v39_surprise"
REVISION = "general_ood_v39_surprise"
PROFILES = {"easy_surface", "heavy_traffic", "mega_event", "slippery_rush"}
TRACE_KEYS = ("effective_beta", "q_std", "q_mean", "alpha", "policy_entropy",
              "reward_normalization_std", "q_loss", "policy_loss")


def validate_run(data, manifest, variant, paired=False):
    expected = {
        "revision": REVISION, "environment": "Hopper-v4", "variant": variant,
        "training_seed": 1, "training_budget": manifest["training_interactions"],
        "loaded_training_context_count": manifest["training_interactions"],
        "checkpoint": f"{REVISION}_Hopper-v4_{variant}_seed1_final.pt",
        "evaluation_only": True, "restore_training_context_before_each_evaluation_block": True,
        **{k: manifest[k] for k in ("checkpoint_selection", "evaluation_seed_start", "evaluation_episodes",
                                  "fixed_profile_episodes", "ood_intensity", "ood_profile_family",
                                  "ood_schedule", "ood_switch_probability", "ood_switch_steps",
                                  "optimizer_updates_during_evaluation")},
    }
    if paired:
        expected.update(evaluation_revision="episode-seeded-ood", ood_rng_reseeded_on_episode_reset=True,
                        loaded_training_update_steps=manifest["optimizer_updates_per_formal_cell"])
    else:
        expected.update(stage="formal", training_switch_steps=manifest["training_switch_steps"],
                        diagnostic_interval=manifest["diagnostic_interval"])
        base = json.loads((ROOT / "experiments/general_ood_v38_belief_20261005/dispatch_manifest.json").read_text())["common_configs"]["Hopper-v4"]
        assert data["config"] == dict(base, penalty_scale=0.0 if variant == "fixed-minus2" else 1)
        training = data["training"]
        assert training["interactions"] == manifest["training_interactions"]
        assert training["optimizer_update_steps"] == manifest["optimizer_updates_per_formal_cell"]
        blocks = training["diagnostic_blocks"]
        assert [b["step"] for b in blocks] == list(range(1000, 100001, 1000))
        assert [b["updates"] for b in blocks] == [745] + [1000] * 99
        assert all(math.isfinite(v) for b in blocks for k, v in b.items() if k != "recent_completed_episode_return")
        assert [e["step"] for e in training["mode_events"]] == list(range(0, 100000, 5000))
    for key, value in expected.items():
        assert data[key] == value, (variant, key)
    assert set(data["results"]) == {"id", "markov_50x"}
    assert set(data["fixed_profile_results"]) == PROFILES
    for results, count in ((data["results"], 20), (data["fixed_profile_results"], 5)):
        for name, block in results.items():
            assert len(block["rewards"]) == len(block["lengths"]) == count, (variant, name)
            assert all(math.isfinite(x) for x in block["rewards"]), (variant, name)
            assert all(0 < x <= 1000 for x in block["lengths"]), (variant, name)
            assert math.isclose(mean(block["rewards"]), block["mean"], rel_tol=1e-10, abs_tol=1e-10)
            if name != "id":
                assert sum(block["profile_counts"].values()) == sum(block["lengths"]) + count
    ood = data["results"]["markov_50x"]
    assert data["ood_transition_count"] == sum(ood["lengths"])
    assert data["ood_non_id_transition_count"] == sum(n for p, n in ood["profile_counts"].items() if p != "id")
    return data


def floor(data):
    value, profile = min((r["mean"], p) for p, r in data["fixed_profile_results"].items())
    return {"profile": profile, "mean_return": value}


def weighted_metrics(blocks):
    count = sum(b["updates"] for b in blocks)
    keys = set(blocks[0]) - {"step", "updates", "recent_completed_episode_return"}
    return {k: sum(b[k] * b["updates"] for b in blocks) / count for k in sorted(keys)}


def summarize(manifest, destination):
    original, paired, artifacts, missing = {}, {}, [], []
    for phase, runs, suffix in (("formal", original, ""), ("paired_eval", paired, "_paired_eval")):
        for variant in manifest["variants"]:
            path = destination / phase / "Hopper-v4/seed1" / f"{REVISION}_Hopper-v4_{variant}_seed1{suffix}.json"
            if not path.exists():
                missing.append({"phase": phase, "variant": variant})
                continue
            runs[variant] = validate_run(json.loads(path.read_text()), manifest, variant, paired=phase == "paired_eval")
            artifacts.append({"path": str(path.relative_to(destination)), "bytes": path.stat().st_size})
    output = {"revision": REVISION, "n_training_seeds": 1, "training_seed": 1,
              "statistics_unit": "one trained policy per arm; evaluation episodes are not training replicates",
              "missing_cells": missing, "matrix_complete": not missing, "artifacts": artifacts,
              "decision": "pending-missing-cells"}
    if missing:
        return output
    reference = original["raw-adaptive"]
    for variant in manifest["variants"]:
        assert original[variant]["runtime"] == reference["runtime"] == paired[variant]["runtime"]
        assert original[variant]["training"]["mode_events"] == reference["training"]["mode_events"]
        # Seeding the OOD stream changes Markov schedules only in these noise-free profiles.
        for name, old, new in (("id", original[variant]["results"]["id"], paired[variant]["results"]["id"]),
                               *[(p, original[variant]["fixed_profile_results"][p], paired[variant]["fixed_profile_results"][p]) for p in sorted(PROFILES)]):
            assert old["lengths"] == new["lengths"], (variant, name)
            assert all(math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-8) for a, b in zip(old["rewards"], new["rewards"])), (variant, name)
    methods = {}
    for variant, data in paired.items():
        blocks = original[variant]["training"]["diagnostic_blocks"]
        methods[variant] = {
            "checkpoint": data["checkpoint"],
            "original_returns": {m: r["mean"] for m, r in original[variant]["results"].items()},
            "paired_returns": {m: r["mean"] for m, r in data["results"].items()},
            "episode_lengths": {m: mean(r["lengths"]) for m, r in data["results"].items()},
            "fixed_profiles": {p: r["mean"] for p, r in data["fixed_profile_results"].items()},
            "fixed_profile_floor": floor(data),
            "ood_non_id_transition_count": data["ood_non_id_transition_count"],
            "ood_transition_count": data["ood_transition_count"],
            "training_seconds": original[variant]["training"]["seconds"],
            "diagnostics_update_weighted": weighted_metrics(blocks),
            "last_20k_diagnostics_update_weighted": weighted_metrics(blocks[-20:]),
            "last_diagnostic_block": blocks[-1],
            "effective_beta_block_mean_range_after_first_1k": [min(b["effective_beta"] for b in blocks[1:]), max(b["effective_beta"] for b in blocks[1:])],
        }
    candidate = paired["relative-adaptive"]
    contrasts = {v: {**{m: candidate["results"][m]["mean"] - d["results"][m]["mean"] for m in ("id", "markov_50x")},
                     "fixed_profile_floor": floor(candidate)["mean_return"] - floor(d)["mean_return"]}
                 for v, d in paired.items() if v != "relative-adaptive"}
    constants = [paired[v] for v in ("fixed-minus2", "fixed-minus3")]
    conditions = {
        "id_improves_over_raw": contrasts["raw-adaptive"]["id"] > 0,
        "nominal_markov_return_improves_over_raw": contrasts["raw-adaptive"]["markov_50x"] > 0,
        "relative_policy_has_non_id_exposure": candidate["ood_non_id_transition_count"] > 0,
        "id_matches_best_constant": candidate["results"]["id"]["mean"] >= max(d["results"]["id"]["mean"] for d in constants),
        "markov_matches_best_constant": candidate["results"]["markov_50x"]["mean"] >= max(d["results"]["markov_50x"]["mean"] for d in constants),
        "fixed_floor_matches_best_constant": floor(candidate)["mean_return"] >= max(floor(d)["mean_return"] for d in constants),
    }
    raw, fixed3 = original["raw-adaptive"], original["fixed-minus3"]
    output.update(
        methods=methods, relative_minus_control=contrasts, frozen_return_conditions=conditions,
        raw_vs_fixed_minus3={
            "original_evaluation_equal": raw["results"] == fixed3["results"] and raw["fixed_profile_results"] == fixed3["fixed_profile_results"],
            "paired_evaluation_equal": paired["raw-adaptive"]["results"] == paired["fixed-minus3"]["results"] and paired["raw-adaptive"]["fixed_profile_results"] == paired["fixed-minus3"]["fixed_profile_results"],
            "max_absolute_training_trace_differences": {k: max(abs(a[k] - b[k]) for a, b in zip(raw["training"]["diagnostic_blocks"], fixed3["training"]["diagnostic_blocks"])) for k in TRACE_KEYS},
        },
        id_and_fixed_profile_evaluation_preserved=True,
        decision="passes-return-conditions-needs-stability-judgment" if all(conditions.values()) else "hold-no-benefit-over-fixed-penalty",
        limitations=["One selected development seed; no independent confirmation or seed-level CI.",
                     "Original Markov streams were not fully episode-paired; use repaired evaluation for comparisons.",
                     "Raw and fixed-minus3 end before OOD switches; their nominal Markov scores contain no non-ID exposure.",
                     "1000-step means do not establish individual shock response timing; belief updates only during training.",
                     "Do not pool 100k v39 endpoints with 300k v38 or claim NAU/SINDy benefits."],
    )
    return output


def main():
    result = summarize(json.loads((STUDY / "dispatch_manifest.json").read_text()), DESTINATION)
    path = DESTINATION / "summary.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(result["decision"], "saved", path)
    for variant, data in result.get("methods", {}).items():
        print(variant, data["paired_returns"], "floor", data["fixed_profile_floor"])


if __name__ == "__main__":
    main()
