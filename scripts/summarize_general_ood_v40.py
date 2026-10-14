"""Summarize the strength control and observed switch response of one seed."""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, stdev
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.summarize_general_ood_v39 import PROFILES, floor, weighted_metrics

REVISION = "general_ood_v40_strength"
STUDY = ROOT / "experiments/general_ood_v40_strength_20261006"
DESTINATION = ROOT / "results" / REVISION
SWITCH_KEYS = ("surprise", "belief_reset_probability", "belief_effective_window", "effective_beta")
SCALES = {"relative-adaptive": 1.0, "fixed-minus2": 0.0, "fixed-minus2p3": 0.3}


def validate_run(data, manifest, variant):
    expected = {
        "revision": REVISION, "stage": "formal", "environment": "Hopper-v4",
        "variant": variant, "training_seed": 1,
        "training_budget": manifest["training_interactions"],
        "loaded_training_context_count": manifest["training_interactions"],
        "checkpoint": f"{REVISION}_Hopper-v4_{variant}_seed1_final.pt",
        "evaluation_only": True, "restore_training_context_before_each_evaluation_block": True,
        "ood_rng_reseeded_each_episode": True,
        **{k: manifest[k] for k in ("checkpoint_selection", "training_switch_steps", "diagnostic_interval",
                                  "evaluation_seed_start", "evaluation_episodes", "fixed_profile_episodes",
                                  "ood_intensity", "ood_profile_family", "ood_schedule", "ood_switch_probability",
                                  "ood_switch_steps", "optimizer_updates_during_evaluation")},
    }
    for key, value in expected.items():
        assert data[key] == value, (variant, key)
    base = json.loads((ROOT / "experiments/general_ood_v38_belief_20261005/dispatch_manifest.json").read_text())["common_configs"]["Hopper-v4"]
    assert data["config"] == dict(base, penalty_scale=SCALES[variant]), variant
    assert data["config"]["enable_rollout_surprise"] and data["config"]["surprise_window"] == 256
    training = data["training"]
    assert training["interactions"] == 100000
    assert training["optimizer_update_steps"] == manifest["optimizer_updates_per_formal_cell"] == 99745
    blocks = training["diagnostic_blocks"]
    assert [b["step"] for b in blocks] == list(range(1000, 100001, 1000))
    assert [b["updates"] for b in blocks] == [745] + [1000] * 99
    assert all(math.isfinite(v) for b in blocks for k, v in b.items() if k != "recent_completed_episode_return")
    modes = training["mode_events"]
    assert [e["step"] for e in modes] == list(range(0, 100000, 5000))
    diagnostics = training["switch_diagnostics"]
    assert all(diagnostics[k] == v for k, v in manifest["switch_diagnostics"].items()
               if k != "distinguish_actual_change_from_same_mode_redraw")
    events = diagnostics["events"]
    assert len(events) == 19
    for event, previous, current in zip(events, modes, modes[1:]):
        assert event["step"] == current["step"]
        assert event["previous_mode"] == previous["mode"] and event["mode"] == current["mode"]
        assert event["actual_change"] == (previous["mode"] != current["mode"])
        assert [(b["relative_start"], b["relative_end"]) for b in event["bins"]] == [
            (i, i + 50) for i in range(-250, 500, 50)]
        for b in event["bins"]:
            assert b["interactions"] == b["updates"] == 50
            assert all(math.isfinite(v) for v in b["means"].values())
            assert all(math.isfinite(v) for ext in b["extrema"].values() for v in ext.values())
            if variant != "relative-adaptive":
                beta = -2.0 - SCALES[variant]
                assert math.isclose(b["means"]["effective_beta"], beta, abs_tol=1e-12)
                assert b["extrema"]["effective_beta"] == {"min": beta, "max": beta}
    if variant != "relative-adaptive":
        warmup = data["config"]["bapr_warmup_iters"]
        assert warmup == manifest["common_beta_warmup_optimizer_updates"] == 100
        expected_mean = -2.0 - SCALES[variant] * (99745 - warmup) / 99745
        assert math.isclose(weighted_metrics(blocks)["effective_beta"], expected_mean, abs_tol=1e-12)
    assert set(data["results"]) == {"id", "markov_50x"}
    assert set(data["fixed_profile_results"]) == PROFILES
    for results, count in ((data["results"], 20), (data["fixed_profile_results"], 5)):
        for name, block in results.items():
            assert len(block["rewards"]) == len(block["lengths"]) == count
            assert all(math.isfinite(x) for x in block["rewards"])
            assert all(0 < x <= 1000 for x in block["lengths"])
            assert math.isclose(mean(block["rewards"]), block["mean"], rel_tol=1e-10, abs_tol=1e-10)
            if name != "id":
                assert sum(block["profile_counts"].values()) == sum(block["lengths"]) + count
    ood = data["results"]["markov_50x"]
    assert data["ood_transition_count"] == sum(ood["lengths"])
    assert data["ood_non_id_transition_count"] == sum(n for p, n in ood["profile_counts"].items() if p != "id")
    return data


def bin_means(bins):
    updates = sum(b["updates"] for b in bins)
    return {key: sum(b["means"][key] * b["updates"] for b in bins) / updates for key in SWITCH_KEYS}


def switch_summary(events, inferred_belief):
    records = []
    for event in events:
        pre = [b for b in event["bins"] if b["relative_end"] <= 0]
        post = [b for b in event["bins"] if b["relative_start"] >= 0]
        before, after = bin_means(pre), bin_means(post)
        records.append({
            **{k: event[k] for k in ("step", "previous_mode", "mode", "actual_change")},
            "pre250": before, "first50": bin_means(post[:1]), "post500": after,
            "post_minus_pre": {k: after[k] - before[k] for k in SWITCH_KEYS},
            "post_peak_reset_probability": max(b["extrema"]["belief_reset_probability"]["max"] for b in post),
            "episode_end_rate_pre": sum(b["episode_endings"] for b in pre) / sum(b["interactions"] for b in pre),
            "episode_end_rate_post": sum(b["episode_endings"] for b in post) / sum(b["interactions"] for b in post),
        })
    groups = {}
    for name, change in (("actual_change", True), ("same_mode_redraw", False)):
        selected = [e for e in records if e["actual_change"] == change]
        groups[name] = {
            "event_count": len(selected),
            **{window: {k: mean(e[window][k] for e in selected) for k in SWITCH_KEYS}
               for window in ("pre250", "first50", "post500", "post_minus_pre")},
            "post_peak_reset_range": [min(e["post_peak_reset_probability"] for e in selected),
                                      max(e["post_peak_reset_probability"] for e in selected)],
        }
    return {"inferred_belief": inferred_belief, "groups": groups, "events": records}


def paired_differences(candidate, control):
    differences = [a - b for a, b in zip(candidate["rewards"], control["rewards"])]
    return {"mean": mean(differences), "sample_std": stdev(differences),
            "wins": sum(x > 0 for x in differences), "ties": differences.count(0),
            "episode_count": len(differences), "values": differences}


def summarize(manifest, destination):
    runs, artifacts, missing = {}, [], []
    for variant in manifest["variants"]:
        path = destination / "formal/Hopper-v4/seed1" / f"{REVISION}_Hopper-v4_{variant}_seed1.json"
        if not path.exists():
            missing.append(variant)
            continue
        runs[variant] = validate_run(json.loads(path.read_text()), manifest, variant)
        artifacts.append({"path": str(path.relative_to(destination)), "bytes": path.stat().st_size})
    output = {"revision": REVISION, "n_training_seeds": 1, "training_seed": 1,
              "statistics_unit": "one policy per arm; ordered evaluation episodes are not training replicates",
              "matrix_complete": not missing, "missing_cells": missing, "artifacts": artifacts,
              "evaluation_json_bytes": sum(a["bytes"] for a in artifacts), "decision": "pending-missing-cells"}
    if missing:
        return output
    candidate = runs["relative-adaptive"]
    assert all(d["runtime"] == candidate["runtime"] for d in runs.values())
    assert all(d["training"]["mode_events"] == candidate["training"]["mode_events"] for d in runs.values())
    methods = {variant: {
        "returns": {m: r["mean"] for m, r in data["results"].items()},
        "fixed_profiles": {p: r["mean"] for p, r in data["fixed_profile_results"].items()},
        "fixed_profile_floor": floor(data), "ood_non_id_transition_count": data["ood_non_id_transition_count"],
        "ood_transition_count": data["ood_transition_count"], "training_seconds": data["training"]["seconds"],
        "diagnostics_update_weighted": weighted_metrics(data["training"]["diagnostic_blocks"]),
        "last_diagnostic_block": data["training"]["diagnostic_blocks"][-1],
        "switch_response": switch_summary(data["training"]["switch_diagnostics"]["events"], variant != "fixed-minus2p3"),
    } for variant, data in runs.items()}
    contrasts = {variant: {
        "paired_episode_differences": {m: paired_differences(candidate["results"][m], data["results"][m])
                                       for m in ("id", "markov_50x")},
        "fixed_profile_floor": floor(candidate)["mean_return"] - floor(data)["mean_return"],
        "fixed_profiles": {p: candidate["fixed_profile_results"][p]["mean"] - data["fixed_profile_results"][p]["mean"]
                           for p in sorted(PROFILES)},
    } for variant, data in runs.items() if variant != "relative-adaptive"}
    conditions = {
        **{f"{mode}_matches_both_constants": all(candidate["results"][mode]["mean"] >= d["results"][mode]["mean"]
                                                for v, d in runs.items() if v != "relative-adaptive")
           for mode in ("id", "markov_50x")},
        "fixed_floor_matches_both_constants": all(c["fixed_profile_floor"] >= 0 for c in contrasts.values()),
        "candidate_has_non_id_exposure": candidate["ood_non_id_transition_count"] > 0,
    }
    output.update(methods=methods, relative_minus_control=contrasts, frozen_return_conditions=conditions,
                  decision="passes-returns-needs-switch-and-stability-judgment" if all(conditions.values()) else "hold-no-benefit-over-strength-matched-controls",
                  surprise_observation={"reward_and_q_spread": "recent 256-transition rollout, supplied on every update",
                                        "reward_normalizer": "EMA variance of replay reward batches",
                                        "critic_l1_drift": "global current/target critic norms"},
                  limitations=["One selected development seed; no independent confirmation or training-seed CI.",
                               "Two same-mode redraws are descriptive controls, not a causal contrast.",
                               "Belief updates during training only; evaluation uses context updates, with block-level restoration and within-block history carryover.",
                               "Fifty-step extrema do not provide exact peak timing. Do not pool with v38's 300k endpoints."])
    return output


def main():
    result = summarize(json.loads((STUDY / "dispatch_manifest.json").read_text()), DESTINATION)
    path = DESTINATION / "summary.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(result["decision"], "saved", path)
    for variant, data in result.get("methods", {}).items():
        print(variant, data["returns"], "floor", data["fixed_profile_floor"])


if __name__ == "__main__":
    main()
