"""Keep post-intervention effects and ID context dependence separate."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_general_ood_v136_warmup_context as worker
from summarize_general_ood_v129 import aggregate


def validate(row, original, smoke=False):
    json.dumps(row, allow_nan=False)
    seed, source = row["training_seed"], row["source"]
    spec = worker.stage_spec(seed, smoke)
    assert seed in (3, 4) and row["revision"] == worker.REVISION
    assert row["spec"] == spec and row["smoke"] is smoke
    assert row["evaluation_device"] == "cpu" and not row["cuda_initialized"] and not row["replay_loaded"]
    assert row["optimizer_updates"] == row["new_training_interactions"] == 0
    assert not row["source_readout_refitted"] and row["checkpoint_selection"] == "predeclared-final-only"
    assert row["reward_changes"] == [] and not row["policy_receives_profile_labels"]
    assert row["ablation_scope"] == "post-warmup inference context, not training representation"
    assert (original["training_seed"], original["arm"], original["smoke"]) == (seed, "mse-control", False)
    assert source["revision"] == original["revision"] == worker.v135.v134.v133.REVISION
    assert original["checkpoint_selection"] == "final-not-best" and source["checkpoint"] == original["checkpoint"]
    assert source["context_count"] == original["training"]["interactions"] == 300000
    assert source["initial_context_normalizer"]["count"] == 300000
    assert source["training_steps"] == original["training"]["optimizer_updates"] == 299745
    assert source["config"] == dict(original["config"], device="cpu")
    assert all(row["runtime"][k] == original["runtime"][k] for k in row["runtime"])
    profiles = worker.v135.make_ood_profiles("Hopper-v4", intensity=50., family="structured")
    assert row["profiles"] == profiles
    profiles = {p["name"]: p for p in profiles}
    pairs, expected = row["pairs"], worker.cells(spec)
    assert [{k: p[k] for k in expected[0]} for p in pairs] == expected
    episodes = []
    for pair in pairs:
        assert pair["prefix_equal"] and pair["prefix_max_abs_difference"] == 0.
        online, held = pair["episodes"]
        assert [e["context_mode"] for e in pair["episodes"]] == list(worker.CONTEXT_MODES)
        assert online["context_at_intervention"] == held["context_at_intervention"]
        assert online["warmup_end_physical_state"] == held["warmup_end_physical_state"]
        assert online["warmup_return"] == held["warmup_return"]
        assert pair["prefix_steps"] == min(online["length"], 201) == min(held["length"], 201)
        for e in pair["episodes"]:
            assert all(e[k] == pair[k] for k in expected[0])
            length, post = e["length"], e["post_warmup_steps"]
            assert 0 < length <= spec["max_steps"] and e["native_reward_verified"]
            assert e["warmup_steps"] == min(length, 200) and post == max(length-200, 0)
            assert e["intervention_reached"] is (post > 0)
            assert e["context_updates"] == (length if e["context_mode"] == "online" else min(length, 200))
            assert e["mean_context_l2"] > 0.
            if post:
                assert e["context_at_intervention"] is not None and e["post_warmup_mean_velocity"] is not None
                assert e["mean_post_context_change_l2"] == 0. if e["context_mode"] == "held-context" else e["mean_post_context_change_l2"] >= 0.
            else:
                assert e["post_warmup_return"] == 0. and e["post_warmup_mean_velocity"] is None
                assert e["mean_post_context_change_l2"] is None
            np.testing.assert_allclose(e["episode_return"], e["warmup_return"]+e["post_warmup_return"], rtol=1e-12, atol=1e-12)
            assert e["noise_scale"] == (0. if e["mode"] == "deterministic" else 1.)
            assert e["mean_action_noise_l2"] == 0. if e["mode"] == "deterministic" else e["mean_action_noise_l2"] > 0.
            assert smoke or (not e["diagnostic_cutoff"] and (e["physical_termination"] or e["time_limit_truncation"]))
            target = e["condition"].removeprefix("abrupt/")
            events = e["profile_events"]
            schedule = [(1, "id")]+([(201, target)] if target != "id" and post else [])
            assert [(v["step"], v["profile"]) for v in events] == schedule and e["switches"] == len(events)-1
            counts = {"id": e["warmup_steps"], target: post} if target != "id" and post else {"id": length}
            assert e["profile_counts"] == counts and e["non_id_steps"] == (post if target != "id" else 0)
            for event in events:
                np.testing.assert_allclose(list(event["actual_multipliers"].values()),
                    [profiles[event["profile"]].get(k, 1.) for k in event["actual_multipliers"]], rtol=1e-12, atol=1e-12)
            episodes.append(e)
    assert row["evaluation_interactions"] == sum(e["length"] for e in episodes)
    return episodes


def summarize(rows, sources):
    assert len(rows) == len(sources) == 2
    indexed, originals = {r["training_seed"]: r for r in rows}, {r["training_seed"]: r for r in sources}
    assert set(indexed) == set(originals) == {3, 4}
    records, paired = [], []
    for seed in (3, 4):
        row = indexed[seed]
        episodes = validate(row, originals[seed])
        groups = {}
        for condition in worker.CONDITIONS:
            for mode in ("deterministic", "sampled"):
                subsets = {}
                for context in worker.CONTEXT_MODES:
                    subset = [e for e in episodes if (e["condition"], e["mode"], e["context_mode"]) == (condition, mode, context)]
                    failures = Counter(k for e in subset for k in e["final_physical_state"]["unhealthy_conditions"])
                    subsets[context] = dict(aggregate(subset),
                        mean_post_warmup_steps=float(np.mean([e["post_warmup_steps"] for e in subset])),
                        mean_post_warmup_return=float(np.mean([e["post_warmup_return"] for e in subset])),
                        intervention_reached=sum(e["intervention_reached"] for e in subset),
                        non_id_steps=sum(e["non_id_steps"] for e in subset), unhealthy_conditions=dict(failures))
                groups[f"{condition}/{mode}"] = subsets
                selected = [p for p in row["pairs"] if (p["condition"], p["mode"]) == (condition, mode)]
                paired.append(dict(training_seed=seed, condition=condition, mode=mode, pairs=len(selected),
                    online_minus_held={key: float(np.mean([p["episodes"][0][key]-p["episodes"][1][key] for p in selected]))
                                      for key in ("post_warmup_steps", "post_warmup_return")},
                    post_step_wins=sum(p["episodes"][0]["post_warmup_steps"] > p["episodes"][1]["post_warmup_steps"] for p in selected),
                    post_step_ties=sum(p["episodes"][0]["post_warmup_steps"] == p["episodes"][1]["post_warmup_steps"] for p in selected),
                    post_step_losses=sum(p["episodes"][0]["post_warmup_steps"] < p["episodes"][1]["post_warmup_steps"] for p in selected)))
        records.append(dict(training_seed=seed, groups=groups, seconds=row["seconds"],
            verified_prefixes=len(row["pairs"]), early_failure_episodes=sum(not e["intervention_reached"] for e in episodes)))
    return dict(revision=worker.REVISION, independent_training_seeds=2, frozen_models=2,
        evaluation_episodes=sum(2*len(r["pairs"]) for r in rows), verified_prefixes=sum(len(r["pairs"]) for r in rows),
        evaluation_interactions=sum(r["evaluation_interactions"] for r in rows), new_training_interactions=0,
        optimizer_updates=0, records=records, paired_differences=paired,
        interpretation="Post-warmup inference intervention with exact matched prefixes. ID hold control remains separate; early failures retained. Two training seeds, no strong-baseline or training-ablation claim.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads((args.result_dir/f"seed{seed}/warmup_context.json").read_text()) for seed in (3, 4)]
    sources = [json.loads((args.source_dir/f"seed{seed}_mse-control/source.json").read_text()) for seed in (3, 4)]
    result = summarize(rows, sources)
    (args.result_dir/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    for row in result["paired_differences"]:
        if row["mode"] == "sampled":
            print(row["training_seed"], row["condition"], row["online_minus_held"])


if __name__ == "__main__":
    main()
