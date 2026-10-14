"""Paired inference-context effects, separated by physical condition and seed."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_general_ood_v135_ood_context as worker
from summarize_general_ood_v129 import aggregate


def validate(row, original, smoke=False):
    json.dumps(row, allow_nan=False)
    seed, context = row["training_seed"], row["context_mode"]
    spec = worker.stage_spec(seed, context, smoke)
    assert seed in (3, 4) and context in worker.CONTEXT_MODES
    assert row["revision"] == worker.REVISION and row["spec"] == spec and row["smoke"] is smoke
    assert row["evaluation_device"] == "cpu" and not row["cuda_initialized"] and not row["replay_loaded"]
    assert row["optimizer_updates"] == row["new_training_interactions"] == 0
    assert row["checkpoint_selection"] == "predeclared-final-only" and not row["source_readout_refitted"]
    assert row["reward_changes"] == [] and not row["policy_receives_profile_labels"]
    assert row["ablation_scope"] == "inference context, not training representation"
    source = row["source"]
    assert source["revision"] == original["revision"] == worker.v134.v133.REVISION
    assert (original["training_seed"], original["arm"], original["smoke"]) == (seed, "mse-control", False)
    assert original["checkpoint_selection"] == "final-not-best" and source["checkpoint"] == original["checkpoint"]
    assert source["context_count"] == original["training"]["interactions"] == 300000
    assert source["training_steps"] == original["training"]["optimizer_updates"] == 299745
    assert source["initial_context_normalizer"]["count"] == 300000
    assert source["config"] == dict(original["config"], device="cpu")
    assert all(row["runtime"][k] == original["runtime"][k] for k in row["runtime"])
    profiles = worker.make_ood_profiles("Hopper-v4", intensity=50., family="structured")
    assert row["profiles"] == profiles
    profiles = {p["name"]: p for p in profiles}
    episodes, expected = row["evaluations"], worker.cells(spec)
    assert [{k: e[k] for k in expected[0]} for e in episodes] == expected
    for e in episodes:
        length, condition, events = e["length"], e["condition"], e["profile_events"]
        assert 0 < length <= spec["max_steps"] and e["native_reward_verified"]
        assert e["context_mode"] == context and e["context_updates"] == (length if context == "online" else 0)
        assert e["mean_context_l2"] == 0. if context == "zero-context" else e["mean_context_l2"] > 0.
        assert e["noise_scale"] == (0. if e["mode"] == "deterministic" else 1.)
        assert e["mean_action_noise_l2"] == 0. if e["mode"] == "deterministic" else e["mean_action_noise_l2"] > 0.
        assert smoke or (not e["diagnostic_cutoff"] and (e["physical_termination"] or e["time_limit_truncation"]))
        assert events[0]["step"] == 1 and e["switches"] == len(events)-1
        counts = {}
        for i, event in enumerate(events):
            name = event["profile"]
            assert name in profiles
            np.testing.assert_allclose(list(event["actual_multipliers"].values()),
                [profiles[name].get(k, 1.) for k in event["actual_multipliers"]], rtol=1e-12, atol=1e-12)
            end = events[i+1]["step"] if i+1 < len(events) else length+1
            assert end > event["step"]
            counts[name] = counts.get(name, 0)+end-event["step"]
        assert counts == e["profile_counts"] and sum(counts.values()) == length
        assert e["non_id_steps"] == sum(n for name, n in counts.items() if name != "id")
        if condition == "markov":
            assert events[0]["profile"] == "id"
            assert [event["step"] for event in events] == list(range(1, length+1, 50))
            assert all(a["profile"] != b["profile"] for a, b in zip(events, events[1:]))
        else:
            assert len(events) == 1 and events[0]["profile"] == condition.removeprefix("static/")
    assert row["evaluation_interactions"] == sum(e["length"] for e in episodes)
    return episodes


def summarize(rows, sources):
    combinations = [(seed, context) for seed in (3, 4) for context in worker.CONTEXT_MODES]
    indexed = {(r["training_seed"], r["context_mode"]): r for r in rows}
    originals = {r["training_seed"]: r for r in sources}
    assert len(rows) == 4 and set(indexed) == set(combinations)
    assert len(sources) == 2 and set(originals) == {3, 4}
    records, paired = [], []
    total_steps = 0
    for seed, context in combinations:
        row = indexed[seed, context]
        episodes = validate(row, originals[seed])
        total_steps += row["evaluation_interactions"]
        groups = {}
        for condition in worker.CONDITIONS:
            for mode in ("deterministic", "sampled"):
                subset = [e for e in episodes if (e["condition"], e["mode"]) == (condition, mode)]
                groups[f"{condition}/{mode}"] = dict(aggregate(subset),
                    minimum_length=min(e["length"] for e in subset),
                    non_id_steps=sum(e["non_id_steps"] for e in subset),
                    mean_switches=float(np.mean([e["switches"] for e in subset])),
                    mean_context_l2=float(np.mean([e["mean_context_l2"] for e in subset])),
                    mean_action_noise_l2=float(np.mean([e["mean_action_noise_l2"] for e in subset])))
        records.append(dict(training_seed=seed, context_mode=context, groups=groups,
                            evaluation_interactions=row["evaluation_interactions"], seconds=row["seconds"]))
    for seed in (3, 4):
        online, zero = [indexed[seed, context] for context in worker.CONTEXT_MODES]
        assert online["source"] == zero["source"] and online["runtime"] == zero["runtime"]
        for condition in worker.CONDITIONS:
            for mode in ("deterministic", "sampled"):
                pairs = [(a, b) for a, b in zip(online["evaluations"], zero["evaluations"])
                         if (a["condition"], a["mode"]) == (condition, mode)]
                # Longer survival exposes more switches; compare only the shared schedule prefix.
                for a, b in pairs:
                    common_length = min(a["length"], b["length"])
                    assert [e for e in a["profile_events"] if e["step"] <= common_length] == [
                        e for e in b["profile_events"] if e["step"] <= common_length]
                paired.append(dict(training_seed=seed, condition=condition, mode=mode, pairs=len(pairs),
                    online_minus_zero={key: float(np.mean([a[key]-b[key] for a, b in pairs]))
                                       for key in ("length", "episode_return", "mean_velocity")},
                    length_wins=sum(a["length"] > b["length"] for a, b in pairs),
                    length_ties=sum(a["length"] == b["length"] for a, b in pairs),
                    length_losses=sum(a["length"] < b["length"] for a, b in pairs),
                    fewer_falls=sum(b["physical_termination"] for a, b in pairs)-sum(a["physical_termination"] for a, b in pairs)))
    return dict(revision=worker.REVISION, independent_training_seeds=2, frozen_models=2,
        evaluation_episodes=sum(len(r["evaluations"]) for r in rows), evaluation_interactions=total_steps,
        new_training_interactions=0, optimizer_updates=0, records=records, paired_differences=paired,
        interpretation="Paired inference-context ablation at intensity50; ID and OOD remain separate. Reset/action-noise repeats are not training seeds. No training-ablation or strong-baseline superiority claim.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads((args.result_dir/f"seed{seed}_{context}/ood_context.json").read_text())
            for seed in (3, 4) for context in worker.CONTEXT_MODES]
    sources = [json.loads((args.source_dir/f"seed{seed}_mse-control/source.json").read_text()) for seed in (3, 4)]
    result = summarize(rows, sources)
    (args.result_dir/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    for row in result["records"]:
        print(f"seed{row['training_seed']} {row['context_mode']}: "+", ".join(
            f"{c}={row['groups'][c+'/sampled']['mean_length']:.2f}" for c in worker.CONDITIONS))


if __name__ == "__main__":
    main()
