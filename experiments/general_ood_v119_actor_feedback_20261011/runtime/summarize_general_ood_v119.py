"""Paired environment-seed returns for the frozen actor-feedback pilot."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

import run_general_ood_v119_actor_feedback as worker


def interval(values):
    values = np.asarray(values, dtype=float)
    assert values.shape == (3,) and np.isfinite(values).all()
    mean = float(values.mean())
    half = float(t.ppf(.975, 2)*values.std(ddof=1)/np.sqrt(3))
    return dict(per_seed_differences=values.tolist(), mean_difference=mean, ci95=[mean-half, mean+half],
                unit="environment seed conditional on one source policy", n=3)


def summarize(records):
    index = {(r["seed"], r["arm"], r["mode"]): r for r in records}
    changed = [m for m in worker.v118.v117.MODES if m != "normal"]
    def value(seed, arm, metric, modes=None):
        modes = changed if modes is None else modes
        selected = [index[seed, arm, mode] for mode in modes]
        if metric == "post_mean_reward":
            return float(np.mean([r["windows"]["post"]["mean_reward"] for r in selected]))
        if metric == "final_mode_return":
            return float(np.mean([r["evaluation"][r["mode"]]["mean_return"] for r in selected]))
        assert metric == "final_normal_return_after_changed_training"
        return float(np.mean([r["evaluation"]["normal"]["mean_return"] for r in selected]))
    metrics = ("post_mean_reward", "final_mode_return", "final_normal_return_after_changed_training")
    per_seed = {str(s): {a: dict(
        changed={m: value(s, a, m) for m in metrics},
        normal={m: value(s, a, m, ["normal"]) for m in metrics[:2]})
        for a in worker.ARMS} for s in worker.SEEDS}
    comparisons = {}
    for baseline in worker.ARMS[1:]:
        comparisons[baseline] = dict(changed={m: interval([
            value(s, worker.ARMS[0], m)-value(s, baseline, m) for s in worker.SEEDS]) for m in metrics},
            normal={m: interval([value(s, worker.ARMS[0], m, ["normal"])-value(s, baseline, m, ["normal"])
                                  for s in worker.SEEDS]) for m in metrics[:2]},
            per_mode={mode: interval([value(s, worker.ARMS[0], "post_mean_reward", [mode])-
                value(s, baseline, "post_mean_reward", [mode]) for s in worker.SEEDS]) for mode in changed})
    return dict(per_seed=per_seed, matched_minus_controls=comparisons)


def run(directory):
    records, metadata, files = [], None, []
    for seed in worker.SEEDS:
        for arm in worker.ARMS:
            path = directory/f"seed{seed}_{arm}/actor_feedback.json"
            data = json.loads(path.read_text())
            assert data["revision"] == worker.REVISION and data["passed"] and not data["smoke"]
            assert data["spec"] == worker.SPEC and (data["seed"], data["arm"]) == (seed, arm)
            assert data["actual_transitions"] == data["optimizer_updates"] == 30000
            assert data["reference_metadata_unchanged"] and data["readout_controls_actor"]
            assert not data["reference_refitted"] and not data["normal_thresholds_recalibrated"] and not data["readout_adopted"]
            assert data["optimizer_updates_during_evaluation"] == data["counterfactual_step_calls"] == 0
            assert [(r["seed"], r["arm"], r["mode"]) for r in data["records"]] == [
                (c["seed"], c["arm"], c["mode"]) for c in worker.cells(seed, arm)]
            if metadata is None:
                metadata = data["reference_metadata"]
            assert metadata == data["reference_metadata"]
            records.extend(data["records"])
            files.append(dict(path=str(path), bytes=path.stat().st_size))
    for seed in worker.SEEDS:
        initial = [r["first_transition"] for r in records if r["seed"] == seed]
        assert all(r == initial[0] for r in initial), "different initial transitions across arms"
    result = dict(revision=worker.REVISION, passed=True, paths=len(records), source_training_seed=1,
        actual_transitions=sum(r["actual_transitions"] for r in records),
        optimizer_updates=sum(r["optimizer_updates"] for r in records),
        evaluation_interactions=sum(r["evaluation_interactions"] for r in records),
        frozen_metadata_matches_all_shards=True, cross_arm_first_transition_equal=True,
        readout_controls_actor=True, readout_adopted=False, scalar_files=files,
        **summarize(records), decision="closed-loop-pilot-requires-joint-return-and-normal-cost-assessment",
        limitations=["n=3 environment seeds, not independently trained policies; exploratory intervals are not multiplicity adjusted.",
                     "Known mild Hopper modes and one source; different reference budgets and divergent post-feedback trajectories."])
    assert result["actual_transitions"] == result["optimizer_updates"] == 360000
    path = directory/"summary.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    run(parser.parse_args().result_dir)
