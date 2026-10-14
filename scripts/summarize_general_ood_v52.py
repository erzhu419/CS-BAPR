"""Matched warm-continuation accounting and conditional paired evaluation."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scripts.run_general_ood_v52 import BUDGETS, REVISION, SOURCE_NAME, stage_spec
from scripts.v52_agent import VARIANTS

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "results" / REVISION


def paired(candidate, control):
    differences = np.asarray(candidate["rewards"]) - control["rewards"]
    rng = np.random.RandomState(52)
    boot = differences[rng.randint(0, len(differences), size=(2000, len(differences)))].mean(axis=1)
    return dict(mean_difference=float(differences.mean()), positive_pairs=int(np.sum(differences > 0)),
        pairs=len(differences), paired_range=[float(differences.min()), float(differences.max())],
        conditional_episode_bootstrap95=np.quantile(boot, [.025, .975]).tolist())


def summarize(directory, stage):
    paths = {v: directory / f"{REVISION}_{v}.json" for v in VARIANTS}
    missing = [v for v, p in paths.items() if not p.exists()]
    if missing:
        return dict(revision=REVISION, stage=stage, matrix_complete=False, missing_variants=missing)
    runs = {v: json.loads(p.read_text()) for v, p in paths.items()}
    base, budget, spec = runs["fixed-minus2"], BUDGETS[stage], stage_spec(stage)
    for variant, data in runs.items():
        assert data["revision"] == REVISION and data["stage"] == stage and data["variant"] == variant
        assert data["source_checkpoint"] == SOURCE_NAME and data["continuation_seed"] == 1
        assert data["shared_metadata"] == base["shared_metadata"] and data["config"] == base["config"]
        assert data["runtime"] == base["runtime"]
        for key in ("replay_refill_shared", "reference_frozen", "penalty_enters_actor_loss_only",
                    "continuation_checkpoint_resumable", "restore_training_context_before_each_evaluation_episode"):
            assert data[key] is True
        assert data["source_training_state_available"] is False and data["optimizer_updates_during_evaluation"] == 0
        assert data["evaluation_seed_start"] == spec["evaluation_start"] and data["added_training_budget"] == budget
        assert data["initial_replay_size"] == len(spec["fit"]) * 3072
        training = data["training"]
        assert training["interactions"] == training["optimizer_updates"] == budget
        assert training["context_count"] == 100000 + budget
        assert training["replay_size"] == data["initial_replay_size"] + budget
        assert training["mode_events"] == base["training"]["mode_events"]
        assert training["first_transition"] == base["training"]["first_transition"], "unmatched-first-transition"
        assert [r["step"] for r in training["mode_events"]] == list(range(0, budget, 5000))
        blocks = training["diagnostic_blocks"]
        assert sum(b["updates"] for b in blocks) == budget
        for b in blocks:
            assert all(math.isfinite(x) for k, x in b.items() if k != "mode")
            assert -3. <= b["effective_beta"] <= -2. and 0 <= b["weighted_lambda_effective"] <= 1
            coefficient = {"fixed-minus2": 0., "fixed-minus2p3": .3,
                           "noise-age": b["age_coefficient"], "age-departure": b["departure_coefficient"]}[variant]
            assert abs(b["effective_beta"] + 2. + coefficient) < 1e-10, "wrong-actor-penalty"
        for group, count in (("results", spec["episodes"]), ("fixed_profile_results", spec["fixed_episodes"])):
            assert set(data[group]) == set(base[group])
            for r in data[group].values():
                assert len(r["rewards"]) == len(r["lengths"]) == count
                assert all(math.isfinite(x) for x in r["rewards"]) and all(0 < n <= 1000 for n in r["lengths"])
                assert abs(np.mean(r["rewards"]) - r["mean"]) < 1e-10
    return dict(revision=REVISION, stage=stage, matrix_complete=True,
        decision="operational-preflight-passed" if stage == "preflight" else "complete-development-pilot-awaits-return-analysis",
        added_training_interactions=4 * budget,
        shared_normal_interactions=base["shared_metadata"]["fit_calibration_interactions"] + base["shared_metadata"]["null_interactions"],
        evaluation_interactions=sum(sum(r["lengths"]) for d in runs.values()
            for group in ("results", "fixed_profile_results") for r in d[group].values()),
        source_controls_match=True, shared_metadata=base["shared_metadata"],
        methods={v: dict(returns={k: r["mean"] for k, r in d["results"].items()},
            fixed_profiles={k: r["mean"] for k, r in d["fixed_profile_results"].items()},
            mean_beta=float(np.average([b["effective_beta"] for b in d["training"]["diagnostic_blocks"]],
                                      weights=[b["updates"] for b in d["training"]["diagnostic_blocks"]])))
            for v, d in runs.items()},
        contrasts={v: {group: {k: paired(runs["age-departure"][group][k], runs[v][group][k])
            for k in base[group]} for group in ("results", "fixed_profile_results")}
            for v in VARIANTS if v != "age-departure"},
        limitations=["One selected seed1 starting policy and one continuation RNG seed; this is development, not training-seed confirmation.",
            "Confidence intervals condition on trained policies and evaluation episodes, not training variability.",
            "Training uses stochastic normal calibration; no claim of calibrated OOD-mode probabilities."])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=BUDGETS, default="formal")
    args = parser.parse_args()
    result = summarize(DESTINATION / args.stage, args.stage)
    p = DESTINATION / ("preflight_summary.json" if args.stage == "preflight" else "summary.json")
    p.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(result.get("decision", "pending-missing-variants"), "saved", p)


if __name__ == "__main__":
    main()
