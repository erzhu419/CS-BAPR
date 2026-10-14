"""Summarize fixed-window support and paired density-score attribution."""
import argparse
import json
from pathlib import Path
from statistics import mean

CONDITION = "suffix_disagreement_mix"


def decompose(scores):
    a, b, c, d = scores
    return dict(old_score=a, diverse_score=d, total=d-a,
        mean_effect=((b-a)+(d-c))/2, covariance_effect=((c-a)+(d-b))/2,
        interaction=d-b-c+a)


def summarize(data):
    assert data["passed"] and not data["smoke"]
    assert (data["checked_windows"], data["checked_transitions"], data["diagonal_attribution_comparisons"]) == (20, 1084, 120)
    original = [r for r in data["records"] if r["role"] == "original_v95_failure_pair"]
    updating = [r for r in original if r["updating"]]
    assert len(original) == 18 and len(updating) == 9
    output = []
    for r in updating:
        def score(m, v):
            return r["crossed_scores"][m+"_mean__"+v+"_covariance"][CONDITION]["mean_true_minus_best_other_log_density"]
        output.append(dict(seed=r["seed"], mode=r["mode"], window=r["window"],
            diverse_recursive_margin=r["saved_recursive"]["diverse_recipe"]["mean_true_minus_best_other_score"],
            diverse_neighbors=r["diagnostics"]["diverse_recipe"]["neighbor_partition"],
            diverse_dominant_seed_fraction=max(r["diagnostics"]["diverse_recipe"]["neighbor_seed_fractions"].values()),
            instantaneous_score_decomposition=decompose([score("old_recipe", "old_recipe"),
                score("diverse_recipe", "old_recipe"), score("old_recipe", "diverse_recipe"), score("diverse_recipe", "diverse_recipe")])))
    subsets = dict(frozen=[r for r in original if not r["updating"]], updating=updating,
        remaining=[r for r in updating if r["saved_recursive"]["diverse_recipe"]["mean_true_minus_best_other_score"] <= 0],
        repaired=[r for r in updating if r["saved_recursive"]["diverse_recipe"]["mean_true_minus_best_other_score"] > 0])
    aggregates = {k: dict(windows=len(rows), recipes={recipe: dict(
        extension_neighbor_fraction=mean(r["diagnostics"][recipe]["neighbor_partition"]["v97_extension"] for r in rows),
        nearest_radius=mean(r["diagnostics"][recipe]["support"]["nearest_radius"]["mean"] for r in rows),
        prediction_error_rms=mean(r["diagnostics"][recipe]["prediction_error_rms"] for r in rows))
        for recipe in ("old_recipe", "diverse_recipe")}) for k, rows in subsets.items()}
    unsupported = [r for r in output if r["seed"] in (135191, 135192)]
    assert len(unsupported) == 6
    available = sum(r["diagnostics"]["diverse_recipe"]["oracle_observations_available"] for r in data["records"])
    assert available == 18
    return dict(revision=data["revision"], passed=True, aggregates=aggregates, updating_windows=output,
        six_windows_maximum_extension_neighbor_fraction=max(r["diverse_neighbors"]["v97_extension"] for r in unsupported),
        six_windows_minimum_dominant_seed_fraction=min(r["diverse_dominant_seed_fraction"] for r in unsupported),
        oracle_windows=available, oracle_unavailable_windows=2,
        decomposition="symmetric two-order mean/covariance substitution of instantaneous best-other density margins",
        averaging="equal selected window; not pooled transitions", decision="HOLD", readout_controls_actor=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(json.loads((args.result_dir/"coverage_attribution.json").read_text()))
    path = args.result_dir/"summary.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(path)
