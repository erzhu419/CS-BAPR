"""Local mean/covariance/history substitutions; no readout selection."""
import argparse
from itertools import permutations
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_general_ood_v99 import decompose

RECIPES = ("incumbent_suffix", "age_balanced")
CONDITION = "suffix_disagreement_mix"


def decompose_three(scores):
    effects = [0., 0., 0.]
    for order in permutations(range(3)):
        key = [0, 0, 0]
        for factor in order:
            old = scores[tuple(key)]
            key[factor] = 1
            effects[factor] += (scores[tuple(key)]-old)/6
    return dict(incumbent_score=scores[0, 0, 0], age_balanced_score=scores[1, 1, 1],
        total=scores[1, 1, 1]-scores[0, 0, 0],
        history_effect=effects[0], mean_effect=effects[1], covariance_effect=effects[2])


def summarize(data):
    assert data["passed"] and not data["smoke"]
    assert (data["checked_windows"], data["checked_transitions"], data["diagonal_posterior_comparisons"]) == (12, 656, 72)
    records = []
    for r in data["records"]:
        instant, cube = [], {}
        for mean_recipe, cov_recipe in ((0, 0), (1, 0), (0, 1), (1, 1)):
            arm = RECIPES[mean_recipe]+"_mean__"+RECIPES[cov_recipe]+"_covariance"
            instant.append(r["crossed_scores"][arm][CONDITION]["mean_true_minus_best_other_log_density"])
        for history in (0, 1):
            for mean_recipe in (0, 1):
                for cov_recipe in (0, 1):
                    arm = RECIPES[history]+"_history__"+RECIPES[mean_recipe]+"_mean__"+RECIPES[cov_recipe]+"_covariance"
                    cube[history, mean_recipe, cov_recipe] = r["posterior_scores"][arm][CONDITION]["mean_true_minus_best_other_score"]
        instantaneous = decompose(instant)
        instantaneous["incumbent_score"] = instantaneous.pop("old_score")
        instantaneous["age_balanced_score"] = instantaneous.pop("diverse_score")
        records.append(dict(**{k: r[k] for k in ("role", "seed", "mode", "updating", "window")},
            instantaneous_decomposition=instantaneous, posterior_decomposition=decompose_three(cube),
            mixed_posterior_scores={k: v[CONDITION] for k, v in r["posterior_scores"].items()}))
    return dict(revision=data["revision"], passed=True, condition=CONDITION, records=records,
        decomposition="symmetric instantaneous substitutions and six-order history/mean/covariance contributions to fixed-window margins",
        decision="HOLD", readout_controls_actor=False, readout_adopted=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(json.loads((args.result_dir/"failure_attribution.json").read_text()))
    path = args.result_dir/"summary.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(path)
