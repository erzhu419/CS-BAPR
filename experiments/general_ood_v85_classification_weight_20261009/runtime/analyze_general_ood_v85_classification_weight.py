"""Select one covariance weight by five-way reference classification loss."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_general_ood_v84_reference_discrimination as v84

REVISION = "general_ood_v85_classification_weight"


def classification_nll(weight, folds):
    truth = np.arange(len(v84.MODES))
    losses = []
    for offsets, cross, raw in folds:
        ll = v84.scores(offsets, weight*raw+(1-weight)*cross)[0]
        losses.append(float(np.mean(logsumexp(ll, axis=2)-ll[:, truth, truth])))
    return float(np.mean(losses))


def select_weight(folds, density_weight):
    objective = lambda w: classification_nll(w, folds)
    result = minimize_scalar(objective, bounds=(0., 1.), method="bounded")
    assert result.success, result.message
    candidates = [(name, weight, objective(weight)) for name, weight in (
        ("pure_cross_seed", 0.), ("raw", 1.), ("v83_density_selected", density_weight))]
    candidates.append(("bounded", float(result.x), float(result.fun)))
    name, weight, loss = min(candidates, key=lambda row: row[2])
    return dict(raw_weight=weight, classification_nll=loss, chosen_candidate=name,
        candidates=[dict(name=n, raw_weight=w, classification_nll=l) for n, w, l in candidates],
        optimizer_evaluations=int(result.nfev), solver="scipy.optimize.minimize_scalar bounded plus endpoints and v83 weight",
        objective="equal-seed / equal-true-mode uniform-prior five-way classification NLL",
        independent_confirmation=False)


def compact(stats):
    result = {k: v for k, v in stats.items() if k != "per_mode"}
    result["per_mode"] = {m: {k: v for k, v in s.items() if k != "pairs"} for m, s in stats["per_mode"].items()}
    return result


def run(args):
    sys.path.insert(0, str(args.v83_dir))
    from scripts import v83_cross_seed_covariance as v83

    torch.set_num_threads(2)
    source = args.v83_dir/"run_checkpoints/cross-seed-covariance"
    saved_selection = json.loads((source/"nested_selection.json").read_text())
    previous = json.loads((args.v84_dir/"run_results/diagnostic/reference_discrimination.json").read_text())
    assert saved_selection["revision"] == v83.REVISION and previous["revision"] == v84.REVISION
    bank = torch.load(args.v76_dir/"run_checkpoints/formal/behavior_coverage_references.pt",
                      map_location="cpu", weights_only=False)["references"]["updating_mixed"].bank
    groups = v83.groups_from_blocks(saved_selection["reference_blocks"], bank.tree.n)
    density_weight = saved_selection["selection"]["raw_weight"]
    assert density_weight == previous["conditions"]["selected_mix"]["raw_weight"]
    folds, metadata = [], []
    for expected in saved_selection["selection"]["folds"]:
        cached = torch.load(source/f"validation/outer_seed{expected['seed']}.pt", map_location="cpu", weights_only=False)
        assert cached["revision"] == v83.REVISION and cached["ridge_alpha"] == v83.ALPHA
        values = cached["values"]
        assert values[3]["seed"] == expected["seed"] and values[3]["validation_rows"] == 512
        _, _, offsets, cross, raw = v84.reconstruct_fold(bank, groups, values)
        folds.append((offsets, cross, raw))
        metadata.append(values[3])
    assert len(folds) == 6 and [m["seed"] for m in metadata] == [f["seed"] for f in previous["folds"]]
    for name, weight in (("raw", 1.), ("cross_seed", 0.), ("selected_mix", density_weight)):
        np.testing.assert_allclose(classification_nll(weight, folds),
            previous["conditions"][name]["classification_nll"], rtol=1e-10, atol=1e-10)
    print("[VERIFY] three fixed-weight classification losses reproduce v84", flush=True)
    selection = select_weight(folds, density_weight)
    weight = selection["raw_weight"]
    print(f"[SELECT] raw_weight={weight:.10f} classification_nll={selection['classification_nll']:.8f}", flush=True)
    records, collected, changes = [], [], []
    for values, meta, old in zip(folds, metadata, previous["folds"]):
        ll, energy, logdet = v84.scores(values[0], weight*values[2]+(1-weight)*values[1])
        stats = v84.summarize(ll, energy, logdet)
        collected.append((ll, energy, logdet))
        conditions = {c: compact(s) for c, s in old["conditions"].items()}
        conditions["classification_mix"] = compact(stats)
        records.append(dict(meta, conditions=conditions))
        for mode in v84.MODES:
            new, baseline = stats["per_mode"][mode], old["conditions"]["selected_mix"]["per_mode"][mode]
            changes.append(dict(seed=meta["seed"], mode=mode,
                classification_nll_delta=new["classification_nll"]-baseline["classification_nll"],
                accuracy_delta=new["accuracy"]-baseline["accuracy"]))
        print(f"[FOLD] seed{meta['seed']}: classification_nll={stats['classification_nll']:.8f}", flush=True)
    aggregate = v84.summarize(*(np.concatenate([v[i] for v in collected]) for i in range(3)))
    np.testing.assert_allclose(aggregate["classification_nll"], selection["classification_nll"], rtol=1e-10, atol=1e-10)
    conditions = {c: compact(s) for c, s in previous["conditions"].items()}
    conditions["classification_mix"] = dict(raw_weight=weight, **compact(aggregate))
    result = dict(revision=REVISION, source_revision=v84.REVISION, selection=selection,
        conditions=conditions, folds=records, fold_mode_changes_vs_v83=changes,
        queries=3072, true_mode_labels=15360, candidate_scores_per_weight=76800,
        reused_control_conditions=["raw", "cross_seed", "selected_mix"],
        verification=dict(fixed_weight_classification_losses_match_v84=True,
            selected_aggregate_matches_objective=True),
        covariance_components_refitted=False, conditional_means_refitted=False,
        global_forward_refitted=False, scaling_refitted=False, temporal_rules_changed=False,
        evaluation_paths_loaded=False, normal_null_loaded=False, readout_controls_actor=False,
        environment_steps=0, optimizer_updates=0, independent_confirmation=False,
        decision="reference-selection-pilot-not-replay-admission",
        limitations=["Selection and diagnostics use the same six reference folds; no independent confirmation.",
            "Outer labels leave mean/covariance fits but enter shared weight selection.",
            "Instantaneous uniform-prior classification is not recursive query accuracy or RL returns."])
    args.result_dir.mkdir(parents=True, exist_ok=True)
    path = args.result_dir/"classification_weight.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] {REVISION}: {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("v84-dir", "v83-dir", "v76-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    run(parser.parse_args())
