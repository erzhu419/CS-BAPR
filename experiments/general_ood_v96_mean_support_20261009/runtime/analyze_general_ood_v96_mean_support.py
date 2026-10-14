"""Server-only paired support attribution of the nine v95 updating failures."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_general_ood_v95_fresh_seed as v95

v94 = v95.v94
REVISION = "general_ood_v96_mean_support"
CANDIDATE = "suffix_disagreement_mix"
FAILURES = (
    (135190, "slippery", "first_post50"),
    (135190, "high_grav", "first_post50"),
    (135190, "compound", "first_post50"),
    (135191, "slippery", "first_post50"),
    (135192, "slippery", "first_post50"),
    (135192, "slippery", "last64"),
    (135192, "high_grav", "first_post50"),
    (135192, "compound", "first_post50"),
    (135192, "compound", "last64"),
)
SCALARS = ("raw_mode_accuracy", "mean_true_prediction_error_rms",
           "mean_true_minus_best_other_log_density", "mean_energy_advantage",
           "mean_covariance_volume_advantage")


def select_failures(records):
    index = {(r["seed"], r["mode"], r["updating"]): r for r in records}
    assert len(index) == len(records) == 30
    failed = {(r["seed"], r["mode"], w) for r in records if r["updating"] and r["mode"] != "normal"
              for w in ("first_post50", "last64")
              if r["observers"][CANDIDATE][w]["mean_true_minus_best_other_score"] <= 0}
    assert failed == set(FAILURES), "v95 failure population changed"
    return [(seed, mode, window, {u: index[seed, mode, u] for u in (False, True)})
            for seed, mode, window in FAILURES]


def error_directions(error, gaps):
    error_squared = np.einsum("ni,ni->n", error, error)
    output = {}
    for mode, gap in gaps.items():
        gap_squared = np.einsum("ni,ni->n", gap, gap)
        separable = gap_squared > 0
        directional = separable & (error_squared > 0)
        dot = np.einsum("ni,ni->n", error, gap)
        output[mode] = dict(
            separable_fraction=float(separable.mean()),
            mean_error_to_gap_ratio=float(np.sqrt(error_squared[separable]/gap_squared[separable]).mean())
                if separable.any() else None,
            mean_projection_toward_competitor=float((dot[separable]/gap_squared[separable]).mean())
                if separable.any() else None,
            mean_cosine_toward_competitor=float((dot[directional]/np.sqrt(
                error_squared[directional]*gap_squared[directional])).mean()) if directional.any() else None)
    return output


def diagnose(query, family, previous, window, source, support_worker):
    mode = previous["mode"]
    truth = v94.MODES.index(mode)
    inputs = family.bank.features.inputs(query)
    prediction = support_worker.local_prediction(family.mean, inputs)
    fixed, _, covariances, _ = family.geometry(inputs)
    direct = np.stack([prediction["linear"][m] for m in v94.MODES], axis=1)
    np.testing.assert_allclose(direct, fixed, rtol=1e-10, atol=1e-10)
    variants = dict(linear=fixed, constant=np.stack([prediction["constant"][m]
                    for m in v94.MODES], axis=1), global_forward=np.zeros_like(fixed))
    conditions = {c: {} for c in v94.CONDITIONS}
    for variant, centers in variants.items():
        terms = v94.terms_for_covariances(query, family.bank, centers, covariances)
        for condition in v94.CONDITIONS:
            attribution = source.v78.v77.attribution(query, terms[condition], family.bank, mode, 0, len(inputs))
            if variant == "linear":
                v94.v90.verify_baseline(attribution, previous["attribution"][condition][window])
                expected = previous["attribution"][condition][window]
                for key in ("mean_nearest_other_observation_template_gap",
                            "prediction_error_exceeds_template_gap_fraction"):
                    np.testing.assert_allclose(attribution[key], expected[key], rtol=1e-9, atol=1e-8)
                conditions[condition][variant] = attribution
            else:
                conditions[condition][variant] = {k: attribution[k] for k in SCALARS}
    scaled = (inputs-family.mean.features.input_mean)/family.mean.features.input_scale
    _, neighbors = family.mean.tree.query(scaled, k=64)
    groups = family.groups[neighbors]
    targets, design = family.bank.targets(query), family.bank.design(query)
    error = design@family.bank.coefficients[mode]+fixed[:, truth]-targets
    oracle = query["oracle_observations"]
    np.testing.assert_array_equal(oracle[:, truth], query["next_states"])
    gaps = {m: (oracle[:, i]-oracle[:, truth])/family.bank.target_scale
            for i, m in enumerate(v94.MODES) if i != truth}
    return dict(transitions=len(inputs),
        saved_recursive={c: {k: previous["observers"][c][window][k]
            for k in ("mode_accuracy", "mean_true_minus_best_other_score")} for c in v94.CONDITIONS},
        support={k: support_worker.statistics(v) for k, v in prediction["support"].items()},
        neighbor_seed_fractions={str(int(s)): float(np.mean(groups == s)) for s in np.unique(family.groups)},
        added_neighbor_fraction=float(np.mean(neighbors >= family.bank.tree.n)),
        local_reference_fit_rms=support_worker.statistics(prediction["fit_rms"][:, truth]),
        local_reference_loo_rms=support_worker.statistics(prediction["loo_rms"][:, truth]),
        coordinate_prediction_error_rms=np.sqrt(np.mean(error**2, axis=0)).tolist(),
        coordinate_prediction_error_mean=error.mean(axis=0).tolist(),
        error_directions=error_directions(error, gaps), conditions=conditions)


def summarize(records):
    result = {}
    for updating in (False, True):
        windows = [r["updating" if updating else "frozen"] for r in records]
        result["updating" if updating else "frozen"] = dict(
            mean_support={k: float(np.mean([w["support"][k]["mean"] for w in windows]))
                          for k in windows[0]["support"]},
            mean_added_neighbor_fraction=float(np.mean([w["added_neighbor_fraction"] for w in windows])),
            mean_local_reference_loo_rms=float(np.mean([w["local_reference_loo_rms"]["mean"] for w in windows])),
            mean_neighbor_seed_fractions={s: float(np.mean([w["neighbor_seed_fractions"][s] for w in windows]))
                                         for s in windows[0]["neighbor_seed_fractions"]},
            conditions={c: {variant: {k: float(np.mean([w["conditions"][c][variant][k] for w in windows]))
                                     for k in SCALARS} for variant in ("linear", "constant", "global_forward")}
                        for c in v94.CONDITIONS})
    return result


def run(args):
    sys.path.insert(0, str(args.v83_dir))
    from scripts import analyze_general_ood_v83_cross_seed_covariance as source
    from scripts import analyze_general_ood_v79_linear_support as support_worker

    torch.set_num_threads(2)
    originals = []
    for seed in v95.SEEDS:
        saved = json.loads((args.v95_dir/f"run_results/seed{seed}/fresh_seed.json").read_text())
        assert saved["revision"] == v95.REVISION and saved["seed"] == seed and saved["spec"] == v95.SPEC
        originals.extend(saved["records"])
    population = select_failures(originals)
    if args.smoke:
        population = population[:1]
    family, _, references = v95.load_readouts(args, source)
    metadata = deepcopy({c: r.metadata() for c, r in references.items()})
    ranges = {w: (a, b) for w, a, b in source.v78.v77.windows()}
    records = []
    for seed, mode, window, pair in population:
        start, stop = ranges[window]
        record = dict(seed=seed, mode=mode, window=window)
        for updating, previous in pair.items():
            stored = args.v95_dir/f"run_checkpoints/seed{seed}/traces"
            trace = source.v76.v68.load_trace(source.v76.trace_path(stored, previous))
            assert len(trace["states"]) == v95.SPEC["steps"]
            keys = (*source.v76.v74.KEYS, "oracle_observations")
            query = {k: trace[k][start:stop] for k in keys}
            record["updating" if updating else "frozen"] = diagnose(
                query, family, previous, window, source, support_worker)
        records.append(record)
        print(f"[PAIR] {len(records)}/{len(population)} seed{seed} {mode} {window}: v95 scores reproduced", flush=True)
    assert metadata == {c: r.metadata() for c, r in references.items()}
    rows = sum(r[p]["transitions"] for r in records for p in ("frozen", "updating"))
    assert rows == (100 if args.smoke else 956)
    result = dict(revision=REVISION, source_revision=v95.REVISION, passed=True, smoke=args.smoke,
        paired_windows=len(records), checked_windows=2*len(records), checked_transitions=rows,
        source_attribution_comparisons=6*len(records), source_prediction_and_density_statistics_match=True,
        direct_local_linear_predictions_match=True, reference_metadata_unchanged=True,
        environment_steps=0, optimizer_updates=0, reference_refitted=False,
        normal_thresholds_recalibrated=False, coefficient_tuned=False, readout_controls_actor=False,
        decision="development-mean-support-attribution-not-readout-adoption", records=records,
        summary=summarize(records), averaging="equal window; not pooled transitions",
        limitations=["Failure-selected development windows from three environment seeds and one source policy.",
            "Frozen and updating windows share seed, mode and time, not identical post-update states/actions.",
            "Neighbor leverage and error direction are attribution, not causal proof of support failure.",
            "Local leave-one-out errors reuse query-selected neighbors; not held-out deployment errors.",
            "Constant/global means reuse the linear candidate's covariance as diagnostic controls only.",
            "Zero oracle gaps are excluded from ratio/direction averages; separable fractions are reported."])
    args.result_dir.mkdir(parents=True, exist_ok=True)
    path = args.result_dir/("native_smoke.json" if args.smoke else "mean_support.json")
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] {REVISION}: {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("v95-dir", "v94-dir", "v83-dir", "v76-dir", "v78-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
