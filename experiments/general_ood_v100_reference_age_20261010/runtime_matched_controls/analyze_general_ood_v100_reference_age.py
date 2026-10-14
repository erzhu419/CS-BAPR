"""Equal-row-budget normal-prefix versus OOD-suffix mean-support pilot."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_general_ood_v99_coverage_attribution as v99

v98 = v99.v98
REVISION = "general_ood_v100_reference_age"
RECIPES = ("diverse_suffix", "late_prefix", "age_balanced")


def extensions(traces, spec):
    switch, stop, stride = (spec[k] for k in ("switch_step", "steps", "stride"))
    rows = 2*len(np.arange(switch, stop, stride))
    start = switch-rows
    assert start >= 0
    selections = dict(diverse_suffix=[("high_grav", np.arange(switch, stop, stride)),
        ("compound", np.arange(switch, stop, stride))], late_prefix=[("normal", np.arange(start, switch))],
        age_balanced=[("normal", np.arange(start, switch, 2)),
            ("high_grav", np.arange(switch, stop, 2*stride)), ("compound", np.arange(switch, stop, 2*stride))])
    payloads, groups, sources = {}, {}, {}
    for recipe, selected in selections.items():
        payloads[recipe] = {m: {k: [] for k in v98.v97.KEYS} for m in v98.v94.MODES}
        labels, provenance = [], []
        offset = 0
        for seed, population in traces.items():
            for mode, indices in selected:
                trace = population[mode]
                oracle = trace["oracle_observations"]
                np.testing.assert_array_equal(oracle[indices, v98.v94.MODES.index(mode)], trace["next_states"][indices])
                for i, m in enumerate(v98.v94.MODES):
                    for k in v98.v97.KEYS:
                        values = oracle[indices, i] if k == "next_states" else trace[k][indices]
                        payloads[recipe][m][k].append(values)
                labels.append(np.full(len(indices), seed, dtype=np.int64))
                provenance.append(dict(seed=seed, source_mode=mode, start=int(indices[0]),
                    stop=int(indices[-1]+1), stride=int(indices[1]-indices[0]), rows=len(indices),
                    row_range=[offset, offset+len(indices)]))
                offset += len(indices)
        payloads[recipe] = {m: {k: np.concatenate(v) for k, v in data.items()} for m, data in payloads[recipe].items()}
        groups[recipe], sources[recipe] = np.concatenate(labels), provenance
        assert len(groups[recipe]) == rows*len(traces)
    return payloads, groups, sources


def evaluate_window(query, raw, mean, groups, old_rows, mode, worker):
    inputs = raw.features.inputs(query)
    prediction = worker.local_prediction(mean, inputs)
    truth = v98.v94.MODES.index(mode)
    error = raw.design(query)@raw.coefficients[mode]+prediction["linear"][mode]-raw.targets(query)
    _, neighbors = mean.tree.query((inputs-mean.features.input_mean)/mean.features.input_scale, k=64)
    gap, direction = None, None
    if "oracle_observations" in query:
        oracle = query["oracle_observations"]
        np.testing.assert_array_equal(oracle[:, truth], query["next_states"])
        gaps = {m: (oracle[:, i]-oracle[:, truth])/raw.target_scale for i, m in enumerate(v98.v94.MODES) if i != truth}
        gap = float(np.min(np.stack([np.sqrt(np.mean(g**2, axis=1)) for g in gaps.values()]), axis=0).mean())
        direction = v99.v96.error_directions(error, gaps)
    return dict(prediction_error_rms=float(np.sqrt(np.mean(error**2, axis=1)).mean()),
        support={k: worker.statistics(v) for k, v in prediction["support"].items()},
        neighbor_partition=v99.neighbor_partition(neighbors, old_rows, raw.tree.n),
        neighbor_seed_fractions={str(int(s)): float(np.mean(groups[neighbors] == s)) for s in np.unique(groups)},
        local_reference_loo_rms=float(prediction["loo_rms"][:, truth].mean()),
        oracle_observations_available=gap is not None, nearest_oracle_gap_rms=gap, error_directions=direction)


def evaluation_windows(records, ranges):
    population = []
    for cell in records:
        if cell["group"].startswith("v95_development_"):
            population.extend((cell, w, a, b) for w, a, b in ranges)
        elif (cell["group"] == "v62_development_replay" and cell["seed"] == 134020
                and cell["updating"] and cell["mode"] in ("normal", "slippery")):
            population.extend((cell, w, a, b) for w, a, b in ranges if w == "last64")
    assert len(population) == 62
    return population


def run(args):
    sys.path.insert(0, str(args.v83_dir))
    from scripts import analyze_general_ood_v83_cross_seed_covariance as source
    from scripts import analyze_general_ood_v79_linear_support as worker

    torch.set_num_threads(2)
    traces = {s: {m: source.v76.v68.load_trace(source.v76.trace_path(args.v97_dir/f"run_checkpoints/seed{s}/traces",
        next(c for c in v98.v97.cells(s) if c["mode"] == m))) for m in ("normal", "high_grav", "compound")}
        for s in v98.v97.SEEDS}
    payloads, groups, sources = extensions(traces, v98.v97.SPEC)
    previous = json.loads((args.v98_dir/"run_results/replay_diverse_recipe/regime_replay.json").read_text())
    assert previous["passed"] and previous["evaluation_paths"] == 95
    original, _, _ = v98.v95.load_readouts(args, source)
    old_payload, old_groups, _ = v98.load_extensions(SimpleNamespace(**vars(args), recipe="diverse_recipe"))
    for m in v98.v94.MODES:
        for k in v98.v97.KEYS:
            np.testing.assert_array_equal(payloads["diverse_suffix"][m][k], old_payload[m][k])
    np.testing.assert_array_equal(groups["diverse_suffix"], old_groups)
    banks = {r: v98.expand(original, payloads[r], groups[r], r) for r in RECIPES}
    assert all((raw.tree.n, mean.tree.n, len(np.unique(labels))) == (27400, 29600, 8)
               for raw, mean, labels in banks.values())
    inputs = {source.v78.v77.cell_key(c): p for c, p, _ in v98.evaluation_inputs(args, source)}
    population = evaluation_windows(previous["records"], source.v78.v77.windows())
    if args.smoke:
        population = [p for p in population if p[0]["seed"] == 135190 and p[0]["mode"] in ("slippery", "high_grav") and p[1] == "first_post50"]
        assert len(population) == 4
    records = []
    for cell, window, start, stop in population:
        trace = source.v76.v68.load_trace(inputs[source.v78.v77.cell_key(cell)])
        keys = list(v98.v97.KEYS)
        if "oracle_observations" in trace:
            keys.append("oracle_observations")
        query = {k: trace[k][start:stop] for k in keys}
        diagnostics = {r: evaluate_window(query, raw, mean, labels, original.bank.tree.n, cell["mode"], worker)
                       for r, (raw, mean, labels) in banks.items()}
        for c in v98.v94.CONDITIONS:
            np.testing.assert_allclose(diagnostics["diverse_suffix"]["prediction_error_rms"],
                cell["attribution"][c][window]["mean_true_prediction_error_rms"], rtol=1e-9, atol=1e-8)
        records.append(dict(**{k: cell[k] for k in ("group", "seed", "mode", "updating")},
            window=window, transitions=stop-start, diagnostics=diagnostics))
        print(f"[AGE] {len(records)}/{len(population)} {cell['seed']} {cell['mode']} {window} updating={cell['updating']}", flush=True)
    rows = sum(r["transitions"] for r in records)
    assert (len(records), rows) == ((4, 200) if args.smoke else (62, 3548))
    result = dict(revision=REVISION, source_revision=v98.REVISION, passed=True, smoke=args.smoke,
        checked_windows=len(records), checked_transitions=rows, incumbent_rms_comparisons=3*len(records),
        additional_reference_inputs_per_recipe=4400, raw_rows_per_recipe=27400, mean_rows_per_recipe=29600,
        reference_sources=sources, records=records, environment_steps=0, optimizer_updates=0,
        reference_errors_refitted=False, normal_thresholds_recalibrated=False, covariance_scored=False,
        readout_controls_actor=False, readout_adopted=False, decision="development-reference-age-mean-pilot",
        limitations=["One source policy, known-mode simulator labels and already inspected development windows.",
            "This tests reference support and prediction means, not recalibrated densities, belief accuracy or RL returns.",
            "Prefix-only sampling has denser temporal rows than suffix-only; the age-balanced recipe retains two OOD suffix sources.",
            "The two historical v62 windows lack oracle observations."])
    args.result_dir.mkdir(parents=True, exist_ok=True)
    path = args.result_dir/("native_smoke.json" if args.smoke else "reference_age.json")
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] {REVISION}: {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("v98-dir", "v97-dir", "v95-dir", "v94-dir", "v83-dir", "v78-dir", "v76-dir", "v74-dir",
                 "v73-dir", "v72-dir", "v71-dir", "v69-dir", "v70-dir", "v61-dir", "v62-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
