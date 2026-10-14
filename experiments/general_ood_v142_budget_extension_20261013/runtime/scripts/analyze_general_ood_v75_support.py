"""Scalar-only support attribution on the server-resident v74 evaluation paths."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from scripts import probe_general_ood_v74_density_coverage as v74

REVISION = "general_ood_v75_support"
NEIGHBORS = 64


def clipped_fraction(states):
    assert states.ndim == 2 and states.shape[1] == 11
    # Native Hopper-v4 exposes five positions followed by six clipped velocities.
    saturated = np.isclose(np.abs(states[:, -6:]), 10, rtol=0, atol=1e-12)
    return float(np.mean(np.any(saturated, axis=1)))


def support(query, bank, base_count):
    inputs = bank.features.inputs(query)
    assert inputs.shape[1] == 14
    normalized = (inputs-bank.features.input_mean)/bank.features.input_scale
    distances, indices = bank.tree.query(normalized, k=NEIGHBORS)
    differences = normalized[:, None, :]-bank.tree.data[indices]
    return dict(mean_nearest_radius=float(distances[:, 0].mean()),
                mean_radius64=float(distances[:, -1].mean()),
                median_radius64=float(np.median(distances[:, -1])),
                p90_radius64=float(np.quantile(distances[:, -1], .9)),
                added_neighbor_fraction=float(np.mean(indices >= base_count)),
                mean_neighbor_state_rms=float(np.sqrt(np.mean(differences[:, :, :11]**2, axis=(1, 2))).mean()),
                mean_neighbor_action_rms=float(np.sqrt(np.mean(differences[:, :, 11:]**2, axis=(1, 2))).mean()))


def cell_key(cell):
    return tuple(cell[k] for k in ("group", "seed", "mode", "updating"))


def run(args):
    torch.set_num_threads(2)
    saved = torch.load(args.v74_dir/"run_checkpoints/preflight/density_coverage_references.pt",
                       map_location="cpu", weights_only=False)
    assert saved["revision"] == v74.REVISION and saved["spec"] == v74.SPEC
    references = saved["references"]
    base_count = references["v72"]["window16"].bank.density_transitions
    scores = json.loads((args.v74_dir/"run_results/screen/density_coverage.json").read_text())
    assert scores["revision"] == v74.REVISION and scores["spec"] == v74.SPEC
    score_index = {cell_key(r): r for r in scores["records"]}
    reference_clipping = {}
    for condition, readouts in references.items():
        bank = readouts["window16"].bank
        assert bank is readouts["recursive"].bank
        physical = bank.tree.data*bank.features.input_scale+bank.features.input_mean
        reference_clipping[condition] = dict(base_transitions=base_count,
            added_transitions=len(physical)-base_count,
            base_clipped_velocity_fraction=clipped_fraction(physical[:base_count, :11]),
            added_clipped_velocity_fraction=clipped_fraction(physical[base_count:, :11])
                if len(physical) > base_count else None)
    records = []
    stored = args.v74_dir/"run_checkpoints/preflight/traces"
    for cell, path in v74.evaluation_inputs(args, stored):
        trace = v74.v68.load_trace(path)
        assert len(trace["states"]) == v74.SPEC["steps"]
        previous = score_index[cell_key(cell)]
        windows = {}
        for name, start, stop in (("first_post50", v74.SPEC["switch_step"], v74.SPEC["switch_step"]+50),
                                  ("last64", v74.SPEC["steps"]-64, v74.SPEC["steps"])):
            query = {k: trace[k][start:stop] for k in v74.KEYS}
            conditions = {}
            for condition, readouts in references.items():
                conditions[condition] = support(query, readouts["window16"].bank, base_count)
                conditions[condition]["recursive_mode_accuracy"] = previous["observers"][condition]["recursive"][name]["mode_accuracy"]
            windows[name] = dict(transitions=stop-start,
                clipped_input_velocity_fraction=clipped_fraction(query["states"]),
                clipped_target_velocity_fraction=clipped_fraction(query["next_states"]),
                conditions=conditions)
        records.append(dict(**cell, windows=windows))
        print(f"[SUPPORT] {cell['group']} seed{cell['seed']} {cell['mode']} updating={cell['updating']}", flush=True)
    assert {cell_key(r) for r in records} == set(score_index) and len(records) == len(score_index)
    result = dict(revision=REVISION, source_revision=v74.REVISION, environment_steps=0,
        optimizer_updates=0, reference_refitted=False, temporal_rules_changed=False,
        neighbors=NEIGHBORS, input_scaling="unchanged v74/v71 standardized state and executed action",
        reference_clipping=reference_clipping, records=records,
        limitations=["Development attribution on already inspected trajectories, not a new confirmation test.",
                     "Saturated observed velocity indicates clipping; it cannot recover the underlying velocity.",
                     "Neighbor distances describe support, not causal proof of observer failure."])
    args.result_dir.mkdir(parents=True, exist_ok=True)
    path = args.result_dir/"support_attribution.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] scalar support attribution: {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("v74-dir", "v73-dir", "v72-dir", "v71-dir", "v69-dir", "v70-dir", "v61-dir", "v62-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    run(parser.parse_args())
