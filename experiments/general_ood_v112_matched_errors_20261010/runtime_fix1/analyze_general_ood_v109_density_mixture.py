"""Fixed equal-weight reference-model density mixtures before belief inference."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_general_ood_v108_modewise_mean as v108

v107, v98, summary = v108.v107, v108.v98, v108.summary
REVISION = "general_ood_v109_density_mixture"
CONTROLS = ("old", "new", "old_normal_new_ood")
CANDIDATES = ("normal_density_mixture", "all_mode_density_mixture")
ARMS = (*CONTROLS, *CANDIDATES)
SOURCES = {a: {m: (["old", "new"] if a == "all_mode_density_mixture" or m == "normal" else ["new"])
              for m in v108.MODES} for a in CANDIDATES}


def compose(old, new):
    hard = v108.compose(old, new)
    # Mix normalized probabilities, not log densities or posteriors.
    mixture = np.logaddexp(old, new)-np.log(2.)
    return {a: hard[a] for a in CONTROLS} | dict(
        normal_density_mixture=np.column_stack((mixture[:, 0], new[:, 1:])),
        all_mode_density_mixture=mixture)


def run(args):
    sys.path.insert(0, str(args.v83_dir))
    from scripts import analyze_general_ood_v83_cross_seed_covariance as source

    torch.set_num_threads(2)
    previous = json.loads((args.v108_dir/"run_results/formal/modewise_mean.json").read_text())
    assert previous["revision"] == v108.REVISION and previous["passed"] and not previous["smoke"]
    assert (previous["checked_paths"], previous["checked_transitions"],
        previous["homogeneous_calibration_comparisons"]) == (95, 570000, 6)
    assert previous["covariance_reference_unchanged"] and previous["complete_history_preserved"]
    population = v107.select_paths(previous["records"], True) if args.smoke else previous["records"]
    inputs = v98.evaluation_inputs(args, source)
    assert [summary.key(c) for c, _, _ in inputs] == [summary.key(r) for r in previous["records"]]
    paths = {summary.key(c): (i, p) for i, (c, p, _) in enumerate(inputs)}
    directories = dict(old=args.v98_dir/"run_checkpoints/diverse_recipe", new=args.v106_dir/"run_checkpoints/standard")
    bank_metadata = {a: {c: previous["reference_metadata"][a][c]["bank"] for c in summary.CONDITIONS}
                     for a in CONTROLS}
    bank_metadata.update({a: {c: dict(revision=REVISION, mode_density_sources=SOURCES[a],
        component_weights="uniform over listed sources for each mode", mixture_stage="density before recursive posterior",
        component_density_covariance_reference="v98 diverse_recipe", weights_fitted=False)
        for c in summary.CONDITIONS} for a in CANDIDATES})
    null_cells = [c for c in source.v76.v69.reference_cells(source.v76.v69.SPEC) if c["role"] == "normal_null"]
    nulls = [source.v76.v68.load_trace(source.v76.v69.trace_path(args.v69_dir/"run_checkpoints/reference/traces", c))
             for c in null_cells]
    assert len(nulls) == 2 and all(len(t["dones"]) == 6000 for t in nulls)
    assert {c["updating"] for c in null_cells} == {False, True}
    null_ll = [{a: v107.load_likelihoods(directories[a]/"terms"/f"normal_null_{i}.pt",
        v107.RECIPES[a], ("normal_null", i), 29600 if a == "old" else 34000) for a in v107.RECIPES}
        for i in range(2)]
    references = v108.calibrate(nulls, null_ll, bank_metadata, assemble=compose)
    for arm in CONTROLS:
        for c in summary.CONDITIONS:
            assert references[arm][c].metadata() == previous["reference_metadata"][arm][c]
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(dict(revision=REVISION, mode_density_sources=SOURCES, normal_null_cells=null_cells,
        normal_null_transitions=12000, phases={a: {c: r.phase for c, r in refs.items()}
        for a, refs in references.items()}), args.checkpoint_dir/"phases.pt")
    print("[CALIBRATE] five densities; same 12000 null transitions; nine control calibrations reproduced", flush=True)
    records, comparisons = [], 0
    for cell in population:
        index, path = paths[summary.key(cell)]
        trace = source.v76.v68.load_trace(path)
        assert len(trace["dones"]) == 6000
        ll = {a: v107.load_likelihoods(directories[a]/"terms"/f"eval_{index:02d}.pt",
            v107.RECIPES[a], summary.key(cell), 29600 if a == "old" else 34000) for a in v107.RECIPES}
        arms, episodes = {a: {} for a in ARMS}, {a: {} for a in ARMS}
        for c in summary.CONDITIONS:
            for arm, values in compose(ll["old"][c], ll["new"][c]).items():
                readout = source.v76.v73.observe_likelihoods(values, trace["dones"], references[arm][c], "recursive")
                arms[arm][c] = source.v76.v73.window_statistics(trace, values, readout, cell["mode"])
                if cell["mode"] == "normal" and c == v108.PRIMARY:
                    episodes[arm][c] = v107.episode_diagnostics(trace, readout)
            for arm in CONTROLS:
                v107.verify_diagonal(arms[arm][c], previous["records"][index]["arms"][arm][c])
                comparisons += 1
        records.append(dict(**{k: cell[k] for k in ("group", "seed", "mode", "updating")},
            arms=arms, normal_episodes=episodes if cell["mode"] == "normal" else None))
        print(f"[REPLAY] {len(records)}/{len(population)} {summary.key(cell)}", flush=True)
    assert len(records) == (4 if args.smoke else 95)
    result = dict(revision=REVISION, passed=True, smoke=args.smoke, checked_paths=len(records),
        checked_transitions=6000*len(records), control_path_condition_comparisons=comparisons,
        control_calibration_comparisons=9, normal_null_transitions_per_arm_condition=12000,
        mode_density_sources=SOURCES, summary=v108.summarize(records), records=records,
        reference_metadata={a: {c: r.metadata() for c, r in refs.items()} for a, refs in references.items()},
        mixture_weights=[.5, .5], mixture_weights_fitted=False,
        mixture_stage="per-transition mode density; component redrawn independently, not a persistent latent branch",
        episode_diagnostic_conditions=[v108.PRIMARY], likelihood_shift_source="v108 unchanged cached components",
        environment_steps=0, optimizer_updates=0, component_likelihoods_recomputed=False,
        geometry_refitted=False, calibration_refitted=True, covariance_reference_unchanged=True,
        temporal_rules_changed=False, complete_history_preserved=True, labels_used_for_mixture_selection=False,
        readout_controls_actor=False, readout_adopted=False, decision="development-density-mixture-not-adoption",
        limitations=["Development-only paths, one source policy and known simulator modes.",
            "Old/new reference models share data and are not independent ensemble members; total reference budgets differ.",
            "Normal calibration reuses seed135060 frozen/updating histories, not independent confirmation.",
            "No online actor implementation or return evidence; window mean risks are not false-positive rates."])
    args.result_dir.mkdir(parents=True, exist_ok=True)
    path = args.result_dir/("native_smoke.json" if args.smoke else "density_mixture.json")
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] {REVISION}: {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("v108-dir", "v106-dir", "v98-dir", "v97-dir", "v95-dir", "v94-dir", "v83-dir", "v78-dir",
                 "v76-dir", "v74-dir", "v73-dir", "v72-dir", "v71-dir", "v69-dir", "v70-dir", "v61-dir",
                 "v62-dir", "checkpoint-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
