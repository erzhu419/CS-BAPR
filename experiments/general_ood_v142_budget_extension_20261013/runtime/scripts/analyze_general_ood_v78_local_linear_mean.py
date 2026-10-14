"""Isolated conditional-mean intervention on all saved v76 development paths."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from scripts import analyze_general_ood_v77_mean_covariance as v77
from scripts.v73_recursive_mode_readout import RecursiveModeReference, observe_likelihoods
from scripts.v77_mean_covariance import MeanCovarianceBank, check_shared_forward
from scripts.v78_local_linear_mean import CONDITIONS, CONTROLS, LocalLinearDensityBank, LocalLinearMean, correction_terms

v76 = v77.v76
REVISION = "general_ood_v78_local_linear_mean"


def verify_controls(observers, previous):
    for condition, source in CONTROLS.items():
        for window, values in observers[condition].items():
            for key in ("mean_coefficient", "mode_accuracy", "mean_true_minus_best_other_score",
                        "instantaneous_mode_accuracy", "instantaneous_mean_true_minus_best_other_log_density"):
                np.testing.assert_allclose(values[key], previous["observers"][source][window][key], rtol=1e-9, atol=1e-9)
            assert values["predicted_mode_fractions"] == previous["observers"][source][window]["predicted_mode_fractions"]


def freeze_references(args, old, banks, correction):
    saved = torch.load(args.v77_dir/"run_checkpoints/attribution-final/hybrid_phases.pt", map_location="cpu", weights_only=False)
    assert saved["revision"] == v77.REVISION and saved["spec"] == v76.SPEC
    constant = RecursiveModeReference()
    constant.bank = MeanCovarianceBank(banks, "updating_mean_frozen_cov")
    constant.phase, constant.normal_null_calibration_transitions = saved["phases"]["updating_mean_frozen_cov"], saved["normal_null_transitions"]
    references = dict(frozen_anchor=old["frozen_mixed"], constant_mean_frozen_cov=constant,
                      constant_mean_updating_cov=old["updating_mixed"])
    cache = args.checkpoint_dir/"linear_phases.pt"
    if cache.is_file():
        saved = torch.load(cache, map_location="cpu", weights_only=False)
        assert saved["revision"] == REVISION and saved["spec"] == v76.SPEC
        for condition, phase in saved["phases"].items():
            reference = RecursiveModeReference()
            reference.bank = LocalLinearDensityBank(banks, correction, CONDITIONS[condition][1])
            reference.phase, reference.normal_null_calibration_transitions = phase, saved["normal_null_transitions"]
            references[condition] = reference
    else:
        stored = args.v69_dir/"run_checkpoints/reference/traces"
        nulls = [v76.v68.load_trace(v76.v69.trace_path(stored, c)) for c in v76.v69.reference_cells(v76.v69.SPEC)
                 if c["role"] == "normal_null"]
        for condition, (mean_source, covariance_source) in CONDITIONS.items():
            if mean_source == "updating_linear":
                references[condition] = RecursiveModeReference.calibrate(LocalLinearDensityBank(banks, correction, covariance_source), nulls)
                print(f"[CALIBRATE] {condition}: same saved normal-null population", flush=True)
        torch.save(dict(revision=REVISION, spec=v76.SPEC, normal_null_transitions=constant.normal_null_calibration_transitions,
            phases={c: r.phase for c, r in references.items() if c not in CONTROLS}), cache)
    assert set(references) == set(CONDITIONS)
    assert all(r.normal_null_calibration_transitions == constant.normal_null_calibration_transitions for r in references.values())
    return references


def replay_cell(cell):
    return dict(cell, group="v76_development_replay" if cell["group"] == "v76_fresh_updating" else cell["group"])


def evaluate(args, inputs, references, banks, correction, scores):
    records = []
    if args.resume_from:
        saved = json.loads(args.resume_from.read_text())
        assert saved["revision"] == REVISION and saved["spec"] == v76.SPEC
        records = saved["records"]
        assert [v77.cell_key(r) for r in records] == [v77.cell_key(replay_cell(c)) for c, _ in inputs[:len(records)]]
    for cell, path in inputs[len(records):]:
        trace = v76.v68.load_trace(path)
        assert len(trace["states"]) == v76.SPEC["steps"]
        terms = correction_terms(trace, banks, correction)
        observers, attribution = {}, {}
        for condition in CONDITIONS:
            ll = terms[condition]["likelihoods"]
            all_windows = v76.v73.window_statistics(trace, ll,
                observe_likelihoods(ll, trace["dones"], references[condition], "recursive"), cell["mode"])
            observers[condition] = {w: all_windows[w] for w in v77.WINDOWS}
            attribution[condition] = {w: v77.attribution(trace, terms[condition], banks["frozen"], cell["mode"], a, b)
                                     for w, a, b in v77.windows()}
        record = dict(**replay_cell(cell), observers=observers, attribution=attribution)
        verify_controls(observers, scores[v77.cell_key(record)])
        normal = record if cell["mode"] == "normal" else next(r for r in records if r["group"] == record["group"]
            and r["seed"] == cell["seed"] and r["updating"] == cell["updating"] and r["mode"] == "normal")
        record["change_minus_nochange"] = {c: {w: observers[c][w]["mean_coefficient"]-normal["observers"][c][w]["mean_coefficient"]
            for w in v76.v68.BASE.CHANGE_WINDOWS} for c in CONDITIONS}
        records.append(record)
        (args.checkpoint_dir/"progress.json").write_text(json.dumps(dict(revision=REVISION, spec=v76.SPEC, records=records), allow_nan=False)+"\n")
        print(f"[EVAL] {len(records)}/{len(inputs)} {record['group']} seed{cell['seed']} {cell['mode']} updating={cell['updating']}", flush=True)
    return records


def run(args):
    torch.set_num_threads(2)
    saved = torch.load(args.v76_dir/"run_checkpoints/formal/behavior_coverage_references.pt", map_location="cpu", weights_only=False)
    scores = json.loads((args.v77_dir/"run_results/attribution-final/mean_covariance_attribution.json").read_text())
    assert saved["revision"] == v76.REVISION and scores["revision"] == v77.REVISION
    assert saved["spec"] == scores["spec"] == v76.SPEC
    old = saved["references"]
    banks = {"frozen": old["frozen_mixed"].bank, "updating": old["updating_mixed"].bank}
    check_shared_forward(banks)
    correction = LocalLinearMean(banks["updating"])
    inputs = v76.evaluation_inputs(args, args.v76_dir/"run_checkpoints/formal/traces")
    index = {v77.cell_key(r): r for r in scores["records"]}
    assert len(inputs) == len(index) == 65 and {v77.cell_key(replay_cell(c)) for c, _ in inputs} == set(index)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        cell, path = next((c, p) for c, p in inputs if c["group"] == "v76_fresh_updating" and c["mode"] == "high_grav")
        trace = v76.v68.load_trace(path)
        for _, a, b in v77.windows():
            query = {k: trace[k][a:b] for k in v76.v74.KEYS}
            terms = correction_terms(query, banks, correction)
            for condition, source in CONTROLS.items():
                np.testing.assert_allclose(terms[condition]["likelihoods"],
                    MeanCovarianceBank(banks, source).log_likelihoods(query), rtol=1e-11, atol=1e-9)
            for condition, (mean_source, covariance_source) in CONDITIONS.items():
                if mean_source == "updating_linear":
                    np.testing.assert_allclose(terms[condition]["likelihoods"],
                        LocalLinearDensityBank(banks, correction, covariance_source).log_likelihoods(query), rtol=1e-11, atol=1e-9)
        result = dict(revision=REVISION, passed=True, evaluation_paths=65, checked_transitions=114,
            checked_windows=2, constant_controls_match_v77_densities=True, linear_batch_matches_readout=True,
            environment_steps=0, optimizer_updates=0)
        name = "native_smoke.json"
    else:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        references = freeze_references(args, old, banks, correction)
        records = evaluate(args, inputs, references, banks, correction, index)
        result = dict(revision=REVISION, source_revision=v77.REVISION, spec=v76.SPEC, conditions=CONDITIONS,
            environment_steps=0, optimizer_updates=0, readout_controls_actor=False,
            forward_models_refitted=False, scaling_refitted=False, temporal_rules_changed=False,
            conditional_covariances_changed=False, conditional_mean_rule_changed=True,
            all_paths_development_replays=True, constant_controls_reproduce_v77_statistics=True,
            reference_metadata={c: {k: val for k, val in r.metadata().items() if k != "phase"} for c, r in references.items()},
            records=records, summary=v77.summarize(records, CONDITIONS), decision="development-mean-intervention-not-confirmation",
            limitations=["Development replay, one source training seed, mild known modes and privileged saved reference labels.",
                "Local affine residual approximation is a hypothesis; covariance is not refitted after changing the mean.",
                "Template-gap diagnostics are unavailable on 17 historical v61/v62 paths.",
                "No actor-feedback or RL return evidence; a selected correction needs fresh confirmation."])
        name = "local_linear_mean.json"
    path = args.result_dir/name
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] {REVISION}: {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("v77-dir", "v76-dir", "v74-dir", "v73-dir", "v72-dir", "v71-dir", "v69-dir", "v70-dir", "v61-dir", "v62-dir", "checkpoint-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
