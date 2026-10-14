"""Server-only factorial development replay; no simulation or actor updates."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from scripts import probe_general_ood_v76_behavior_coverage as v76
from scripts.v73_recursive_mode_readout import RecursiveModeReference, observe_likelihoods
from scripts.v77_mean_covariance import COMBINATIONS, DIAGONALS, MODES, MeanCovarianceBank, check_shared_forward, factorial_terms

REVISION = "general_ood_v77_mean_covariance"
WINDOWS = tuple(dict.fromkeys((*v76.v68.BASE.NORMAL_WINDOWS, *v76.v68.BASE.CHANGE_WINDOWS)))


def cell_key(cell):
    return tuple(cell[k] for k in ("group", "seed", "mode", "updating"))


def windows():
    return (("first_post50", v76.SPEC["switch_step"], v76.SPEC["switch_step"]+50),
            ("last64", v76.SPEC["steps"]-64, v76.SPEC["steps"]))


def attribution(trace, terms, bank, mode, start, stop):
    truth = MODES.index(mode)
    likelihoods, energy, logdet = (terms[k][start:stop] for k in ("likelihoods", "energy", "logdet"))
    others = likelihoods.copy()
    others[:, truth] = -np.inf
    competitor = others.argmax(axis=1)
    rows = np.arange(stop-start)
    margin = likelihoods[:, truth]-others.max(axis=1)
    energy_advantage = .5*(energy[rows, competitor]-energy[:, truth])
    volume_advantage = .5*(logdet[rows, competitor]-logdet[:, truth])
    np.testing.assert_allclose(energy_advantage+volume_advantage, margin, rtol=1e-10, atol=1e-9)
    error = terms["mean_error_rms"][start:stop, truth]
    result = dict(raw_mode_accuracy=float(np.mean(likelihoods.argmax(axis=1) == truth)),
        mean_true_minus_best_other_log_density=float(margin.mean()),
        mean_energy_advantage=float(energy_advantage.mean()),
        mean_covariance_volume_advantage=float(volume_advantage.mean()),
        best_other_mode_fractions={m: float(np.mean(competitor == i)) for i, m in enumerate(MODES)},
        mean_true_prediction_error_rms=float(error.mean()),
        observation_template_gap_available="oracle_observations" in trace,
        mean_nearest_other_observation_template_gap=None,
        prediction_error_exceeds_template_gap_fraction=None,
        fixed_competitor={m: dict(
            mean_log_density_margin=float((likelihoods[:, truth]-likelihoods[:, i]).mean()),
            mean_energy_advantage=float(.5*(energy[:, i]-energy[:, truth]).mean()),
            mean_covariance_volume_advantage=float(.5*(logdet[:, i]-logdet[:, truth]).mean()),
            positive_margin_fraction=float(np.mean(likelihoods[:, truth] > likelihoods[:, i])))
            for i, m in enumerate(MODES) if i != truth})
    if result["observation_template_gap_available"]:
        oracle, actual = trace["oracle_observations"][start:stop], trace["next_states"][start:stop]
        np.testing.assert_array_equal(oracle[:, truth], actual)
        gaps = np.sqrt(np.mean(((oracle-actual[:, None, :])/bank.target_scale)**2, axis=2))
        gaps[:, truth] = np.inf
        nearest = gaps.min(axis=1)
        result["mean_nearest_other_observation_template_gap"] = float(nearest.mean())
        result["prediction_error_exceeds_template_gap_fraction"] = float(np.mean(error > nearest))
    return result


def check_diagonal_statistics(observers, previous):
    for condition, source in DIAGONALS.items():
        for window, values in observers[condition].items():
            for key in ("mean_coefficient", "mode_accuracy", "mean_true_minus_best_other_score",
                        "instantaneous_mode_accuracy", "instantaneous_mean_true_minus_best_other_log_density"):
                np.testing.assert_allclose(values[key], previous["observers"][source][window][key],
                                           rtol=1e-9, atol=1e-9)
            assert values["predicted_mode_fractions"] == previous["observers"][source][window]["predicted_mode_fractions"]


def summarize(records, conditions=COMBINATIONS):
    output = {}
    for group in dict.fromkeys(r["group"] for r in records):
        normal = [r for r in records if r["group"] == group and r["mode"] == "normal"]
        changed = [r for r in records if r["group"] == group and r["mode"] != "normal"]
        output[group] = {}
        for condition in conditions:
            maximum = max(r["observers"][condition][w]["mean_coefficient"] for r in normal for w in v76.v68.BASE.NORMAL_WINDOWS)
            contrasts = {w: sum(r["change_minus_nochange"][condition][w] > 0 for r in changed) for w in v76.v68.BASE.CHANGE_WINDOWS}
            margins = {w: sum(r["observers"][condition][w]["mean_true_minus_best_other_score"] > 0 for r in changed)
                       for w in v76.v68.BASE.CHANGE_WINDOWS}
            output[group][condition] = dict(changed_pairs=len(changed), largest_normal_mean=maximum,
                positive_risk_contrast_pairs=contrasts, positive_true_mode_margin_pairs=margins,
                mean_changed_accuracy={w: float(np.mean([r["observers"][condition][w]["mode_accuracy"] for r in changed]))
                    for w in v76.v68.BASE.CHANGE_WINDOWS},
                screen_pass=maximum <= .1 and all(n == len(changed) for n in (*contrasts.values(), *margins.values())))
    return output


def freeze_references(args, old, banks):
    cache = args.checkpoint_dir/"hybrid_phases.pt"
    references = {c: old[source] for c, source in DIAGONALS.items()}
    if cache.is_file():
        saved = torch.load(cache, map_location="cpu", weights_only=False)
        assert saved["revision"] == REVISION and saved["spec"] == v76.SPEC
        for condition, phase in saved["phases"].items():
            ref = RecursiveModeReference()
            ref.bank, ref.phase = MeanCovarianceBank(banks, condition), phase
            ref.normal_null_calibration_transitions = saved["normal_null_transitions"]
            references[condition] = ref
    else:
        stored = args.v69_dir/"run_checkpoints/reference/traces"
        nulls = [v76.v68.load_trace(v76.v69.trace_path(stored, c)) for c in v76.v69.reference_cells(v76.v69.SPEC)
                 if c["role"] == "normal_null"]
        for condition in COMBINATIONS:
            if condition not in references:
                references[condition] = RecursiveModeReference.calibrate(MeanCovarianceBank(banks, condition), nulls)
                print(f"[CALIBRATE] {condition}: same saved normal-null population", flush=True)
        torch.save(dict(revision=REVISION, spec=v76.SPEC,
            normal_null_transitions=old["frozen_mixed"].normal_null_calibration_transitions,
            phases={c: r.phase for c, r in references.items() if c not in DIAGONALS}), cache)
    assert set(references) == set(COMBINATIONS)
    assert all(r.normal_null_calibration_transitions == old["frozen_mixed"].normal_null_calibration_transitions
               for r in references.values())
    return references


def evaluate(args, inputs, references, banks, scores):
    records = []
    if args.resume_from:
        saved = json.loads(args.resume_from.read_text())
        assert saved["revision"] == REVISION and saved["spec"] == v76.SPEC
        records = saved["records"]
        assert [cell_key(r) for r in records] == [cell_key(dict(c, group=("v76_development_replay"
            if c["group"] == "v76_fresh_updating" else c["group"]))) for c, _ in inputs[:len(records)]]
    for cell, path in inputs[len(records):]:
        trace = v76.v68.load_trace(path)
        assert len(trace["states"]) == v76.SPEC["steps"]
        terms = factorial_terms(trace, banks)
        observers, details = {}, {}
        for condition in COMBINATIONS:
            ll = terms[condition]["likelihoods"]
            all_windows = v76.v73.window_statistics(trace, ll,
                observe_likelihoods(ll, trace["dones"], references[condition], "recursive"), cell["mode"])
            observers[condition] = {w: all_windows[w] for w in WINDOWS}
            details[condition] = {w: attribution(trace, terms[condition], banks["frozen"], cell["mode"], a, b)
                                  for w, a, b in windows()}
        check_diagonal_statistics(observers, scores[cell_key(cell)])
        replay_cell = dict(cell, group="v76_development_replay" if cell["group"] == "v76_fresh_updating" else cell["group"])
        record = dict(**replay_cell, observers=observers, attribution=details)
        normal = record if cell["mode"] == "normal" else next(r for r in records if r["group"] == record["group"]
            and r["seed"] == cell["seed"] and r["updating"] == cell["updating"] and r["mode"] == "normal")
        record["change_minus_nochange"] = {c: {w: observers[c][w]["mean_coefficient"]-normal["observers"][c][w]["mean_coefficient"]
            for w in v76.v68.BASE.CHANGE_WINDOWS} for c in COMBINATIONS}
        records.append(record)
        (args.checkpoint_dir/"progress.json").write_text(json.dumps(
            dict(revision=REVISION, spec=v76.SPEC, records=records), allow_nan=False)+"\n")
        print(f"[EVAL] {len(records)}/{len(inputs)} {record['group']} seed{cell['seed']} {cell['mode']} updating={cell['updating']}", flush=True)
    return records


def run(args):
    torch.set_num_threads(2)
    saved = torch.load(args.v76_dir/"run_checkpoints/formal/behavior_coverage_references.pt", map_location="cpu", weights_only=False)
    scores = json.loads((args.v76_dir/"run_results/screen/behavior_coverage.json").read_text())
    assert saved["revision"] == scores["revision"] == v76.REVISION
    assert saved["spec"] == scores["spec"] == v76.SPEC
    old = saved["references"]
    banks = {"frozen": old["frozen_mixed"].bank, "updating": old["updating_mixed"].bank}
    check_shared_forward(banks)
    inputs = v76.evaluation_inputs(args, args.v76_dir/"run_checkpoints/formal/traces")
    index = {cell_key(r): r for r in scores["records"]}
    assert len(inputs) == len(index) == 65 and {cell_key(c) for c, _ in inputs} == set(index)
    assert all(path.is_file() for _, path in inputs), "missing-server-evaluation-trace"
    args.result_dir.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        cell, path = next((c, p) for c, p in inputs if c["group"] == "v76_fresh_updating" and c["mode"] == "high_grav")
        trace = v76.v68.load_trace(path)
        checks = 0
        for _, a, b in windows():
            query = {k: trace[k][a:b] for k in v76.v74.KEYS}
            terms = factorial_terms(query, banks)
            for condition in COMBINATIONS:
                expected = MeanCovarianceBank(banks, condition).log_likelihoods(query)
                np.testing.assert_allclose(terms[condition]["likelihoods"], expected, rtol=1e-11, atol=1e-9)
            for condition, source in DIAGONALS.items():
                np.testing.assert_allclose(terms[condition]["likelihoods"], old[source].bank.log_likelihoods(query), rtol=1e-11, atol=1e-9)
            checks += b-a
        result = dict(revision=REVISION, passed=True, evaluation_paths_available=len(inputs),
            checked_transitions=checks, checked_windows=2, seed=cell["seed"], mode=cell["mode"],
            factorial_matches_hybrid_density=True, diagonal_matches_saved_banks=True,
            environment_steps=0, optimizer_updates=0)
        name = "native_smoke.json"
    else:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        references = freeze_references(args, old, banks)
        records = evaluate(args, inputs, references, banks, index)
        result = dict(revision=REVISION, source_revision=v76.REVISION, spec=v76.SPEC,
            conditions=COMBINATIONS, environment_steps=0, optimizer_updates=0,
            forward_models_refitted=False, scaling_refitted=False, conditional_moments_refitted=False,
            hybrid_normal_null_thresholds_recalibrated=True, temporal_rules_changed=False,
            readout_controls_actor=False, all_paths_development_replays=True,
            diagonals_reproduce_v76_statistics=True,
            reference_metadata={c: {k: val for k, val in r.metadata().items() if k != "phase"}
                                for c, r in references.items()},
            records=records, summary=summarize(records), decision="development-attribution-not-confirmation",
            limitations=["Already inspected trajectories, one source training seed and mild known modes.",
                "Historical v61/v62 traces lack saved counterfactual observations; their template-gap diagnostics are unavailable.",
                "Hybrid covariances are conditioned on their own saved state/action neighborhoods, not refitted around hybrid means.",
                "Privileged counterfactual observations only diagnose prediction error and template gaps offline.",
                "This is not an actor-feedback or RL return comparison; a selected change needs fresh confirmation."])
        name = "mean_covariance_attribution.json"
    path = args.result_dir/name
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] {REVISION}: {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("v76-dir", "v74-dir", "v73-dir", "v72-dir", "v71-dir", "v69-dir", "v70-dir", "v61-dir", "v62-dir", "checkpoint-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
