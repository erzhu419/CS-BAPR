"""Summarize matched reference-age means without inferring belief gains."""
import argparse
import json
from pathlib import Path
from statistics import mean

RECIPES = ("diverse_suffix", "late_prefix", "age_balanced")


def aggregate(rows):
    baseline = [r["diagnostics"]["diverse_suffix"]["prediction_error_rms"] for r in rows]
    recipes = {}
    for recipe in RECIPES:
        values = [r["diagnostics"][recipe] for r in rows]
        errors = [v["prediction_error_rms"] for v in values]
        recipes[recipe] = dict(prediction_error_rms=mean(errors),
            extension_neighbor_fraction=mean(v["neighbor_partition"]["v97_extension"] for v in values),
            nearest_radius=mean(v["support"]["nearest_radius"]["mean"] for v in values),
            mean_error_change=mean(a-b for a, b in zip(errors, baseline)),
            improved_windows=sum(a < b-1e-10 for a, b in zip(errors, baseline)),
            worsened_windows=sum(a > b+1e-10 for a, b in zip(errors, baseline)))
    return dict(windows=len(rows), recipes=recipes)


def summarize(data, attribution):
    assert data["passed"] and not data["smoke"] and attribution["passed"]
    assert (data["checked_windows"], data["checked_transitions"], data["incumbent_rms_comparisons"]) == (62, 3548, 186)
    assert not data["covariance_scored"] and not data["readout_controls_actor"]
    rows = data["records"]
    failure_keys = {(r["seed"], r["mode"], r["window"]) for r in attribution["updating_windows"]}
    frozen = [r for r in rows if r["group"] == "v95_development_frozen"]
    updating = [r for r in rows if r["group"] == "v95_development_updating"]
    failures = [r for r in updating if (r["seed"], r["mode"], r["window"]) in failure_keys]
    unsupported = [r for r in failures if r["seed"] in (135191, 135192)]
    assert (len(rows), len(frozen), len(updating), len(failures), len(unsupported)) == (62, 30, 30, 9, 6)
    subsets = dict(all=rows, v95_frozen=frozen, v95_updating=updating,
        updating_changed=[r for r in updating if r["mode"] != "normal"],
        updating_normal=[r for r in updating if r["mode"] == "normal"],
        original_failures=failures, six_unsupported=unsupported)
    per_seed = {str(seed): aggregate([r for r in updating if r["seed"] == seed and r["mode"] != "normal"])
                for seed in (135190, 135191, 135192)}
    selected = failures+[r for r in rows if r["group"] == "v62_development_replay"]
    return dict(revision=data["revision"], passed=True,
        aggregates={name: aggregate(rs) for name, rs in subsets.items()},
        updating_changed_per_seed=per_seed,
        selected_windows=[dict(**{k: r[k] for k in ("group", "seed", "mode", "updating", "window")},
            recipes={recipe: dict(prediction_error_rms=r["diagnostics"][recipe]["prediction_error_rms"],
                extension_neighbor_fraction=r["diagnostics"][recipe]["neighbor_partition"]["v97_extension"])
                for recipe in RECIPES}) for r in selected],
        averaging="equal window; prediction RMS is a within-window mean of per-transition coordinate RMS",
        decision="HOLD: age-balanced density replay candidate only; prefix-only not selected",
        covariance_scored=False, readout_controls_actor=False, readout_adopted=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--v99-summary", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(json.loads((args.result_dir/"reference_age.json").read_text()),
        json.loads(args.v99_summary.read_text()))
    path = args.result_dir/"summary.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(path)
