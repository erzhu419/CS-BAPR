from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import analyze_general_ood_v101_age_replay as worker
from scripts import summarize_general_ood_v101 as summary_worker
from scripts import analyze_general_ood_v83_cross_seed_covariance as source
from scripts.v83_cross_seed_covariance import LocalLinearMean
from test_general_ood_v58 import shared_start
from test_general_ood_v68 import trace
from test_general_ood_v78 import banks
from test_general_ood_v89 import sources
from test_general_ood_v91 import expanded
from test_general_ood_v94 import fitted
from test_general_ood_v98 import expanded_coverage, fitted_coverage


def test_new_reference_metadata_and_inherited_density_math(fitted_coverage):
    _, old, _ = fitted_coverage
    family = worker.AgeBalancedDensity(old.bank, old.mean, old.groups, old.errors, old.raw_weight)
    family.recipe = worker.RECIPE
    result = worker.v98.gaussian_check(family, source)
    assert result["physical_gaussian_checks"] == 75
    assert result["direct_member_checks"] == 2*len(family.members)
    metadata = family.metadata(worker.v98.v94.CONDITIONS[1])
    assert metadata["reference_revision"] == worker.REVISION
    assert metadata["reference_payload_revision"] == worker.v100.REVISION
    assert metadata["recipe"] == "age_balanced"


@pytest.mark.parametrize("change", ["settings", "counts"])
def test_calibration_protocol_checks_settings_and_actual_phase_populations(change):
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root/"results/general_ood_v98_regime_replay/fit_diverse_recipe/reference_fit.json").read_text())
    metadata = deepcopy(data["reference_metadata"])
    assert worker.verify_calibration_protocol(metadata, data)
    one = metadata[worker.v98.v94.CONDITIONS[0]]
    if change == "settings":
        one["settings"]["quantile"] = .90
    else:
        one["phase"]["initial"]["counts"][0] += 1
    with pytest.raises(AssertionError):
        worker.verify_calibration_protocol(metadata, data)


def test_promoted_mean_check_detects_changed_recipe(expanded_coverage, monkeypatch):
    _, _, raw, mean, _ = expanded_coverage
    population, queries, records = [], {}, []
    for updating in (False, True):
        for mode in ("slippery", "high_grav"):
            cell = dict(seed=worker.v98.v95.SEEDS[0], mode=mode, updating=updating,
                group="v95_development_updating" if updating else "v95_development_frozen")
            path = f"{mode}_{updating}"
            query = trace(135190+int(updating), mode, 8)
            queries[path] = query
            population.append((cell, path, {}))
            prediction = LocalLinearMean(mean).predict(raw.features.inputs(query))[mode]
            error = raw.design(query)@raw.coefficients[mode]+prediction-raw.targets(query)
            records.append(dict(**cell, window="first_post50", diagnostics={worker.RECIPE: dict(
                prediction_error_rms=float(np.sqrt(np.mean(error**2, axis=1)).mean()))}))
    monkeypatch.setattr(worker.v98, "evaluation_inputs", lambda *a: population)
    monkeypatch.setattr(source.v76.v68, "load_trace", lambda p: queries[p])
    monkeypatch.setattr(source.v78.v77, "windows", lambda: (("first_post50", 0, 8),))
    pilot = dict(records=records)
    assert worker.verify_pilot_means(SimpleNamespace(), raw, mean, pilot, source) == 4
    pilot["records"][0]["diagnostics"][worker.RECIPE]["prediction_error_rms"] += 1.
    with pytest.raises(AssertionError):
        worker.verify_pilot_means(SimpleNamespace(), raw, mean, pilot, source)


def test_full_comparison_keeps_seed_windows_controls_and_rejects_mismatched_pairs():
    root = Path(__file__).resolve().parents[1]
    old = json.loads((root/"results/general_ood_v98_regime_replay/replay_diverse_recipe/regime_replay.json").read_text())
    candidate = deepcopy(old)
    candidate.update(revision=worker.REVISION, recipe=worker.RECIPE, calibration_protocol_matches_incumbent=True)
    result = summary_worker.summarize(candidate, old)
    assert len(result["v95_updating_changed_windows"]) == 24
    assert len(result["v62_tail_controls"]) == 2
    assert all(r["arms"]["incumbent_suffix"] == r["arms"]["age_balanced"] for r in result["v95_updating_changed_windows"])
    for seed in (135190, 135191, 135192):
        data = result["populations"][f"v95_updating_seed{seed}"]
        assert data["incumbent_suffix"] == data["age_balanced"]
    candidate["records"][0], candidate["records"][1] = candidate["records"][1], candidate["records"][0]
    with pytest.raises(AssertionError):
        summary_worker.summarize(candidate, old)
    candidate["records"][0], candidate["records"][1] = candidate["records"][1], candidate["records"][0]
    candidate["records"][0]["change_minus_nochange"]["v94_unchanged__suffix_support_mix"]["first_post50"] += 1.
    with pytest.raises(AssertionError):
        summary_worker.summarize(candidate, old)
