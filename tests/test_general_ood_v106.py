from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import analyze_general_ood_v106_standard_mean_replay as worker
from test_general_ood_v58 import shared_start
from test_general_ood_v68 import trace
from test_general_ood_v78 import banks
from test_general_ood_v89 import sources
from test_general_ood_v91 import expanded
from test_general_ood_v94 import fitted
from test_general_ood_v98 import expanded_coverage, fitted_coverage
from test_general_ood_v98 import test_replay_preserves_full_history_controls_normal_pairs_and_completed_resume as replay_check


@pytest.fixture(scope="module")
def family(fitted_coverage):
    incumbent = fitted_coverage[1]
    payloads = {b: {m: trace(135230, m, 80) for m in worker.v98.v94.MODES} for b in worker.v105.v104.BEHAVIORS}
    means, groups = worker.v105.extend_means(incumbent.mean, payloads, incumbent.groups, np.full(80, 135230))
    return worker.ExpandedMeanDensity(incumbent, means["standard"], groups["standard"])


def test_expanded_mean_keeps_all_covariance_members_and_old_moment_center(family):
    query = trace(135232, "compound", 4)
    inputs = family.bank.features.inputs(query)
    old_fixed, old_members, old_cov, _ = family.incumbent.geometry(inputs)
    fixed, members, cov, _ = family.geometry(inputs)
    np.testing.assert_array_equal(members, old_members)
    assert family.mean.tree.n == family.incumbent.mean.tree.n+80
    assert len(family.mean_groups) == len(family.groups)+80
    assert not np.allclose(fixed, old_fixed)
    for c in worker.v98.v94.CONDITIONS:
        np.testing.assert_array_equal(cov[c], old_cov[c])
    np.testing.assert_allclose(cov[worker.v98.v94.CONDITIONS[1]], cov[worker.v98.v94.CONDITIONS[0]]+
        worker.v98.v94.v93.displacement_second_moment(old_fixed, old_members), rtol=1e-10, atol=1e-10)
    changed = deepcopy(query); changed["next_states"] += 100
    np.testing.assert_array_equal(family.geometry(family.bank.features.inputs(changed))[0], fixed)
    _, checks = worker.v103.physical_checks(query, family, fixed, cov)
    assert checks == 45
    meta = family.metadata(worker.v98.v94.CONDITIONS[1])
    assert meta["mean_fitting_rows"] == family.mean.tree.n
    assert meta["error_reference_rows"] == meta["covariance_mean_reference_rows"] == family.incumbent.mean.tree.n
    assert meta["mean_source_seeds"] == meta["covariance_source_seeds"]+1


def test_full_history_controls_and_completed_resume(family, shared_start, tmp_path, monkeypatch):
    replay_check((None, family, None), shared_start, tmp_path, monkeypatch)


def test_calibration_accepts_only_intentional_mean_size_change():
    root = Path(__file__).resolve().parents[1]
    old = json.loads((root/"results/general_ood_v98_regime_replay/replay_diverse_recipe/regime_replay.json").read_text())
    metadata = deepcopy(old["reference_metadata"])
    for c in worker.v98.v94.CONDITIONS:
        metadata[c]["bank"].update(mean_fitting_rows=34000, covariance_mean_reference_rows=29600)
    assert worker.verify_calibration_protocol(metadata, old)
    metadata[worker.v98.v94.CONDITIONS[1]]["switch_hazard"] += .01
    with pytest.raises(AssertionError):
        worker.verify_calibration_protocol(metadata, old)


def test_full_replay_means_must_match_pilot():
    root = Path(__file__).resolve().parents[1]
    pilot = json.loads((root/"results/general_ood_v105_behavior_support/formal/behavior_support.json").read_text())
    result = json.loads((root/"results/general_ood_v98_regime_replay/replay_diverse_recipe/regime_replay.json").read_text())
    index = {worker.v105.key(r): r for r in result["records"]}
    for row in pilot["records"]:
        for c in worker.v98.v94.CONDITIONS:
            index[worker.v105.key(row)]["attribution"][c][row["window"]]["mean_true_prediction_error_rms"] = row["diagnostics"]["standard"]["prediction_error_rms"]
    assert worker.verify_replay_means(result, pilot) == 186
    result["records"][-1]["attribution"][worker.v98.v94.CONDITIONS[0]]["last64"]["mean_true_prediction_error_rms"] += .01
    with pytest.raises(AssertionError):
        worker.verify_replay_means(result, pilot)
