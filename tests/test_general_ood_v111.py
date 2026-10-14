from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import analyze_general_ood_v111_fresh_attribution as worker
from scripts import summarize_general_ood_v110 as merger
from test_general_ood_v58 import shared_start
from test_general_ood_v68 import trace
from test_general_ood_v78 import banks
from test_general_ood_v89 import sources
from test_general_ood_v91 import expanded
from test_general_ood_v94 import fitted
from test_general_ood_v98 import expanded_coverage, fitted_coverage
from test_general_ood_v106 import family


@pytest.fixture(scope="module")
def parts():
    root = Path(__file__).resolve().parents[1]/"results/general_ood_v110_fresh_mixture"
    return [json.loads((root/f"seed{s}/fresh_seed.json").read_text()) for s in worker.v110.SEEDS]


def test_confirmation_merge_uses_three_environment_seeds_and_rejects_changed_candidate(parts):
    result = merger.summarize(parts)
    assert result["paths"] == 30 and result["changed_windows"] == 48
    paired = result["paired_updating_accuracy"]
    assert paired["mean_delta"] == pytest.approx(.006927083333333288)
    assert paired["confidence_interval_95"] == pytest.approx([-.018413662860782862, .032267829527449436])
    assert result["populations"]["updating"]["normal_density_mixture"][worker.v110.v109.v108.PRIMARY]["maximum_normal_mean_risk"] == pytest.approx(.007587803360817263)
    changed = deepcopy(parts); changed[1]["normal_thresholds_recalibrated"] = True
    with pytest.raises(AssertionError):
        merger.summarize(changed)


def test_failure_attribution_includes_all_four_failures_frozen_pairs_and_normal_controls(parts):
    records = [r for part in parts for r in part["records"]]
    selected = worker.select_windows(records)
    assert len(selected) == 14 and sum(r["mode"] == "normal" for r in selected) == 6
    assert sum(r["mode"] != "normal" and r["updating"] for r in selected) == 4
    assert sum(r["mode"] != "normal" and not r["updating"] for r in selected) == 4
    assert {r["mode"] for r in worker.select_windows(records, True)} == {"normal", "compound"}


def test_oracle_only_changes_centers_and_preserves_both_production_densities(family):
    seed, mode = 135252, "compound"
    query = trace(seed, mode, 4)
    query["oracle_observations"] = np.stack([trace(seed, m, 4)["next_states"] for m in worker.v98.v94.MODES], axis=1)
    result, terms = worker.diagnose(query, family, mode)
    for c in worker.summary.CONDITIONS:
        oracle = result["densities"][c]["oracle"]
        assert oracle["true_prediction_error_rms"] == pytest.approx(0., abs=1e-12)
        for a, model in (("old", family.incumbent), ("new", family)):
            direct, _ = model.terms(query)
            for k in direct[c]:
                np.testing.assert_allclose(terms[a][c][k], direct[c][k], rtol=1e-10, atol=1e-8)
            np.testing.assert_array_equal(terms[a][c]["logdet"], terms["oracle"][c]["logdet"])
            density = result["densities"][c][a]
            assert density["mean_energy_advantage"]+density["mean_volume_advantage"] == pytest.approx(density["mean_true_minus_best_other_log_density"])
        shift = result["densities"][c]["normal_mixture_attribution"]
        assert shift["normal_log_density_change_vs_old"]-shift["heavy_log_density_change_vs_old"] == pytest.approx(shift["normal_minus_heavy_margin_change"])
        assert 0 <= shift["mean_old_normal_component_responsibility"] <= 1
        assert -np.log(2)-1e-12 <= shift["mean_mixture_log_penalty_vs_best_normal"] <= 1e-12
    assert result["support"]["old"]["added_neighbor_fraction"] == 0
    assert 0 <= result["support"]["new"]["added_neighbor_fraction"] <= 1


def test_oracle_truth_must_match_actual_executed_transition(family):
    query = trace(135252, "normal", 4)
    query["oracle_observations"] = np.stack([trace(135252, m, 4)["next_states"] for m in worker.v98.v94.MODES], axis=1)
    query["oracle_observations"][0, 0, 0] += 1
    with pytest.raises(AssertionError):
        worker.oracle_centers(query, family.bank, "normal")
