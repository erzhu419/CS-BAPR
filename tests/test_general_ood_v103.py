from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import analyze_general_ood_v103_frozen_covariance as worker
from scripts import analyze_general_ood_v83_cross_seed_covariance as source
from scripts import summarize_general_ood_v103 as summary
from scripts.v83_cross_seed_covariance import subset_bank
from test_general_ood_v58 import shared_start
from test_general_ood_v68 import trace
from test_general_ood_v78 import banks
from test_general_ood_v89 import sources
from test_general_ood_v91 import expanded
from test_general_ood_v94 import fitted
from test_general_ood_v98 import expanded_coverage, fitted_coverage
from test_general_ood_v98 import test_replay_preserves_full_history_controls_normal_pairs_and_completed_resume as replay_check


@pytest.fixture(scope="module")
def hybrid(fitted_coverage):
    old = fitted_coverage[1]
    new_mean = subset_bank(old.mean, np.ones(old.mean.tree.n, dtype=bool))
    new_mean.residuals = {m: old.mean.residuals[m]+.17 for m in worker.v98.v94.MODES}
    candidate = SimpleNamespace(bank=old.bank, mean=new_mean, groups=old.groups)
    return worker.FrozenCovarianceDensity(old, candidate)


def test_replaces_mean_without_recentering_disagreement_or_using_query_target(hybrid):
    query = trace(135221, "compound", 4)
    inputs = hybrid.bank.features.inputs(query)
    old_fixed, old_members, old_cov, _ = hybrid.incumbent.geometry(inputs)
    fixed, members, cov, _ = hybrid.geometry(inputs)
    np.testing.assert_allclose(fixed, old_fixed+.17, rtol=1e-10, atol=1e-10)
    np.testing.assert_array_equal(members, old_members)
    for c in worker.v98.v94.CONDITIONS:
        np.testing.assert_array_equal(cov[c], old_cov[c])
    altered_moment = worker.v98.v94.v93.displacement_second_moment(fixed, members)
    assert not np.allclose(cov[worker.v98.v94.CONDITIONS[1]], cov[worker.v98.v94.CONDITIONS[0]]+altered_moment)
    changed = deepcopy(query); changed["next_states"] += 999
    np.testing.assert_array_equal(hybrid.geometry(hybrid.bank.features.inputs(changed))[0], fixed)
    terms, checked = worker.physical_checks(query, hybrid, fixed, cov)
    assert checked == 45
    assert not np.allclose(terms[worker.v98.v94.CONDITIONS[1]]["likelihoods"], hybrid.incumbent.terms(query)[0][worker.v98.v94.CONDITIONS[1]]["likelihoods"])
    meta = hybrid.metadata(worker.v98.v94.CONDITIONS[1])
    assert meta["complete_covariance_geometry_unchanged"] and not meta["full_error_bank_recomputed"]


def test_hybrid_replay_uses_complete_history_controls_and_resumes(hybrid, shared_start, tmp_path, monkeypatch):
    replay_check((None, hybrid, None), shared_start, tmp_path, monkeypatch)


def test_smoke_entrypoint_does_not_pass_output_recipe_or_directory_to_reference_loader(hybrid, tmp_path, monkeypatch):
    args = SimpleNamespace(v83_dir=tmp_path, v98_dir=tmp_path/"v98", v101_dir=tmp_path/"v101",
        checkpoint_dir=tmp_path/"new-checkpoints", result_dir=tmp_path/"results", stage="smoke")
    for path in (args.v98_dir/"run_results/replay_diverse_recipe/regime_replay.json",
                 args.v101_dir/"run_results/replay/regime_replay.json"):
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(dict(passed=True, evaluation_paths=95, records=[], inner_fits=[])))
    monkeypatch.setattr(worker.v98.v95, "load_readouts", lambda *a: (hybrid.incumbent, None, None))
    def load(reference_args, original, source):
        assert not hasattr(reference_args, "recipe") and not hasattr(reference_args, "checkpoint_dir")
        assert reference_args.v98_dir == args.v98_dir and reference_args.v101_dir == args.v101_dir
        return dict(incumbent_suffix=hybrid.incumbent, age_balanced=hybrid), {}
    monkeypatch.setattr(worker.v102, "load_families", load)
    monkeypatch.setattr(worker, "smoke", lambda *a: dict(passed=True))
    worker.run(args)
    saved = json.loads((args.result_dir/"native_smoke.json").read_text())
    assert saved["recipe"] == worker.RECIPE and saved["stage"] == "smoke" and saved["passed"]


def test_comparison_retains_three_arms_and_detects_mean_or_control_mismatch():
    root = Path(__file__).resolve().parents[1]
    old = json.loads((root/"results/general_ood_v98_regime_replay/replay_diverse_recipe/regime_replay.json").read_text())
    age = json.loads((root/"results/general_ood_v101_age_replay/replay/regime_replay.json").read_text())
    candidate = deepcopy(age)
    candidate.update(revision=worker.REVISION, recipe=worker.RECIPE,
        complete_covariance_geometry_unchanged=True, covariance_moment_recentered=False)
    actual = summary.summarize(candidate, old, age)
    assert set(actual["populations"]["v95_updating"]) == {"incumbent_suffix", "age_balanced", "frozen_covariance"}
    assert actual["reference_budget"] == dict(incumbent_distinct_extension_inputs=4400,
        age_balanced_distinct_extension_inputs=4400, overlap=2200,
        hybrid_distinct_extension_inputs=6600, equal_used_reference_input_budget=False,
        additional_environment_steps=0)
    candidate["records"][0]["attribution"][worker.v98.v94.CONDITIONS[1]]["last64"]["mean_true_prediction_error_rms"] += .01
    with pytest.raises(AssertionError):
        summary.summarize(candidate, old, age)
    candidate = deepcopy(age)
    candidate.update(revision=worker.REVISION, recipe=worker.RECIPE,
        complete_covariance_geometry_unchanged=True, covariance_moment_recentered=False)
    candidate["records"][0]["observers"]["v94_unchanged__suffix_support_mix"]["full"]["mean_coefficient"] += .01
    with pytest.raises(AssertionError):
        summary.summarize(candidate, old, age)
