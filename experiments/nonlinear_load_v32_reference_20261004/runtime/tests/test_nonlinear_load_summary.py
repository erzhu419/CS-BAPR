from copy import deepcopy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from summarize_nonlinear_load_v10 import gate_passes, paired_diagnostics
from summarize_nonlinear_load_v12 import validate_reproduction
from summarize_nonlinear_load_v13 import expansion_decision, validate_radius_reproduction


def test_added_value_gate_requires_both_training_seeds_at_id_and_50x():
    differences = {str(a): {"matched_training_seed_wins": 2, "mean": 0.001} for a in (1, 50)}
    assert gate_passes(differences)
    failure = deepcopy(differences)
    failure["50"] = {"matched_training_seed_wins": 1, "mean": 0.1}
    assert not gate_passes(failure)
    failure = deepcopy(differences)
    failure["1"] = {"matched_training_seed_wins": 1, "mean": 0.1}
    assert not gate_passes(failure)


def test_path_win_count_does_not_replace_mean_and_retains_the_worst_path():
    learner, reference, seeds = [1.0] * 99 + [-200.0], [0.0] * 100, list(range(50000, 50100))
    result = paired_diagnostics(learner, reference, seeds)
    assert result["path_wins"] == 99 and result["mean"] == -1.01
    assert result["worst_five_paths"][0] == {"evaluation_seed": 50099, "return_gap": -200.0}


def test_new_path_reproduction_does_not_trust_pass_flag_or_drop_endpoint():
    source = {"actor": {"1": -1.0, "50": -10.0}}
    result = {"passed": True, "evaluation_seed": 50000, "severities": [1, 50],
              "rtol": 1e-4, "atol": 1e-3,
              "checks": {"actor": {a: {"actual": v, "source": v, "return_gap": 0.0}
                                     for a, v in source["actor"].items()}}}
    validate_reproduction(result, source, "test-input")
    wrong = deepcopy(result)
    wrong["checks"]["actor"]["50"].update(actual=-11.0, return_gap=-1.0)
    with pytest.raises(AssertionError):
        validate_reproduction(wrong, source, "test-input")
    missing = deepcopy(result)
    del missing["checks"]["actor"]["1"]
    with pytest.raises(AssertionError):
        validate_reproduction(missing, source, "test-input")


def test_radius_expansion_requires_each_radius_to_pass_without_pooling():
    results = {r: {"replication_gates": {"no-symbolic": True, "bapr": True, "resac": True},
                   "learner_added_value_gate_passed": True} for r in ("0.5", "1.0")}
    assert expansion_decision(results)["full_expansion_confirmed"]
    results["1.0"]["learner_added_value_gate_passed"] = False
    decision = expansion_decision(results)
    assert decision["learner_comparison_passed_at_both_expanded_radii"]
    assert not decision["full_expansion_confirmed"]
    results["0.5"]["replication_gates"]["bapr"] = False
    assert not expansion_decision(results)["learner_comparison_passed_at_both_expanded_radii"]


def test_radius_reproduction_recomputes_last_path_and_reported_gaps():
    seeds = list(range(60000, 60100))
    values = {str(a): {"evaluation_seeds": seeds, "returns": [-float(a)] * 100} for a in (1, 50)}
    source = {"amplitude_results": values, "feedback_controls": {"known_inverse": {"amplitude_results": values}}}
    data = {"radius_results": {"0.2": deepcopy(source)}, "reproduction": {
        "passed": True, "radius": 0.2, "source_revision": "nonlinear_load_v12_paths",
        "evaluation_seeds": seeds, "severities": [1, 50], "rtol": 1e-4, "atol": 1e-3,
        "maximum_absolute_return_gaps": {name: {"1": 0.0, "50": 0.0} for name in ("actor", "known_inverse")}}}
    validate_radius_reproduction(data, source, seeds, "test-input")
    wrong = deepcopy(data)
    wrong["radius_results"]["0.2"]["feedback_controls"]["known_inverse"]["amplitude_results"]["50"]["returns"][-1] -= 1
    with pytest.raises(AssertionError):
        validate_radius_reproduction(wrong, source, seeds, "test-input")
    wrong = deepcopy(data)
    wrong["reproduction"]["maximum_absolute_return_gaps"]["actor"]["1"] = 1.0
    with pytest.raises(AssertionError):
        validate_radius_reproduction(wrong, source, seeds, "test-input")
