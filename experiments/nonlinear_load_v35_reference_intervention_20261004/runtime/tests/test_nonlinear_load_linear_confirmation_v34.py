from copy import deepcopy
import sys

import numpy as np
import pytest

from csbapr.envs.lqr_env import A, B
from csbapr.envs.sustained_load_env import SustainedLoadEnv
from scripts.diagnose_nonlinear_load_v33 import CostReference
from scripts.diagnose_nonlinear_load_v34 import confirm, CONTROLLERS, PREFLIGHT_START, REVISION, SEED_START
from scripts.summarize_nonlinear_load_v34 import decide, validate


def source():
    fit = {"gain": np.zeros((2, 6)).tolist(), "dynamics": A.tolist(), "load_matrix": B.tolist(),
           "linear_action_matrix": (2.5 * B).tolist(), "cubic_action_matrix": (2.5 * B).tolist()}
    reference = CostReference(fit, SustainedLoadEnv.specification())
    return {"reference_identification": {"fits": {"cubic": fit}},
            "environment_settings": SustainedLoadEnv.specification(), "source_candidate_gain": reference.gain.tolist()}


def test_changed_candidate_gain_rejected_before_rollouts():
    data = source()
    data["source_candidate_gain"][0][4] += .001
    with pytest.raises(AssertionError):
        confirm(data, 2, PREFLIGHT_START)


def fixture_run():
    data = source()
    manifest = {"source_revision": "nonlinear_load_v33_cost_reference", "source_config": {"fixed": True},
                "environment_settings": data["environment_settings"],
                "source_candidate_gains": {"21": deepcopy(data["source_candidate_gain"])},
                "runtime_requirement": {"numpy": "2.2.6", "scipy": "1.15.3", "python_minor_versions": ["3.10", "3.11"]}}
    seeds = list(range(PREFLIGHT_START, PREFLIGHT_START + 2))
    controls = {name: {"evaluation_seeds": seeds, "returns": [-1., -1.], "mean_cost": 1.,
                       "state_costs_by_window": [[.25]*4]*2, "command_costs_by_window": [[0.]*4]*2,
                       "force_rms": [0., 0.], "maximum_load_components": [0., 0.], "force_change_counts": [0., 0.]}
                for name in CONTROLLERS}
    block = {"controllers": controls, "linear_minus_baselines": {
        name: {"per_path_cost_changes": [0., 0.], "mean_total_cost_change": 0.}
        for name in CONTROLLERS if name != "cost_linear"}}
    run = {"revision": REVISION, "source_revision": manifest["source_revision"], "training_seed": 21,
           "source_symbolic_constraint": "on", "preflight_only": True, "source_config": manifest["source_config"],
           "environment_settings": data["environment_settings"], "evaluation_seeds": seeds,
           "radii": [.2, .5, 1.], "severities": [1, 50], "source_controls_reproduced": True,
           "source_candidate_gain_reproduced": True, "references_changed": False,
           "new_training_interactions": 0, "new_identification_interactions": 0, "optimizer_updates": 0,
           "checkpoint_access": False, "nonlinear_action_components_solved": 0, "new_evaluation_episodes": 48,
           "reproduction_episodes": 48, "evaluation_interactions": 19200,
           "reference_specification": {"rule": "clip(-gain*observation, -1, 1)",
                                       "gain": data["source_candidate_gain"], "load_persistence": .9},
           "runtime": {"numpy": "2.2.6", "scipy": "1.15.3", "python": sys.version},
           "radius_results": {str(radius): {str(load): deepcopy(block) for load in (1, 50)} for radius in (.2, .5, 1.)}}
    return run, manifest


@pytest.mark.parametrize("failure", (None, "paths", "gain", "paired_cost"))
def test_validator_detects_protocol_or_cost_mismatch(failure):
    run, manifest = fixture_run()
    if failure == "paths":
        run["evaluation_seeds"] = [SEED_START, SEED_START + 1]
    elif failure == "gain":
        run["reference_specification"]["gain"][0][4] += .001
    elif failure == "paired_cost":
        run["radius_results"]["0.2"]["1"]["linear_minus_baselines"]["fitted_local"]["mean_total_cost_change"] = -1.
    if failure:
        with pytest.raises(AssertionError):
            validate(run, manifest, 21, preflight=True)
    else:
        validate(run, manifest, 21, preflight=True)


@pytest.mark.parametrize("failure", (None, "id", "ood"))
def test_gate_keeps_every_seed_block_and_descriptive_controls_separate(failure):
    runs = [{"radius_results": {str(r): {str(load): {"linear_minus_baselines": {
        "fitted_local": {"mean_total_cost_change": -.1}, "fitted_inverse": {"mean_total_cost_change": .1}}}
        for load in (1, 50)} for r in (.2, .5, 1.)}} for _ in (21, 22)]
    if failure:
        runs[1]["radius_results"]["0.2"]["1" if failure == "id" else "50"]["linear_minus_baselines"]["fitted_local"]["mean_total_cost_change"] = 0.
    result = decide(runs)
    assert result["reference_intervention_admitted"] == (failure is None)
    assert not result["learner_replication_admitted"] and result["historical_hold_unchanged"]
