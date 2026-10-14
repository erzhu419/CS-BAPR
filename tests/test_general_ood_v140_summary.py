from copy import deepcopy

import pytest

from scripts import summarize_general_ood_v140 as merger
from test_general_ood_v137_summary import make_rows as id_rows
from test_general_ood_v138_summary import make_rows as component_rows


def make_rows():
    components, _ = component_rows()
    controls = [r for r in components if r["training_profile"] in ("friction-only", "gravity-only")]
    controls += [r for r in id_rows() if r["training_profile"] == "id"]
    for control in controls:
        control["config"].update(lr_actor=1e-4, lr_critic=3e-4)
        for block in control["training"]["diagnostic_blocks"]:
            block["metrics"] = dict(td_mse=1., alpha=.1, q_mean=100.)
    rows = deepcopy(controls)
    for row in rows:
        row.update(revision=merger.worker.REVISION, spec=merger.worker.stage_spec(row["training_profile"]))
        row["config"]["lr_critic"] = merger.worker.CRITIC_LR
    return rows, controls


def test_paired_summary_keeps_all_negative_results_and_budgets():
    result = merger.summarize(*make_rows())
    assert result["training_interactions"] == result["reused_control_training_interactions"] == 1800000
    assert result["optimizer_updates"] == 1798470 and result["final_evaluation_episodes"] == 120
    assert len(result["records"]) == 6 and result["independent_training_seeds"] == 2
    assert all(r["sampled_length_delta"] == 0 for r in result["records"])
    assert all(r["control"]["revision"] in (merger.worker.v137.REVISION, merger.worker.v138.REVISION) for r in result["records"])


@pytest.mark.parametrize("failure", ("missing_control", "wrong_actor_rate", "wrong_critic_rate", "noise", "budget"))
def test_rejects_unmatched_control_or_more_than_one_config_change(failure):
    rows, controls = make_rows()
    if failure == "missing_control":
        controls.pop()
    elif failure == "wrong_actor_rate":
        rows[0]["config"]["lr_actor"] = 3e-5
    elif failure == "wrong_critic_rate":
        for r in rows:
            r["config"]["lr_critic"] = 3e-4
    elif failure == "noise":
        controls[0]["final_evaluation"]["episodes"][-1]["noise_seed"] += 1
    else:
        controls[0]["training"]["interactions"] = 100000
    with pytest.raises(AssertionError):
        merger.summarize(rows, controls)
