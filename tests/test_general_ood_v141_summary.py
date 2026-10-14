from copy import deepcopy

import pytest

from scripts import summarize_general_ood_v141 as merger
from test_general_ood_v140_summary import make_rows as critic_rows


def make_rows():
    _, controls = critic_rows()
    rows = deepcopy(controls)
    for row in rows:
        row.update(revision=merger.worker.REVISION, spec=merger.worker.stage_spec(row["training_profile"]))
        row["config"]["lr_actor"] = merger.worker.ACTOR_LR
    return rows, controls


def test_actor_summary_keeps_primary_control_budget_and_all_seeds():
    result = merger.summarize(*make_rows())
    assert result["training_interactions"] == result["reused_control_training_interactions"] == 1800000
    assert result["optimizer_updates"] == 1798470 and result["final_evaluation_episodes"] == 120
    assert len(result["records"]) == 6


@pytest.mark.parametrize("key,value", [("lr_actor", 1e-4), ("lr_critic", 1e-4)])
def test_rejects_unmatched_rate_change(key, value):
    rows, controls = make_rows()
    for row in rows:
        row["config"][key] = value
    with pytest.raises(AssertionError):
        merger.summarize(rows, controls)
