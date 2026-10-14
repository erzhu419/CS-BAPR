from copy import deepcopy

import pytest

from scripts import summarize_general_ood_v142 as merger
from test_general_ood_v141_summary import make_rows as source_rows


def make_rows():
    sources, _ = source_rows()
    rows = deepcopy(sources)
    for source, row in zip(sources, rows):
        profile = row["training_profile"]
        source["resumable_checkpoint"] = row["source_checkpoint"] = "server/prefix/latest.pt"
        for episode in source["final_evaluation"]["episodes"]:
            episode["final_physical_state"] = dict(height=.69 if episode["physical_termination"] else 1.,
                max_abs_qvel=3., unhealthy_conditions=["height"] if episode["physical_termination"] else [])
        row.update(revision=merger.worker.REVISION, spec=merger.worker.stage_spec(profile), source_revision=source["revision"],
            source_runtime=source["runtime"], source_interactions=300000, source_optimizer_updates=299745,
            actual_new_training_interactions=300000, actual_new_optimizer_updates=300000,
            trained_from_scratch=False, training_mode="fixed-profile-continuation", final_context_count=600000)
        row["training"]["interactions"], row["training"]["optimizer_updates"] = 600000, 599745
        row["training"]["diagnostic_blocks"] += [dict(step=step, interactions=5000, updates=5000,
            metrics=dict(td_mse=1., q_mean=100., alpha=.1)) for step in range(305000, 600001, 5000)]
        template = deepcopy(row["training"]["periodic_evaluations"][-1])
        row["training"]["periodic_evaluations"] += [dict(template, step=step) for step in range(300000, 600000, 25000)]
        row["final_evaluation"]["episodes"] = deepcopy(source["final_evaluation"]["episodes"])
    return sources, rows


def test_budget_summary_preserves_all_sources_and_failure_reasons():
    result = merger.summarize(*make_rows())
    assert result["actual_new_training_interactions"] == result["actual_new_optimizer_updates"] == 1800000
    assert result["cumulative_training_interactions"] == 3600000 and result["cumulative_optimizer_updates"] == 3598470
    assert result["final_models"] == 6 and result["independent_training_seeds"] == 2


@pytest.mark.parametrize("failure", ("prefix", "rate", "budget", "boundary", "noise", "source_checkpoint"))
def test_rejects_broken_lineage_or_unmatched_continuation(failure):
    sources, rows = make_rows()
    if failure == "prefix":
        rows[0]["training"]["diagnostic_blocks"][0]["metrics"]["td_mse"] += 1
    elif failure == "rate":
        rows[0]["config"]["lr_actor"] = 1e-4
    elif failure == "budget":
        rows[0]["actual_new_training_interactions"] = 600000
    elif failure == "boundary":
        rows[0]["training"]["periodic_evaluations"].pop(12)
    elif failure == "noise":
        rows[0]["final_evaluation"]["episodes"][-1]["noise_seed"] += 1
    else:
        rows[0]["source_checkpoint"] = "server/prefix/final.pt"
    with pytest.raises(AssertionError):
        merger.summarize(sources, rows)
