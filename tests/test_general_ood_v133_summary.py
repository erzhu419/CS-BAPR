from copy import deepcopy
import pytest

from scripts import summarize_general_ood_v133 as merger
from test_general_ood_v132_summary import make_rows


def make_extension_rows():
    sources = make_rows()
    extensions = deepcopy(sources)
    for source, row in zip(sources, extensions):
        row.update(revision=merger.worker.REVISION, training_driver_revision=merger.worker.v132.REVISION,
            spec=merger.worker.stage_spec(row["arm"]), source_revision=source["revision"],
            source_checkpoint=source["resumable_checkpoint"], source_runtime=source["runtime"],
            source_interactions=100000, actual_new_training_interactions=200000,
            actual_new_optimizer_updates=200000, trained_from_scratch=False,
            checkpoint=source["checkpoint"].replace("server/", "extended/"),
            resumable_checkpoint=source["resumable_checkpoint"].replace("server/", "extended/"))
        row["training"].update(interactions=300000, optimizer_updates=299745)
        row["training"]["diagnostic_blocks"] += [dict(step=step, interactions=5000, updates=5000,
            time_limit_truncations=0, metrics=dict(td_mse=2.)) for step in range(105000, 300001, 5000)]
        row["training"]["periodic_evaluations"] += [dict(step=step) for step in range(125000, 300000, 25000)]
        length = 1000 if row["arm"] == "mse-control" else (150 if row["training_seed"] == 3 else 900)
        final = row["final_evaluation"]
        for episode in final["episodes"]:
            episode.update(length=length, episode_return=float(length), physical_termination=length < 1000,
                           time_limit_truncation=length == 1000)
        final.update(mean_return=float(length), mean_episode_length=float(length),
                     physical_terminations=5 if length < 1000 else 0, environment_step_calls=5*length)
    return sources, extensions


def test_counts_new_budget_separately_and_retains_both_negative_huber_pairs():
    result = merger.summarize(*make_extension_rows())
    assert result["actual_new_training_interactions"] == result["actual_new_optimizer_updates"] == 800000
    assert result["reused_source_interactions"] == 400000 and result["independent_training_seeds"] == 2
    assert result["final_evaluation_episodes"] == 20
    assert all(r["huber_minus_mse"]["mean_episode_length"] < 0 for r in result["paired_differences"])


@pytest.mark.parametrize("failure", ["duplicate", "wrong_source", "changed_prefix", "wrong_budget", "wrong_mask", "nonfinite", "bad_final_mean"])
def test_rejects_unmatched_lineage_or_counts(failure):
    sources, rows = make_extension_rows()
    if failure == "duplicate":
        rows[-1]["training_seed"] = 3
    elif failure == "wrong_source":
        rows[-1]["source_checkpoint"] = "smoke/latest.pt"
    elif failure == "changed_prefix":
        rows[-1]["training"]["diagnostic_blocks"][0]["metrics"]["td_mse"] = 3.
    elif failure == "wrong_budget":
        rows[-1]["actual_new_optimizer_updates"] = 199999
    elif failure == "wrong_mask":
        rows[-1]["spec"]["bootstrap_mask"] = "terminated-or-truncated"
    elif failure == "nonfinite":
        rows[-1]["training"]["diagnostic_blocks"][-1]["metrics"]["td_mse"] = float("nan")
    else:
        rows[-1]["final_evaluation"]["mean_episode_length"] = 999.
    with pytest.raises((AssertionError, ValueError)):
        merger.summarize(sources, rows)
