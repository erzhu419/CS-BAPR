from copy import deepcopy
import pytest

from scripts import summarize_general_ood_v134 as merger
from test_general_ood_v133_summary import make_extension_rows


def make_evaluation_rows():
    _, sources = make_extension_rows()
    rows = []
    for source in sources:
        seed, arm = source["training_seed"], source["arm"]
        spec = merger.worker.stage_spec(seed, arm)
        episodes = []
        for cell in merger.worker.cells(spec):
            length = 1000 if arm == "mse-control" else 147
            episodes.append(dict(cell, length=length, episode_return=float(length), mean_velocity=1.,
                physical_termination=length < 1000, time_limit_truncation=length == 1000, diagnostic_cutoff=False,
                noise_scale=0. if cell["mode"] == "deterministic" else 1., mean_policy_std=.2,
                mean_action_noise_l2=.1 if cell["mode"] == "sampled" else 0.))
        rows.append(dict(revision=merger.worker.REVISION, training_seed=seed, arm=arm, spec=spec, smoke=False,
            evaluation_device="cpu", cuda_initialized=False, replay_loaded=False, optimizer_updates=0,
            new_training_interactions=0, checkpoint_selection="predeclared-final-only", source_readout_refitted=False,
            reward_or_physics_changes=[], cpu_anchor_requires_gpu_trajectory_parity=False,
            runtime=source["runtime"], seconds=1., evaluations=episodes,
            evaluation_interactions=sum(e["length"] for e in episodes), source=dict(revision=source["revision"],
                checkpoint=source["checkpoint"], context_count=300000, training_steps=299745,
                initial_context_normalizer=dict(count=300000), config=dict(source["config"], device="cpu"),
                runtime=source["runtime"], gpu_anchor_evaluation=source["final_evaluation"])))
    return rows, sources


def test_summary_keeps_negative_outcomes_and_cpu_gpu_anchor_difference():
    rows, sources = make_evaluation_rows()
    result = merger.summarize(rows, sources)
    assert result["independent_training_seeds"] == 2 and result["evaluation_episodes"] == 340
    assert result["optimizer_updates"] == result["new_training_interactions"] == 0
    assert result["records"][0]["groups"]["new-reset/sampled"]["time_limits"] == 60
    assert result["records"][1]["groups"]["new-reset/sampled"]["falls"] == 60
    assert result["records"][3]["cpu_minus_gpu_anchor"]["mean_length"] != 0.


@pytest.mark.parametrize("failure", ["missing", "wrong_snapshot", "wrong_noise", "nonfinite", "updates", "cutoff"])
def test_rejects_incomplete_or_unmatched_evaluation(failure):
    rows, sources = deepcopy(make_evaluation_rows())
    if failure == "missing":
        rows.pop()
    elif failure == "wrong_snapshot":
        rows[0]["source"]["checkpoint"] = "periodic/best.pt"
    elif failure == "wrong_noise":
        rows[0]["evaluations"][-1]["noise_seed"] += 1
    elif failure == "nonfinite":
        rows[0]["evaluations"][0]["mean_velocity"] = float("nan")
    elif failure == "updates":
        rows[0]["optimizer_updates"] = 1
    else:
        rows[0]["evaluations"][0]["diagnostic_cutoff"] = True
    with pytest.raises((AssertionError, ValueError)):
        merger.summarize(rows, sources)
