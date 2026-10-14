from copy import deepcopy
import pytest

from scripts import summarize_general_ood_v132 as merger


def make_rows():
    rows = []
    for seed in merger.worker.SEEDS:
        for arm in merger.worker.ARMS:
            length = 200 if arm == "mse-control" else 100
            episodes = [dict(seed=reset, length=length, episode_return=float(length),
                mean_forward_velocity=1., physical_termination=True, time_limit_truncation=False)
                for reset in range(150000, 150005)]
            blocks = [dict(step=step, interactions=5000, updates=4745 if step == 5000 else 5000,
                time_limit_truncations=0, metrics=dict(td_mse=2.)) for step in range(5000, 100001, 5000)]
            rows.append(dict(revision=merger.worker.REVISION, training_seed=seed, arm=arm, smoke=False,
                spec=merger.worker.stage_spec(arm), config=dict(weight_reg=0.), runtime={}, reward_settings={},
                trained_from_scratch=True, training_mode="normal-only", effective_beta=-2.,
                checkpoint_selection="final-not-best", periodic_evaluation_selects_checkpoint=False,
                source_readout_refitted=False, optimizer_updates_during_evaluation=0,
                checkpoint=f"server/seed{seed}_{arm}/final.pt", resumable_checkpoint=f"server/seed{seed}_{arm}/latest.pt", seconds=1.,
                training=dict(interactions=100000, optimizer_updates=99745, diagnostic_blocks=blocks,
                              periodic_evaluations=[dict(step=step) for step in (0, 25000, 50000, 75000)]),
                final_evaluation=dict(episodes=episodes, mean_return=float(length), mean_episode_velocity=1.,
                                     mean_episode_length=float(length), physical_terminations=5, environment_step_calls=5*length)))
    return rows


def test_summary_keeps_negative_pairs_and_training_seed_unit():
    result = merger.summarize(make_rows())
    assert result["independent_training_seeds"] == 2 and result["final_evaluation_episodes"] == 20
    assert result["interactions"] == 400000 and result["optimizer_updates"] == 398980
    assert result["training_time_limits"] == 0 and result["final_physical_terminations"] == 20
    assert all(r["huber_minus_mse"]["mean_episode_length"] == -100. for r in result["paired_differences"])


@pytest.mark.parametrize("failure", ["missing_arm", "duplicate_seed", "wrong_mask", "mismatched_config", "bad_mean", "nonfinite"])
def test_summary_rejects_incomplete_or_unmatched_evidence(failure):
    rows = deepcopy(make_rows())
    if failure == "missing_arm":
        rows.pop()
    elif failure == "duplicate_seed":
        rows[-1]["training_seed"] = 3
    elif failure == "wrong_mask":
        rows[-1]["spec"]["bootstrap_mask"] = "terminated-or-truncated"
    elif failure == "mismatched_config":
        rows[-1]["config"]["weight_reg"] = .01
    elif failure == "bad_mean":
        rows[-1]["final_evaluation"]["mean_episode_length"] = 999.
    else:
        rows[-1]["training"]["diagnostic_blocks"][0]["metrics"]["td_mse"] = float("nan")
    with pytest.raises((AssertionError, ValueError)):
        merger.summarize(rows)
