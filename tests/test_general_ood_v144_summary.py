from copy import deepcopy
import pytest

from scripts import summarize_general_ood_v144 as merger
from test_general_ood_v142_summary import make_rows as reference_rows


def make_rows():
    old, new = reference_rows()
    references = {(steps, r["training_seed"], r["training_profile"]): r
                  for steps, group in ((300000, old), (600000, new)) for r in group}
    rows = []
    for source in new:
        seed, profile = source["training_seed"], source["training_profile"]
        snapshots = []
        for steps in (300000, 600000):
            final = deepcopy(references[steps, seed, profile]["final_evaluation"])
            final["protocol"]["context_mode"] = "not-applicable"
            for e in final["episodes"]:
                e.update(context_mode="not-applicable", context_updates=0)
            snapshots.append(dict(step=steps, optimizer_updates=steps-255,
                checkpoint=f"server/step{steps}.zip", final_evaluation=final))
        rows.append(dict(revision=merger.worker.REVISION, training_seed=seed, training_profile=profile,
            smoke=False, spec=merger.worker.stage_spec(profile), config=merger.worker.SETTINGS,
            training_device="cuda", runtime=dict(stable_baselines3="2.7.0"),
            cumulative_training_interactions=600000, optimizer_updates=599745, observation_dim=11,
            use_context=False, use_symbolic_constraints=False, reward_normalization=False,
            policy_receives_profile_labels=False, reward_changes=[], healthy_termination_changes=[],
            bootstrap_mask="terminated-only", replay_handle_timeout_termination=True,
            optimizer_updates_during_evaluation=0, checkpoint_selection="predeclared300k600k-not-best",
            physics=source["physics"], reward_settings=source["reward_settings"], seconds=1., snapshots=snapshots,
            training=dict(step=600000, snapshots=snapshots,
                blocks=[dict(step=n, interactions=5000, updates=4745 if n == 5000 else 5000, metrics={})
                        for n in range(5000, 600001, 5000)],
                periodic_evaluations=[dict(step=n) for n in range(0, 600000, 25000)])))
    return rows, references


def test_formal_evidence_matches_budgets_and_preserves_seed_specific_results():
    result = merger.summarize(*make_rows())
    assert result["training_interactions"] == 3600000 and result["evaluation_episodes"] == 240
    assert result["independent_training_seeds"] == 2 and len(result["records"]) == 6
    assert all(r["snapshots"][-1]["modes"]["sampled"]["sac"]["time_limits"] == 15 for r in result["records"] if r["profile"] == "id")


@pytest.mark.parametrize("failure", ("budget", "noise", "physics", "missing_snapshot", "updates", "wrong_reference"))
def test_rejects_unmatched_formal_or_reference_evidence(failure):
    rows, references = make_rows()
    row = rows[0]
    if failure == "budget":
        row["optimizer_updates"] -= 1
    elif failure == "noise":
        row["snapshots"][0]["final_evaluation"]["episodes"][1]["noise_seed"] += 1
    elif failure == "physics":
        row["physics"]["actual_multipliers"]["gravity"] = 2.
    elif failure == "missing_snapshot":
        row["snapshots"].pop()
    elif failure == "updates":
        row["snapshots"][0]["final_evaluation"]["optimizer_updates"] = 1
    else:
        references[300000, 3, "friction-only"]["training_profile"] = "id"
    with pytest.raises(AssertionError):
        merger.summarize(rows, references)
