from copy import deepcopy

import pytest

from scripts import summarize_general_ood_v143 as merger
from test_general_ood_v142_summary import make_rows as source_rows


def make_rows():
    _, raw_sources = source_rows()
    sources = {(s["training_seed"], s["training_profile"]): s for s in raw_sources}
    for source in sources.values():
        for episode in source["final_evaluation"]["episodes"]:
            episode["final_physical_state"]["angle"] = 0.
    rows = []
    for case, (seed, origin, target) in zip(merger.worker.CASE_NAMES, merger.worker.CASES):
        source = sources[seed, origin]
        episodes = deepcopy(sources[seed, target]["final_evaluation"]["episodes"])
        for e in episodes:
            state = dict(qpos=[0., e["final_physical_state"]["height"], 0., 0., 0., 0.], qvel=[0.]*6)
            steps = [i for i in range(1, e["length"]+1) if i % 4 == 0 or i == e["length"]][-17:]
            e["terminal_trace"] = dict(steps=e["length"], sampled_step_stride=4, max_snapshots=17,
                healthy_height_bounds=[.7, None], healthy_angle_bounds=[-.2, .2],
                actuator_joint_names=["thigh_joint", "leg_joint", "foot_joint"],
                max_abs_normalized_action=.2, action_saturation_fraction=[0.]*3, mean_abs_normalized_action=[.1]*3,
                tail=[dict(step=i, before=state, after=state, joint_torque=[2., 3., 4.],
                    physical_termination=e["physical_termination"] and i == e["length"],
                    time_limit_truncation=e["time_limit_truncation"] and i == e["length"]) for i in steps])
        rows.append(dict(revision=merger.worker.REVISION, case=case, smoke=False,
            source=dict(revision=source["revision"], checkpoint=source["checkpoint"], seed=seed, profile=origin,
                config=source["config"], training_interactions=600000, optimizer_updates=599745),
            runtime=source["runtime"], evaluation_device="cpu", cuda_initialized=False, replay_loaded=False,
            new_training_interactions=0, optimizer_updates=0, reward_changes=[], healthy_termination_changes=[], physics_profile_changes=[],
            spec=dict(merger.worker.v137.evaluation_spec(), case=case, source_profile=origin, evaluation_profile=target, smoke=False),
            episodes=episodes, evaluation_interactions=sum(e["length"] for e in episodes), seconds=1.))
    return rows, sources


def test_summary_separates_matched_cases_and_positive_controls():
    result = merger.summarize(*make_rows())
    assert result["evaluation_episodes"] == 160 and result["matched_reference_episodes"] == 120
    assert result["new_training_interactions"] == result["optimizer_updates"] == 0
    assert result["records"][-1]["case"] == "seed4_id_to_gravity-only"


def test_native_numeric_variation_is_retained_without_changing_endings():
    rows, sources = make_rows()
    episode = rows[5]["episodes"][0]
    episode["episode_return"] += .05
    episode["final_physical_state"]["height"] += .01
    episode["terminal_trace"]["tail"][-1]["after"]["qpos"][1] += .01
    result = merger.summarize(rows, sources)
    assert result["matched_reference_lengths_terminations"] == 120
    delta = result["records"][5]["reference_max_abs_deltas"]
    assert delta["scalar"]["episode_return"] == pytest.approx(.05)
    assert delta["physical"]["height"] == pytest.approx(.01)


@pytest.mark.parametrize("failure", ("updates", "checkpoint", "physics", "trace_end", "matched_length"))
def test_rejects_changed_model_protocol_or_missing_trace(failure):
    rows, sources = make_rows()
    row = rows[0]
    if failure == "updates":
        row["optimizer_updates"] = 1
    elif failure == "checkpoint":
        row["source"]["checkpoint"] = "server/latest.pt"
    elif failure == "physics":
        row["episodes"][0]["profile_events"][0]["actual_multipliers"]["gravity"] = 50.
    elif failure == "trace_end":
        row["episodes"][0]["terminal_trace"]["tail"].pop()
    else:
        sources[3, "friction-only"]["final_evaluation"]["episodes"][0]["length"] -= 1
    with pytest.raises(AssertionError):
        merger.summarize(rows, sources)
