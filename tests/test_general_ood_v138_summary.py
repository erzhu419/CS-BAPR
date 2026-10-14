from copy import deepcopy

import pytest

from scripts import summarize_general_ood_v138 as merger
from test_general_ood_v137_summary import make_rows as previous_rows


def make_rows():
    originals = previous_rows()
    controls = [r for r in originals if r["training_profile"] in ("id", "mega_event")]
    rows = deepcopy(originals)
    mapping = dict(zip(("id", "easy_surface", "mega_event"), merger.worker.PROFILES))
    for row in rows:
        profile = mapping[row["training_profile"]]
        settings = merger.worker.profile_settings(profile)
        multipliers = {k: settings[k] for k in merger.worker.FACTORS.values()}
        row.update(revision=merger.worker.REVISION, training_profile=profile, spec=merger.worker.stage_spec(profile))
        row["checkpoint"] = f"server/seed{row['training_seed']}_{profile}/final.pt"
        row["physics"] = dict(profile=profile, actual_multipliers=multipliers)
        for e in row["training"]["periodic_evaluations"]:
            e.update(profile=profile, actual_multipliers=multipliers)
        row["final_evaluation"]["profile"] = profile
        for e in row["final_evaluation"]["episodes"]:
            e.update(condition="static/"+profile, profile_counts={profile: e["length"]}, non_id_steps=e["length"],
                     profile_events=[dict(step=1, profile=profile, actual_multipliers=multipliers)])
    return rows, controls


def test_summary_keeps_matched_controls_and_negative_component_outcomes():
    result = merger.summarize(*make_rows())
    assert result["training_interactions"] == 1800000 and result["optimizer_updates"] == 1798470
    assert result["reused_control_training_interactions"] == 1200000
    assert result["final_evaluation_episodes"] == 120 and result["independent_training_seeds"] == 2
    assert result["records"][2]["modes"]["sampled"]["falls"] == 15
    assert result["control_records"][0]["modes"]["sampled"]["time_limits"] == 15
    assert result["control_records"][1]["modes"]["sampled"]["falls"] == 15


@pytest.mark.parametrize("failure", ["missing", "two_factors", "wrong_final_profile", "wrong_periodic_profile", "control_budget", "control_noise", "control_config", "control_missing"])
def test_rejects_wrong_component_or_unmatched_control(failure):
    rows, controls = make_rows()
    if failure == "missing":
        rows.pop()
    elif failure == "two_factors":
        rows[0]["physics"]["actual_multipliers"]["gravity"] = 2.659
    elif failure == "wrong_final_profile":
        rows[0]["final_evaluation"]["episodes"][0]["condition"] = "static/mega_event"
    elif failure == "wrong_periodic_profile":
        rows[0]["training"]["periodic_evaluations"][0]["profile"] = "mega_event"
    elif failure == "control_budget":
        controls[0]["training"]["interactions"] = 100000
    elif failure == "control_noise":
        controls[0]["final_evaluation"]["protocol"]["action_noise_seeds"] = [138000]
    elif failure == "control_config":
        controls[0]["config"]["weight_reg"] = .01
    else:
        controls.pop()
    with pytest.raises((AssertionError, ValueError)):
        merger.summarize(rows, controls)
