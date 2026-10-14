from copy import deepcopy

import pytest

from scripts import summarize_general_ood_v145 as merger
from test_general_ood_v144_summary import make_rows as controls


def make_rows():
    reference = next(r for r in controls()[0] if r["training_profile"] == "id")
    reference["physics"].update(body_mass=[0., 1.], geom_friction=[[1., .01, .001]], gravity=[0., 0., -9.81])
    rows = []
    for seed in (3, 4):
        for profile in merger.worker.PROFILES:
            factor = merger.worker.GRAVITY[profile]
            row = deepcopy(reference)
            row.update(revision=merger.worker.REVISION, spec=merger.worker.stage_spec(profile),
                training_seed=seed, training_profile=profile, actual_new_training_interactions=325000 if seed == 3 else 600000)
            row["physics"].update(profile=profile, gravity=[0., 0., -9.81*factor],
                actual_multipliers=dict(body_mass=1., gravity=factor, friction=1.))
            for snapshot in row["snapshots"]:
                own = snapshot["final_evaluation"]
                own["profile"] = profile
                for e in own["episodes"]:
                    e.update(condition="static/"+profile, profile_counts={profile: e["length"]}, non_id_steps=e["length"],
                        profile_events=[dict(step=1, profile=profile, actual_multipliers=dict(body_mass=1., gravity=factor, friction=1.))])
                peak = deepcopy(own)
                peak["profile"] = "gravity-only"
                for e in peak["episodes"]:
                    e.update(condition="static/gravity-only", profile_counts={"gravity-only": e["length"]},
                        profile_events=[dict(step=1, profile="gravity-only", actual_multipliers=dict(body_mass=1., gravity=2.6591479484724942, friction=1.))])
                own["heldout_peak_evaluation"] = peak
            rows.append(row)
    return rows, reference


def test_resumed_prefixes_are_counted_once_and_domains_stay_separate():
    result = merger.summarize(*make_rows())
    assert result["cumulative_training_interactions"] == 2400000 and result["evaluation_episodes"] == 320
    assert [r["restored_prefix_interactions"] for r in result["records"]] == [275000, 275000, 0, 0]
    assert set(result["records"][0]["snapshots"][0]["domains"]) == {"own", "heldout_peak"}


@pytest.mark.parametrize("failure", ("budget", "physics", "peak", "noise", "prefix", "update"))
def test_rejects_mismatched_calibration_evidence(failure):
    rows, reference = make_rows()
    row = rows[0]
    if failure == "budget": row["optimizer_updates"] -= 1
    elif failure == "physics": row["physics"]["body_mass"][1] = 2.
    elif failure == "peak": row["snapshots"][0]["final_evaluation"]["heldout_peak_evaluation"]["episodes"][0]["profile_events"][0]["actual_multipliers"]["gravity"] = 1.5
    elif failure == "noise": row["snapshots"][0]["final_evaluation"]["episodes"][1]["noise_seed"] += 1
    elif failure == "prefix": row["training"]["blocks"].pop(0)
    else: row["snapshots"][1]["final_evaluation"]["optimizer_updates"] = 1
    with pytest.raises(AssertionError): merger.summarize(rows, reference)
