from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts import summarize_general_ood_v147 as merger


def source():
    root = Path(__file__).resolve().parents[1]
    return json.loads((root/"results/general_ood_v147_randomized_recovery/seed5_no-pulse/source.json").read_text())


def test_actual_matched_protocol_is_accepted():
    merger.validate(source())


@pytest.mark.parametrize("error", ("updates", "budget", "peak", "prefix", "schedule", "selection", "coverage"))
def test_changed_protocol_or_evidence_is_rejected(error):
    row = source()
    final = row["snapshots"][1]["final_evaluation"]
    if error == "updates": row["optimizer_updates"] -= 1
    if error == "budget": row["actual_new_training_interactions"] -= 1
    if error == "peak": final["protocol"]["peak_gravity"] = 2.
    if error == "prefix":
        final["episodes"][20]["prefix_markers"][0]["cumulative_return"] += 1.
    if error == "schedule": final["episodes"][20]["profile_events"][1]["step"] += 1
    if error == "selection": row["checkpoint_selection"] = "best"
    if error == "coverage": row["recovery_coverage"]["baseline_steps"] -= 1
    row["training"]["snapshots"] = deepcopy(row["snapshots"])
    with pytest.raises(AssertionError):
        merger.validate(row)


def test_short_recovery_does_not_count_as_full_healthy_survival():
    e = dict(length=250, episode_return=250., mean_velocity=1., reset_seed=1, noise_seed=None,
        physical_termination=True, time_limit_truncation=False,
        recovery=dict(exposed=True, recovery_evaluable=True, recovered=True))
    values = merger.metrics([e], {(1, None): dict(length=1000)})
    assert values["recovered"] == values["delayed_falls_after_recovery"] == 1
    assert values["healthy_full_survival"] == 0 and values["mean_paired_length_delta"] == -750.
