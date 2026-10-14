from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts import summarize_general_ood_v149 as merger


def inputs():
    root = Path(__file__).resolve().parents[1]
    row = json.loads((root/"results/general_ood_v149_warm_recovery/seed6_randomized-recovery/source.json").read_text())
    source = json.loads((root/"results/general_ood_v147_randomized_recovery/seed6_no-pulse/source.json").read_text())
    return row, source


def test_actual_warm_protocol_is_accepted():
    merger.validate(*inputs())


@pytest.mark.parametrize("error", ("source", "replay", "optimizer", "budget", "snapshot", "prefix", "peak"))
def test_wrong_provenance_or_protocol_is_rejected(error):
    row, source = inputs()
    if error == "source": row["source"]["checkpoint"] = "other/source.pt"
    if error == "replay": row["initialization"]["replay_size"] = 0
    if error == "optimizer": row["initialization"]["optimizer_steps"]["actor"] = [0]
    if error == "budget": row["continuation_training_interactions"] = 99999
    if error == "snapshot": row["snapshots"][0]["step"] = 660000
    if error == "prefix": row["snapshots"][1]["final_evaluation"]["episodes"][20]["prefix_markers"][0]["cumulative_return"] += 1.
    if error == "peak": row["snapshots"][1]["final_evaluation"]["protocol"]["peak_gravity"] = 2.
    row["training"]["source"] = deepcopy(row["source"])
    row["training"]["initialization"] = deepcopy(row["initialization"])
    row["training"]["snapshots"] = deepcopy(row["snapshots"])
    with pytest.raises(AssertionError):
        merger.validate(row, source)
