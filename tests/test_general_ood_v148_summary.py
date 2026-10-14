import json
from pathlib import Path

import pytest

from scripts import summarize_general_ood_v148 as merger


def inputs():
    root = Path(__file__).resolve().parents[1]
    case = "seed5_randomized-recovery"
    row = json.loads((root/f"results/general_ood_v148_known_pulse/smoke_paired_{case}/known_pulse.json").read_text())
    source = json.loads((root/f"results/general_ood_v147_randomized_recovery/{case}/source.json").read_text())
    return row, source


def test_same_run_smoke_is_accepted():
    merger.validate(*inputs())


@pytest.mark.parametrize("error", ("peak", "prefix", "source", "updates"))
def test_changed_native_control_or_source_is_rejected(error):
    row, source = inputs()
    if error == "peak": row["episodes"][2]["profile_events"][1]["actual_multipliers"]["gravity"] = 2.6591479484724942
    if error == "prefix": row["episodes"][2]["prefix_markers"][0]["cumulative_return"] += 1.
    if error == "source": row["source"]["checkpoint"] = "other/step600000.zip"
    if error == "updates": row["optimizer_updates"] = 1
    with pytest.raises(AssertionError):
        merger.validate(row, source)
