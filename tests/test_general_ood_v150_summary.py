import json
from pathlib import Path

import pytest

from scripts import summarize_general_ood_v150 as summary

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def evidence():
    row = json.loads((ROOT/"results/general_ood_v150_noise_timing/smoke_seed6_randomized-recovery/noise_timing.json").read_text())
    source = json.loads((ROOT/"results/general_ood_v149_warm_recovery/seed6_randomized-recovery/source.json").read_text())
    return row, source


def test_native_smoke_is_valid(evidence):
    summary.validate(*evidence)


@pytest.mark.parametrize("mutation", ("clock", "budget", "source", "noise", "prefix", "gravity"))
def test_changed_protocol_or_evidence_is_rejected(evidence, mutation):
    row, source = evidence
    pulse = next(e for e in row["episodes"] if e["condition"] == "pulse-quiet_after100_pulse10")
    if mutation == "clock":
        row["spec"]["oracle_clock"] = False
    elif mutation == "budget":
        row["optimizer_updates"] = 1
    elif mutation == "source":
        row["source"]["checkpoint"] = row["source"]["checkpoint"].replace("700000", "650000")
    elif mutation == "noise":
        pulse["noise_scale_counts"]["0.0"] += 1
    elif mutation == "prefix":
        pulse["prefix_markers"][0]["cumulative_return"] += 1
    else:
        pulse["profile_events"][1]["actual_multipliers"]["gravity"] = 2.
    with pytest.raises(AssertionError):
        summary.validate(row, source)
