from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import analyze_general_ood_v107_normal_attribution as worker
from scripts import summarize_general_ood_v106 as summary
from scripts.v73_recursive_mode_readout import RecursiveModeReference
from test_general_ood_v58 import shared_start


def test_population_includes_normal_pairs_failures_and_history():
    root = Path(__file__).resolve().parents[1]
    records = json.loads((root/"results/general_ood_v106_standard_mean_replay/replay/regime_replay.json").read_text())["records"]
    selected = worker.select_paths(records)
    assert len(selected) == 18 and sum(r["mode"] == "normal" for r in selected) == 7
    smoke = worker.select_paths(records, True)
    assert {(r["updating"], r["mode"]) for r in smoke} == {
        (updating, mode) for updating in (True, False) for mode in ("normal", "compound")}
    assert {r["updating"] for r in selected if r["seed"] == 135192 and r["mode"] == "normal"} == {True, False}


def test_crossed_calibration_preserves_posterior_and_episode_history(shared_start):
    references = {}
    for m in worker.RECIPES:
        ref = RecursiveModeReference()
        ref.phase = deepcopy(shared_start[4])
        references[m] = ref
    likelihoods = dict(old=np.tile([0., -1., -2., -3., -4.], (8, 1)),
                       new=np.tile([0., -2., -1., -4., -3.], (8, 1)))
    dones = np.arange(8) % 4 == 3
    actual = worker.crossed_readouts(likelihoods, references, dones)
    for m in worker.RECIPES:
        a, b = (actual[f"{m}_mean__{c}_calibration"] for c in worker.RECIPES)
        np.testing.assert_array_equal(a["mode_scores"], b["mode_scores"])
        episodes = worker.episode_diagnostics(dict(dones=dones), a)
        assert [(r["start"], r["stop"]) for r in episodes] == [(0, 4), (4, 8)]
        assert episodes[1]["mean_risk"] == pytest.approx(a["risk"][4:].mean())
        assert not np.array_equal(a["mode_scores"][4], a["mode_scores"][0])


def test_cache_loader_rejects_wrong_mean_bank(tmp_path):
    key = ("test", 1, "normal", True)
    saved = dict(revision=worker.v98.REVISION, recipe=worker.RECIPES["new"], key=key,
        raw_rows=27400, mean_rows=34000, terms={c: dict(likelihoods=np.zeros((6000, 5)))
        for c in worker.v98.v94.CONDITIONS})
    path = tmp_path/"terms.pt"
    torch.save(saved, path)
    assert len(worker.load_likelihoods(path, worker.RECIPES["new"], key, 34000)) == 3
    with pytest.raises(AssertionError):
        worker.load_likelihoods(path, worker.RECIPES["new"], key, 29600)


def test_comparison_rejects_wrong_pilot():
    root = Path(__file__).resolve().parents[1]
    new = json.loads((root/"results/general_ood_v106_standard_mean_replay/replay/regime_replay.json").read_text())
    old = json.loads((root/"results/general_ood_v98_regime_replay/replay_diverse_recipe/regime_replay.json").read_text())
    pilot = json.loads((root/"results/general_ood_v105_behavior_support/formal/behavior_support.json").read_text())
    actual = summary.summarize(new, old, pilot)
    assert actual["passed"] and actual["six_old_failures"]["expanded_standard"]["mode_accuracy"] > actual["six_old_failures"]["incumbent"]["mode_accuracy"]
    new["records"][-1]["attribution"][worker.v98.v94.CONDITIONS[0]]["last64"]["mean_true_prediction_error_rms"] += .01
    with pytest.raises(AssertionError):
        summary.summarize(new, old, pilot)
