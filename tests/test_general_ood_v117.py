from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import analyze_general_ood_v117_temporal_classifier_replay as worker
from scripts import v73_recursive_mode_readout as temporal
from test_general_ood_v114 import frozen_classifier
from test_general_ood_v115 import family, shared_start, trace, banks, sources, expanded, fitted, expanded_coverage, fitted_coverage


def null_population():
    nulls = [dict(states=np.zeros((2000, 11)), dones=np.arange(2000) % 1000 == 999) for _ in range(2)]
    values = [{a: np.tile([4., 0., 0., 0., 0.], (2000, 1))*(i+1) for i, a in enumerate(worker.ARMS)} for _ in nulls]
    return nulls, values


def test_transition_scores_use_real_data_not_oracles_and_preserve_prefix(family):
    query = trace(135252, "compound", 20)
    mean = worker.v115.SharedResidualMean(len(family.bank.features.input_mean)).eval()
    torch.manual_seed(117)
    torch.nn.init.normal_(mean.layers[-1].weight, std=.01)
    classifiers = {a: frozen_classifier() for a in worker.ARMS[:2]}
    before = worker.transition_scores(family.bank, mean, classifiers, query)
    changed = deepcopy(query)
    changed["oracle_observations"] = np.full((20, 5, 11), 1e8)
    other = worker.transition_scores(family.bank, mean, classifiers, changed)
    for arm in worker.ARMS:
        np.testing.assert_array_equal(before[arm], other[arm])
    changed["next_states"][10:] += 3
    other = worker.transition_scores(family.bank, mean, classifiers, changed)
    for arm in worker.ARMS:
        np.testing.assert_array_equal(before[arm][:10], other[arm][:10])
        assert not np.allclose(before[arm][10:], other[arm][10:])


def test_normal_only_calibration_matches_existing_algorithm_and_rejects_protocol_change():
    nulls, values = null_population()
    refs = worker.calibrate_scores(nulls, values)
    for arm in worker.ARMS:
        bank = SimpleNamespace(log_likelihoods=lambda t: next(v[arm] for n, v in zip(nulls, values) if t is n))
        direct = temporal.RecursiveModeReference.calibrate(bank, nulls)
        assert refs[arm].phase.metadata() == direct.phase.metadata()
    metadata = worker.reference_metadata(refs)
    incumbent = deepcopy(metadata[worker.ARMS[0]])
    worker.verify_temporal_protocol(metadata, incumbent)
    incumbent["history_reset"] = "reset every episode"
    with pytest.raises(AssertionError):
        worker.verify_temporal_protocol(metadata, incumbent)
    assert all(m["normal_null_calibration_transitions"] == 4000 for m in metadata.values())
    assert all("empirical" in m["mode_weights"] for m in metadata.values())


def test_future_cannot_change_history_and_posterior_persists_across_reset():
    nulls, values = null_population()
    ref = worker.calibrate_scores(nulls, values)[worker.ARMS[0]]
    scores = values[0][worker.ARMS[0]].copy()
    scores[980:1050] = [0., 0., 0., 0., 4.]
    before = deepcopy(ref.phase.metadata())
    series = temporal.observe_likelihoods(scores, nulls[0]["dones"], ref, "recursive")
    altered = scores.copy(); altered[1100:] = [0., 8., 0., 0., 0.]
    other = temporal.observe_likelihoods(altered, nulls[0]["dones"], ref, "recursive")
    for field in series:
        np.testing.assert_array_equal(series[field][:1100], other[field][:1100])
    reset_scores = temporal.temporal_scores(scores[1000:1001], "recursive")
    assert series["mode_scores"][1000, 4] > reset_scores[0, 4]
    assert before == ref.phase.metadata()


def test_all_smoke_failures_have_normal_controls_and_incumbent_population_not_filtered():
    for seed, mode, updating in worker.SMOKE_KEYS:
        assert (seed, "normal", updating) in worker.SMOKE_KEYS
    assert len(worker.SMOKE_KEYS) == 10
    assert (135190, "compound", True) in worker.SMOKE_KEYS
    assert (135252, "compound", True) in worker.SMOKE_KEYS
    assert (135192, "high_grav", True) in worker.SMOKE_KEYS
    windows = {w: dict(mode_accuracy=.5, mean_coefficient=.1, mean_true_minus_best_other_score=1.,
                      above_threshold_fraction=.2) for w in ("first_post50", "last64", "first800", "full", "last1000")}
    normal = dict(group="test", seed=1, updating=True, mode="normal", arms={a: deepcopy(windows) for a in worker.ARMS},
                  incumbent={worker.PRIMARY: deepcopy(windows)},
                  normal_episodes={a: [dict(mean_risk=.3)] for a in worker.ARMS})
    changed = dict(normal, mode="compound", normal_episodes=None)
    changed["arms"] = deepcopy(normal["arms"])
    changed["arms"][worker.ARMS[0]]["first_post50"]["mode_accuracy"] = 1.
    summary = worker.summarize([normal, changed])["all"]
    assert summary[worker.ARMS[0]]["early_accuracy"] == 1.
    assert summary["incumbent_v98"]["early_accuracy"] == .5
    assert summary[worker.ARMS[0]]["worst_normal_episode_mean_risk"] == .3
    assert summary[worker.ARMS[0]]["normal_full_above_threshold_fraction"] == .2
