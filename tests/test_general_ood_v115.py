from copy import deepcopy

import numpy as np
import pytest
import torch

from scripts import analyze_general_ood_v115_shared_residual_mean as worker
from test_general_ood_v114 import family, shared_start, trace, banks, sources, expanded, fitted, expanded_coverage, fitted_coverage


def test_reference_arrays_match_frozen_coordinates_and_balanced_targets(family):
    bank, groups = family.mean, family.mean_groups
    x, y, selected_groups = worker.reference_arrays(bank, groups, True)
    rows = worker.v113.selected_rows(groups, True)
    np.testing.assert_array_equal(x, bank.tree.data[rows].astype(np.float32))
    np.testing.assert_array_equal(y, np.concatenate([bank.residuals[m][rows] for m in worker.MODES], axis=1).astype(np.float32))
    model, metadata = worker.fit_mean(bank, groups, "cpu", smoke=True)
    assert model.layers[-1].out_features == 55
    assert metadata["weighted_mode_fractions"] == [.2]*5
    assert list(metadata["weighted_seed_fractions"].values()) == pytest.approx([1/len(np.unique(groups))]*len(np.unique(groups)))
    assert metadata["optimizer_updates"] == 2*((len(x)+511)//512)


def test_source_holdout_excluded_from_all_mean_training_and_selection(family):
    validation, model = worker.heldout_validation(family, "cpu", smoke=True)
    assert validation["validation_source_seeds"] == [135230]
    assert 135230 not in validation["training"]["source_seeds"]
    changed = deepcopy(family)
    for mode in worker.MODES:
        changed.mean.residuals[mode][changed.mean_groups == 135230] += 10
    other, other_model = worker.heldout_validation(changed, "cpu", smoke=True)
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, other_model.state_dict()[key], rtol=0, atol=0)
    assert validation["training"] == other["training"]
    assert validation["aggregate"] != other["aggregate"]
    assert all(validation["aggregate"][a] != other["aggregate"][a] for a in worker.ARMS)


def test_zero_correction_and_inference_use_only_state_action(family):
    bank = family.bank
    query = trace(135252, "compound", 5)
    model = worker.SharedResidualMean(len(bank.features.input_mean)).eval()
    inputs = bank.features.inputs(query)
    np.testing.assert_array_equal(worker.residual_prediction(model, bank, inputs), np.zeros((5, 5, 11)))
    torch.manual_seed(4)
    torch.nn.init.normal_(model.layers[-1].weight, std=.01)
    before = worker.residual_prediction(model, bank, inputs)
    changed = deepcopy(query); changed["next_states"] += 100
    np.testing.assert_array_equal(before, worker.residual_prediction(model, bank, bank.features.inputs(changed)))
    changed["actions"][0] += .5
    after = worker.residual_prediction(model, bank, bank.features.inputs(changed))
    np.testing.assert_array_equal(before[1:], after[1:])
    assert not np.allclose(before[0], after[0])


def test_prediction_metrics_retain_per_mode_failures_and_zero_gap_rows():
    targets = np.zeros((4, 5, 11))
    targets[:, :, 0] = np.arange(5)
    perfect = worker.prediction_statistics(targets, targets.copy())
    assert perfect["mean_prediction_mse"] == 0 and perfect["paired_nearest_mode_accuracy"] == 1
    predicted = targets.copy(); predicted[:, [0, 4]] = predicted[:, [4, 0]]
    swapped = worker.prediction_statistics(targets, predicted)
    assert swapped["paired_nearest_mode_accuracy"] == .6
    assert swapped["per_mode"]["normal"]["paired_nearest_mode_accuracy"] == 0
    assert swapped["per_mode"]["compound"]["paired_nearest_mode_accuracy"] == 0
    targets[:, 1] = targets[:, 0]
    tied = worker.prediction_statistics(targets, targets.copy())
    assert tied["zero_oracle_gap_fraction"] == .4
    assert tied["error_exceeds_nearest_oracle_gap_fraction"] == 0
