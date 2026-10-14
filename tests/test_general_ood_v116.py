from copy import deepcopy

import numpy as np
import pytest
import torch

from scripts import analyze_general_ood_v116_matched_neural_classifier as worker
from test_general_ood_v114 import frozen_classifier
from test_general_ood_v115 import family, shared_start, trace, banks, sources, expanded, fitted, expanded_coverage, fitted_coverage


def test_crossfit_excludes_whole_source_and_reuses_only_matching_cache(family, tmp_path, monkeypatch):
    bank, groups = family.incumbent.mean, family.groups
    rows, errors, folds = worker.crossfit_errors(bank, groups, "cpu", True, tmp_path)
    assert errors.shape == (len(rows), 5, 11)
    for fold in folds:
        assert fold["seed"] not in fold["training_source_seeds"]
        assert 135230 not in fold["training_source_seeds"]
        assert fold["heldout_rows"] == np.sum(groups[rows] == fold["seed"])
    seed = folds[0]["seed"]
    from scripts.v83_cross_seed_covariance import subset_bank
    reduced = subset_bank(bank, groups != seed)
    model, _ = worker.v115.fit_mean(reduced, groups[groups != seed], "cpu", True)
    selected = groups[rows] == seed
    inputs = bank.tree.data[rows[selected]]*bank.features.input_scale+bank.features.input_mean
    direct = np.stack([bank.residuals[m][rows[selected]] for m in worker.MODES], axis=1)
    direct -= worker.v115.residual_prediction(model, bank, inputs)
    np.testing.assert_array_equal(errors[selected], direct)
    monkeypatch.setattr(worker.v115, "fit_mean", lambda *a, **kw: pytest.fail("cached folds must not retrain"))
    other_rows, other, cached = worker.crossfit_errors(bank, groups, "cpu", True, tmp_path)
    np.testing.assert_array_equal(rows, other_rows)
    np.testing.assert_array_equal(errors, other)
    assert all(f["reused_cache"] for f in cached)
    saved = torch.load(tmp_path/f"errors_seed{seed}.pt", weights_only=False)
    saved["smoke"] = False
    torch.save(saved, tmp_path/f"errors_seed{seed}.pt")
    with pytest.raises(AssertionError):
        worker.crossfit_errors(bank, groups, "cpu", True, tmp_path)


def test_outer_classifier_is_unchanged_by_validation_source_targets(family, tmp_path):
    matched, metadata, folds = worker.fit_reference_classifier(family.incumbent.mean, family.groups, "cpu", True, tmp_path)
    before = deepcopy(matched)
    model, _ = worker.v115.fit_mean(family.incumbent.mean, family.groups, "cpu", True)
    validation = worker.validation_scores(family, model, matched, frozen_classifier(), True)
    changed = deepcopy(family)
    for mode in worker.MODES:
        changed.mean.residuals[mode][changed.mean_groups == 135230] += 10
    other = worker.validation_scores(changed, model, matched, frozen_classifier(), True)
    assert metadata["source_seeds"] == np.unique(family.groups).tolist()
    assert not set(metadata["source_seeds"]) & set(validation["validation_source_seeds"])
    assert validation["aggregate"] != other["aggregate"]
    for obj, previous, fields in ((matched[0], before[0], ("mean_", "scale_")),
                                  (matched[1], before[1], ("coef_", "intercept_"))):
        for field in fields:
            np.testing.assert_array_equal(getattr(obj, field), getattr(previous, field))


def test_load_requires_formal_frozen_mean_states(tmp_path):
    model = worker.v115.SharedResidualMean(14)
    path = tmp_path/"means.pt"
    saved = dict(revision=worker.v115.REVISION, smoke=False, settings=worker.v115.SETTINGS,
        validation_model=model.state_dict(), final_model=model.state_dict(),
        validation_training=dict(epochs=100), final_training=dict(epochs=100))
    torch.save(saved, path)
    models, _ = worker.load_frozen_means(path, 14)
    for name, loaded in models.items():
        assert not loaded.training
        for key, value in loaded.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
    saved["smoke"] = True
    torch.save(saved, path)
    with pytest.raises(AssertionError):
        worker.load_frozen_means(path, 14)


def test_replacing_classifier_preserves_nearest_mean_and_real_transition_dependence():
    rng = np.random.RandomState(116)
    observed, predicted = rng.normal(size=(7, 11)), rng.normal(size=(7, 5, 11))
    original, matched = frozen_classifier(), frozen_classifier()
    matched[1].coef_[0] *= -1
    scores = worker.score_arms(matched, original, observed, predicted)
    expected = worker.v113.scores(original, observed, predicted)
    np.testing.assert_array_equal(scores["original_classifier"], expected["quadratic_logistic"])
    np.testing.assert_array_equal(scores["nearest_mean"], expected["nearest_mean"])
    assert not np.allclose(scores["matched_classifier"], scores["original_classifier"])
    altered = observed.copy(); altered[3:] += 3
    changed = worker.score_arms(matched, original, altered, predicted)
    for arm in worker.ARMS:
        np.testing.assert_array_equal(scores[arm][:3], changed[arm][:3])
        assert not np.allclose(scores[arm][3:], changed[arm][3:])
