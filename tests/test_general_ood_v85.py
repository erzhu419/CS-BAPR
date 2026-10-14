from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import logsumexp

from scripts import analyze_general_ood_v85_classification_weight as worker


def folds():
    rng = np.random.default_rng(85)
    result = []
    for _ in range(3):
        errors = rng.normal(size=(8, 5, 5, 2))
        a, b = (rng.normal(size=(8, 5, 2, 2)) for _ in range(2))
        cross = a@a.transpose(0, 1, 3, 2)+.1*np.eye(2)
        raw = b@b.transpose(0, 1, 3, 2)+.1*np.eye(2)
        result.append((errors, cross, raw))
    return result


def test_objective_is_equal_seed_mode_classification_not_true_density():
    values = folds()
    expected = []
    for offsets, cross, raw in values:
        ll = worker.v84.scores(offsets, .3*raw+.7*cross)[0]
        expected.append(np.mean(logsumexp(ll, axis=-1)-ll[:, np.arange(5), np.arange(5)]))
    assert worker.classification_nll(.3, values) == pytest.approx(np.mean(expected))
    assert worker.classification_nll(.3, list(reversed(values))) == pytest.approx(np.mean(expected))
    own_nll = np.mean([-worker.v84.scores(e, .3*r+.7*c)[0][:, np.arange(5), np.arange(5)].mean() for e, c, r in values])
    assert worker.classification_nll(.3, values) != pytest.approx(own_nll)


def test_bounded_candidate_competes_with_endpoints_and_frozen_density_weight():
    selected = worker.select_weight(folds(), .24619454443573643)
    assert 0. <= selected['raw_weight'] <= 1.
    assert selected['classification_nll'] == min(c['classification_nll'] for c in selected['candidates'])
    assert {c['name'] for c in selected['candidates']} == {'pure_cross_seed', 'raw', 'v83_density_selected', 'bounded'}
    assert not selected['independent_confirmation']


def test_old_weight_is_retained_if_numerical_candidate_is_worse(monkeypatch):
    monkeypatch.setattr(worker, 'classification_nll', lambda w, _: (w-.25)**2)
    monkeypatch.setattr(worker, 'minimize_scalar', lambda fun, **kw:
        SimpleNamespace(success=True, message='', x=.8, fun=fun(.8), nfev=1))
    selected = worker.select_weight([], .25)
    assert selected['chosen_candidate'] == 'v83_density_selected'
    assert selected['raw_weight'] == .25 and selected['classification_nll'] == 0.


def test_compact_keeps_probability_diagnostics_without_repeating_pair_attribution():
    offsets, cross, _ = folds()[0]
    stats = worker.v84.summarize(*worker.v84.scores(offsets, cross))
    compact = worker.compact(stats)
    assert compact['classification_nll'] == stats['classification_nll']
    for mode in worker.v84.MODES:
        assert 'pairs' not in compact['per_mode'][mode]
        assert compact['per_mode'][mode]['predicted_mode_fractions'] == stats['per_mode'][mode]['predicted_mode_fractions']
