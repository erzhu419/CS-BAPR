"""Experimental capped predictive filters; not used by the training agent."""
from collections import deque
from math import lgamma

import numpy as np

from csbapr.belief.tracker import BeliefTracker

SETTINGS = dict(max_run_length=20, hazard_rate=.02, base_variance=1., variance_growth=.05)
FILTERS = ("half-normal-fixed", "student-fixed", "student-adaptive")


def student_log_density(value, location, scale_squared, degrees):
    normalization = np.asarray([lgamma((d + 1) / 2) - lgamma(d / 2) for d in degrees])
    return (normalization - .5 * np.log(degrees * np.pi * scale_squared)
            - (degrees + 1) / 2 * np.log1p((value - location) ** 2 / (degrees * scale_squared)))


class FixedStudent(BeliefTracker):
    def _compute_log_likelihood(self, surprise):
        variance = self.base_var / (1 + self.var_growth * np.arange(self.max_H))
        return student_log_density(surprise, 0., variance / 2, np.full(self.max_H, 4.))


class AdaptiveStudent(BeliefTracker):
    """Normal-inverse-gamma predictive likelihood with at most H-1 past samples.

    The capped bin pools mature paths, using their common recent H-1 samples.
    This finite-memory approximation is not unbounded BOCD parameter inference.
    """
    def __init__(self, **settings):
        super().__init__(**settings)
        self.history = deque(maxlen=self.max_H - 1)

    def reset(self):
        super().reset()
        self.history.clear()

    def predictive_parameters(self):
        past = np.asarray(list(self.history)[::-1])
        n = np.minimum(np.arange(self.max_H), len(past))
        sums = np.r_[0., np.cumsum(past)][n]
        squares = np.r_[0., np.cumsum(past ** 2)][n]
        sample_mean = sums / np.maximum(n, 1)
        kappa, alpha = 1. + n, 2. + n / 2
        # Prior mu=0, kappa=1, alpha=2, beta=base_var/2: fresh variance=base_var.
        beta = (self.base_var / 2 + .5 * np.maximum(squares - sums * sample_mean, 0.)
                + .5 * n / kappa * sample_mean ** 2)
        return sums / kappa, beta * (kappa + 1) / (alpha * kappa), 2 * alpha

    def _compute_log_likelihood(self, surprise):
        return student_log_density(surprise, *self.predictive_parameters())

    def update(self, surprise):
        super().update(surprise)
        # Current observation must not enter its own predictive likelihood.
        self.history.append(float(surprise))


def make_filters():
    return dict(zip(FILTERS, (BeliefTracker(**SETTINGS), FixedStudent(**SETTINGS), AdaptiveStudent(**SETTINGS))))


def replay(scores, thresholds):
    signal = np.log1p(scores / thresholds)
    result = {}
    for name, tracker in make_filters().items():
        rows = []
        for value in signal:
            tracker.update(float(value))
            penalty = float(tracker.belief @ np.exp(-.1 * np.arange(tracker.max_H)))
            rows.append((tracker.belief[0], tracker.effective_window, penalty, -2. - penalty, tracker.entropy))
        result[name] = np.asarray(rows)
    return signal, result
