"""Capped run-length filter for a nonnegative surprise observation.

A reset occurs before the current observation. Its predictive variance is
broader than a mature run's, so a surprise spike favors short run lengths.
This is a fixed half-normal observation model, not full BOCD parameter inference.
"""

import numpy as np


class BeliefTracker:
    """Maintain run-length probabilities; the last bin includes longer runs."""
    def __init__(self, max_run_length: int = 20, hazard_rate: float = 0.05,
                 base_variance: float = 0.1, variance_growth: float = 0.05):
        self.max_H = max_run_length
        self.hazard = hazard_rate
        self.base_var = base_variance
        self.var_growth = variance_growth
        self.belief = np.ones(max_run_length) / max_run_length

    def reset(self):
        self.belief = np.ones(self.max_H) / self.max_H

    def update(self, surprise: float):
        growth = (1 - self.hazard) * self.belief
        prior = np.zeros(self.max_H)
        prior[0] = self.hazard * self.belief.sum()
        prior[1:] = growth[:-1]
        # Keep mature-run mass at the cap instead of dropping it each update.
        prior[-1] += growth[-1]
        with np.errstate(divide="ignore"):
            log_weights = np.log(prior) + self._compute_log_likelihood(surprise)
        weights = np.exp(log_weights - np.max(log_weights))
        self.belief = weights / weights.sum()

    def _compute_log_likelihood(self, surprise: float) -> np.ndarray:
        variances = self.base_var / (1 + self.var_growth * np.arange(self.max_H))
        return 0.5 * np.log(2 / (np.pi * variances)) - surprise**2 / (2 * variances)

    @property
    def effective_window(self) -> float:
        return np.sum(np.arange(self.max_H) * self.belief)

    @property
    def entropy(self) -> float:
        p = self.belief[self.belief > 1e-10]
        return -np.sum(p * np.log(p))
