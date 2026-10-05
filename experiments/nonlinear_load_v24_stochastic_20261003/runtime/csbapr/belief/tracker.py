"""
BeliefTracker — BOCD-style Bayesian belief over run-lengths.

Directly copied from BAPR/bapr_components.py (proven convergent).
Corresponds to CSBAPR.lean Part IV (L346-386): Belief update (identical to BA-PR).
"""

import numpy as np


class BeliefTracker:
    """
    Bayesian Online Change Detection (Adams & MacKay 2007)
    Maintains run-length posterior ρ(h).
    Corresponds to BAPR.lean: update_belief / normalization_const
    """
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
        """
        BAPR.lean: update_belief ρ L Z
        1. Compute likelihood L(h, ξ)
        2. Reweight ρ(h) * L(h)
        3. Normalize
        4. Mix in changepoint prior (BOCD hazard shift)
        """
        L = self._compute_likelihood(surprise)
        unnorm = self.belief * L
        Z = unnorm.sum()
        if Z > 1e-10:
            self.belief = unnorm / Z
        else:
            self.belief = np.ones(self.max_H) / self.max_H

        growth_prob = self.belief * (1 - self.hazard)
        changepoint_prob = self.belief.sum() * self.hazard

        new_belief = np.zeros(self.max_H)
        new_belief[0] = changepoint_prob
        new_belief[1:] = growth_prob[:-1]
        total = new_belief.sum()
        self.belief = new_belief / total if total > 1e-10 else np.ones(self.max_H) / self.max_H

    def _compute_likelihood(self, surprise: float) -> np.ndarray:
        variances = self.base_var + self.var_growth * np.arange(self.max_H)
        return np.exp(-surprise**2 / (2 * variances))

    @property
    def effective_window(self) -> float:
        return np.sum(np.arange(self.max_H) * self.belief)

    @property
    def entropy(self) -> float:
        p = self.belief[self.belief > 1e-10]
        return -np.sum(p * np.log(p))
