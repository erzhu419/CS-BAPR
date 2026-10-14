"""Startup-noise moment approximations; not wired into RL training."""
from collections import deque

import numpy as np

from scripts.v44_belief import student_log_density
from scripts.v46_belief import RobustAdaptiveStudent


class NoiseLikelihoodStudent(RobustAdaptiveStudent):
    """Add known observation uncertainty to the df4 predictive variance."""
    def __init__(self, **settings):
        super().__init__(**settings)
        self.observation_variance = 0.

    def reset(self):
        super().reset()
        self.observation_variance = 0.

    def _compute_log_likelihood(self, surprise):
        location, scale2, degrees = self.predictive_parameters()
        return student_log_density(surprise, location, scale2 + self.observation_variance / 2, degrees)

    def update(self, surprise, observation_variance=0.):
        self.observation_variance = float(observation_variance)
        super().update(surprise)


class NoiseWeightedStudent(NoiseLikelihoodStudent):
    """Fractional-history NIG moments plus startup observation uncertainty.

    Precision-weighted sufficient statistics and df4 moment matching are an
    approximation, not an exact heteroscedastic conjugate posterior or BOCD.
    Run length and hazard still count physical transitions, not precision mass.
    """
    def __init__(self, **settings):
        super().__init__(**settings)
        self.precisions = deque(maxlen=self.max_H - 1)

    def reset(self):
        super().reset()
        self.precisions.clear()

    def predictive_parameters(self):
        past = np.asarray(list(self.history)[::-1])
        weights = np.asarray(list(self.precisions)[::-1])
        count = np.minimum(np.arange(self.max_H), len(past))
        mass = np.r_[0., np.cumsum(weights)][count]
        sums = np.r_[0., np.cumsum(weights * past)][count]
        squares = np.r_[0., np.cumsum(weights * past ** 2)][count]
        sample_mean = sums / np.where(mass > 0, mass, 1.)
        kappa, alpha = 1. + mass, 2. + mass / 2
        beta = (self.base_var / 2 + .5 * np.maximum(squares - sums * sample_mean, 0.)
                + .5 * mass / kappa * sample_mean ** 2)
        scale2, degrees = beta * (kappa + 1) / (alpha * kappa), 2 * alpha
        variance = scale2 * degrees / (degrees - 2)
        return sums / kappa, variance / 2, np.full_like(degrees, 4.)

    def update(self, surprise, observation_variance=0.):
        # Likelihood uses only preceding samples; append precision with its value.
        super().update(surprise, observation_variance)
        self.precisions.append(self.base_var / (self.base_var + observation_variance))
