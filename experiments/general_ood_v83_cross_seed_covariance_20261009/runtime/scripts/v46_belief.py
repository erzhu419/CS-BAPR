"""Variance-matched heavy-tail candidate; not wired into RL training."""
import numpy as np

from scripts.v44_belief import AdaptiveStudent


class RobustAdaptiveStudent(AdaptiveStudent):
    """Learn recent location/variance but retain Student-t df=4 tails.

    This is a moment-matched predictive approximation, not the conjugate
    Gaussian posterior predictive distribution used by the v44 control.
    """
    def predictive_parameters(self):
        location, scale2, degrees = super().predictive_parameters()
        variance = scale2 * degrees / (degrees - 2)
        return location, variance / 2, np.full_like(degrees, 4.)
