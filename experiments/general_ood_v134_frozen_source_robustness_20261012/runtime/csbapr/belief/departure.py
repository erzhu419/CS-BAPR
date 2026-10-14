"""Normal-calibrated departure readout, separate from change-point run length."""
from collections import deque

import numpy as np

SETTINGS = dict(knot_interval=16, age_radius=8, quantile=.95, history_steps=16)


class NormalAgeReference:
    def __init__(self, ages, upper, counts, normal_start_count):
        self.ages = np.asarray(ages)
        self.upper = np.asarray(upper)
        self.counts = list(counts)
        self.normal_start_count = int(normal_start_count)

    @classmethod
    def fit_normal(cls, signal, episode_ages):
        signal, episode_ages = np.asarray(signal), np.asarray(episode_ages)
        maximum = int(episode_ages.max())
        knots = np.unique(np.r_[np.arange(0, maximum + 1, SETTINGS["knot_interval"]), maximum])
        upper, counts = [], []
        for age in knots:
            values = signal[np.abs(episode_ages - age) <= SETTINGS["age_radius"]]
            assert len(values), "empty-normal-age-neighborhood"
            upper.append(float(np.quantile(values, SETTINGS["quantile"])))
            counts.append(len(values))
        return cls(knots, upper, counts, np.sum(episode_ages == 0))

    @classmethod
    def global95(cls):
        return cls([0], [np.log(2.)], [0], 0)

    def threshold(self, episode_age):
        return np.interp(episode_age, self.ages, self.upper)

    def metadata(self):
        return dict(ages=self.ages.tolist(), upper_signal=self.upper.tolist(), counts=self.counts,
                    normal_start_count=self.normal_start_count, settings=SETTINGS)


class DepartureReadout:
    def __init__(self, reference):
        self.reference = reference
        self.history = deque(maxlen=SETTINGS["history_steps"])
        self.risk = self.instant_risk = self.threshold = 0.

    def update(self, signal, episode_age):
        self.threshold = float(self.reference.threshold(episode_age))
        excess = max(float(signal) - self.threshold, 0.)
        self.instant_risk = float(-np.expm1(-excess))
        self.history.append(self.instant_risk)
        self.risk = float(np.mean(self.history))
        return self.risk

    @property
    def beta(self):
        return -2. - self.risk
