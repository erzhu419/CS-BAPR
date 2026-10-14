"""Normal-calibrated multivariate signed residual windows."""
from collections import deque

import numpy as np

from csbapr.belief.departure import DepartureReadout, NormalAgeReference, SETTINGS
from scripts import probe_general_ood_v43 as v43

WINDOW_STEPS = 16
COVARIANCE_RIDGE = 1e-3


def normalized_residuals(model, trace):
    calibration = model.calibrations["global-centered"]
    return (model.errors(trace) - calibration.means[0]) / calibration.scales[0]


def rolling_means(residuals):
    return np.asarray([residuals[max(0, i - WINDOW_STEPS + 1):i + 1].mean(axis=0)
                       for i in range(len(residuals))])


class SignedWindowReference:
    @classmethod
    def fit_normal(cls, model, fitting, calibration):
        reference = cls()
        moments = np.concatenate([rolling_means(normalized_residuals(model, trace)) for trace in fitting])
        reference.center = moments.mean(axis=0)
        centered = moments - reference.center
        reference.covariance = centered.T @ centered / len(centered)
        dimension = len(reference.center)
        reference.ridge = COVARIANCE_RIDGE * np.trace(reference.covariance) / dimension
        reference.precision = np.linalg.inv(reference.covariance + reference.ridge * np.eye(dimension))
        cal_moments = np.concatenate([rolling_means(normalized_residuals(model, trace)) for trace in calibration])
        cal_ages = np.concatenate([v43.episode_steps(trace["dones"]) for trace in calibration])
        scores = reference.scores(cal_moments)
        reference.score95 = float(np.quantile(scores, SETTINGS["quantile"]))
        reference.age_reference = NormalAgeReference.fit_normal(np.log1p(scores / reference.score95), cal_ages)
        reference.fit_transitions, reference.calibration_transitions = len(moments), len(cal_moments)
        return reference

    def scores(self, moments):
        centered = moments - self.center
        return np.einsum("ni,ij,nj->n", centered, self.precision, centered) / len(self.center)

    def metadata(self):
        return dict(window_steps=WINDOW_STEPS, covariance_ridge_relative=COVARIANCE_RIDGE,
            covariance_ridge_absolute=float(self.ridge), center=self.center.tolist(), covariance=self.covariance.tolist(),
            global_score95=self.score95, age_reference=self.age_reference.metadata(),
            fit_transitions=self.fit_transitions, calibration_transitions=self.calibration_transitions,
            coordinate_selection="all observation coordinates", history_reset="independent trajectory only")


class SignedWindowReadout:
    def __init__(self, model, reference):
        self.model, self.reference = model, reference
        self.history = deque(maxlen=WINDOW_STEPS)
        self.departure = DepartureReadout(reference.age_reference)
        self.signal = self.score = 0.

    def observe(self, state, executed_action, next_state, episode_age):
        trace = dict(states=np.asarray([state]), executed_actions=np.asarray([executed_action]),
                     next_states=np.asarray([next_state]))
        self.history.append(normalized_residuals(self.model, trace)[0])
        self.score = float(self.reference.scores(np.asarray([np.mean(self.history, axis=0)]))[0])
        self.signal = float(np.log1p(self.score / self.reference.score95))
        return self.departure.update(self.signal, episode_age)


def observe_trace(trace, model, reference, switch_step):
    ages = v43.episode_steps(trace["dones"])
    scores = reference.scores(rolling_means(normalized_residuals(model, trace)))
    signals = np.log1p(scores / reference.score95)
    departure = DepartureReadout(reference.age_reference)
    coefficients = np.asarray([departure.update(signal, age) for signal, age in zip(signals, ages)])
    windows = dict(full=(0, len(ages)), prefix=(0, switch_step), first_post50=(switch_step, switch_step + 50),
                   post=(switch_step, len(ages)), last64=(len(ages) - 64, len(ages)))
    return {name: dict(mean_coefficient=float(coefficients[start:end].mean()),
        mean_signal=float(signals[start:end].mean())) for name, (start, end) in windows.items()}
