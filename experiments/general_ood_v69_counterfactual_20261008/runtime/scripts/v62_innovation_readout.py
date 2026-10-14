"""Causal signed-residual innovation jointly calibrated with persistent level."""
from collections import deque

import numpy as np

from csbapr.belief.departure import NormalAgeReference, SETTINGS
from scripts import probe_general_ood_v43 as v43
from scripts.probe_general_ood_v56_reset_calibration import episode_ordinals
from scripts.v54_window_readout import COVARIANCE_RIDGE, WINDOW_STEPS, normalized_residuals, rolling_means
from scripts.v57_phase_readout import PhaseAgeReference, PhaseDepartureReadout

NAMES = ("level_recalibrated", "level_plus_innovation")


def window_features(residuals):
    moments = rolling_means(residuals)
    innovations = np.zeros_like(moments)
    start = 2 * WINDOW_STEPS - 1
    innovations[start:] = moments[start:] - moments[start-WINDOW_STEPS:-WINDOW_STEPS]
    return moments, innovations


class InnovationReference:
    @classmethod
    def fit_normal(cls, model, pooled, fitting, calibration):
        reference = cls()
        reference.pooled = pooled
        innovations = np.concatenate([window_features(normalized_residuals(model, t))[1] for t in fitting])
        dimension = innovations.shape[1]
        # Zero-centered second moment gives either direction the same departure score.
        reference.covariance = innovations.T @ innovations / len(innovations)
        reference.ridge = COVARIANCE_RIDGE * np.trace(reference.covariance) / dimension
        reference.precision = np.linalg.inv(reference.covariance + reference.ridge * np.eye(dimension))
        features = [window_features(normalized_residuals(model, t)) for t in calibration]
        scores = np.concatenate([reference.innovation_scores(i) for _, i in features])
        reference.score95 = float(np.quantile(scores, SETTINGS["quantile"]))
        assert reference.score95 > 0
        reference.fit_transitions = len(innovations)
        reference.calibration_transitions = len(scores)
        signals = [reference.signals(m, i) for m, i in features]
        ages = np.concatenate([v43.episode_steps(t["dones"]) for t in calibration])
        ordinals = np.concatenate([episode_ordinals(t["dones"]) for t in calibration])
        reference.phases = {}
        for name in NAMES:
            signal = np.concatenate([s[name] for s in signals])
            curves = []
            for mask in (ordinals == 0, ordinals > 0):
                assert mask.any(), "missing-normal-episode-phase"
                curve = NormalAgeReference.fit_normal(signal[mask], ages[mask])
                assert curve.ages[0] == 0 and curve.ages[-1] == 999, "incomplete-normal-phase-age-coverage"
                curves.append(curve)
            reference.phases[name] = PhaseAgeReference(*curves)
        return reference

    def innovation_scores(self, innovations):
        return np.einsum("ni,ij,nj->n", innovations, self.precision, innovations) / innovations.shape[1]

    def signals(self, moments, innovations):
        level = self.pooled.scores(moments) / self.pooled.score95
        innovation = self.innovation_scores(innovations) / self.score95
        return dict(level_recalibrated=np.log1p(level),
                    level_plus_innovation=np.log1p(np.maximum(level, innovation)),
                    level_ratio=level, innovation_ratio=innovation)

    def metadata(self):
        return dict(window_steps=WINDOW_STEPS, innovation_warmup=2*WINDOW_STEPS,
            innovation_center="zero", innovation_covariance=self.covariance.tolist(),
            covariance_ridge_relative=COVARIANCE_RIDGE, covariance_ridge_absolute=float(self.ridge),
            innovation_score95=self.score95, fit_transitions=self.fit_transitions,
            calibration_transitions=self.calibration_transitions,
            combination="log1p(max(level/level95, innovation/innovation95))",
            phases={k: v.metadata() for k, v in self.phases.items()},
            history_reset="independent trajectory only", coordinate_selection="all observation coordinates")


class OnlineInnovationReadout:
    def __init__(self, model, reference, name="level_plus_innovation"):
        assert name in NAMES
        self.model, self.reference, self.name = model, reference, name
        self.history = deque(maxlen=2*WINDOW_STEPS)
        self.departure = PhaseDepartureReadout(reference.phases[name])
        self.signal = self.level_ratio = self.innovation_ratio = 0.

    def observe(self, state, executed_action, next_state, episode_age):
        trace = dict(states=np.asarray([state]), executed_actions=np.asarray([executed_action]),
                     next_states=np.asarray([next_state]))
        self.history.append(normalized_residuals(self.model, trace)[0])
        history = np.asarray(self.history)
        moment = history[-WINDOW_STEPS:].mean(axis=0)
        innovation = moment-history[:WINDOW_STEPS].mean(axis=0) if len(history) == 2*WINDOW_STEPS else np.zeros_like(moment)
        values = self.reference.signals(moment[None], innovation[None])
        self.signal = float(values[self.name][0])
        self.level_ratio, self.innovation_ratio = float(values["level_ratio"][0]), float(values["innovation_ratio"][0])
        return self.departure.update(self.signal, episode_age)
