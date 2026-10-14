"""Temporal mode inference around the unchanged v72 conditional density."""
import numpy as np
from scipy.special import logsumexp

from scripts import v68_mode_likelihood as original

MODES = original.MODES
SWITCH_HAZARD = .02
NAMES = ("window16", "recursive")


class RecursiveModeFilter:
    def __init__(self):
        self.log_posterior = np.full(len(MODES), -np.log(len(MODES)))

    def update(self, log_likelihood):
        # Uniform redraw allows recovery without an online change-time label.
        prior = np.logaddexp(np.log1p(-SWITCH_HAZARD)+self.log_posterior,
                             np.log(SWITCH_HAZARD/len(MODES)))
        row = np.asarray(log_likelihood, dtype=float)
        scores = prior+row-row.max()
        self.log_posterior = scores-logsumexp(scores)
        return self.log_posterior.copy()


def temporal_scores(log_likelihoods, name):
    if name == "window16":
        return original.rolling_means(log_likelihoods)
    assert name == "recursive"
    sensor = RecursiveModeFilter()
    return np.asarray([sensor.update(row) for row in log_likelihoods])


class RecursiveModeReference:
    @classmethod
    def calibrate(cls, bank, nulls):
        reference = cls()
        reference.bank = bank
        signals = np.concatenate([original.mode_evidence(temporal_scores(bank.log_likelihoods(t), "recursive"))[2]
                                  for t in nulls])
        ages = np.concatenate([original.v43.episode_steps(t["dones"]) for t in nulls])
        phases = np.concatenate([original.episode_ordinals(t["dones"]) > 0 for t in nulls])
        curves = [original.NormalAgeReference.fit_normal(signals[phases == p], ages[phases == p])
                  for p in (False, True)]
        assert all(c.ages[0] == 0 and c.ages[-1] == 999 for c in curves)
        reference.phase = original.PhaseAgeReference(*curves)
        reference.normal_null_calibration_transitions = len(signals)
        return reference

    def metadata(self):
        return dict(bank=self.bank.metadata(), window_steps=None, switch_hazard=SWITCH_HAZARD,
            transition_prior="(1-h)*previous posterior + h/5; uniform redraw including same mode",
            actual_change_probability=SWITCH_HAZARD*(len(MODES)-1)/len(MODES),
            mode_weights="recursive finite-mode posterior from untempered transition log densities",
            initial_prior="uniform", history_reset="independent trajectory only; persists across episode reset",
            departure_signal="softplus(log(mean changed posterior mass / normal posterior mass)); not a Bayes factor",
            normal_null_calibration_transitions=self.normal_null_calibration_transitions,
            settings=original.SETTINGS, phase=self.phase.metadata(),
            mode_labels_used_online=False, switch_step_used_online=False, actor_feedback=False)


def observe_likelihoods(log_likelihoods, dones, reference, name):
    scores = temporal_scores(log_likelihoods, name)
    weights, contrast, signals = original.mode_evidence(scores)
    sensor = original.PhaseDepartureReadout(reference.phase)
    risks, thresholds = [], []
    for age, signal in zip(original.v43.episode_steps(dones), signals):
        risks.append(sensor.update(signal, int(age)))
        thresholds.append(sensor.departure.threshold)
    return dict(mode_scores=scores, mode_weights=weights, log_alternative_contrast=contrast,
                signal=signals, risk=np.asarray(risks), threshold=np.asarray(thresholds))


class OnlineRecursiveReadout:
    def __init__(self, reference):
        self.reference = reference
        self.filter = RecursiveModeFilter()
        self.departure = original.PhaseDepartureReadout(reference.phase)

    def observe(self, state, executed_action, next_state, episode_age):
        trace = dict(states=np.asarray([state]), executed_actions=np.asarray([executed_action]),
                     next_states=np.asarray([next_state]))
        self.mode_scores = self.filter.update(self.reference.bank.log_likelihoods(trace)[0])
        self.mode_weights, self.log_alternative_contrast, self.signal = original.mode_evidence(self.mode_scores)
        return self.departure.update(float(self.signal), episode_age)
