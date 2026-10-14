"""Age references conditioned on an observed initial versus later episode."""
import numpy as np

from csbapr.belief.departure import DepartureReadout, NormalAgeReference
from scripts import probe_general_ood_v55_learning_calibration as v55
from scripts.probe_general_ood_v56_reset_calibration import episode_ordinals


class PhaseAgeReference:
    def __init__(self, initial, after_reset):
        self.initial, self.after_reset = initial, after_reset

    def metadata(self):
        return dict(initial=self.initial.metadata(), after_reset=self.after_reset.metadata(),
                    selection="observed prior episode boundary only")


def fit_references(model, original, traces):
    pooled = v55.calibrate_learning(model, original, traces)
    values = [(*v55.signals_and_ages(model, original, trace), episode_ordinals(trace["dones"])) for trace in traces]
    signals, ages, ordinals = (np.concatenate([v[i] for v in values]) for i in range(3))
    curves = []
    for mask in (ordinals == 0, ordinals > 0):
        assert mask.any(), "missing-normal-episode-phase"
        curve = NormalAgeReference.fit_normal(signals[mask], ages[mask])
        assert curve.ages[0] == 0 and curve.ages[-1] == 999, "incomplete-normal-phase-age-coverage"
        curves.append(curve)
    return pooled, PhaseAgeReference(*curves)


class PhaseDepartureReadout:
    def __init__(self, reference):
        self.reference = reference
        self.departure = DepartureReadout(reference.initial)
        self.seen_transition = self.after_reset = False

    def update(self, signal, episode_age):
        if self.seen_transition and episode_age == 0:
            self.after_reset = True
        self.seen_transition = True
        self.departure.reference = self.reference.after_reset if self.after_reset else self.reference.initial
        return self.departure.update(signal, episode_age)


def observe_trace(trace, model, pooled, phase, switch_step):
    signals, ages = v55.signals_and_ages(model, pooled, trace)
    departure = PhaseDepartureReadout(phase)
    coefficients = np.asarray([departure.update(signal, age) for signal, age in zip(signals, ages)])
    windows = dict(full=(0, len(ages)), prefix=(0, switch_step), first_post50=(switch_step, switch_step + 50),
                   post=(switch_step, len(ages)), last64=(len(ages) - 64, len(ages)))
    return {name: dict(mean_coefficient=float(coefficients[start:end].mean()),
        mean_signal=float(signals[start:end].mean())) for name, (start, end) in windows.items()}
