"""Normal signed-vector geometry conditioned on observed phase and episode age."""
from collections import deque

import numpy as np

from csbapr.belief.departure import NormalAgeReference, SETTINGS
from scripts import probe_general_ood_v43 as v43
from scripts.probe_general_ood_v56_reset_calibration import episode_ordinals
from scripts.v54_window_readout import COVARIANCE_RIDGE, WINDOW_STEPS, normalized_residuals, rolling_means
from scripts.v57_phase_readout import PhaseAgeReference, PhaseDepartureReadout

NAMES = ("phase_center", "phase_center_covariance")


class VectorAgeCurve:
    def __init__(self, ages, centers, precisions, counts, ridges):
        self.ages, self.centers, self.precisions = np.asarray(ages), np.asarray(centers), np.asarray(precisions)
        self.counts, self.ridges = list(counts), np.asarray(ridges)

    @classmethod
    def fit_normal(cls, moments, ages):
        assert ages.min() == 0 and ages.max() == 999, "incomplete-normal-phase-age-coverage"
        knots = np.unique(np.r_[np.arange(0,1000,SETTINGS["knot_interval"]),999])
        centers, precisions, counts, ridges = [], [], [], []
        dimension = moments.shape[1]
        for age in knots:
            values = moments[np.abs(ages-age)<=SETTINGS["age_radius"]]
            assert len(values), "empty-normal-vector-age-neighborhood"
            center = values.mean(axis=0)
            centered = values-center
            covariance = centered.T@centered/len(values)
            ridge = COVARIANCE_RIDGE*np.trace(covariance)/dimension
            centers.append(center)
            precisions.append(np.linalg.inv(covariance+ridge*np.eye(dimension)))
            counts.append(len(values))
            ridges.append(ridge)
        return cls(knots,centers,precisions,counts,ridges)

    def scores(self, moments, ages, global_precision):
        right = np.clip(np.searchsorted(self.ages,ages,side="right"),1,len(self.ages)-1)
        left = right-1
        weight = (ages-self.ages[left])/(self.ages[right]-self.ages[left])
        center = (1-weight[:,None])*self.centers[left]+weight[:,None]*self.centers[right]
        offset = moments-center
        dimension = moments.shape[1]
        center_score = np.einsum("ni,ij,nj->n",offset,global_precision,offset)/dimension
        # Interpolate positive-definite precision matrices without fitting on the current transition.
        a = np.einsum("ni,nij,nj->n",offset,self.precisions[left],offset)
        b = np.einsum("ni,nij,nj->n",offset,self.precisions[right],offset)
        return dict(phase_center=center_score,phase_center_covariance=((1-weight)*a+weight*b)/dimension)

    def metadata(self):
        return dict(ages=self.ages.tolist(),normal_fit_counts=self.counts,
                    ridge_min=float(self.ridges.min()),ridge_max=float(self.ridges.max()))


class PhaseVectorReference:
    @classmethod
    def fit_normal(cls, model, pooled, fitting, calibration):
        reference = cls()
        reference.global_precision = pooled.precision.copy()
        fit = [(rolling_means(normalized_residuals(model,t)),v43.episode_steps(t["dones"]),
                episode_ordinals(t["dones"])>0) for t in fitting]
        moments, ages, after_reset = (np.concatenate([t[i] for t in fit]) for i in range(3))
        reference.curves = {}
        for phase in (False,True):
            mask = after_reset == phase
            assert mask.any(), "missing-normal-episode-phase"
            reference.curves[phase] = VectorAgeCurve.fit_normal(moments[mask],ages[mask])
        reference.fit_transitions = len(moments)
        cal = [(rolling_means(normalized_residuals(model,t)),v43.episode_steps(t["dones"]),
                episode_ordinals(t["dones"])>0) for t in calibration]
        moments, ages, after_reset = (np.concatenate([t[i] for t in cal]) for i in range(3))
        scores = reference.scores(moments,ages,after_reset)
        reference.score95 = {name:float(np.quantile(values,SETTINGS["quantile"])) for name,values in scores.items()}
        assert all(v>0 for v in reference.score95.values())
        reference.phases = {}
        for name in NAMES:
            signals = np.log1p(scores[name]/reference.score95[name])
            curves = [NormalAgeReference.fit_normal(signals[after_reset==phase],ages[after_reset==phase])
                      for phase in (False,True)]
            reference.phases[name] = PhaseAgeReference(*curves)
        reference.calibration_transitions = len(moments)
        return reference

    def scores(self, moments, ages, after_reset):
        output = {name:np.empty(len(moments)) for name in NAMES}
        for phase in (False,True):
            mask = after_reset == phase
            scores = self.curves[phase].scores(moments[mask],ages[mask],self.global_precision)
            for name in NAMES:
                output[name][mask] = scores[name]
        return output

    def signals(self, moments, ages, after_reset):
        return {name:np.log1p(score/self.score95[name]) for name,score in self.scores(moments,ages,after_reset).items()}

    def metadata(self):
        return dict(window_steps=WINDOW_STEPS,coordinate_dimension=self.global_precision.shape[0],
            fit_transitions=self.fit_transitions,calibration_transitions=self.calibration_transitions,
            settings=SETTINGS,covariance_ridge_relative=COVARIANCE_RIDGE,score95=self.score95,
            vector_curves={"after_reset" if k else "initial":v.metadata() for k,v in self.curves.items()},
            departure_curves={k:v.metadata() for k,v in self.phases.items()},
            interpolation="linear center and positive-definite precision",history_reset="independent trajectory only",
            phase_selection="observed prior episode boundary only",matrix_storage="server reference checkpoint only")


class OnlinePhaseVectorReadout:
    def __init__(self, model, reference, name="phase_center_covariance"):
        assert name in NAMES
        self.model, self.reference, self.name = model, reference, name
        self.history = deque(maxlen=WINDOW_STEPS)
        self.departure = PhaseDepartureReadout(reference.phases[name])
        self.signal = 0.

    def observe(self, state, executed_action, next_state, episode_age):
        trace = dict(states=np.asarray([state]),executed_actions=np.asarray([executed_action]),next_states=np.asarray([next_state]))
        self.history.append(normalized_residuals(self.model,trace)[0])
        phase = self.departure.after_reset or (self.departure.seen_transition and episode_age == 0)
        moment = np.mean(self.history,axis=0)[None]
        self.signal = float(self.reference.signals(moment,np.asarray([episode_age]),np.asarray([phase]))[self.name][0])
        return self.departure.update(self.signal,episode_age)
