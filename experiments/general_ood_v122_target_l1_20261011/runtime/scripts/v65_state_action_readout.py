"""Normal signed residual geometry conditioned on causal state/action windows."""
from collections import deque

import numpy as np
from scipy.spatial import cKDTree
from sklearn.covariance import oas

from csbapr.belief.departure import NormalAgeReference, SETTINGS
from scripts import probe_general_ood_v43 as v43
from scripts.probe_general_ood_v42 import SCALE_FLOOR
from scripts.probe_general_ood_v56_reset_calibration import episode_ordinals
from scripts.v54_window_readout import COVARIANCE_RIDGE, WINDOW_STEPS, normalized_residuals, rolling_means
from scripts.v57_phase_readout import PhaseAgeReference, PhaseDepartureReadout

NEIGHBORS = 64
NAMES = ("state_action_center", "state_action_center_covariance")


class StateActionReference:
    @classmethod
    def fit_normal(cls, model, pooled, fitting, calibration):
        assert model.use_action, "state-action-reference-requires-executed-action"
        reference = cls()
        reference.global_precision = pooled.precision.copy()
        inputs, moments = [], []
        for trace in fitting:
            selected = np.arange(WINDOW_STEPS-1, len(trace["dones"]), WINDOW_STEPS)
            inputs.append(rolling_means(model.inputs(trace))[selected])
            moments.append(rolling_means(normalized_residuals(model, trace))[selected])
        inputs = np.concatenate(inputs)
        reference.moments = np.concatenate(moments)
        reference.input_mean = inputs.mean(axis=0)
        reference.input_scale = np.maximum(inputs.std(axis=0), SCALE_FLOOR)
        assert len(inputs) >= NEIGHBORS, "insufficient-normal-reference-windows"
        reference.tree = cKDTree((inputs-reference.input_mean)/reference.input_scale)
        reference.fit_transitions = sum(len(t["dones"]) for t in fitting)
        values = [(rolling_means(normalized_residuals(model,t)), rolling_means(model.inputs(t)),
                   v43.episode_steps(t["dones"]), episode_ordinals(t["dones"])>0) for t in calibration]
        moments, inputs, ages, phases = (np.concatenate([v[i] for v in values]) for i in range(4))
        scores, diagnostic = reference.scores(moments, inputs)
        reference.score95 = {name:float(np.quantile(v, SETTINGS["quantile"])) for name,v in scores.items()}
        assert all(v>0 for v in reference.score95.values())
        reference.phases = {}
        for name in NAMES:
            signals = np.log1p(scores[name]/reference.score95[name])
            curves = [NormalAgeReference.fit_normal(signals[phases==phase], ages[phases==phase])
                      for phase in (False,True)]
            assert all(c.ages[0]==0 and c.ages[-1]==999 for c in curves)
            reference.phases[name] = PhaseAgeReference(*curves)
        reference.calibration_transitions = len(moments)
        reference.neighbor_radius95 = float(np.quantile(diagnostic["neighbor_radius"], SETTINGS["quantile"]))
        reference.calibration_geometry = dict(mean_shrinkage=float(diagnostic["shrinkage"].mean()),
            mean_covariance_trace=float(diagnostic["covariance_trace"].mean()))
        return reference

    def geometry(self, inputs):
        distances, indices = self.tree.query((inputs-self.input_mean)/self.input_scale, k=NEIGHBORS)
        values = self.moments[indices]
        centers = values.mean(axis=1)
        dimension = self.moments.shape[1]
        precisions, shrinkage, traces = [], [], []
        for neighbors in values:
            # OAS estimates shrinkage from saved normal neighbors, never from the query residual.
            covariance, weight = oas(neighbors)
            trace = np.trace(covariance)
            precisions.append(np.linalg.inv(covariance+COVARIANCE_RIDGE*trace/dimension*np.eye(dimension)))
            shrinkage.append(weight)
            traces.append(trace)
        return centers, np.asarray(precisions), dict(neighbor_radius=distances[:,-1],
            shrinkage=np.asarray(shrinkage), covariance_trace=np.asarray(traces))

    def scores(self, moments, inputs):
        centers, precisions, diagnostic = self.geometry(inputs)
        offsets = moments-centers
        dimension = moments.shape[1]
        scores = dict(state_action_center=np.einsum("ni,ij,nj->n",offsets,self.global_precision,offsets)/dimension,
            state_action_center_covariance=np.einsum("ni,nij,nj->n",offsets,precisions,offsets)/dimension)
        return scores, diagnostic

    def evaluate(self, moments, inputs):
        scores, diagnostic = self.scores(moments,inputs)
        return {name:np.log1p(v/self.score95[name]) for name,v in scores.items()}, diagnostic

    def metadata(self):
        return dict(window_steps=WINDOW_STEPS,neighbors=NEIGHBORS,normal_reference_windows=len(self.moments),
            coordinate_dimension=self.moments.shape[1],input_dimension=self.tree.m,
            fit_transitions=self.fit_transitions,calibration_transitions=self.calibration_transitions,
            conditioning="past/current 16-transition mean of state and executed action; normal-fit scaling",
            fit_window_sampling="disjoint full windows; no episode-reset history clearing",
            covariance_estimator="sklearn.covariance.oas on saved normal neighbor residuals",
            covariance_ridge_relative=COVARIANCE_RIDGE,settings=SETTINGS,score95=self.score95,
            neighbor_radius95=self.neighbor_radius95,calibration_geometry=self.calibration_geometry,
            departure_curves={name:phase.metadata() for name,phase in self.phases.items()},
            neighbor_distance_controls_risk=False,history_reset="independent trajectory only",
            reference_storage="server checkpoint only")


class OnlineStateActionReadout:
    def __init__(self, model, reference, name="state_action_center_covariance"):
        assert name in NAMES
        self.model,self.reference,self.name = model,reference,name
        self.history,self.input_history = deque(maxlen=WINDOW_STEPS),deque(maxlen=WINDOW_STEPS)
        self.departure = PhaseDepartureReadout(reference.phases[name])
        self.signal = 0.

    def observe(self, state, executed_action, next_state, episode_age):
        trace = dict(states=np.asarray([state]),executed_actions=np.asarray([executed_action]),next_states=np.asarray([next_state]))
        self.history.append(normalized_residuals(self.model,trace)[0])
        self.input_history.append(self.model.inputs(trace)[0])
        signals,_ = self.reference.evaluate(np.mean(self.history,axis=0)[None],np.mean(self.input_history,axis=0)[None])
        self.signal = float(signals[self.name][0])
        return self.departure.update(self.signal,episode_age)
