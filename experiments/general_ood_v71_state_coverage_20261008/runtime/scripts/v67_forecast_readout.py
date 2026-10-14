"""Normal-only causal residual forecasts added to the fixed v66 level channel."""
from collections import deque

import numpy as np
from sklearn.covariance import oas

from csbapr.belief.departure import NormalAgeReference, SETTINGS
from scripts import probe_general_ood_v43 as v43
from scripts.probe_general_ood_v42 import RIDGE, SCALE_FLOOR
from scripts.probe_general_ood_v56_reset_calibration import episode_ordinals
from scripts.v54_window_readout import COVARIANCE_RIDGE, WINDOW_STEPS, normalized_residuals, rolling_means
from scripts.v57_phase_readout import PhaseAgeReference, PhaseDepartureReadout
from scripts.v66_covariance_mix_readout import NAME as LEVEL_NAME

NAMES = ("forecast_only", "level_plus_forecast")
MAIN = "level_plus_forecast"


def forecast_inputs(model,trace,residuals):
    past = np.zeros_like(residuals)
    past[1:] = rolling_means(residuals)[:-1]
    history_fraction = np.minimum(np.arange(len(residuals)),WINDOW_STEPS)/WINDOW_STEPS
    return np.column_stack((model.inputs(trace),past,history_fraction))


class NormalResidualForecast:
    @classmethod
    def fit_normal(cls,model,traces):
        forecast = cls()
        pairs = [(normalized_residuals(model,t),t) for t in traces]
        inputs = np.concatenate([forecast_inputs(model,t,r) for r,t in pairs])
        targets = np.concatenate([r for r,_ in pairs])
        forecast.input_mean = inputs.mean(axis=0)
        forecast.input_scale = np.maximum(inputs.std(axis=0),SCALE_FLOOR)
        forecast.target_scale = np.maximum(targets.std(axis=0),SCALE_FLOOR)
        design = forecast.design(inputs)
        penalty = len(inputs)*RIDGE*np.eye(design.shape[1])
        penalty[0,0] = 0.
        forecast.coefficients = np.linalg.solve(design.T@design+penalty,design.T@(targets/forecast.target_scale))
        forecast.fit_transitions = len(inputs)
        forecast.fit_mse = float(np.mean((targets-forecast.predict(inputs))**2))
        forecast.fit_constant_mse = float(np.mean((targets-targets.mean(axis=0))**2))
        return forecast

    def design(self,inputs):
        return np.column_stack((np.ones(len(inputs)),(inputs-self.input_mean)/self.input_scale))

    def predict(self,inputs):
        return self.design(inputs)@self.coefficients*self.target_scale

    def errors(self,model,trace):
        residuals = normalized_residuals(model,trace)
        return residuals-self.predict(forecast_inputs(model,trace,residuals))

    def metadata(self):
        return dict(input_dimension=len(self.input_mean),feature_count=len(self.coefficients),degree=1,
            ridge_mean_loss_coefficient=RIDGE,fit_transitions=self.fit_transitions,
            fit_mse=self.fit_mse,fit_constant_mse=self.fit_constant_mse,
            inputs="current state/executed action, previous 16 residual mean, prior history fraction",
            current_residual_used_as_prediction_input=False,history_reset="independent trajectory only")


class ForecastReference:
    @classmethod
    def fit_normal(cls,model,level,fitting,calibration):
        reference = cls()
        reference.level = level
        reference.forecast = NormalResidualForecast.fit_normal(model,fitting)
        moments = np.concatenate([rolling_means(reference.forecast.errors(model,t)) for t in fitting])
        reference.center = moments.mean(axis=0)
        covariance,reference.shrinkage = oas(moments)
        reference.covariance_trace = float(np.trace(covariance))
        reference.ridge = COVARIANCE_RIDGE*reference.covariance_trace/moments.shape[1]
        reference.precision = np.linalg.inv(covariance+reference.ridge*np.eye(moments.shape[1]))
        reference.fit_transitions = len(moments)
        errors = [reference.forecast.errors(model,t) for t in calibration]
        moments = np.concatenate([rolling_means(e) for e in errors])
        scores = reference.scores(moments)
        reference.score95 = float(np.quantile(scores,SETTINGS["quantile"]))
        assert reference.score95>0
        ages = np.concatenate([v43.episode_steps(t["dones"]) for t in calibration])
        phases = np.concatenate([episode_ordinals(t["dones"])>0 for t in calibration])
        signals = np.log1p(scores/reference.score95)
        curves = [NormalAgeReference.fit_normal(signals[phases==p],ages[phases==p]) for p in (False,True)]
        assert all(c.ages[0]==0 and c.ages[-1]==999 for c in curves)
        reference.phase = PhaseAgeReference(*curves)
        reference.calibration_transitions = len(scores)
        reference.calibration_forecast_mse = float(np.mean(np.concatenate(errors)**2))
        return reference

    def scores(self,moments):
        centered = moments-self.center
        return np.einsum("ni,ij,nj->n",centered,self.precision,centered)/moments.shape[1]

    def signals(self,moments):
        return np.log1p(self.scores(moments)/self.score95)

    def metadata(self):
        return dict(window_steps=WINDOW_STEPS,forecast=self.forecast.metadata(),
            coordinate_dimension=len(self.center),fit_transitions=self.fit_transitions,
            calibration_transitions=self.calibration_transitions,calibration_forecast_mse=self.calibration_forecast_mse,
            covariance_estimator="normal-fit OAS forecast-window covariance",oas_shrinkage=float(self.shrinkage),
            covariance_trace=self.covariance_trace,covariance_ridge_relative=COVARIANCE_RIDGE,
            covariance_ridge_absolute=self.ridge,score95=self.score95,settings=SETTINGS,phase=self.phase.metadata(),
            combination="maximum of separately calibrated v66 and forecast 16-step mean risks",
            inherited_level_refitted=False,current_residual_used_as_prediction_input=False,
            history_reset="independent trajectory only",matrix_storage="server reference checkpoint only")


def observe_series(trace,model,reference):
    residuals = normalized_residuals(model,trace)
    errors = residuals-reference.forecast.predict(forecast_inputs(model,trace,residuals))
    forecast_signal = reference.signals(rolling_means(errors))
    level_signals,_ = reference.level.evaluate(rolling_means(residuals),rolling_means(model.inputs(trace)))
    level_signal = level_signals[LEVEL_NAME]
    level_sensor = PhaseDepartureReadout(reference.level.phases[LEVEL_NAME])
    forecast_sensor = PhaseDepartureReadout(reference.phase)
    ages = v43.episode_steps(trace["dones"])
    level_risk,forecast_risk,thresholds = [],[],[]
    for age,a,b in zip(ages,level_signal,forecast_signal):
        level_risk.append(level_sensor.update(a,int(age)))
        forecast_risk.append(forecast_sensor.update(b,int(age)))
        thresholds.append(forecast_sensor.departure.threshold)
    level_risk,forecast_risk = np.asarray(level_risk),np.asarray(forecast_risk)
    return dict(level_signal=level_signal,forecast_signal=forecast_signal,level_risk=level_risk,
        forecast_risk=forecast_risk,combined_risk=np.maximum(level_risk,forecast_risk),
        forecast_threshold=np.asarray(thresholds))


class OnlineForecastReadout:
    def __init__(self,model,reference,name=MAIN):
        assert name in NAMES
        self.model,self.reference,self.name = model,reference,name
        self.history,self.input_history,self.error_history = (deque(maxlen=WINDOW_STEPS) for _ in range(3))
        self.level_sensor = PhaseDepartureReadout(reference.level.phases[LEVEL_NAME])
        self.forecast_sensor = PhaseDepartureReadout(reference.phase)
        self.level_signal=self.forecast_signal=self.level_risk=self.forecast_risk=0.

    def observe(self,state,executed_action,next_state,episode_age):
        trace = dict(states=np.asarray([state]),executed_actions=np.asarray([executed_action]),next_states=np.asarray([next_state]))
        inputs = self.model.inputs(trace)[0]
        past = np.mean(self.history,axis=0) if self.history else np.zeros_like(state,dtype=float)
        features = np.r_[inputs,past,len(self.history)/WINDOW_STEPS]
        self.prediction = self.reference.forecast.predict(features[None])[0]
        residual = normalized_residuals(self.model,trace)[0]
        self.error_history.append(residual-self.prediction)
        self.history.append(residual)
        self.input_history.append(inputs)
        self.forecast_signal = float(self.reference.signals(np.mean(self.error_history,axis=0)[None])[0])
        signals,_ = self.reference.level.evaluate(np.mean(self.history,axis=0)[None],np.mean(self.input_history,axis=0)[None])
        self.level_signal = float(signals[LEVEL_NAME][0])
        self.level_risk = self.level_sensor.update(self.level_signal,episode_age)
        self.forecast_risk = self.forecast_sensor.update(self.forecast_signal,episode_age)
        return self.forecast_risk if self.name=="forecast_only" else max(self.level_risk,self.forecast_risk)
