"""Frozen normal dynamics and an actor-only penalty intervention."""
from copy import deepcopy

import numpy as np

from csbapr.agent import CSBAPRAgent
from csbapr.belief.departure import DepartureReadout
from csbapr.belief.surprise import RelativeSurpriseComputer
from scripts import probe_general_ood_v42 as v42
from scripts import probe_general_ood_v43 as v43
from scripts import run_general_ood_v38 as v38
from scripts.v44_belief import SETTINGS
from scripts.v49_belief import NoiseLikelihoodStudent

VARIANTS = ("fixed-minus2", "fixed-minus2p3", "noise-age", "age-departure")


class NormalDynamics(v43.ReferenceDynamics):
    def fit_references(self, fitting, calibration):
        x = self.inputs(fitting)
        y = np.asarray(fitting["next_states"], dtype=np.float64) - fitting["states"]
        self.input_mean = x.mean(axis=0)
        self.input_scale = np.maximum(x.std(axis=0), v42.SCALE_FLOOR)
        self.target_scale = np.maximum(y.std(axis=0), v42.SCALE_FLOOR)
        design = self.features(x)
        penalty = len(x) * v42.RIDGE * np.eye(design.shape[1])
        penalty[0, 0] = 0.
        self.coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ (y / self.target_scale))
        errors = self.errors(calibration)
        # Stochastic Hopper episodes can be short; only the used calibration is fitted.
        self.calibrations = {"global-centered": v43.ResidualCalibration(
            "global-centered", errors, calibration["episode_steps"], self.target_scale)}
        self.fit_mse = float(np.mean((self.errors(fitting) / self.target_scale) ** 2))
        self.calibration_mse = float(np.mean((errors / self.target_scale) ** 2))
        return self


class TransitionReadouts:
    def __init__(self, model, reference, startup_noise):
        self.model, self.reference, self.startup_noise = model, reference, startup_noise
        self.departure = DepartureReadout(reference)
        self.age = NoiseLikelihoodStudent(**SETTINGS)
        self.signal = self.age_coefficient = 0.

    def observe(self, state, executed_action, next_state, episode_age):
        trace = dict(states=np.asarray([state]), executed_actions=np.asarray([executed_action]),
                     next_states=np.asarray([next_state]))
        scores, thresholds = self.model.calibrations["global-centered"].scores(
            self.model.errors(trace), np.asarray([episode_age]))
        self.signal = float(np.log1p(scores[0] / thresholds[0]))
        self.departure.update(self.signal, episode_age)
        self.age.update(self.signal, self.startup_noise if episode_age == 0 else 0.)
        self.age_coefficient = float(self.age.belief @ np.exp(-.1 * np.arange(self.age.max_H)))


class ContinuationAgent(CSBAPRAgent):
    def __init__(self, env, variant, device="cuda"):
        config = v38.make_config("Hopper-v4", "fixed-penalty", device)
        config.penalty_scale = 1.
        config.state_dim, config.action_dim = env.observation_space.shape[0], env.action_space.shape[0]
        super().__init__(config.state_dim, config.action_dim, config)
        self.surprise_computer = RelativeSurpriseComputer(ema_alpha=config.surprise_ema)
        self.variant, self.readouts = variant, None

    @property
    def penalty_coefficient(self):
        if self.variant == "fixed-minus2":
            return 0.
        if self.variant == "fixed-minus2p3":
            return .3
        if self.variant == "noise-age":
            return self.readouts.age_coefficient
        return self.readouts.departure.risk

    def _compute_q_loss_bapr_fallback(self, *args, **kwargs):
        result = super()._compute_q_loss_bapr_fallback(*args, **kwargs)
        # This return slot only enters beta in the inherited actor loss.
        return (*result[:3], self.penalty_coefficient, result[4])

    def start_shared(self, checkpoint, shared):
        self.load(str(checkpoint))
        self.replay_buffer = deepcopy(shared["replay"])
        self.restore_context(deepcopy(shared["context"]))
        self.readouts = TransitionReadouts(shared["model"], shared["reference"], shared["startup_noise"])

