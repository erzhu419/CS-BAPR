"""A known-mode dynamics bank; no online mode labels or actor feedback."""
from collections import deque

import numpy as np
from scipy.special import logsumexp
from sklearn.covariance import oas

from csbapr.belief.departure import NormalAgeReference, SETTINGS
from scripts import probe_general_ood_v43 as v43
from scripts.probe_general_ood_v42 import DynamicsResidual, RIDGE, SCALE_FLOOR
from scripts.probe_general_ood_v53_behavior_coverage import MODES
from scripts.probe_general_ood_v56_reset_calibration import episode_ordinals
from scripts.v54_window_readout import COVARIANCE_RIDGE, WINDOW_STEPS, rolling_means
from scripts.v57_phase_readout import PhaseAgeReference, PhaseDepartureReadout

NAME = "known_mode_likelihood"


class ModeLikelihoodBank:
    @classmethod
    def fit(cls, fitting, density_calibration):
        assert tuple(fitting) == tuple(density_calibration) == MODES
        bank = cls()
        bank.features = DynamicsResidual(2, True)
        inputs = np.concatenate([bank.features.inputs(fitting[m]) for m in MODES])
        targets = np.concatenate([fitting[m]["next_states"]-fitting[m]["states"] for m in MODES])
        bank.features.input_mean = inputs.mean(axis=0)
        bank.features.input_scale = np.maximum(inputs.std(axis=0), SCALE_FLOOR)
        bank.target_scale = np.maximum(targets.std(axis=0), SCALE_FLOOR)
        bank.coefficients, bank.centers, bank.precisions, bank.log_determinants, bank.statistics = {}, {}, {}, {}, {}
        for mode in MODES:
            trace = fitting[mode]
            design = bank.design(trace)
            target = bank.targets(trace)
            penalty = len(target)*RIDGE*np.eye(design.shape[1])
            penalty[0, 0] = 0.
            bank.coefficients[mode] = np.linalg.solve(design.T@design+penalty, design.T@target)
            errors = bank.targets(density_calibration[mode])-bank.design(density_calibration[mode])@bank.coefficients[mode]
            bank.centers[mode] = errors.mean(axis=0)
            covariance, shrinkage = oas(errors)
            ridge = COVARIANCE_RIDGE*np.trace(covariance)/len(bank.target_scale)
            covariance += ridge*np.eye(len(bank.target_scale))
            sign, logdet = np.linalg.slogdet(covariance)
            assert sign == 1 and np.isfinite(logdet)
            bank.precisions[mode] = np.linalg.inv(covariance)
            # All densities are in the same physical delta-state coordinates.
            bank.log_determinants[mode] = float(logdet+2*np.log(bank.target_scale).sum())
            bank.statistics[mode] = dict(fit_transitions=len(target), density_calibration_transitions=len(errors),
                fit_mse_in_common_target_units=float(np.mean((target-design@bank.coefficients[mode])**2)),
                density_calibration_mse_in_common_target_units=float(np.mean(errors**2)),
                oas_shrinkage=float(shrinkage), covariance_ridge_absolute=float(ridge),
                log_determinant_physical_units=bank.log_determinants[mode])
        return bank

    def design(self, trace):
        return self.features.features(self.features.inputs(trace))

    def targets(self, trace):
        return (np.asarray(trace["next_states"], dtype=float)-trace["states"])/self.target_scale

    def log_likelihoods(self, trace):
        design, targets = self.design(trace), self.targets(trace)
        constant = len(self.target_scale)*np.log(2*np.pi)
        output = []
        for mode in MODES:
            errors = targets-design@self.coefficients[mode]-self.centers[mode]
            energy = np.einsum("ni,ij,nj->n", errors, self.precisions[mode], errors)
            output.append(-.5*(energy+self.log_determinants[mode]+constant))
        return np.column_stack(output)

    def metadata(self):
        return dict(modes=list(MODES), degree=2, use_executed_action=True,
            input_dimension=len(self.features.input_mean), feature_count=len(self.coefficients["normal"]),
            ridge_mean_loss_coefficient=RIDGE, scaling="pooled fitting inputs and target deltas; shared by every mode",
            residual_distribution="mode-specific independent density-calibration mean and OAS covariance",
            covariance_ridge_relative=COVARIANCE_RIDGE, likelihood_units="physical delta-state Gaussian density including log determinant",
            statistics=self.statistics, matrix_storage="server reference checkpoint only")


def mode_evidence(mean_log_likelihoods):
    weights = np.exp(mean_log_likelihoods-logsumexp(mean_log_likelihoods, axis=-1, keepdims=True))
    log_bayes_factor = logsumexp(mean_log_likelihoods[..., 1:], axis=-1)-np.log(len(MODES)-1)-mean_log_likelihoods[..., 0]
    return weights, log_bayes_factor, np.logaddexp(0., log_bayes_factor)


class ModeLikelihoodReference:
    @classmethod
    def fit(cls, fitting, density_calibration, normal_null_calibration):
        reference = cls()
        reference.bank = ModeLikelihoodBank.fit(fitting, density_calibration)
        signals = np.concatenate([mode_evidence(rolling_means(reference.bank.log_likelihoods(t)))[2]
                                  for t in normal_null_calibration])
        ages = np.concatenate([v43.episode_steps(t["dones"]) for t in normal_null_calibration])
        phases = np.concatenate([episode_ordinals(t["dones"]) > 0 for t in normal_null_calibration])
        curves = [NormalAgeReference.fit_normal(signals[phases == p], ages[phases == p]) for p in (False, True)]
        assert all(c.ages[0] == 0 and c.ages[-1] == 999 for c in curves), "incomplete-normal-phase-age-coverage"
        reference.phase = PhaseAgeReference(*curves)
        reference.normal_null_calibration_transitions = len(signals)
        return reference

    def metadata(self):
        return dict(bank=self.bank.metadata(), window_steps=WINDOW_STEPS,
            mode_weights="uniform-prior softmax of 16-step MEAN log densities; tempered diagnostic weights, not sequential belief",
            alternative="equal mixture of four known changed modes versus normal",
            departure_signal="softplus(log alternative-to-normal likelihood ratio)",
            normal_null_calibration_transitions=self.normal_null_calibration_transitions,
            settings=SETTINGS, phase=self.phase.metadata(), history_reset="independent trajectory only",
            mode_labels_used_online=False, switch_step_used_online=False, actor_feedback=False)


def observe_series(trace, reference):
    mean_likelihoods = rolling_means(reference.bank.log_likelihoods(trace))
    weights, log_bayes_factor, signals = mode_evidence(mean_likelihoods)
    sensor = PhaseDepartureReadout(reference.phase)
    risks, thresholds = [], []
    for age, signal in zip(v43.episode_steps(trace["dones"]), signals):
        risks.append(sensor.update(signal, int(age)))
        thresholds.append(sensor.departure.threshold)
    return dict(mean_log_likelihoods=mean_likelihoods, mode_weights=weights, log_bayes_factor=log_bayes_factor,
        signal=signals, risk=np.asarray(risks), threshold=np.asarray(thresholds))


class OnlineModeLikelihoodReadout:
    def __init__(self, reference):
        self.reference = reference
        self.history = deque(maxlen=WINDOW_STEPS)
        self.departure = PhaseDepartureReadout(reference.phase)

    def observe(self, state, executed_action, next_state, episode_age):
        trace = dict(states=np.asarray([state]), executed_actions=np.asarray([executed_action]), next_states=np.asarray([next_state]))
        self.history.append(self.reference.bank.log_likelihoods(trace)[0])
        self.mode_weights, self.log_bayes_factor, self.signal = mode_evidence(np.mean(self.history, axis=0))
        return self.departure.update(float(self.signal), episode_age)
