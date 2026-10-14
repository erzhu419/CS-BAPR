"""Fixed v78 means with local reference leave-one-out error second moments."""
import numpy as np
from sklearn.covariance import oas
from sklearn.linear_model import Ridge

from scripts.probe_general_ood_v42 import RIDGE
from scripts.v54_window_readout import COVARIANCE_RIDGE
from scripts.v65_state_action_readout import NEIGHBORS
from scripts.v77_mean_covariance import MODES, density_terms

CONDITION = "linear_mean_loo_error_cov"
ALPHA = NEIGHBORS*RIDGE


def fit_reference(inputs, targets, query):
    offsets = inputs-query
    model = Ridge(alpha=ALPHA, fit_intercept=True, solver="cholesky").fit(offsets, targets)
    centered = offsets-offsets.mean(axis=0)
    gram = centered.T@centered+ALPHA*np.eye(centered.shape[1])
    hat = 1/NEIGHBORS+np.einsum("ij,ji->i", centered, np.linalg.solve(gram, centered.T))
    loo_errors = (targets-model.predict(offsets))/(1-hat[:, None])
    displacement = offsets.mean(axis=0)
    leverage = 1/NEIGHBORS+displacement@np.linalg.solve(gram, displacement)
    return model.intercept_, loo_errors, float(leverage)


def error_covariance(errors):
    covariance, shrinkage = oas(errors, assume_centered=True)
    dimension = errors.shape[1]
    return covariance+COVARIANCE_RIDGE*np.trace(covariance)/dimension*np.eye(dimension), float(shrinkage)


class PredictionErrorDensityBank:
    def __init__(self, bank):
        self.bank = bank
        self.paired_residuals = np.concatenate([bank.residuals[m] for m in MODES], axis=1)

    def geometry(self, inputs):
        bank = self.bank
        normalized = (inputs-bank.features.input_mean)/bank.features.input_scale
        _, indices = bank.tree.query(normalized, k=NEIGHBORS)
        centers, covariances, shrinkages, errors, leverages = [], [], [], [], []
        for query, neighbors in zip(normalized, indices):
            center, loo, leverage = fit_reference(bank.tree.data[neighbors], self.paired_residuals[neighbors], query)
            per_mode = [error_covariance(values) for values in np.split(loo, len(MODES), axis=1)]
            centers.append(center)
            covariances.append([c for c, _ in per_mode])
            shrinkages.append([s for _, s in per_mode])
            errors.append([np.sqrt(np.mean(values**2)) for values in np.split(loo, len(MODES), axis=1)])
            leverages.append(leverage)
        return (dict(zip(MODES, np.split(np.asarray(centers), len(MODES), axis=1))),
            dict(zip(MODES, np.asarray(covariances).transpose(1, 0, 2, 3))),
            dict(shrinkage=np.asarray(shrinkages), reference_loo_rms=np.asarray(errors),
                 query_leverage=np.asarray(leverages)))

    def terms(self, trace):
        bank = self.bank
        centers, covariances, diagnostic = self.geometry(bank.features.inputs(trace))
        targets, design = bank.targets(trace), bank.design(trace)
        result = {k: [] for k in ("energy", "logdet", "likelihoods", "mean_error_rms")}
        for mode in MODES:
            errors = targets-design@bank.coefficients[mode]-centers[mode]
            energy, logdet, likelihoods = density_terms(errors, covariances[mode], bank.target_scale)
            for key, values in (("energy", energy), ("logdet", logdet), ("likelihoods", likelihoods),
                                ("mean_error_rms", np.sqrt(np.mean(errors**2, axis=1)))):
                result[key].append(values)
        diagnostic["covariance_trace"] = np.column_stack([np.trace(covariances[m], axis1=1, axis2=2) for m in MODES])
        return {k: np.column_stack(v) for k, v in result.items()}, diagnostic

    def log_likelihoods(self, trace):
        return self.terms(trace)[0]["likelihoods"]

    def metadata(self):
        return dict(modes=list(MODES), conditional_mean="unchanged v78 local linear ridge residual prediction",
            conditional_mean_source="updating_mixed", neighbors=NEIGHBORS, ridge_alpha=ALPHA,
            intercept_penalized=False, covariance_source="same updating-bank reference neighbors",
            covariance_estimator="OAS second moment of local reference leave-one-out prediction errors",
            error_second_moment_assume_centered=True, covariance_ridge_relative=COVARIANCE_RIDGE,
            covariance_formula="OAS(LOO errors, assume_centered=True) + 0.001*trace(OAS)/11*I",
            leave_one_out="fixed 64-neighbor set and alpha; hold each reference row out of the local residual fit",
            query_residual_used_for_geometry=False, query_leverage_inflation=False,
            conditional_mean_changed=False, conditional_covariance_changed=True,
            forward_refitted=False, scaling_refitted=False,
            likelihood_units="physical delta-state Gaussian density including log determinant")
