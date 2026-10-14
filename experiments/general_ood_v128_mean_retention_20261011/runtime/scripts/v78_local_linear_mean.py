"""Local linear mean correction; saved conditional covariances stay unchanged."""
import numpy as np
from sklearn.linear_model import Ridge

from scripts.probe_general_ood_v42 import RIDGE
from scripts.v65_state_action_readout import NEIGHBORS
from scripts.v77_mean_covariance import MODES, density_terms

CONDITIONS = {
    "frozen_anchor": ("frozen_constant", "frozen"),
    "constant_mean_frozen_cov": ("updating_constant", "frozen"),
    "linear_mean_frozen_cov": ("updating_linear", "frozen"),
    "constant_mean_updating_cov": ("updating_constant", "updating"),
    "linear_mean_updating_cov": ("updating_linear", "updating"),
}
CONTROLS = {"frozen_anchor": "frozen_mean_frozen_cov",
            "constant_mean_frozen_cov": "updating_mean_frozen_cov",
            "constant_mean_updating_cov": "updating_mean_updating_cov"}


class LocalLinearMean:
    def __init__(self, bank):
        self.bank = bank
        self.paired_residuals = np.concatenate([bank.residuals[m] for m in MODES], axis=1)

    def predict(self, inputs):
        bank = self.bank
        normalized = (inputs-bank.features.input_mean)/bank.features.input_scale
        _, indices = bank.tree.query(normalized, k=NEIGHBORS)
        centers = []
        for query, neighbors in zip(normalized, indices):
            offsets = bank.tree.data[neighbors]-query
            model = Ridge(alpha=NEIGHBORS*RIDGE, fit_intercept=True, solver="cholesky")
            model.fit(offsets, self.paired_residuals[neighbors])
            centers.append(model.intercept_)
        return dict(zip(MODES, np.split(np.asarray(centers), len(MODES), axis=1)))


def correction_terms(trace, banks, correction):
    bank = banks["frozen"]
    inputs, design, targets = bank.features.inputs(trace), bank.design(trace), bank.targets(trace)
    linear = correction.predict(inputs)
    output = {c: {k: [] for k in ("energy", "logdet", "likelihoods", "mean_error_rms")}
              for c in CONDITIONS}
    for mode in MODES:
        geometry = {s: b.geometry(inputs, mode)[:2] for s, b in banks.items()}
        centers = {"frozen_constant": geometry["frozen"][0],
                   "updating_constant": geometry["updating"][0], "updating_linear": linear[mode]}
        errors = {s: targets-design@bank.coefficients[mode]-center for s, center in centers.items()}
        for condition, (mean_source, cov_source) in CONDITIONS.items():
            error = errors[mean_source]
            energy, logdet, likelihoods = density_terms(error, geometry[cov_source][1], bank.target_scale)
            for key, values in (("energy", energy), ("logdet", logdet), ("likelihoods", likelihoods),
                                ("mean_error_rms", np.sqrt(np.mean(error**2, axis=1)))):
                output[condition][key].append(values)
    return {c: {k: np.column_stack(v) for k, v in terms.items()} for c, terms in output.items()}


class LocalLinearDensityBank:
    def __init__(self, banks, correction, covariance_source):
        self.banks, self.correction, self.covariance_source = banks, correction, covariance_source

    def log_likelihoods(self, trace):
        bank = self.banks["updating"]
        inputs, design, targets = bank.features.inputs(trace), bank.design(trace), bank.targets(trace)
        centers = self.correction.predict(inputs)
        output = []
        for mode in MODES:
            covariance = self.banks[self.covariance_source].geometry(inputs, mode)[1]
            errors = targets-design@bank.coefficients[mode]-centers[mode]
            output.append(density_terms(errors, covariance, bank.target_scale)[2])
        return np.column_stack(output)

    def metadata(self):
        return dict(modes=list(MODES), conditional_mean="64-neighbor local linear ridge residual prediction",
            conditional_mean_source="updating_mixed", conditional_covariance_source=self.covariance_source,
            conditional_covariance_changed=False, forward_refitted=False, scaling_refitted=False,
            ridge_mean_loss_coefficient=RIDGE, ridge_alpha=NEIGHBORS*RIDGE, intercept_penalized=False,
            input_dimension=len(self.banks["updating"].features.input_mean), neighbors=NEIGHBORS,
            fitting="saved reference residuals only; no query next-state or mode label",
            likelihood_units="physical delta-state Gaussian density including log determinant")
