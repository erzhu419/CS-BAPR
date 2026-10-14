"""Cross conditional residual means and covariances from the saved v76 banks."""
import numpy as np

from scripts.v73_recursive_mode_readout import MODES

COMBINATIONS = {
    "frozen_mean_frozen_cov": ("frozen", "frozen"),
    "updating_mean_frozen_cov": ("updating", "frozen"),
    "frozen_mean_updating_cov": ("frozen", "updating"),
    "updating_mean_updating_cov": ("updating", "updating"),
}
DIAGONALS = {"frozen_mean_frozen_cov": "frozen_mixed",
             "updating_mean_updating_cov": "updating_mixed"}


def check_shared_forward(banks):
    frozen, updating = banks["frozen"], banks["updating"]
    for mode in MODES:
        np.testing.assert_array_equal(frozen.coefficients[mode], updating.coefficients[mode])
    for name in ("input_mean", "input_scale"):
        np.testing.assert_array_equal(getattr(frozen.features, name), getattr(updating.features, name))
    np.testing.assert_array_equal(frozen.target_scale, updating.target_scale)


def density_terms(errors, covariance, target_scale):
    sign, logdet = np.linalg.slogdet(covariance)
    assert np.all(sign > 0), "non-positive-hybrid-covariance"
    solved = np.linalg.solve(covariance, errors[..., None])[..., 0]
    energy = np.einsum("ni,ni->n", errors, solved)
    constant = len(target_scale)*np.log(2*np.pi)+2*np.log(target_scale).sum()
    return energy, logdet, -.5*(energy+logdet+constant)


def factorial_terms(trace, banks):
    bank = banks["frozen"]
    inputs, design, targets = bank.features.inputs(trace), bank.design(trace), bank.targets(trace)
    output = {c: {k: [] for k in ("energy", "logdet", "likelihoods", "mean_error_rms")}
              for c in COMBINATIONS}
    for mode in MODES:
        geometry = {s: b.geometry(inputs, mode)[:2] for s, b in banks.items()}
        errors = {s: targets-design@bank.coefficients[mode]-geometry[s][0] for s in banks}
        for condition, (mean_source, cov_source) in COMBINATIONS.items():
            error = errors[mean_source]
            energy, logdet, likelihoods = density_terms(error, geometry[cov_source][1], bank.target_scale)
            for key, values in (("energy", energy), ("logdet", logdet), ("likelihoods", likelihoods),
                                ("mean_error_rms", np.sqrt(np.mean(error**2, axis=1)))):
                output[condition][key].append(values)
    return {c: {k: np.column_stack(v) for k, v in terms.items()} for c, terms in output.items()}


class MeanCovarianceBank:
    def __init__(self, banks, condition):
        self.banks, self.condition = banks, condition

    def log_likelihoods(self, trace):
        mean_source, cov_source = COMBINATIONS[self.condition]
        mean_bank, cov_bank = self.banks[mean_source], self.banks[cov_source]
        inputs = mean_bank.features.inputs(trace)
        design, targets = mean_bank.design(trace), mean_bank.targets(trace)
        centers = mean_bank.local_centers(inputs) if mean_source != cov_source else None
        output = []
        for mode in MODES:
            center, covariance, _ = cov_bank.geometry(inputs, mode)
            if centers is not None:
                center = centers[mode]
            errors = targets-design@mean_bank.coefficients[mode]-center
            output.append(density_terms(errors, covariance, mean_bank.target_scale)[2])
        return np.column_stack(output)

    def metadata(self):
        mean_source, cov_source = COMBINATIONS[self.condition]
        return dict(modes=list(MODES), conditional_mean_source=mean_source,
                    conditional_covariance_source=cov_source, neighbors=64,
                    forward_refitted=False, scaling_refitted=False,
                    mean_and_covariance_refitted=False,
                    likelihood_units="physical delta-state Gaussian density including log determinant")
