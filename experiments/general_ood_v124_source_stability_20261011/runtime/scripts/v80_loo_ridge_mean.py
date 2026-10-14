"""Reference-only local ridge selection with unchanged v78 covariances."""
import numpy as np

from scripts.probe_general_ood_v42 import RIDGE
from scripts.v65_state_action_readout import NEIGHBORS
from scripts.v77_mean_covariance import MODES
from scripts.v78_local_linear_mean import LocalLinearDensityBank

MEAN_LOSS_COEFFICIENTS = (1., .1, .01, RIDGE)
ALPHAS = tuple(NEIGHBORS*c for c in MEAN_LOSS_COEFFICIENTS)
CONDITIONS = {"loo_mean_frozen_cov": "frozen", "loo_mean_updating_cov": "updating"}


def fit_neighborhood(inputs, targets, query):
    x_mean, y_mean = inputs.mean(axis=0), targets.mean(axis=0)
    x, y = inputs-x_mean, targets-y_mean
    eigenvalues, vectors = np.linalg.eigh(x.T@x)
    projected = x@vectors
    inverse = 1/(np.asarray(ALPHAS)[:, None]+eigenvalues)
    slopes = np.einsum("ij,aj,jk->aik", vectors, inverse, projected.T@y)
    errors = y[None, :, :]-np.einsum("ni,aik->ank", x, slopes)
    hat_diagonal = 1/NEIGHBORS+np.einsum("nj,aj->an", projected**2, inverse)
    loo_errors = errors/(1-hat_diagonal[:, :, None])
    losses = np.mean(loo_errors**2, axis=(1, 2))
    # Descending strengths select the stronger fit in an exact loss tie.
    selected = int(np.argmin(losses))
    displacement = query-x_mean
    weights = 1/NEIGHBORS-projected@(inverse[selected]*(vectors.T@(-displacement)))
    return y_mean+displacement@slopes[selected], dict(
        selected_alpha=ALPHAS[selected], loo_mse=losses,
        prediction_weight_l1=float(np.abs(weights).sum()),
        ridge_query_leverage=float(1/NEIGHBORS+np.sum((displacement@vectors)**2*inverse[selected])))


class LooRidgeMean:
    def __init__(self, bank):
        self.bank = bank
        self.paired_residuals = np.concatenate([bank.residuals[m] for m in MODES], axis=1)

    def predict_details(self, inputs):
        bank = self.bank
        normalized = (inputs-bank.features.input_mean)/bank.features.input_scale
        _, indices = bank.tree.query(normalized, k=NEIGHBORS)
        centers, details = [], []
        for query, neighbors in zip(normalized, indices):
            center, detail = fit_neighborhood(bank.tree.data[neighbors], self.paired_residuals[neighbors], query)
            centers.append(center)
            details.append(detail)
        return dict(zip(MODES, np.split(np.asarray(centers), len(MODES), axis=1))), {
            k: np.asarray([d[k] for d in details]) for k in details[0]}

    def predict(self, inputs):
        return self.predict_details(inputs)[0]


class LooRidgeDensityBank(LocalLinearDensityBank):
    def metadata(self):
        meta = super().metadata()
        meta.pop("ridge_alpha")
        meta.pop("ridge_mean_loss_coefficient")
        meta.update(conditional_mean="64-neighbor local ridge residual prediction selected by reference leave-one-out MSE",
            ridge_alpha_candidates=list(ALPHAS), mean_loss_coefficient_candidates=list(MEAN_LOSS_COEFFICIENTS),
            selection="minimum mean squared leave-one-out error across 64 reference rows and all five normalized residual vectors",
            shared_strength_across_modes=True, exact_tie="stronger ridge", query_next_state_used_for_selection=False)
        return meta
