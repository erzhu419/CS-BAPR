"""Conditional residual densities around an unchanged v71 forward bank."""
from copy import deepcopy

import numpy as np
from scipy.spatial import cKDTree

from scripts import v68_mode_likelihood as original
from scripts.v66_covariance_mix_readout import normal_geometry
from scripts.v65_state_action_readout import NEIGHBORS

MODES = original.MODES


class ConditionalMeanBank(original.ModeLikelihoodBank):
    @classmethod
    def fit_density(cls, forward, calibration):
        bank = cls()
        bank.__dict__.update(deepcopy(forward.__dict__))
        assert tuple(calibration) == MODES
        inputs = bank.features.inputs(calibration["normal"])
        assert len(inputs) >= NEIGHBORS
        bank.tree = cKDTree((inputs-bank.features.input_mean)/bank.features.input_scale)
        bank.residuals = {}
        for mode in MODES:
            np.testing.assert_array_equal(bank.features.inputs(calibration[mode]), inputs)
            bank.residuals[mode] = bank.targets(calibration[mode])-bank.design(calibration[mode])@bank.coefficients[mode]
            np.testing.assert_allclose(bank.residuals[mode].mean(axis=0), forward.centers[mode], rtol=0, atol=1e-12)
        bank.density_transitions = len(inputs)
        return bank

    def local_centers(self, inputs):
        _, indices = self.tree.query((inputs-self.features.input_mean)/self.features.input_scale, k=NEIGHBORS)
        return {m: self.residuals[m][indices].mean(axis=1) for m in MODES}

    def log_likelihoods(self, trace):
        centers = self.local_centers(self.features.inputs(trace))
        design, targets = self.design(trace), self.targets(trace)
        constant = len(self.target_scale)*np.log(2*np.pi)
        output = []
        for mode in MODES:
            errors = targets-design@self.coefficients[mode]-centers[mode]
            energy = np.einsum("ni,ij,nj->n", errors, self.precisions[mode], errors)
            output.append(-.5*(energy+self.log_determinants[mode]+constant))
        return np.column_stack(output)

    def metadata(self):
        meta = super().metadata()
        meta["forward_statistics"] = meta.pop("statistics")
        meta.update(residual_distribution="64-neighbor conditional residual mean; unchanged global covariance",
            neighbors=NEIGHBORS, density_calibration_transitions=self.density_transitions,
            conditioning="current state and executed action only; frozen v71 input scaling",
            geometry_source="matched v71 density-calibration states and raw forward residuals",
            query_residual_used_for_geometry=False, forward_refitted=False)
        return meta


class ConditionalDensityBank(ConditionalMeanBank):
    def geometry(self, inputs, mode):
        return normal_geometry(self.tree, self.residuals[mode], self.features.input_mean,
                               self.features.input_scale, inputs)

    def log_likelihoods(self, trace):
        inputs = self.features.inputs(trace)
        design, targets = self.design(trace), self.targets(trace)
        constant = len(self.target_scale)*np.log(2*np.pi)+2*np.log(self.target_scale).sum()
        output = []
        for mode in MODES:
            centers, covariance, _ = self.geometry(inputs, mode)
            errors = targets-design@self.coefficients[mode]-centers
            sign, logdet = np.linalg.slogdet(covariance)
            assert np.all(sign > 0), "non-positive-conditional-density-covariance"
            solved = np.linalg.solve(covariance, errors[..., None])[..., 0]
            energy = np.einsum("ni,ni->n", errors, solved)
            output.append(-.5*(energy+logdet+constant))
        return np.column_stack(output)

    def metadata(self):
        meta = super().metadata()
        meta["residual_distribution"] = "64-neighbor conditional residual mean and regularized local OAS covariance"
        return meta


def calibrate_bank(bank, nulls):
    reference = original.ModeLikelihoodReference()
    reference.bank = bank
    signals = np.concatenate([original.mode_evidence(original.rolling_means(bank.log_likelihoods(t)))[2] for t in nulls])
    ages = np.concatenate([original.v43.episode_steps(t["dones"]) for t in nulls])
    phases = np.concatenate([original.episode_ordinals(t["dones"]) > 0 for t in nulls])
    curves = [original.NormalAgeReference.fit_normal(signals[phases == p], ages[phases == p]) for p in (False, True)]
    assert all(c.ages[0] == 0 and c.ages[-1] == 999 for c in curves)
    reference.phase = original.PhaseAgeReference(*curves)
    reference.normal_null_calibration_transitions = len(signals)
    return reference
