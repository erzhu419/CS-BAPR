"""Append matched density labels without changing the v72 forward model."""
from copy import deepcopy

import numpy as np
from scipy.spatial import cKDTree

from scripts.v72_conditional_likelihood import ConditionalDensityBank, MODES


class DensityCoverageBank(ConditionalDensityBank):
    @classmethod
    def extend(cls, original, extensions, condition):
        assert tuple(extensions) == MODES
        bank = cls()
        bank.__dict__.update(deepcopy(original.__dict__))
        inputs = bank.features.inputs(extensions["normal"])
        design = bank.features.features(inputs)
        bank.base_density_transitions = original.density_transitions
        bank.added_density_transitions = len(inputs)
        bank.coverage_condition = condition
        standardized = (inputs-bank.features.input_mean)/bank.features.input_scale
        bank.tree = cKDTree(np.concatenate((original.tree.data, standardized)))
        for mode in MODES:
            np.testing.assert_array_equal(bank.features.inputs(extensions[mode]), inputs)
            extra = bank.targets(extensions[mode])-design@bank.coefficients[mode]
            bank.residuals[mode] = np.concatenate((original.residuals[mode], extra))
        bank.density_transitions = bank.base_density_transitions+bank.added_density_transitions
        return bank

    def metadata(self):
        meta = super().metadata()
        meta.update(geometry_source="v72 base density labels plus matched v74 offline counterfactual suffix labels",
            base_density_transitions=self.base_density_transitions,
            added_density_transitions=self.added_density_transitions,
            coverage_condition=self.coverage_condition, conditional_density_refitted=True,
            forward_refitted=False, scaling_refitted=False)
        return meta
