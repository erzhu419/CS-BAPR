"""
Symbolic World Model — PySINDy Wrapper for CS-BAPR.

Corresponds to CSBAPR.lean Part VI (L461-595):
- BasisLibrary (L463) → PySINDy feature_library
- SINDyCoeffs (L470) → model.coefficients_
- SINDyIdentifies (L502) → fit quality < ε₁
- sindy_implies_jacobian_consistency_nD (L518) → bridge theorem

Reference: sindy-rl/sindy_rl/dynamics.py:EnsembleSINDyDynamicsModel
Reference: sindy-rl/sindy_rl/sindy_utils.py:get_affine_lib
"""

import numpy as np
import warnings

try:
    import pysindy as ps
except ImportError:
    ps = None
    warnings.warn("PySINDy not installed. Install with: pip install pysindy")


class SymbolicWorldModel:
    """
    SINDy-based symbolic dynamics model.
    
    Identifies sparse analytic formula f_sym from trajectory data.
    In CS-BAPR, f_sym approximates the environment dynamics (NOT the policy).
    
    Lean mapping:
    - fit() → SINDyIdentifies (L502): SINDy identifies f_sym with error ε₁
    - predict() → SINDyCoeffs.reconstruct (L475): Θ(x) · ξ
    - deriv_error() → ε₁ estimation for OOD bound
    
    ⚠️ SINDy identifies ENVIRONMENT DYNAMICS ds/dt = f(s), not policy π(s).
    Policy π is constrained via Jacobian Consistency Loss to align with ∇f_sym.
    """
    def __init__(self, n_state: int, n_control: int = 0,
                 poly_degree: int = 2, threshold: float = 0.1,
                 discrete_time: bool = True,
                 feature_names: list = None):
        if ps is None:
            raise ImportError("PySINDy required. Install: pip install pysindy")

        self.n_state = n_state
        self.n_control = n_control
        self.discrete_time = discrete_time
        self.poly_degree = poly_degree
        self.threshold = threshold
        self.feature_names = feature_names
        self.coeffs = None
        self.sparsity = None

        # Build feature library
        # Use standard polynomial on concatenated (state, control) inputs
        # For pysindy 1.7.x compatibility, avoid GeneralizedLibrary
        self.library = ps.PolynomialLibrary(degree=poly_degree)

        # STLSQ optimizer with sparsity threshold
        self.optimizer = ps.STLSQ(threshold=threshold)

        self.model = ps.SINDy(
            feature_library=self.library,
            optimizer=self.optimizer,
            feature_names=feature_names,
            discrete_time=discrete_time
        )

    def _build_affine_library(self):
        """
        Control-affine library: x_{t+1} = p₁(x) + p₂(x)·u
        Reference: sindy_utils.get_affine_lib
        """
        polyLib = ps.PolynomialLibrary(
            degree=self.poly_degree,
            include_bias=True,
            include_interaction=True
        )
        affineLib = ps.PolynomialLibrary(
            degree=1, include_bias=False, include_interaction=False
        )

        # State library: only state variables
        inputs_state = np.arange(self.n_state + self.n_control)
        inputs_state[-self.n_control:] = 0  # mask out control

        # Control library: only control variables
        inputs_control = np.arange(self.n_state + self.n_control)
        inputs_control[:self.n_state] = self.n_state + self.n_control - 1

        inputs_per_library = np.array([inputs_state, inputs_control], dtype=int)

        return ps.GeneralizedLibrary(
            [polyLib, affineLib],
            tensor_array=None,
            inputs_per_library=inputs_per_library,
        )

    def fit(self, X, U=None, X_dot=None, t=None, multiple_trajectories=False):
        """
        Fit SINDy model.
        
        Corresponds to Lean: SINDyIdentifies (L502)
        
        Args:
            X: State data. If discrete_time and multiple_trajectories: list of [T, n_state]
               Otherwise: [T, n_state]
            U: Control data (optional). Same format as X.
            X_dot: State derivatives (only needed for continuous time)
            t: Time step dt (for discrete) or time array
            multiple_trajectories: if True, X and U are lists
        """
        fit_kwargs = {}
        if multiple_trajectories:
            fit_kwargs['multiple_trajectories'] = True

        if self.discrete_time:
            self.model.fit(X, u=U, t=t, **fit_kwargs)
        else:
            self.model.fit(X, x_dot=X_dot, u=U, t=t, **fit_kwargs)

        self.coeffs = self._get_coefficients()
        self.sparsity = (np.abs(self.coeffs) > 1e-6).mean()

    def _get_coefficients(self):
        """Compatibility shim for PySINDy versions."""
        if hasattr(self.model, 'coefficients_'):
            return self.model.coefficients_  # pysindy >= 2.0
        elif hasattr(self.model, 'coefficients'):
            return self.model.coefficients()  # pysindy 1.7.x
        else:
            raise AttributeError("Cannot find SINDy coefficients")

    def predict(self, x, u=None):
        """
        One-step prediction.
        Corresponds to Lean: SINDyCoeffs.reconstruct (L475)
        """
        return self.model.predict(x, u=u)

    def deriv_error(self, states, true_dynamics, u=None):
        """
        Estimate ε₁ = ‖fderiv(f_real) - fderiv(f_sym)‖
        """
        pred = self.model.predict(states, u=u)
        return np.linalg.norm(pred - true_dynamics, axis=-1).mean()

    def get_feature_names(self):
        """Get feature names for debugging (trap #8: verify order matches torch wrapper)."""
        return self.model.get_feature_names()

    def print_equations(self):
        """Print discovered equations."""
        self.model.print()
