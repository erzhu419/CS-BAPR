"""
Symbolic World Model — PySINDy Wrapper for CS-BAPR.

Corresponds to CSBAPR.lean Part VI (L461-595):
- BasisLibrary (L463) → PySINDy feature_library
- SINDyCoeffs (L470) → model.coefficients()
- SINDyIdentifies (L502) → fit quality < ε₁
- sindy_implies_jacobian_consistency_nD (L518) → bridge theorem

Compatible with PySINDy 2.x API:
- SINDy(optimizer, feature_library) — no feature_names/discrete_time
- fit(x, t, x_dot=, u=, feature_names=) — t is required
- coefficients() — method, not attribute
- predict(x) — no t argument
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

        self.library = ps.PolynomialLibrary(degree=poly_degree)
        self.optimizer = ps.STLSQ(threshold=threshold)

        # PySINDy 2.x: only optimizer + feature_library in __init__
        self.model = ps.SINDy(
            feature_library=self.library,
            optimizer=self.optimizer,
        )

    def fit(self, X, U=None, X_dot=None, t=None, multiple_trajectories=False):
        """
        Fit SINDy model.

        Corresponds to Lean: SINDyIdentifies (L502)

        For discrete-time mode:
          - X is state trajectories, X_dot = X[1:] - X[:-1] computed internally
          - multiple_trajectories: concat list of arrays before fitting

        For continuous-time:
          - X_dot must be provided externally
        """
        fit_kwargs = {}
        if self.feature_names:
            fit_kwargs['feature_names'] = self.feature_names

        # PySINDy 2.x: no multiple_trajectories param. Concat manually.
        if multiple_trajectories and isinstance(X, list):
            if self.discrete_time:
                # Discrete: compute x_dot = x_{t+1} - x_t per trajectory, then concat
                all_x, all_xdot = [], []
                for traj in X:
                    all_x.append(traj[:-1])
                    all_xdot.append(traj[1:] - traj[:-1])
                X_concat = np.vstack(all_x)
                X_dot_concat = np.vstack(all_xdot)
                self.model.fit(X_concat, t=t or 1.0,
                               x_dot=X_dot_concat, **fit_kwargs)
            else:
                # Continuous: X_dot must already be provided
                X_concat = np.vstack(X)
                X_dot_concat = np.vstack(X_dot) if X_dot is not None else None
                self.model.fit(X_concat, t=t or 1.0,
                               x_dot=X_dot_concat, **fit_kwargs)
        else:
            # Single array
            if self.discrete_time and X_dot is None:
                # Auto-compute finite differences
                X_arr = np.asarray(X)
                x_curr = X_arr[:-1]
                x_dot_auto = X_arr[1:] - X_arr[:-1]
                self.model.fit(x_curr, t=t or 1.0,
                               x_dot=x_dot_auto, **fit_kwargs)
            else:
                self.model.fit(X, t=t or 1.0,
                               x_dot=X_dot, **fit_kwargs)

        self.coeffs = self._get_coefficients()
        self.sparsity = (np.abs(self.coeffs) > 1e-6).mean()

    def _get_coefficients(self):
        """Compatibility shim for PySINDy versions."""
        if hasattr(self.model, 'coefficients_'):
            return self.model.coefficients_
        elif callable(getattr(self.model, 'coefficients', None)):
            return self.model.coefficients()
        else:
            raise AttributeError("Cannot find SINDy coefficients")

    def predict(self, x, u=None):
        """
        One-step prediction: x_dot = Θ(x) · ξ
        Corresponds to Lean: SINDyCoeffs.reconstruct (L475)
        """
        return self.model.predict(x)

    def deriv_error(self, states, true_dynamics, u=None):
        """Estimate ε₁ = ‖fderiv(f_real) - fderiv(f_sym)‖"""
        pred = self.model.predict(states)
        return np.linalg.norm(pred - true_dynamics, axis=-1).mean()

    def get_feature_names(self):
        """Get feature names for debugging."""
        return self.model.get_feature_names()

    def print_equations(self):
        """Print discovered equations."""
        self.model.print()
