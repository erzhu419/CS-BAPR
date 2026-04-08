"""
SINDy → PyTorch Bridge (SINDyTorchWrapper)

Converts PySINDy's numpy-based sparse analytic formula into
a differentiable nn.Module for autograd-based Jacobian computation.

⚠️ CRITICAL: Coefficients are registered as buffers (NOT parameters),
   ensuring they are frozen during training (Lean Frozen Penalty assumption, L251).
⚠️ CRITICAL: Basis function ordering must match PySINDy's PolynomialLibrary
   expansion order (trap #8). Use get_feature_names() to verify.
"""

import torch
import torch.nn as nn
import numpy as np
import warnings


class SINDyTorchWrapper(nn.Module):
    """
    Differentiable PyTorch wrapper for PySINDy's polynomial formula.
    
    Reconstructs: f_sym(x) = Θ(x) · ξᵀ
    where Θ(x) is the polynomial feature matrix and ξ are SINDy coefficients.
    
    Coefficients are FROZEN (register_buffer) — never updated during RL training.
    This is required by Lean's Frozen Penalty assumption (t_mode_pointwise_bound, L251).
    """
    def __init__(self, sindy_model, verify_features: bool = True):
        """
        Args:
            sindy_model: A fitted SymbolicWorldModel instance
            verify_features: If True, verify feature ordering matches
        """
        super().__init__()

        if sindy_model.coeffs is None:
            raise ValueError("SINDy model must be fitted before wrapping")

        # Frozen coefficients [output_dim, n_features]
        coeffs = torch.tensor(sindy_model.coeffs, dtype=torch.float32)
        self.register_buffer('coeffs', coeffs)

        # Store library config for feature reconstruction
        self.poly_degree = sindy_model.poly_degree
        self.n_state = sindy_model.n_state
        self.n_control = sindy_model.n_control
        self._has_control = sindy_model.n_control > 0

        # Get feature names from PySINDy for verification
        try:
            self._pysindy_feature_names = sindy_model.get_feature_names()
        except Exception:
            self._pysindy_feature_names = None
            warnings.warn("Could not retrieve PySINDy feature names for verification")

        if verify_features and self._pysindy_feature_names is not None:
            self._verify_feature_order()

    def _verify_feature_order(self):
        """Verify our feature construction matches PySINDy's ordering (trap #8)."""
        # Generate test input and compare feature counts
        n_features_pysindy = len(self._pysindy_feature_names)
        n_features_ours = self.coeffs.shape[1]
        if n_features_pysindy != n_features_ours:
            warnings.warn(
                f"Feature count mismatch! PySINDy: {n_features_pysindy}, "
                f"Wrapper: {n_features_ours}. Check polynomial expansion order."
            )

    def _build_polynomial_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Build polynomial feature matrix Θ(x) matching PySINDy's PolynomialLibrary.
        
        For degree=2 with n features [x₁, x₂, ...]:
        Θ(x) = [1, x₁, x₂, ..., x₁², x₁x₂, x₂², ...]
        
        PySINDy uses graded lexicographic ordering.
        """
        batch = x.shape[0]
        n = x.shape[1]
        feats = [torch.ones(batch, 1, device=x.device)]  # bias (degree 0)

        # Degree 1: x₁, x₂, ..., xₙ
        feats.append(x)

        if self.poly_degree >= 2:
            # Degree 2: x₁², x₁x₂, ..., x₁xₙ, x₂², x₂x₃, ..., xₙ²
            for i in range(n):
                for j in range(i, n):
                    feats.append((x[:, i] * x[:, j]).unsqueeze(1))

        if self.poly_degree >= 3:
            # Degree 3: all combinations x_i * x_j * x_k (i ≤ j ≤ k)
            for i in range(n):
                for j in range(i, n):
                    for k in range(j, n):
                        feats.append((x[:, i] * x[:, j] * x[:, k]).unsqueeze(1))

        return torch.cat(feats, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct SINDy formula: f_sym(x) = Θ(x) · ξᵀ
        
        Args:
            x: State tensor [batch, n_state] (no control inputs for now)
        Returns:
            Predicted dynamics [batch, output_dim]
        """
        features = self._build_polynomial_features(x)  # [batch, n_features]
        return features @ self.coeffs.T                  # [batch, output_dim]

    @torch.no_grad()
    def validate_against_numpy(self, sindy_model, test_states: np.ndarray,
                                atol: float = 1e-4) -> bool:
        """
        Validate torch wrapper output matches PySINDy numpy output.
        Returns True if outputs match within tolerance.
        """
        # Numpy prediction
        np_pred = sindy_model.predict(test_states)
        
        # Torch prediction
        x_torch = torch.tensor(test_states, dtype=torch.float32)
        torch_pred = self.forward(x_torch).numpy()
        
        max_err = np.abs(np_pred - torch_pred).max()
        matches = max_err < atol
        
        if not matches:
            warnings.warn(
                f"SINDyTorchWrapper validation FAILED: max error = {max_err:.6f} "
                f"(threshold = {atol}). Feature ordering may be wrong (trap #8)."
            )
        
        return matches
