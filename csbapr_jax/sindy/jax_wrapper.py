"""SINDy-to-JAX bridge for Jacobian consistency loss.

Converts PySINDy polynomial formula into a pure JAX function
for autograd-based Jacobian computation in the scan-fused training loop.

⚠️ Coefficients are STATIC (not JAX parameters — frozen, matching Lean requirement).
"""
import jax
import jax.numpy as jnp
import numpy as np


class SINDyJAXWrapper:
    """Pure-JAX polynomial reconstruction of SINDy formula.

    f_sym(x) = Θ(x) · ξᵀ where Θ is the polynomial feature matrix.

    Coefficients are stored as plain jnp arrays (NOT trainable params).
    This matches the Lean Frozen Penalty assumption (L251).
    """
    def __init__(self, coeffs: np.ndarray, n_state: int, poly_degree: int = 2):
        """
        Args:
            coeffs: SINDy coefficients [output_dim, n_features]
            n_state: Number of state dimensions
            poly_degree: Polynomial degree used in SINDy
        """
        self.coeffs = jnp.array(coeffs, dtype=jnp.float32)
        self.n_state = n_state
        self.poly_degree = poly_degree

    def _build_features(self, x):
        """Build polynomial feature vector Θ(x) matching PySINDy ordering.

        For degree=2 with n features [x₁, x₂, ...]:
        Θ(x) = [1, x₁, x₂, ..., x₁², x₁x₂, x₂², ...]
        """
        batch = x.shape[0]
        n = x.shape[1]
        feats = [jnp.ones((batch, 1))]  # bias

        # Degree 1
        feats.append(x)

        if self.poly_degree >= 2:
            # Degree 2: graded lexicographic order
            deg2 = []
            for i in range(n):
                for j in range(i, n):
                    deg2.append(x[:, i:i+1] * x[:, j:j+1])
            if deg2:
                feats.append(jnp.concatenate(deg2, axis=1))

        if self.poly_degree >= 3:
            deg3 = []
            for i in range(n):
                for j in range(i, n):
                    for k in range(j, n):
                        deg3.append(x[:, i:i+1] * x[:, j:j+1] * x[:, k:k+1])
            if deg3:
                feats.append(jnp.concatenate(deg3, axis=1))

        return jnp.concatenate(feats, axis=1)

    def predict(self, x):
        """f_sym(x) = Θ(x) · ξᵀ"""
        features = self._build_features(x)
        return features @ self.coeffs.T

    def predict_single(self, x):
        """Single-sample prediction for vmap/jacrev."""
        return self.predict(x[None])[0]


def compute_jacobian_loss_jax(policy_fn, sym_wrapper, obs, key):
    """Compute ‖∇π - ∇f_sym‖² for Jacobian consistency.

    Corresponds to CSBAPR.lean: IsJacobianConsistent_nD (L400).

    Uses JAX autograd to compute gradients of policy mean and SINDy formula
    w.r.t. state, then measures their Frobenius-norm difference.

    Args:
        policy_fn: callable(obs) → (mean, log_std) — policy forward function
        sym_wrapper: SINDyJAXWrapper with frozen coefficients
        obs: [batch, obs_dim] observation batch
        key: PRNG key (for policy sampling, unused here — we use mean)

    Returns:
        Scalar Jacobian consistency loss
    """
    # Policy gradient ∇π w.r.t. obs  (only mean, not log_std)
    def pi_mean_fn(x_single):
        mean, _ = policy_fn(x_single[None])
        return mean[0]

    # SINDy gradient ∇f_sym w.r.t. obs (frozen — no grad to coefficients)
    def sym_fn(x_single):
        return sym_wrapper.predict_single(x_single)

    # Batched Jacobian via vmap+jacrev
    jac_pi = jax.vmap(jax.jacrev(pi_mean_fn))(obs)       # [batch, act, obs]
    jac_sym = jax.vmap(jax.jacrev(sym_fn))(obs)           # [batch, out, obs]

    # Match dimensions: use minimum of act_dim and sym_out_dim
    min_d = min(jac_pi.shape[1], jac_sym.shape[1])
    jac_pi_matched = jac_pi[:, :min_d, :]
    jac_sym_matched = jac_sym[:, :min_d, :]

    # Frobenius norm of difference
    diff = jac_pi_matched - jax.lax.stop_gradient(jac_sym_matched)
    return jnp.mean(diff ** 2)
