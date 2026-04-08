"""
Jacobian Consistency Loss (Γ_sym) for CS-BAPR.

Corresponds to CSBAPR.lean:
- IsJacobianConsistent_nD (L400): ‖fderiv pol x - fderiv f_sym x‖ ≤ ε
- sindy_implies_jacobian_consistency_nD (L518): SINDy → Jacobian consistency

⚠️ CRITICAL: f_sym gradients must be .detach()'d (Frozen Penalty, L251).
   If not detached, violates t_mode_pointwise_bound assumption.

⚠️ Performance: jacrev is O(n²) for large networks (trap #5).
   Use JVP with random vector for O(n) approximation.
"""

import torch
import torch.nn as nn


def compute_jacobian_loss(policy_net: nn.Module, f_sym: nn.Module,
                          states_batch: torch.Tensor) -> torch.Tensor:
    """
    Compute Jacobian consistency loss: ‖∇π - ∇f_sym‖²
    
    Corresponds to CSBAPR.lean: IsJacobianConsistent_nD (L400)
    Implements n-D version (autograd returns gradient vectors).
    When state_dim=1, degenerates to 1D IsJacobianConsistent (L39).
    
    Minimizing this loss drives ε → 0 in the OOD bound.
    
    Args:
        policy_net: Policy network π(s) → action
        f_sym: Frozen SINDy torch wrapper (coefficients immutable)
        states_batch: [batch, state_dim] states from replay buffer
    
    Returns:
        Scalar Jacobian consistency loss (batch-averaged)
    """
    states = states_batch.detach().clone().requires_grad_(True)

    # Policy network gradient ∇π ∈ ℝ^{batch × state_dim}
    pi_output = policy_net(states)
    # Handle (mean, log_std) tuple output from SAC-style policies
    if isinstance(pi_output, tuple):
        pi_output = pi_output[0]  # use mean only
    grad_pi = torch.autograd.grad(
        pi_output.sum(), states, create_graph=True
    )[0]

    # SINDy formula gradient (MUST detach — Frozen Penalty assumption)
    sym_output = f_sym(states)
    grad_sym = torch.autograd.grad(
        sym_output.sum(), states
    )[0]

    # ‖∇π - ∇f_sym‖² — corresponds to Lean's ‖fderiv pol x - fderiv f_sym x‖
    jac_loss = (grad_pi - grad_sym.detach()).pow(2).mean()
    return jac_loss


def compute_jacobian_loss_nD(policy_net: nn.Module, f_sym: nn.Module,
                              states_batch: torch.Tensor) -> torch.Tensor:
    """
    Full n-D Jacobian consistency using torch.func.jacrev.
    
    Corresponds to Part V (L394-460): n-D generalization.
    
    ⚠️ O(n²) cost. For large networks or state dims, use compute_jacobian_loss instead.
    
    Args:
        policy_net: Policy network
        f_sym: Frozen SINDy wrapper
        states_batch: [batch, state_dim]
    
    Returns:
        Scalar Frobenius-norm Jacobian loss
    """
    def policy_fn(x):
        out = policy_net(x.unsqueeze(0))
        if isinstance(out, tuple):
            out = out[0]
        return out.squeeze(0)

    def sym_fn(x):
        return f_sym(x.unsqueeze(0)).squeeze(0)

    # Batched Jacobian computation
    jac_pi = torch.vmap(torch.func.jacrev(policy_fn))(states_batch)
    
    with torch.no_grad():
        jac_sym = torch.vmap(torch.func.jacrev(sym_fn))(states_batch)

    # Frobenius norm of Jacobian difference
    return (jac_pi - jac_sym).norm(dim=(-2, -1)).mean()
