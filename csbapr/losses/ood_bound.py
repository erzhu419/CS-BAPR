"""
OOD Bound Estimation for CS-BAPR.

Corresponds to CSBAPR.lean:
- Part X (L954-1198):
  - deriv_error_propagation_nD (L1070): ε + (L_pol+M)·‖x-x₀‖
  - ood_bound_no_assumption4_nD (L1117): δ + (ε + (L_eff+M)·‖d‖)·‖d‖
  - HasLipschitzFderiv (L1063): M estimation
- Part XI (L1200-1462):
  - composed_deriv_lipschitz_simple (L1417): L_eff = L_h·K_g·B_g + L_g
- Part XIII (L1639-1681):
  - jc_generalization_to_ood (L1663): ε_pop ≤ ε_emp + gap

Key improvement over old code:
- OLD: bound = δ + ε·d + L_nmu/2·d² (required Assumption 4: JC on OOD path)
- NEW: bound = δ + (ε_emp + gap + (L_eff + M)·d)·d (no Assumption 4)
"""

import math
import torch
import torch.nn as nn
import numpy as np


def compute_generalization_gap(deriv_bound_B: float, n_samples: int,
                               confidence_delta: float = 0.05) -> float:
    """
    PAC-style generalization gap for Jacobian Consistency loss.

    Corresponds to CSBAPR.lean: jc_generalization_to_ood (L1663)

    gap = B · √(2·log(2/δ) / N)

    Args:
        deriv_bound_B: Upper bound on derivative functions.
            NAU: max|w| = 1, NMU: 2|c|·|x_max|.
        n_samples: Number of i.i.d. training samples N.
        confidence_delta: Failure probability δ (0.05 → 95% confidence).

    Returns:
        Generalization gap (float).
    """
    if n_samples <= 0:
        return float('inf')
    return deriv_bound_B * math.sqrt(2 * math.log(2.0 / confidence_delta) / n_samples)


def estimate_deriv_bound_B(actor: nn.Module) -> float:
    """
    Estimate the derivative function upper bound B for the actor.

    For NAU_NMU_Actor:
    - NAU component: derivatives are the clamped weights, max |w| ≤ 1
    - NMU component: derivative = 2·c·x, bounded by 2·max|c|·x_max
    - We use a conservative estimate assuming bounded input domain.

    Returns:
        B estimate (float).
    """
    B = 1.0  # conservative default (NAU derivative ≤ 1)
    if hasattr(actor, 'nmu_head'):
        c_max = actor.nmu_head.coeff.abs().max().item()
        # NMU derivative = 2·c·x; for typical RL inputs |x| ≤ 10
        B = max(B, 2 * c_max * 10.0)
    return B


def estimate_physics_smoothness(env_step_fn, states: torch.Tensor,
                                dt: float = 1e-3,
                                n_pairs: int = 100) -> float:
    """
    Estimate M (f_real derivative Lipschitz constant).

    Corresponds to CSBAPR.lean: HasLipschitzFderiv (L1063)
    M = sup_{x₁≠x₂} ‖Df(x₁) - Df(x₂)‖ / ‖x₁ - x₂‖

    Uses finite-difference Jacobian estimation on sampled state pairs.

    Args:
        env_step_fn: Callable s → s' (environment dynamics, one-step).
        states: [N, state_dim] sampled states.
        dt: Finite difference step size.
        n_pairs: Number of random pairs to evaluate.

    Returns:
        M estimate (float). Returns 0.0 if estimation fails.
    """
    if states.shape[0] < 2:
        return 0.0

    state_dim = states.shape[-1]
    device = states.device

    # Estimate Jacobian at each state via finite differences
    def estimate_jacobian(s):
        jac = torch.zeros(state_dim, state_dim, device=device)
        s_np = s.cpu().numpy() if isinstance(s, torch.Tensor) else s
        f_s = env_step_fn(s_np)
        for i in range(state_dim):
            s_plus = s_np.copy()
            s_plus[i] += dt
            f_plus = env_step_fn(s_plus)
            jac[:, i] = torch.tensor((f_plus - f_s) / dt, dtype=torch.float32, device=device)
        return jac

    # Sample pairs and estimate M
    N = min(states.shape[0], n_pairs * 2)
    indices = np.random.permutation(N)
    M_est = 0.0

    for k in range(min(n_pairs, N // 2)):
        i, j = indices[2 * k], indices[2 * k + 1]
        s_i, s_j = states[i], states[j]
        state_diff = torch.norm(s_i - s_j).item()
        if state_diff < 1e-8:
            continue
        jac_i = estimate_jacobian(s_i)
        jac_j = estimate_jacobian(s_j)
        jac_diff = torch.norm(jac_i - jac_j).item()
        M_est = max(M_est, jac_diff / state_diff)

    return M_est


def compute_ood_bound(actor: nn.Module,
                      x_train_boundary: torch.Tensor,
                      x_ood: torch.Tensor,
                      delta: float,
                      epsilon_emp: float,
                      n_train_samples: int,
                      M: float = 0.0,
                      confidence_delta: float = 0.05) -> dict:
    """
    Corrected OOD bound using Part X + XI + XIII.

    Corresponds to CSBAPR.lean: ood_bound_no_assumption4_nD (L1117)

    ‖π(x_ood) - f_real(x_ood)‖ ≤ δ + (ε + (L_eff + M)·‖d‖)·‖d‖

    where ε = ε_emp + generalization_gap.

    No Assumption 4 required: this bound only needs
    (A) JC at training boundary (from loss), (B) NAU architecture (L_eff),
    (C) physics smoothness (M).

    Args:
        actor: NAU_NMU_Actor with compute_L_eff() method.
        x_train_boundary: [state_dim] training domain boundary point x₀.
        x_ood: [state_dim] OOD evaluation point.
        delta: Base accuracy ‖π(x₀) - f_real(x₀)‖ at training boundary.
        epsilon_emp: Empirical Jacobian consistency error (from training loss).
        n_train_samples: Number of training samples N.
        M: Physics smoothness constant (HasLipschitzFderiv, default 0).
        confidence_delta: PAC confidence parameter δ (default 0.05).

    Returns:
        dict with bound, L_eff, gap, epsilon_total, M, d.
    """
    # Part XI: composed architecture L_eff
    if hasattr(actor, 'compute_L_eff'):
        L_eff = actor.compute_L_eff()
    elif hasattr(actor, 'lipschitz_constant'):
        L_eff = actor.lipschitz_constant
    else:
        L_eff = 0.0

    # Part XIII: generalization gap
    B = estimate_deriv_bound_B(actor)
    gap = compute_generalization_gap(B, n_train_samples, confidence_delta)
    epsilon = epsilon_emp + gap

    # Part X: corrected OOD bound (no Assumption 4)
    d = torch.norm(x_ood.float() - x_train_boundary.float()).item()
    bound = delta + (epsilon + (L_eff + M) * d) * d

    return {
        'bound': bound,
        'L_eff': L_eff,
        'gap': gap,
        'epsilon_emp': epsilon_emp,
        'epsilon_total': epsilon,
        'M': M,
        'd': d,
        'B': B,
        'n_samples': n_train_samples,
    }
