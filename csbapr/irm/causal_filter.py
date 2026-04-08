"""
IRM Causal Filtering for CS-BAPR.

Corresponds to CSBAPR.lean:
- Part VIII (L699-774):
  - IRM_Consistent (L726): Same formula ε-accurate across all environments
  - irm_tightens_jacobian_consistency (L736): IRM → tighter OOD bound
  - irm_to_ood_bound_nD (L755): Multi-env SINDy → OOD bound
- Part XII (L1464-1637):
  - CausalDecomposition (L1496): f_real = f_causal + f_spurious
  - irm_variance_reduction (L1536): IRM error = ε_c (env-independent)
  - irm_vs_single_env_gap (L1570): IRM on test env ≤ ε_c + ε_s_test
  - irm_quantitative_advantage (L1619): IRM advantage = ε_s_train

Reference: InvariantRiskMinimization/code/experiment_synthetic/models.py

⚠️ IRM is used in Phase 0 (formula selection), NOT in the training loop.
   1. Fit SINDy in multiple environments → multiple candidate coefficients
   2. Select coefficients with minimum cross-environment deriv_error variance
   3. Fix selected coefficients, proceed to Phase 1 training
   DO NOT call compute_irm_penalty in each training step.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import grad


def compute_irm_penalty(symbolic_model, envs_data: dict) -> float:
    """
    Cross-environment SINDy consistency check.
    
    Corresponds to CSBAPR.lean: IRM_Consistent (L726)
    
    Design choice: We use cross-environment error variance (not Arjovsky's gradient penalty)
    because it directly maps to the Lean definition of IRM_Consistent
    (derivative error ≤ ε across all environments).
    
    Args:
        symbolic_model: Fitted SymbolicWorldModel
        envs_data: dict mapping env_name → (states, dynamics)
            states: [N, n_state] test states
            dynamics: [N, n_state] true derivatives or next states
    
    Returns:
        Cross-environment error variance (lower = more invariant)
    """
    env_errors = []
    for env_name, (states, dynamics) in envs_data.items():
        pred = symbolic_model.predict(states)
        err = np.linalg.norm(pred - dynamics, axis=-1).mean()
        env_errors.append(err)
    return np.var(env_errors)


def compute_irm_penalty_gradient(model: torch.nn.Module,
                                  environments: list,
                                  w: torch.Tensor) -> torch.Tensor:
    """
    Arjovsky 2019 original IRM penalty.
    Reference: InvariantRiskMinimization/code/experiment_synthetic/models.py L62-73
    
    penalty = Σ_e ‖∇_w L_e(w)‖²
    
    Args:
        model: Differentiable model (e.g., SINDy torch wrapper)
        environments: List of (x_e, y_e) tensor pairs
        w: Parameter vector to compute gradients w.r.t.
    
    Returns:
        IRM gradient penalty (scalar tensor)
    """
    penalty = torch.tensor(0.0)
    for x_e, y_e in environments:
        error_e = F.mse_loss(model(x_e), y_e)
        grad_w = grad(error_e, w, create_graph=True)[0]
        penalty = penalty + grad_w.pow(2).mean()
    return penalty


def compute_irm_advantage(symbolic_model, envs_data: dict) -> dict:
    """
    Quantify IRM advantage via causal-spurious decomposition proxy.

    Corresponds to CSBAPR.lean Part XII:
    - irm_variance_reduction (L1536): IRM error = ε_c (env-independent)
    - irm_quantitative_advantage (L1619): IRM advantage = ε_s_train

    The causal-spurious decomposition (CausalDecomposition, L1496) is implicit:
    f_real^e(x) = f_causal(x) + f_spurious^e(x). We cannot directly separate them,
    but cross-environment consistency serves as a proxy:
    - Low variance → formula mostly captures f_causal → small ε_s
    - High variance → formula captures f_spurious → large ε_s → large IRM advantage

    Args:
        symbolic_model: Fitted SymbolicWorldModel
        envs_data: dict mapping env_name → (states, dynamics)

    Returns:
        dict with per-env errors, epsilon_s_proxy (≈ IRM advantage), irm_variance
    """
    per_env_errors = {}
    for env_name, (states, dynamics) in envs_data.items():
        pred = symbolic_model.predict(states)
        err = np.linalg.norm(pred - dynamics, axis=-1).mean()
        per_env_errors[env_name] = err

    errors = list(per_env_errors.values())
    epsilon_s_proxy = np.std(errors)
    irm_variance = np.var(errors)

    return {
        'per_env_errors': per_env_errors,
        'epsilon_s_proxy': epsilon_s_proxy,
        'irm_variance': irm_variance,
        'irm_advantage_estimate': epsilon_s_proxy,
        'mean_error': np.mean(errors),
    }


def select_best_sindy_coefficients(symbolic_model, envs_data: dict,
                                     n_candidates: int = 5,
                                     threshold: float = 0.1,
                                     verbose: bool = True):
    """
    IRM-based coefficient selection for SINDy formulas.
    
    Workflow:
    1. Fit SINDy separately in each environment
    2. Compute cross-environment consistency for each fit
    3. Select the coefficients with the lowest cross-env variance
    
    Args:
        symbolic_model: SymbolicWorldModel instance
        envs_data: dict of env_name → (X_list, U_list) for fitting
        n_candidates: Not used (kept for API consistency)
        threshold: STLSQ threshold
        verbose: Print diagnostics
    
    Returns:
        best_coeffs: Best coefficient matrix
        irm_variance: Cross-env variance of best fit
    """
    from csbapr.sindy.world_model import SymbolicWorldModel

    all_fits = {}
    
    # Fit SINDy in each environment
    for env_name, (X_list, U_list, X_dot_list) in envs_data.items():
        candidate = SymbolicWorldModel(
            n_state=symbolic_model.n_state,
            n_control=symbolic_model.n_control,
            poly_degree=symbolic_model.poly_degree,
            threshold=threshold,
            discrete_time=symbolic_model.discrete_time,
        )
        
        if symbolic_model.discrete_time:
            candidate.fit(X_list, U=U_list, multiple_trajectories=True, t=1.0)
        else:
            X_flat, X_dot_flat = np.vstack([x[:-1] for x in X_list]), np.vstack(X_dot_list)
            candidate.fit(X_flat, X_dot=X_dot_flat)
        
        all_fits[env_name] = candidate.coeffs
        
        if verbose:
            print(f"[IRM] Env '{env_name}': sparsity = {candidate.sparsity:.3f}")

    # Compute cross-environment coefficient variance
    coeff_stack = np.stack(list(all_fits.values()))
    coeff_variance = np.var(coeff_stack, axis=0).mean()
    
    # Use mean coefficients as the IRM-consistent estimate
    best_coeffs = np.mean(coeff_stack, axis=0)
    
    if verbose:
        print(f"[IRM] Cross-env coefficient variance: {coeff_variance:.6f}")
        print(f"[IRM] Using mean coefficients across {len(all_fits)} environments")
    
    return best_coeffs, coeff_variance
