"""Conditional deterministic policy-output bounds with measured boundary errors."""

import math
import torch

from csbapr.losses.jacobian import deterministic_action, action_jacobian


def compute_ood_bound(actor, reference, x_train_boundary, x_ood):
    """No replay-average or sample-count correction is a pointwise certificate."""
    L_pi = actor.compute_L_eff() if hasattr(actor, "compute_L_eff") else math.inf
    M = reference.compute_L_eff() if hasattr(reference, "compute_L_eff") else math.inf
    d = torch.linalg.vector_norm(x_ood - x_train_boundary).item()
    if not math.isfinite(L_pi) or not math.isfinite(M):
        return {"certified": False, "bound": None, "L_eff": L_pi, "M": M, "d": d,
                "reason": "A compared map lacks a finite whole-policy derivative bound"}
    with torch.enable_grad():
        states = x_train_boundary.detach().reshape(1, -1).requires_grad_(True)
        action = deterministic_action(actor, states)
        reference_action = reference(states)
        if action.shape != reference_action.shape:
            raise ValueError("OOD bounds compare maps in the same action space")
        delta = torch.linalg.vector_norm(action - reference_action).item()
        jac_pi = action_jacobian(action, states, create_graph=False)
        jac_ref = action_jacobian(reference_action, states, create_graph=False)
        epsilon = torch.linalg.matrix_norm(jac_pi[0] - jac_ref[0], ord=2).item()
    bound = delta + epsilon * d + 0.5 * (L_pi + M) * d ** 2
    with torch.no_grad():
        actual = torch.linalg.vector_norm(
            deterministic_action(actor, x_ood.reshape(1, -1)) - reference(x_ood.reshape(1, -1))
        ).item()
    return {"certified": True, "bound": bound, "L_eff": L_pi, "M": M, "d": d,
            "delta": delta, "epsilon": epsilon, "actual_output_error": actual}
