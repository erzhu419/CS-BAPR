"""Critic update with a detached action-reference Jacobian discrepancy.

The discrepancy is a batch scalar shared by the run-length modes. Holding
this field fixed preserves a fixed-operator contraction; it does not prove
invariance or convergence of changing actor-critic gradient updates.
"""

import torch
import torch.nn as nn

from csbapr.losses.jacobian import compute_jacobian_loss


def compute_q_loss_csbapr(
    soft_q_net: nn.Module,
    target_soft_q_net: nn.Module,
    policy_net: nn.Module,
    pi_sym_torch: nn.Module,
    state: torch.Tensor,
    action: torch.Tensor,
    reward: torch.Tensor,
    next_state: torch.Tensor,
    done: torch.Tensor,
    new_next_action: torch.Tensor,
    next_log_prob: torch.Tensor,
    reg_norm: torch.Tensor,
    belief_tracker,
    alpha: float,
    gamma: float,
    weight_reg: float,
    weight_sym: float,
    beta_ood: float,
    penalty_decay_rate: float,
    device: str = 'cpu',
    batch_size: int = 256,
    enable_min_q_target: bool = True,
):
    """
    CS-BAPR Q-loss = BA-PR Q-loss + λ_sym · Γ_sym
    
    Γ_sym = compute_jacobian_loss(policy, pi_sym, next_state)
    
    Adapted from BAPR/sac_ensemble_bapr.py:compute_q_loss_bapr
    with the addition of symbolic consistency penalty.
    
    Returns:
        loss: Total Q-loss (scalar)
        predicted_q: [ensemble, batch]
        ood_loss: OOD std penalty (scalar)
        weighted_lambda: scalar (belief-weighted penalty strength)
        sym_penalty_val: scalar (Γ_sym value for logging)
    """
    predicted_q = soft_q_net(state, action)              # [ensemble, batch]
    target_q_next = target_soft_q_net(next_state, new_next_action)  # [ensemble, batch]

    num_critics = soft_q_net.num_critics

    next_log_prob_expanded = next_log_prob.unsqueeze(0).repeat(num_critics, 1)

    # ===== BA-PR: belief-weighted penalty =====
    belief = torch.tensor(belief_tracker.belief, dtype=torch.float32, device=device)
    penalty_schedule = torch.exp(
        -penalty_decay_rate * torch.arange(belief_tracker.max_H, dtype=torch.float32, device=device)
    )
    weighted_lambda = (belief * penalty_schedule).sum()

    reg_norm_expanded = reg_norm.unsqueeze(-1).repeat(1, batch_size)

    # ===== CS-BAPR NEW: symbolic consistency penalty =====
    # sym_penalty: scalar (batch-aggregated Jacobian deviation)
    sym_penalty = compute_jacobian_loss(policy_net, pi_sym_torch, next_state).detach()

    # Group-A1 fix: conservative target via elementwise min over ensemble.
    # When enable_min_q_target = False, fall back to per-critic targets
    # (vanilla SAC ensemble behavior).
    if enable_min_q_target:
        target_q_min = target_q_next.min(dim=0, keepdim=True)[0]  # [1, batch]
        target_q_next = target_q_min.expand(num_critics, -1)       # [ensemble, batch]

    # Target Q with all penalties
    # Note: epistemic penalty moved to policy loss (BAPR design, prevents Q-std explosion)
    target_q_next = (target_q_next
                     - alpha * next_log_prob_expanded      # SAC entropy
                     + weight_reg * reg_norm_expanded       # +κ_ale (aleatoric penalty)
                     - weight_sym * sym_penalty)            # -λ_sym·Γ_sym ← CS-BAPR

    target_q_value = reward + (1 - done) * gamma * target_q_next.unsqueeze(-1)

    ood_loss = predicted_q.std(0).mean()
    q_loss = nn.MSELoss()(predicted_q, target_q_value.squeeze(-1).detach())
    loss = q_loss + beta_ood * ood_loss

    return loss, predicted_q, ood_loss, weighted_lambda.item(), sym_penalty.item()
