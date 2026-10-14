"""Full action-space Jacobian consistency for a frozen policy reference."""

import torch
import torch.nn as nn


def deterministic_action(policy_net: nn.Module, states: torch.Tensor) -> torch.Tensor:
    output = policy_net(states)
    if isinstance(output, tuple):
        return float(getattr(policy_net, "action_range", 1.0)) * torch.tanh(output[0])
    return output


def action_jacobian(output, states, create_graph):
    """One reverse pass per action component; do not sum distinct output rows."""
    rows = [
        torch.autograd.grad(output[:, index].sum(), states,
                            create_graph=create_graph, retain_graph=True)[0]
        for index in range(output.shape[1])
    ]
    return torch.stack(rows, dim=1)


def compute_action_anchor_loss(policy_net: nn.Module, pi_sym: nn.Module,
                               states_batch: torch.Tensor) -> torch.Tensor:
    actions = deterministic_action(policy_net, states_batch)
    with torch.no_grad():
        targets = pi_sym(states_batch)
    if actions.shape != targets.shape:
        raise ValueError("Action anchor requires matching action-space shapes")
    return (actions - targets).square().mean()


def compute_jacobian_loss(policy_net: nn.Module, pi_sym: nn.Module,
                          states_batch: torch.Tensor, input_columns=None) -> torch.Tensor:
    """Mean squared Frobenius error of normalized deterministic-action Jacobians."""
    with torch.enable_grad():
        states = states_batch.detach().requires_grad_(True)
        actions = deterministic_action(policy_net, states)
        reference_actions = pi_sym(states)
        if actions.shape != reference_actions.shape:
            raise ValueError("Jacobian targets must have the same action-space shape")
        jac_pi = action_jacobian(actions, states, create_graph=True)
        jac_sym = action_jacobian(reference_actions, states, create_graph=False).detach()
        difference = jac_pi - jac_sym
        if input_columns is not None:
            difference = difference[..., input_columns]
        return difference.square().sum(dim=(-2, -1)).mean()


def compute_jacobian_loss_nD(policy_net: nn.Module, pi_sym: nn.Module,
                             states_batch: torch.Tensor) -> torch.Tensor:
    def policy_fn(x):
        return deterministic_action(policy_net, x.unsqueeze(0)).squeeze(0)

    def reference_fn(x):
        return pi_sym(x.unsqueeze(0)).squeeze(0)

    if policy_fn(states_batch[0]).shape != reference_fn(states_batch[0]).shape:
        raise ValueError("Jacobian targets must have the same action-space shape")
    jac_pi = torch.vmap(torch.func.jacrev(policy_fn))(states_batch)
    jac_sym = torch.vmap(torch.func.jacrev(reference_fn))(states_batch).detach()
    return (jac_pi - jac_sym).square().sum(dim=(-2, -1)).mean()
