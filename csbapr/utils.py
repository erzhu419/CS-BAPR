"""
CS-BAPR Utilities — soft_update, compute_reg_norm, ReplayBuffer.

Adapted from BAPR/sac_ensemble_bapr.py.
"""

import numpy as np
import torch
import torch.nn as nn


def soft_update(target_net: nn.Module, source_net: nn.Module, tau: float = 0.005):
    """Soft-update target network parameters. Adapted from BAPR."""
    for target_param, param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def compute_reg_norm(model: nn.Module) -> torch.Tensor:
    """
    Compute L1 regularization norm for critic parameters.
    Adapted from BAPR/sac_ensemble_bapr.py:compute_reg_norm
    
    Returns: [ensemble_size] tensor of per-critic L1 norms
    """
    weight_norms = []
    bias_norms = []
    for name, param in model.named_parameters():
        if 'critic' in name:
            if 'weight' in name:
                weight_norms.append(torch.norm(param, p=1, dim=[1, 2]))
            elif 'bias' in name:
                bias_norms.append(torch.norm(param, p=1, dim=[1, 2]))
    
    if not weight_norms:
        return torch.zeros(1, device=next(model.parameters()).device)
    
    reg = torch.sum(torch.stack(weight_norms), dim=0)
    if bias_norms:
        reg = reg + torch.sum(torch.stack(bias_norms[:-1]), dim=0)
    return reg


class ReplayBuffer:
    """
    Standard replay buffer.
    Adapted from BAPR/sac_ensemble_bapr.py:ReplayBuffer.
    """
    def __init__(self, capacity: int = 1_000_000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        indices = np.random.randint(0, len(self.buffer), size=batch_size)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.stack(states), np.stack(actions),
                np.array(rewards, dtype=np.float32),
                np.stack(next_states),
                np.array(dones, dtype=np.float32))

    @property
    def size(self) -> int:
        return len(self.buffer)

    def __len__(self):
        return len(self.buffer)
