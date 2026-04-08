"""
Standard MLP Gaussian Policy for CS-BAPR (fallback when use_nau_actor=False).

Adapted from BAPR/sac_ensemble_bapr.py PolicyNetwork.
Simplified for continuous-state Gymnasium environments.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class GaussianPolicy(nn.Module):
    """
    SAC-style Gaussian policy with reparameterization trick.
    
    Adapted from BAPR's PolicyNetwork but without categorical embeddings.
    Used as fallback when use_nau_actor=False.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 action_range: float = 1.0,
                 log_std_min: float = -20, log_std_max: float = 2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_range = action_range
        self.action_dim = action_dim

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

        # Small init for output heads
        nn.init.uniform_(self.mean_linear.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.mean_linear.bias, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_linear.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_linear.bias, -3e-3, 3e-3)

    def forward(self, state: torch.Tensor):
        features = self.net(state)
        mean = self.mean_linear(features)
        log_std = self.log_std_linear(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def evaluate(self, state: torch.Tensor, epsilon: float = 1e-6):
        """SAC reparameterization trick. Returns (action, log_prob, z, mean, log_std)."""
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample(mean.shape).to(state.device)
        action_0 = torch.tanh(mean + std * z)
        action = self.action_range * action_0

        log_prob = (Normal(mean, std).log_prob(mean + std * z)
                    - torch.log(1. - action_0.pow(2) + epsilon)
                    - np.log(self.action_range))
        log_prob = log_prob.sum(dim=1)

        return action, log_prob, z, mean, log_std

    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        with torch.no_grad():
            mean, log_std = self.forward(state)
            if deterministic:
                action = self.action_range * torch.tanh(mean)
            else:
                std = log_std.exp()
                z = torch.randn_like(mean)
                action = self.action_range * torch.tanh(mean + std * z)
        return action.squeeze(0).cpu().numpy()
