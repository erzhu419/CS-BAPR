"""
Ensemble Q-Network (Vectorized Critic) for CS-BAPR.

Directly adapted from BAPR/sac_ensemble_bapr.py VectorizedLinear + VectorizedCritic.
Simplified to work with continuous state (no embedding layer needed for Gymnasium envs).
"""

import math
import torch
import torch.nn as nn


class VectorizedLinear(nn.Module):
    """Parallelized linear layer across ensemble members."""
    def __init__(self, in_features: int, out_features: int, ensemble_size: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size

        self.weight = nn.Parameter(torch.empty(ensemble_size, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.ensemble_size):
            nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [ensemble, batch, in] or [batch, in]
        return x @ self.weight + self.bias


class EnsembleQNet(nn.Module):
    """
    Vectorized ensemble Q-network for CS-BAPR.
    
    Adapted from BAPR's VectorizedCritic but simplified for continuous-state
    environments (no categorical embedding needed).
    
    Input: (state, action) → Q-values [ensemble_size, batch]
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 num_critics: int = 5):
        super().__init__()
        self.num_critics = num_critics
        
        self.critic = nn.Sequential(
            VectorizedLinear(state_dim + action_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, 1, num_critics),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [batch, state_dim]
            action: [batch, action_dim]
        Returns:
            q_values: [num_critics, batch]
        """
        state_action = torch.cat([state, action], dim=-1)
        # Expand for ensemble: [batch, dim] → [num_critics, batch, dim]
        state_action = state_action.unsqueeze(0).repeat_interleave(self.num_critics, dim=0)
        q_values = self.critic(state_action).squeeze(-1)  # [num_critics, batch]
        return q_values
