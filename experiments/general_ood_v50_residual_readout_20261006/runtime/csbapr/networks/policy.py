"""
Standard MLP Gaussian Policy for CS-BAPR (fallback when use_nau_actor=False).

Adapted from BAPR/sac_ensemble_bapr.py PolicyNetwork.
Simplified for continuous-state Gymnasium environments.
"""

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class GaussianPolicy(nn.Module):
    """
    SAC-style Gaussian policy with reparameterization trick.

    Adapted from BAPR's PolicyNetwork but without categorical embeddings.
    Used as fallback when use_nau_actor=False.

    Args:
        activation: hidden-layer activation function. 'relu' (default) for the
            standard SAC tanh-Gaussian baseline (with ReLU hidden + tanh output);
            'tanh' / 'gelu' / 'softplus' for smooth-activation hidden layers,
            which can give a finite derivative-Lipschitz constant for fixed
            finite weights; the constant still depends on those weights.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 action_range: float = 1.0,
                 log_std_min: float = -20, log_std_max: float = 2,
                 activation: str = "relu"):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_range = action_range
        self.action_dim = action_dim
        self.activation_name = activation

        act_map = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "gelu": nn.GELU,
            "softplus": nn.Softplus,
        }
        if activation not in act_map:
            raise ValueError(f"Unsupported activation: {activation}. Choose from {list(act_map)}")
        Act = act_map[activation]

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            Act(),
            nn.Linear(hidden_dim, hidden_dim),
            Act(),
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

    def compute_L_eff(self):
        """Whole deterministic tanh-MLP bound; other activations are uncertified."""
        if self.activation_name != "tanh":
            return math.inf
        with torch.no_grad():
            curvature = 4.0 / (3.0 * math.sqrt(3.0))
            w1 = torch.linalg.matrix_norm(self.net[0].weight.double(), ord=2).item()
            w2 = torch.linalg.matrix_norm(self.net[2].weight.double(), ord=2).item()
            k_g = w2 * w1
            l_g = curvature * (k_g ** 2 + w2 * w1 ** 2)
            k_h = torch.linalg.matrix_norm(self.mean_linear.weight.double(), ord=2).item()
            return abs(self.action_range) * (curvature * (k_h * k_g) ** 2 + k_h * l_g)


class EquilibriumGaussianPolicy(GaussianPolicy):
    """Tanh Gaussian actor whose deterministic mean is zero at the origin.

    Used where zero state and zero exogenous force define the task equilibrium.
    The std head is unchanged. Centering is differentiable in both mean terms;
    it adds a state-constant shift, so the inherited curvature bound still holds.
    """

    def forward(self, state: torch.Tensor):
        mean, log_std = super().forward(state)
        origin_mean, _ = super().forward(torch.zeros_like(state[:1]))
        return mean - origin_mean, log_std
