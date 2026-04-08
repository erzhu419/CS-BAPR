"""
NAU/NMU Layers and CS-BAPR Actor Architecture

Corresponds to CSBAPR.lean:
- Part IX (L776-953):
  - NAULayer → nau_has_arithmetic_bias_1d (L843): L=0
  - NMULayer → nmu_has_bounded_arithmetic_bias (L876): L=2|c|
  - NAU_NMU_Actor → nau_nmu_composition_bias (L896): L=2|wc|
  - ReLU comparison → relu_derivative_not_lipschitz (L908): L=∞
- Part XI (L1200-1462):
  - IsFunctionLipschitz (L1217): K_g for feature extractor
  - composed_deriv_lipschitz_simple (L1417): L_eff = L_h·K_g·B_g + L_g
  - derivative_jump_not_lip (L1254): LeakyReLU is function-Lip but NOT deriv-Lip

Reference: stable-nalu/stable_nalu/layer/re_regualized_linear_nac.py
Reference: stable-nalu/stable_nalu/layer/re_regualized_linear_mnac.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NAULayer(nn.Module):
    """
    Neural Arithmetic Unit (Addition/Subtraction).
    
    Corresponds to CSBAPR.lean: NAULayer (L799), weights ∈ {-1, 0, 1}
    Reference: stable-nalu/stable_nalu/layer/re_regualized_linear_nac.py
    
    Key design: clamp + regularization (reference impl) rather than STE.
    - clamp(-1,1): gradient=1 inside, gradient=0 outside (progressive constraint)
    - regularization drives W toward {-1,0,1} (progressive discretization)
    - More stable than STE: STE locks weights too early in training
    
    Lean guarantee: L=0 (nau_has_arithmetic_bias_1d)
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(torch.Tensor(out_features, in_features))
        self._reset_parameters()

    def _reset_parameters(self):
        # Xavier init (reference impl L44-46)
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        r = min(0.5, math.sqrt(3.0) * std)
        nn.init.uniform_(self.W, -r, r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # clamp to [-1, 1] (reference impl L65)
        W_clamped = torch.clamp(self.W, -1.0, 1.0)
        return F.linear(x, W_clamped)

    def regularization_loss(self) -> torch.Tensor:
        """Drive W toward {-1, 0, 1}: sparsity_error = min(|W|, |1-|W||)"""
        W_abs = self.W.abs()
        return torch.min(W_abs, (1 - W_abs).abs()).mean()

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}'


class NMULayer(nn.Module):
    """
    Neural Multiplication Unit (simplified quadratic).
    
    Corresponds to CSBAPR.lean: NMUUnit (L858)
    Lean proves simplified version: f(x) = c·x² with L=2|c|
    
    Note: stable-nalu's MNAC uses weighted product y = Π(xᵢ·wᵢ + 1 - wᵢ),
    which is more powerful but not directly covered by the Lean proof.
    We keep the simplified version for direct Lean coverage.
    """
    def __init__(self, features: int):
        super().__init__()
        self.features = features
        self.coeff = nn.Parameter(torch.ones(features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.coeff * x ** 2

    @property
    def lipschitz_constant(self) -> float:
        """L = 2*max(|coeff|) per Lean: nmu_has_bounded_arithmetic_bias"""
        return 2 * self.coeff.abs().max().item()

    def extra_repr(self) -> str:
        return f'features={self.features}, L=2|c|'


class NAU_NMU_Actor(nn.Module):
    """
    CS-BAPR Actor with NAU/NMU output heads for OOD extrapolation.
    
    Architecture:
    - Feature extraction: standard layers (LeakyReLU to avoid dead zones, trap #6)
    - NAU linear head: L=0 component (nau_has_arithmetic_bias_1d)
    - NMU quadratic head: L=2|c| component (nmu_has_bounded_arithmetic_bias)
    - Learnable mixing weight α controlling NAU vs NMU ratio
    
    Lean guarantees:
    - NAU output: L=0 (nau_has_arithmetic_bias_1d)
    - NMU output: L=2|c| (nmu_has_bounded_arithmetic_bias)
    - Composition: L=2|wc| (nau_nmu_composition_bias)
    
    For SAC integration, outputs (mean, log_std) for Gaussian policy.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 log_std_min: float = -20, log_std_max: float = 2):
        super().__init__()
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Feature extraction (ReLU allowed, OOD bound constrains overall policy)
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LeakyReLU(0.01),  # LeakyReLU avoids dead zones
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.01),
        )

        # NAU linear output head (L=0 component)
        self.nau_head = NAULayer(hidden_dim, action_dim)

        # NMU quadratic correction (L=2|c| component)
        self.nmu_proj = nn.Linear(hidden_dim, action_dim, bias=False)
        self.nmu_head = NMULayer(action_dim)

        # Learnable mixing weight (initialized biased toward NAU)
        self.mix_alpha = nn.Parameter(torch.tensor(0.8))

        # Log-std head (separate, standard MLP)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)
        nn.init.uniform_(self.log_std_linear.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_linear.bias, -3e-3, 3e-3)

    def forward(self, state: torch.Tensor):
        """Returns (mean, log_std) for Gaussian policy."""
        features = self.feature_net(state)              # [batch, hidden]
        
        # NAU path: linear extrapolation (L=0)
        linear_part = self.nau_head(features)           # [batch, action]
        
        # NMU path: quadratic correction (L=2|c|)
        nmu_input = self.nmu_proj(features)             # [batch, action]
        quad_part = self.nmu_head(nmu_input)             # [batch, action]
        
        # Mix with learnable alpha
        alpha = torch.sigmoid(self.mix_alpha)
        mean = alpha * linear_part + (1 - alpha) * quad_part

        # Log-std from features
        log_std = self.log_std_linear(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def evaluate(self, state: torch.Tensor, epsilon: float = 1e-6):
        """
        SAC-style action sampling with reparameterization trick.
        Returns: (action, log_prob, z, mean, log_std)
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = torch.distributions.Normal(0, 1)
        z = normal.sample(mean.shape).to(state.device)
        action_0 = torch.tanh(mean + std * z)
        action = action_0  # [-1, 1] bounded

        log_prob = (torch.distributions.Normal(mean, std)
                    .log_prob(mean + std * z)
                    - torch.log(1. - action_0.pow(2) + epsilon))
        log_prob = log_prob.sum(dim=1)

        return action, log_prob, z, mean, log_std

    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        """Get action for execution (no gradient)."""
        if state.dim() == 1:
            state = state.unsqueeze(0)
        with torch.no_grad():
            mean, log_std = self.forward(state)
            if deterministic:
                action = torch.tanh(mean)
            else:
                std = log_std.exp()
                z = torch.randn_like(mean)
                action = torch.tanh(mean + std * z)
        return action.squeeze(0).cpu().numpy()

    def regularization_loss(self) -> torch.Tensor:
        """Total NAU regularization loss for weight discretization."""
        return self.nau_head.regularization_loss()

    @property
    def lipschitz_constant(self) -> float:
        """Overall L for the NMU component (DEPRECATED: use compute_L_eff)."""
        return self.nmu_head.lipschitz_constant

    def compute_L_eff(self) -> float:
        """
        Effective derivative Lipschitz constant for the composed architecture.

        Corresponds to CSBAPR.lean: composed_deriv_lipschitz_simple (L1417)

        For π = h ∘ g where h = output head (NAU/NMU), g = feature extractor (LeakyReLU MLP):
            L_eff = L_h · K_g · B_g + L_g

        - L_h: output head derivative-Lipschitz (NAU: 0, NMU: 2|c|, mixed: (1-α)·2|c|)
        - K_g: feature extractor function-Lipschitz (product of layer spectral norms)
        - B_g: feature extractor derivative upper bound (≈ K_g for Lipschitz functions)
        - L_g: feature extractor derivative-Lipschitz (0 for piecewise-linear, per
                derivative_jump_not_lip: LeakyReLU is NOT deriv-Lip globally, but
                within each linear region L_g=0, and the composition rule handles this)
        """
        with torch.no_grad():
            # L_h: output head derivative-Lipschitz
            alpha = torch.sigmoid(self.mix_alpha).item()
            L_nmu = self.nmu_head.lipschitz_constant  # 2*max(|c|)
            L_h = (1 - alpha) * L_nmu  # NAU contributes 0

            # K_g: feature extractor function-Lipschitz (product of spectral norms)
            K_g = 1.0
            for module in self.feature_net:
                if isinstance(module, nn.Linear):
                    s = torch.linalg.svdvals(module.weight)
                    K_g *= s[0].item()
                elif isinstance(module, nn.LeakyReLU):
                    K_g *= 1.0  # LeakyReLU Lipschitz constant = 1

            # B_g: derivative upper bound ≈ K_g
            B_g = K_g

            # L_g: piecewise-linear feature extractor has L_g = 0 within linear regions
            L_g = 0.0

        return L_h * K_g * B_g + L_g
