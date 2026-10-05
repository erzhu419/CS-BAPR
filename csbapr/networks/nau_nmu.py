"""Arithmetic actor heads; whole-policy bounds require smooth features.

The quadratic branch is NMU-inspired, not the published multiplicative NMU.
Historical LeakyReLU actors remain available without a global derivative bound.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NAULayer(nn.Module):
    """
    Neural Addition Unit (addition/subtraction).
    
    Corresponds to CSBAPR.lean: NAULayer (L799), weights ∈ {-1, 0, 1}
    Reference: stable-nalu/stable_nalu/layer/re_regualized_linear_nac.py
    
    Key design: clamp + regularization (reference impl) rather than STE.
    - clamp(-1,1): gradient=1 inside, gradient=0 outside (progressive constraint)
    - regularization drives W toward {-1,0,1} (progressive discretization)
    - More stable than STE: STE locks weights too early in training
    
    Fixed-weight input derivative curvature is zero, before nonlinear composition.
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
    NMU-inspired quadratic branch f(x) = c*x**2, with curvature 2*max(abs(c)).
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
    Mixed arithmetic Gaussian actor with historical or smooth feature activations.

    The derivative certificate covers the deterministic action tanh(mean),
    not the sampled Gaussian policy, log-standard-deviation head, or return.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 log_std_min: float = -20, log_std_max: float = 2,
                 activation: str = "leaky_relu"):
        super().__init__()
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.mean_clip = 0.0

        if activation not in ("leaky_relu", "tanh"):
            raise ValueError("Arithmetic features support leaky_relu or tanh")
        self.activation_name = activation
        Act = nn.Tanh if activation == "tanh" else lambda: nn.LeakyReLU(0.01)
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            Act(),
            nn.Linear(hidden_dim, hidden_dim),
            Act(),
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
        if getattr(self, "mean_clip", 0.0):
            mean = torch.clamp(mean, -float(self.mean_clip), float(self.mean_clip))

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

    def l_eff_surrogate_loss(self, target: float = 200.0, eps: float = 1e-6):
        """Smooth actor: whole-policy bound with Frobenius norm upper bounds.

        Historical actor: quadratic-branch penalty only, not a certificate.
        """
        if self.activation_name == "tanh" and not self.mean_clip:
            surrogate = self.derivative_bounds(norm="fro")["L_pi"]
            penalty = torch.relu(torch.log(surrogate + eps) - math.log(target + eps)) ** 2
            return penalty, surrogate
        alpha = torch.sigmoid(self.mix_alpha)
        l_h = (1.0 - alpha) * 2.0 * torch.linalg.vector_norm(self.nmu_head.coeff, ord=2)
        l_h = l_h * torch.linalg.matrix_norm(self.nmu_proj.weight, ord="fro") ** 2
        k_g = torch.ones((), dtype=l_h.dtype, device=l_h.device)
        for module in self.feature_net:
            if isinstance(module, nn.Linear):
                k_g = k_g * torch.linalg.matrix_norm(module.weight, ord="fro")
        surrogate = l_h * k_g * k_g
        target_t = torch.as_tensor(float(target), dtype=surrogate.dtype, device=surrogate.device)
        penalty = torch.relu(torch.log(surrogate + eps) - torch.log(target_t + eps)) ** 2
        return penalty, surrogate

    @property
    def lipschitz_constant(self) -> float:
        """Overall L for the NMU component (DEPRECATED: use compute_L_eff)."""
        return self.nmu_head.lipschitz_constant

    def compute_L_eff(self) -> float:
        """Whole deterministic-policy curvature bound, or infinity if uncertified."""
        if self.activation_name != "tanh" or self.mean_clip:
            return math.inf
        with torch.no_grad():
            return self.derivative_bounds()["L_pi"].item()

    def derivative_bounds(self, norm=2):
        """Global bounds for tanh features, quadratic projection, and action squash."""
        if self.activation_name != "tanh" or self.mean_clip:
            raise ValueError("Finite bounds require tanh features and no hard mean clamp")
        matrix_norm = lambda weight: torch.linalg.matrix_norm(weight.double(), ord=norm)
        curvature = 4.0 / (3.0 * math.sqrt(3.0))
        w1 = matrix_norm(self.feature_net[0].weight)
        w2 = matrix_norm(self.feature_net[2].weight)
        k_g = w2 * w1
        l_g = curvature * (k_g ** 2 + w2 * w1 ** 2)
        alpha = torch.sigmoid(self.mix_alpha.double())
        u_norm = matrix_norm(self.nmu_proj.weight)
        c_max = self.nmu_head.coeff.double().abs().max()
        l_h = 2 * (1 - alpha) * c_max * u_norm ** 2
        k_h = alpha * matrix_norm(self.nau_head.W.clamp(-1, 1))
        k_h = k_h + l_h * math.sqrt(self.nau_head.in_features)
        k_pi = k_h * k_g
        l_pi = curvature * k_pi ** 2 + l_h * k_g ** 2 + k_h * l_g
        return {"K_G": k_g, "L_G": l_g, "K_H": k_h, "L_H": l_h,
                "K_pi": k_pi, "L_pi": l_pi}


class KAN_Actor(nn.Module):
    """
    KAN-based actor replacing NAU/NMU heads with learnable spline activations.

    Motivation: KAN combines function discovery + smooth extrapolation in a
    single architecture via per-edge splines, replacing SINDy(discovery) +
    NAU/NMU(extrapolation). Splines are C^k smooth by construction, giving
    bounded derivatives naturally — aligns with CS-BAPR's Lipschitz needs
    without the rigid ±1/0 weight constraint that conflicts with bc_loss.

    Architecture:
      KAN([state_dim, hidden, hidden, action_dim]) → mean
      MLP(hidden) → log_std

    Interface matches NAU_NMU_Actor: forward/evaluate/get_action/regularization_loss.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64,
                 log_std_min: float = -20, log_std_max: float = 2,
                 grid_size: int = 5, spline_order: int = 3):
        super().__init__()
        from efficient_kan import KAN
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # KAN for mean: splines on every edge provide smooth OOD extrapolation
        self.kan_mean = KAN(
            layers_hidden=[state_dim, hidden_dim, action_dim],
            grid_size=grid_size,
            spline_order=spline_order,
        )

        # Separate MLP for log_std (KAN overkill on scalar spread output)
        self.log_std_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim, action_dim),
        )
        nn.init.uniform_(self.log_std_net[-1].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_net[-1].bias, -3e-3, 3e-3)

    def forward(self, state: torch.Tensor):
        mean = self.kan_mean(state)
        log_std = self.log_std_net(state)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def evaluate(self, state: torch.Tensor, epsilon: float = 1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(0, 1)
        z = normal.sample(mean.shape).to(state.device)
        action_0 = torch.tanh(mean + std * z)
        action = action_0
        log_prob = (torch.distributions.Normal(mean, std)
                    .log_prob(mean + std * z)
                    - torch.log(1. - action_0.pow(2) + epsilon))
        log_prob = log_prob.sum(dim=1)
        return action, log_prob, z, mean, log_std

    def get_action(self, state: torch.Tensor, deterministic: bool = False):
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
        """KAN's built-in spline L1 + entropy regularization."""
        return self.kan_mean.regularization_loss(
            regularize_activation=1.0, regularize_entropy=1.0
        )

    def compute_L_eff(self) -> float:
        """No whole-policy spline derivative certificate is implemented."""
        return math.inf
