"""NAU/NMU layers and Actor — JAX/Flax NNX implementation.

Corresponds to CSBAPR.lean Part IX (L776-953):
- nau_lipschitz (L843): L_NAU = 0, weights clamped to [-1, 1]
- nmu_has_bounded_arithmetic_bias (L876): L_NMU = 2|c|

Reference: stable-nalu/stable_nalu/layer/re_regualized_linear_nac.py
"""
import jax
import jax.numpy as jnp
from flax import nnx

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class NAULayer(nnx.Module):
    """Neural Arithmetic Unit (Addition/Subtraction).

    W clamped to [-1, 1] via hard clamp after each forward.
    Lean: nau_lipschitz → L = 0 (additive operation is Lipschitz-0 w.r.t. OOD).
    """
    def __init__(self, in_features: int, out_features: int, *, rngs: nnx.Rngs):
        # Initialize in [-1, 1] range
        self.W = nnx.Param(
            jax.random.uniform(rngs.params(), (in_features, out_features),
                               minval=-1.0, maxval=1.0))

    def __call__(self, x):
        W_clamped = jnp.clip(self.W.value, -1.0, 1.0)
        return x @ W_clamped

    def regularization_loss(self):
        """Sparsity regularization: push weights toward {-1, 0, 1}."""
        W = self.W.value
        return jnp.mean(jnp.minimum(jnp.abs(W), 1.0 - jnp.abs(W)))


class NMULayer(nnx.Module):
    """Neural Multiplication Unit (simplified quadratic).

    f(x) = c * x², where c is a learnable scalar.
    Lean: nmu_has_bounded_arithmetic_bias → L = 2|c|.
    """
    def __init__(self, n_features: int, *, rngs: nnx.Rngs):
        self.c = nnx.Param(jnp.ones((n_features,)) * 0.5)

    def __call__(self, x):
        return self.c.value * x ** 2

    @property
    def lipschitz_constant(self):
        return 2.0 * jnp.abs(self.c.value).max()


class NAU_NMU_Policy(nnx.Module):
    """CS-BAPR policy with NAU/NMU arithmetic inductive bias.

    Architecture:
    - Feature extraction: obs → MLP (LeakyReLU) → h
    - NAU head: h → linear combination (addition)
    - NMU head: h → quadratic features (multiplication)
    - Mix: α·NAU(h) + (1-α)·NMU(h)  (learnable α via sigmoid)
    - Output: mean, log_std for squashed Gaussian

    Optionally accepts context vector (ep_tensor) for ESCP/BAPR.
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256,
                 ep_dim: int = 0, n_layers: int = 2, *, rngs: nnx.Rngs):
        self.ep_dim = ep_dim
        input_dim = obs_dim + ep_dim
        self.act_dim = act_dim

        # Feature extraction MLP
        layers = []
        in_d = input_dim
        for _ in range(n_layers):
            layers.append(nnx.Linear(in_d, hidden_dim, rngs=rngs))
            in_d = hidden_dim
        self.layers = nnx.List(layers)
        self.n_hidden = n_layers

        # NAU head (linear arithmetic)
        self.nau = NAULayer(hidden_dim, act_dim, rngs=rngs)

        # NMU head (quadratic arithmetic) — project to act_dim first
        self.nmu_proj = nnx.Linear(hidden_dim, act_dim, rngs=rngs)
        self.nmu = NMULayer(act_dim, rngs=rngs)

        # Mixing coefficient (learnable)
        self.alpha_logit = nnx.Param(jnp.zeros(act_dim))

        # Log-std head
        self.log_std_head = nnx.Linear(hidden_dim, act_dim, rngs=rngs)

    def __call__(self, obs, ep_tensor=None):
        """Returns (mean, log_std) of the squashed Gaussian."""
        if ep_tensor is not None and self.ep_dim > 0:
            x = jnp.concatenate([obs, ep_tensor], axis=-1)
        else:
            x = obs

        # Feature extraction
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.n_hidden:
                x = nnx.leaky_relu(x, negative_slope=0.01)

        h = x  # [batch, hidden_dim]

        # NAU + NMU heads
        nau_out = self.nau(h)                             # [batch, act_dim]
        nmu_out = self.nmu(self.nmu_proj(h))              # [batch, act_dim]

        # Mix: α·NAU + (1-α)·NMU
        alpha = jax.nn.sigmoid(self.alpha_logit.value)    # [act_dim]
        mean = alpha * nau_out + (1 - alpha) * nmu_out

        log_std = self.log_std_head(h)
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)

        return mean, log_std

    def sample(self, obs, key, ep_tensor=None):
        """Sample action via reparameterization trick. Returns (action, log_prob)."""
        mean, log_std = self(obs, ep_tensor)
        std = jnp.exp(log_std)

        noise = jax.random.normal(key, mean.shape)
        z = mean + std * noise
        action = jnp.tanh(z)

        log_prob = -0.5 * (((z - mean) / std) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi))
        log_prob = log_prob.sum(axis=-1)
        log_prob = log_prob - jnp.sum(jnp.log(1 - action ** 2 + 1e-6), axis=-1)
        return action, log_prob

    def deterministic(self, obs, ep_tensor=None):
        """Deterministic action (mean, tanh-squashed)."""
        mean, _ = self(obs, ep_tensor)
        return jnp.tanh(mean)

    def regularization_loss(self):
        """NAU sparsity regularization."""
        return self.nau.regularization_loss()
