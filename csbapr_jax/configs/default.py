"""CS-BAPR JAX config — extends BAPR Config with symbolic components."""
from dataclasses import dataclass, field
from typing import List
from jax_bapr_reference.configs.default import Config as BAPRConfig


@dataclass
class CSBAPRConfig(BAPRConfig):
    """CS-BAPR hyperparameters.

    Inherits all BAPR config fields plus CS-BAPR-specific:
    - NAU/NMU network options
    - SINDy pre-identification
    - Jacobian consistency loss
    - Symbolic penalty
    """
    algo: str = "csbapr"

    # NAU/NMU actor
    use_nau_actor: bool = True
    nau_reg_weight: float = 0.01        # W sparsity regularization

    # SINDy Phase 0
    sindy_n_explore_episodes: int = 30
    sindy_poly_degree: int = 2
    sindy_threshold: float = 0.1
    sindy_discrete_time: bool = True

    # Jacobian consistency (Lean Part V)
    jac_weight: float = 0.1             # λ_jac — Jacobian consistency in policy loss
    weight_sym: float = 0.05            # λ_sym — symbolic penalty in Q-loss (Γ_sym)

    # Grad clipping (mandatory for Jacobian stability)
    grad_clip: float = 1.0
