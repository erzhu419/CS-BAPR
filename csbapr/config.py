"""
CS-BAPR Hyperparameter Configuration

All parameters centralized here. Maps to Lean symbols per CSBAPR_engineering_doc.md §1.2.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CSBAPRConfig:
    """Complete CS-BAPR hyperparameter set."""

    # ===== Environment =====
    env_name: str = "Pendulum-v1"
    state_dim: int = 3
    action_dim: int = 1
    max_episodes: int = 1000
    max_steps_per_episode: int = 200

    # ===== SAC Core =====
    gamma: float = 0.99           # Lean: p.γ
    tau: float = 0.005            # target network soft update rate
    alpha: float = 0.2            # SAC entropy coefficient
    auto_alpha: bool = True       # auto-tune alpha
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4

    # ===== Network Architecture =====
    hidden_dim: int = 256
    num_critics: int = 5          # ensemble size

    # ===== Replay Buffer =====
    buffer_size: int = 1_000_000
    batch_size: int = 256
    warmup_steps: int = 1000      # random exploration before learning

    # ===== BA-PR Inherited (不变) =====
    max_run_length: int = 20      # BOCD max run length
    hazard_rate: float = 0.02     # BOCD hazard rate
    base_variance: float = 1.0    # BOCD base variance
    surprise_ema: float = 0.1     # surprise EMA decay α
    weight_reg: float = 0.01      # λ_epi — epistemic penalty weight (Lean: p.lam_epi)
    beta_ood: float = 0.1         # OOD Q-std penalty weight
    beta_bc: float = 0.001        # behavior cloning weight (RE-SAC dual reg, prevents policy drift)
    actor_type: str = None        # None→NAU/MLP (via use_nau_actor), 'kan'→KAN actor

    # ===== CS-BAPR New =====
    weight_sym: float = 0.01      # λ_sym — symbolic consistency penalty (Lean: p.lam_sym ≥ 0)
    jac_weight: float = 0.01      # Jacobian consistency weight in policy loss (reduced 0.1→0.01)
    grad_clip: float = 1.0        # gradient clipping for Jacobian loss (trap #1)
    jac_curriculum_start: int = 100  # episodes before JC loss is enabled (curriculum warm-up)
    actor_weight_decay: float = 1e-4  # weight decay on actor optimizer (prevents K_g explosion)

    # ===== SINDy =====
    sindy_threshold: float = 0.1  # STLSQ sparsity threshold
    sindy_lib_degree: int = 2     # polynomial library degree (1-4)
    sindy_discrete_time: bool = True  # prefer discrete mode (trap #10)
    sindy_n_explore_episodes: int = 50  # Phase 0 exploration episodes
    sindy_epsilon_threshold: float = 1.0  # SINDy quality gate

    # ===== NAU/NMU =====
    use_nau_actor: bool = True    # use NAU/NMU actor (False = standard MLP)
    nau_reg_weight: float = 0.01  # NAU weight discretization regularizer

    # ===== IRM =====
    irm_penalty_weight: float = 0.1
    irm_n_envs: int = 3           # minimum environments for IRM filtering

    # ===== OOD Bound (Part X + XIII) =====
    physics_smoothness_M: float = 0.0   # f_real derivative Lipschitz (Lean: HasLipschitzFderiv)
                                         # 0 = conservative (only L_eff contributes to quadratic term)
    ood_confidence_delta: float = 0.05   # PAC confidence δ (0.05 → 95%)
    ood_eval_interval: int = 50          # evaluate OOD bound every N episodes

    # ===== Logging =====
    log_interval: int = 10
    save_interval: int = 100
    log_dir: str = "logs"
    track_L_eff: bool = True      # log L_eff each eval (Part XI)

    # ===== Device =====
    device: str = "auto"          # "auto", "cpu", "cuda"

    def get_device(self):
        import torch
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device
