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
    actor_type: str = None        # None→NAU/MLP (via use_nau_actor), 'kan'→KAN, 'tanh-mlp'→smooth-activation MLP

    # ===== Toggles for the four CS-BAPR base training-stabilization fixes =====
    # These default ON (matching CS-BAPR's recipe). Setting them all OFF gives
    # an honest "pristine BAPR" baseline for ablation.
    enable_min_q_target: bool = True       # Group-A1: elementwise min-Q across ensemble
    enable_reward_ema: bool = True         # Group-A2: divide reward by EMA std (no mean shift)
    enable_entropy_floor: bool = True      # Group-A3: alpha = max(exp(log_alpha), entropy_floor)
    entropy_floor: float = 0.01            # floor value when enable_entropy_floor is True
    enable_rollout_surprise: bool = True   # Group-B7 (v14, part 1): use recent on-policy rollout

    # ===== BAPR v14 fix — rollout-based surprise =====
    # When the training script can supply `recent_rollout` to update(), the
    # surprise signal is computed over the latest `surprise_window`
    # transitions from the on-policy rollout instead of the random replay
    # batch (which can be dominated by stale transitions under non-
    # stationarity). The q_std spike is also one-sided in surprise.py.
    #
    # Default 256 matches batch_size — the rollout-surprise then costs the
    # same critic forward as the previous replay-based path, no extra
    # overhead. (BAPR v14's 1024 default is for its iter-level multi_update
    # which is called ~once per 4000-step iter; CS-BAPR's update() is
    # called every train_freq=20 decisions, ~450 times per ep, so a
    # smaller window keeps total compute bounded.)
    surprise_window: int = 256

    # ===== BAPR v10 fix (P0 + P1) — early-training stability =====
    # P0: force w_lambda=0 for the first bapr_warmup_iters training iterations.
    # Early in training the BOCD belief is uniform and Q-std is tiny, which
    # gives a spurious non-zero w_lambda that over-biases β_eff before the
    # ensemble has diverged. Under NAU's {-1,0,1} weight constraint this
    # bad-basin lock-in is not recoverable; ~40% of NAU seeds crashed at 20x
    # OOD in the 10-seed study before this fix.
    bapr_warmup_iters: int = 100
    # P1: penalty_scale — multiplicative factor on w_lambda in effective_beta.
    # Was hardcoded 5.0 (inherited from BAPR v9); BAPR v10 lowered to 2.0
    # because steady-state w_lambda≈0.3 × scale=5 → β_eff=-3.5 is too
    # conservative. scale=2.0 → β_eff ∈ [-3.2, -2] is much milder.
    penalty_scale: float = 2.0

    # ===== CS-BAPR New =====
    weight_sym: float = 0.01      # λ_sym — symbolic consistency penalty (Lean: p.lam_sym ≥ 0)
    jac_weight: float = 0.01      # Jacobian consistency weight in policy loss (reduced 0.1→0.01)
    grad_clip: float = 1.0        # gradient clipping for Jacobian loss (trap #1)
    jac_curriculum_start: int = 100  # episodes before JC loss is enabled (curriculum warm-up)
    actor_weight_decay: float = 1e-4  # weight decay on actor optimizer (prevents K_g explosion)

    # ===== SINDy =====
    sindy_with_control: bool = False  # if True, fit dx/dt = f(x) + B u (action-aware)
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
