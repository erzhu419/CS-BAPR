"""
CSBAPRAgent — Complete CS-BAPR training agent.

Integrates all components:
- Phase 0: SINDy pre-identification + IRM filtering
- Phase 1: CS-BAPR training loop
  - Surprise → Belief → Q-loss (with Γ_sym) → Policy loss (with Jac loss) → soft_update

Adapted from BAPR/sac_ensemble_bapr.py:SAC_Trainer
with three CS-BAPR additions:
1. SINDy symbolic world model (Phase 0)
2. Jacobian consistency loss in policy update
3. Symbolic consistency penalty (Γ_sym) in Q-loss

⚠️ Temporal ordering (inherited BA-PR + 1 new):
   1. Surprise → Belief → Q-loss (immutable)
   2. Belief frozen within Q-loss
   3. Target network frozen within Q-loss, soft_update at end
   4. SINDy coefficients frozen within entire update() ← CS-BAPR NEW
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
import random
from copy import deepcopy

from csbapr.config import CSBAPRConfig
from csbapr.networks.nau_nmu import NAU_NMU_Actor, KAN_Actor
from csbapr.networks.critic import EnsembleQNet
from csbapr.networks.policy import GaussianPolicy
from csbapr.belief.tracker import BeliefTracker
from csbapr.belief.surprise import SurpriseComputer
from csbapr.context import TransitionContextTracker
from csbapr.losses.jacobian import compute_jacobian_loss, compute_action_anchor_loss
from csbapr.losses.q_loss import compute_q_loss_csbapr
from csbapr.losses.ood_bound import compute_ood_bound
from csbapr.utils import soft_update, compute_reg_norm, ReplayBuffer


class CSBAPRAgent:
    """
    CS-BAPR agent with optional action-valued symbolic feedback.
    
    Extends BAPR's SAC_Trainer with SINDy + NAU/NMU + IRM components.
    """
    def __init__(self, state_dim: int, action_dim: int,
                 config: CSBAPRConfig = None):
        self.config = config or CSBAPRConfig()
        self.device = self.config.get_device()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_context = bool(getattr(self.config, "use_context", False))
        self.context_tracker = None
        self.context_dim = 0
        if self.use_context:
            self.context_tracker = TransitionContextTracker(
                state_dim=state_dim,
                action_dim=action_dim,
                ema_alpha=getattr(self.config, "context_ema_alpha", 0.2),
                clip=getattr(self.config, "context_clip", 5.0),
            )
            self.context_dim = self.context_tracker.context_dim
        self.actor_state_dim = state_dim + self.context_dim

        # ===== Networks =====
        actor_type = getattr(self.config, 'actor_type', None)
        if actor_type == 'kan':
            self.actor = KAN_Actor(
                self.actor_state_dim, action_dim, self.config.hidden_dim
            ).to(self.device)
        elif self.config.use_nau_actor:
            self.actor = NAU_NMU_Actor(
                self.actor_state_dim, action_dim, self.config.hidden_dim,
                activation="tanh" if actor_type == "smooth-nau" else "leaky_relu",
            ).to(self.device)
        elif actor_type in ("tanh-mlp", "gelu-mlp", "softplus-mlp"):
            # Smooth-activation MLP fallback (for the four-way head ablation).
            # The activation is the prefix; e.g., 'tanh-mlp' → tanh hidden layers.
            self.actor = GaussianPolicy(
                self.actor_state_dim, action_dim, self.config.hidden_dim,
                activation=actor_type.split("-")[0],
            ).to(self.device)
        else:
            # Default: ReLU-hidden tanh-output (the SAC baseline used by BAPR).
            self.actor = GaussianPolicy(
                self.actor_state_dim, action_dim, self.config.hidden_dim,
                activation="relu",
            ).to(self.device)

        self.critic = EnsembleQNet(
            self.actor_state_dim, action_dim, self.config.hidden_dim, self.config.num_critics
        ).to(self.device)
        self.target_critic = deepcopy(self.critic)
        if hasattr(self.actor, "mean_clip"):
            self.actor.mean_clip = float(getattr(self.config, "actor_mean_clip", 0.0) or 0.0)

        # ===== Optimizers =====
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=self.config.lr_actor,
            weight_decay=self.config.actor_weight_decay
        )
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config.lr_critic)

        # ===== SAC entropy =====
        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.config.lr_alpha)
        self.alpha = self.config.alpha
        self.target_entropy = -action_dim

        # ===== BA-PR Components (from proven BAPR) =====
        self.belief_tracker = BeliefTracker(
            max_run_length=self.config.max_run_length,
            hazard_rate=self.config.hazard_rate,
            base_variance=self.config.base_variance,
        )
        self.surprise_computer = SurpriseComputer(ema_alpha=self.config.surprise_ema)

        # ===== Replay Buffer =====
        self.replay_buffer = ReplayBuffer(self.config.buffer_size)

        # ===== CS-BAPR: SINDy components (initialized in Phase 0) =====
        self.f_sym_torch = None  # set after sindy_preidentify()
        self.pi_sym_torch = None  # separate action-valued feedback reference
        self.sindy_model = None

        # ===== OOD Bound tracking (Part X + XI + XIII) =====
        self._ood_bound_cache = None  # latest OOD bound estimate
        self._n_train_samples = 0     # sampled transitions, including replay reuse

        # ===== Logging =====
        self.training_steps = 0

    def reset_context(self):
        """Reset episode-local context state for context-enabled methods."""
        if self.context_tracker is not None:
            self.context_tracker.reset_episode()

    def snapshot_context(self):
        """Return a deepcopy-able context snapshot for evaluation rollouts."""
        return deepcopy(self.context_tracker) if self.context_tracker is not None else None

    def restore_context(self, snapshot):
        if snapshot is not None:
            self.context_tracker = snapshot

    def observe_context(self, state, action, reward, next_state, done=False):
        """Update online context without adding to replay."""
        if self.context_tracker is not None:
            self.context_tracker.update(state, action, reward, next_state, done=done)

    def store_transition(self, state, action, reward, next_state, done):
        """
        Store a transition, including the inferred regime context when enabled.

        Context is recorded before and after the transition. This lets the
        critic train each replay sample under the context that was available
        online at collection time, instead of imposing today's regime on old
        off-policy data.
        """
        if self.context_tracker is None:
            self.replay_buffer.push(state, action, reward, next_state, done)
            return
        context = self.context_tracker.context
        next_context = self.context_tracker.update(
            state, action, reward, next_state, done=bool(done)
        )
        self.replay_buffer.push(
            state, action, reward, next_state, done,
            context=context, next_context=next_context,
        )

    def _current_context_np(self) -> np.ndarray:
        if self.context_tracker is None:
            return np.zeros(0, dtype=np.float32)
        return self.context_tracker.context

    def _augment_state_np(self, state: np.ndarray, context: np.ndarray = None) -> np.ndarray:
        if not self.use_context:
            return np.asarray(state, dtype=np.float32)
        if context is None:
            context = self._current_context_np()
        return np.concatenate([
            np.asarray(state, dtype=np.float32).reshape(-1),
            np.asarray(context, dtype=np.float32).reshape(-1),
        ]).astype(np.float32)

    def _augment_state_tensor(self, state: torch.Tensor,
                              context: torch.Tensor = None) -> torch.Tensor:
        if not self.use_context:
            return state
        if context is None:
            context = torch.zeros(
                state.shape[0], self.context_dim,
                dtype=state.dtype, device=state.device,
            )
        return torch.cat([state, context], dim=-1)

    def sindy_preidentify(self, env, policy=None, extra_envs=None, reference_builder=None):
        """Fit controlled dynamics, then optionally build an action-valued reference.

        reference_builder receives the fitted world model and must return a
        frozen policy reference in the actor's normalized action coordinates.
        """
        from csbapr.sindy.data_collector import collect_trajectories
        from csbapr.sindy.world_model import SymbolicWorldModel
        from csbapr.sindy.torch_wrapper import SINDyTorchWrapper
        from csbapr.irm.causal_filter import select_best_sindy_coefficients

        if self.use_context and (self.config.jac_weight > 0 or self.config.weight_sym > 0
                                 or self.config.action_anchor_weight > 0):
            raise ValueError("Symbolic reference must include context before enabling its penalties")
        if reference_builder is None and (self.config.jac_weight > 0 or self.config.weight_sym > 0
                                          or self.config.action_anchor_weight > 0):
            raise ValueError("Symbolic penalties require an action-valued reference_builder")
        n_control = self.action_dim if self.config.sindy_with_control else 0
        if reference_builder is not None and not n_control:
            raise ValueError("Feedback construction requires control-aware identification")
        model = SymbolicWorldModel(
            n_state=self.state_dim, n_control=n_control,
            poly_degree=self.config.sindy_lib_degree,
            threshold=self.config.sindy_threshold,
            discrete_time=self.config.sindy_discrete_time,
        )
        dt = 1.0 if self.config.sindy_discrete_time else float(env.unwrapped.dt)
        envs_data = {}
        for tag, current_env in [("primary", env)] + list(extra_envs or []):
            envs_data[tag] = collect_trajectories(
                current_env, policy=policy,
                n_episodes=self.config.sindy_n_explore_episodes,
                max_steps=self.config.max_steps_per_episode, dt=dt,
            )
        x, u, y = envs_data["primary"]
        model.fit(x, U=u, X_dot=None if model.discrete_time else y,
                  multiple_trajectories=True, t=dt)
        self._irm_report = None
        if extra_envs:
            coeffs, variance = select_best_sindy_coefficients(
                model, envs_data, threshold=self.config.sindy_threshold,
            )
            model.coeffs = coeffs.copy()
            model.model.optimizer.coef_ = coeffs.copy()
            model.sparsity = float((np.abs(coeffs) > 1e-6).mean())
            self._irm_report = {"prediction_error_variance": float(variance),
                                "n_envs": len(envs_data)}
        self.sindy_model = model
        self.f_sym_torch = SINDyTorchWrapper(model).to(self.device).eval()
        self.pi_sym_torch = None
        if reference_builder is not None:
            self.pi_sym_torch = reference_builder(model).to(self.device).eval()
            self.pi_sym_torch.requires_grad_(False)

        x_test, u_test, y_test = collect_trajectories(
            env, policy=policy, n_episodes=5,
            max_steps=self.config.max_steps_per_episode, dt=dt,
        )
        all_x, all_u, all_y = model.training_arrays(
            x_test, u_test, None if model.discrete_time else y_test, True,
        )
        pred_y = model.predict(all_x, all_u)
        ss_res = np.sum((all_y - pred_y) ** 2)
        ss_tot = np.sum((all_y - all_y.mean(axis=0)) ** 2)
        self._sindy_report = {
            "sparsity": model.sparsity,
            "n_coeffs": int(model.coeffs.size),
            "n_nonzero": int((np.abs(model.coeffs) > 1e-6).sum()),
            "n_control": n_control,
            "r_squared": float(1 - ss_res / max(ss_tot, 1e-10)),
            "action_reference": type(self.pi_sym_torch).__name__ if self.pi_sym_torch else None,
        }
        print(f"[Phase 0] controlled SINDy: {self._sindy_report}")
        model.print_equations()

    def update(self, batch_size: int = None, recent_rollout: dict = None):
        """
        Single CS-BAPR update step.
        
        Order (immutable):
        1. Sample from buffer
        2. Compute surprise → update belief
        3. Q-loss with Γ_sym → update critic
        4. Policy loss with Jacobian consistency → update actor
        5. Alpha loss → update entropy coefficient
        6. Soft update target network
        """
        batch_size = batch_size or self.config.batch_size

        if self.replay_buffer.size < batch_size:
            return None
        if self.config.action_anchor_weight > 0 and (self.pi_sym_torch is None or self.use_context):
            raise ValueError("Action anchoring requires a fitted reference in the actor's action coordinates")

        # ===== Sample batch =====
        sample = self.replay_buffer.sample(batch_size)
        if len(sample) == 5:
            states, actions, rewards, next_states, dones = sample
            contexts = next_contexts = None
        else:
            states, actions, rewards, next_states, dones, contexts, next_contexts = sample
        state = torch.FloatTensor(states).to(self.device)
        action = torch.FloatTensor(actions).to(self.device)
        reward = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_states).to(self.device)
        done = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        context = (torch.FloatTensor(contexts).to(self.device)
                   if contexts is not None else None)
        next_context = (torch.FloatTensor(next_contexts).to(self.device)
                        if next_contexts is not None else None)
        state_aug = self._augment_state_tensor(state, context)
        next_state_aug = self._augment_state_tensor(next_state, next_context)

        # Evaluate policy on current states (graph needed for policy loss)
        new_action, log_prob, _, _, _ = self.actor.evaluate(state_aug)

        # Evaluate on next states for target Q — no_grad: target computation
        # must NOT accumulate a computation graph (would never be freed by
        # q_loss.backward() since target is detached, causing a memory leak).
        with torch.no_grad():
            new_next_action, next_log_prob, _, _, _ = self.actor.evaluate(next_state_aug)

        # Reward scaling: divide by running std (no mean shift — preserves sign).
        # Group-A2 fix; toggle via config.enable_reward_ema.
        self._reward_ema_var = getattr(self, '_reward_ema_var', 1.0)
        if getattr(self.config, "enable_reward_ema", True):
            batch_var = reward.var().item()
            self._reward_ema_var = 0.99 * self._reward_ema_var + 0.01 * batch_var
            reward = reward / max(self._reward_ema_var ** 0.5, 1e-6)

        # ===== Step 1: Surprise + Belief (inherited BA-PR) =====
        # By default, surprise uses the current random replay batch. Per BAPR
        # v14 / GPT-5.5 review, this is suboptimal under non-stationarity:
        # the replay batch may be dominated by old-regime transitions, masking
        # the new-regime signal. When `recent_rollout` is supplied (a dict
        # with at least `state` and `action` arrays from the most-recent
        # on-policy rollout), we instead compute Q-std and reward signals
        # from that rollout's most recent surprise_window transitions.
        reg_norm = compute_reg_norm(self.target_critic)
        current_reg_norm = compute_reg_norm(self.critic)

        # v14 rollout-surprise: only enabled when both the toggle is on AND the
        # caller actually provided a recent_rollout dict. Setting
        # config.enable_rollout_surprise = False forces the legacy replay-batch
        # path (Group-B7 ablation).
        use_rollout = (
            getattr(self.config, "enable_rollout_surprise", True)
            and recent_rollout is not None
            and len(recent_rollout.get("reward", [])) >= 8
        )
        if use_rollout:
            surprise_window = int(getattr(self.config, "surprise_window", 1024))
            r_arr = np.asarray(recent_rollout["reward"], dtype=np.float32)
            s_arr = np.asarray(recent_rollout["state"], dtype=np.float32)
            a_arr = np.asarray(recent_rollout["action"], dtype=np.float32)
            tail = min(surprise_window, len(r_arr))
            r_arr = r_arr[-tail:]
            s_arr = s_arr[-tail:]
            a_arr = a_arr[-tail:]
            with torch.no_grad():
                rs = torch.from_numpy(s_arr).to(self.device)
                ra = torch.from_numpy(a_arr).to(self.device)
                if self.use_context and "context" in recent_rollout:
                    rc_arr = np.asarray(recent_rollout["context"], dtype=np.float32)[-tail:]
                    rc = torch.from_numpy(rc_arr).to(self.device)
                else:
                    rc = None
                pre_q = self.critic(self._augment_state_tensor(rs, rc), ra)
                q_std_signal = pre_q.std(dim=0)
            r_tensor = torch.from_numpy(r_arr).unsqueeze(1).to(self.device)
            r_tensor = r_tensor / max(self._reward_ema_var ** 0.5, 1e-6)
            surprise = self.surprise_computer.compute(
                r_tensor, q_std_signal,
                reg_norm_current=current_reg_norm, reg_norm_target=reg_norm,
            )
        else:
            with torch.no_grad():
                pre_q = self.critic(state_aug, action)
                q_std_signal = pre_q.std(dim=0)
            surprise = self.surprise_computer.compute(
                reward, q_std_signal,
                reg_norm_current=current_reg_norm, reg_norm_target=reg_norm,
            )
        self.belief_tracker.update(surprise)

        # ===== Step 2: Alpha loss =====
        if self.config.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            # Group-A3: entropy floor, toggle via config.enable_entropy_floor.
            raw_alpha = self.log_alpha.exp().item()
            floor = float(getattr(self.config, "entropy_floor", 0.01))
            self.alpha = max(raw_alpha, floor) if getattr(self.config, "enable_entropy_floor", True) else raw_alpha
            alpha_cap = getattr(self.config, "alpha_cap", None)
            if alpha_cap is not None and float(alpha_cap) > 0:
                self.alpha = min(self.alpha, float(alpha_cap))
                with torch.no_grad():
                    self.log_alpha.clamp_(max=float(np.log(float(alpha_cap))))

        # ===== Step 3: Q-loss (CS-BAPR, with Γ_sym) =====
        if self.pi_sym_torch is not None and not self.use_context:
            q_loss, predicted_q, ood_loss, w_lambda, sym_p = compute_q_loss_csbapr(
                self.critic, self.target_critic, self.actor,
                self.pi_sym_torch,
                state, action, reward, next_state, done,
                new_next_action, next_log_prob, reg_norm,
                self.belief_tracker,
                alpha=self.alpha,
                gamma=self.config.gamma,
                weight_reg=self.config.weight_reg,
                weight_sym=self.config.weight_sym,
                beta_ood=self.config.beta_ood,
                penalty_decay_rate=0.1,
                device=self.device,
                batch_size=batch_size,
                enable_min_q_target=getattr(self.config, "enable_min_q_target", True),
            )
        else:
            # Fallback: BA-PR Q-loss without Γ_sym
            q_loss, predicted_q, ood_loss, w_lambda, sym_p = self._compute_q_loss_bapr_fallback(
                state_aug, action, reward, next_state_aug, done,
                new_next_action, next_log_prob, reg_norm, batch_size
            )

        q_std_guard_loss_val = 0.0
        q_std_cap = float(getattr(self.config, "critic_q_std_cap", 0.0) or 0.0)
        q_std_guard_weight = float(
            getattr(self.config, "critic_q_std_penalty_weight", 0.0) or 0.0
        )
        if q_std_cap > 0.0 and q_std_guard_weight > 0.0:
            predicted_q_std = predicted_q.std(dim=0)
            q_std_guard_loss = torch.relu(
                torch.log(predicted_q_std + 1e-6) - math.log(q_std_cap)
            ).pow(2).mean()
            q_loss = q_loss + q_std_guard_weight * q_std_guard_loss
            q_std_guard_loss_val = float(q_std_guard_loss.detach().item())

        self.critic_optimizer.zero_grad()
        q_loss.backward()
        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.grad_clip)
        self.critic_optimizer.step()

        # ===== Step 4: Policy loss (with Jacobian consistency) =====
        # Reuse new_action/log_prob from L330: alpha_loss uses log_prob.detach(),
        # and q_loss.backward() only touches critic params — actor graph still intact.
        q_values = self.critic(state_aug, new_action)
        q_mean = q_values.mean(dim=0)
        q_std = q_values.std(dim=0)
        policy_q_std_cap = float(getattr(self.config, "policy_q_std_cap", 0.0) or 0.0)
        q_std_for_policy = (
            torch.clamp(q_std, max=policy_q_std_cap)
            if policy_q_std_cap > 0.0
            else q_std
        )

        # Adaptive beta from belief (directly from BAPR, with v10 fix).
        # P0: during warmup, force w_lambda=0 so β_eff=-2 (pure RE-SAC).
        #     BOCD belief is uniform + Q-std is tiny in early training,
        #     giving a spurious w_lambda that over-conservativizes the
        #     policy before it can explore. Particularly bad for NAU,
        #     whose {-1,0,1} weight constraint locks bad early basins.
        # P1: penalty_scale is configurable (default 2.0 per BAPR v10)
        #     instead of hardcoded 5.0, reducing over-conservatism at
        #     steady-state too.
        w_lambda_eff = 0.0 if self.training_steps < self.config.bapr_warmup_iters else w_lambda
        effective_beta = -2.0 - w_lambda_eff * self.config.penalty_scale
        policy_loss = -(q_mean + effective_beta * q_std_for_policy - self.alpha * log_prob).mean()

        # Behavior cloning regularization (RE-SAC dual reg, prevents policy drift
        # from replay buffer distribution after buffer saturation)
        bc_loss = nn.MSELoss()(new_action, action)
        policy_loss = policy_loss + self.config.beta_bc * bc_loss

        # CS-BAPR: Add Jacobian consistency loss
        jac_loss_val = 0.0
        if self.pi_sym_torch is not None and not self.use_context and self.config.jac_weight > 0:
            jac_loss = compute_jacobian_loss(self.actor, self.pi_sym_torch, state)
            policy_loss = policy_loss + self.config.jac_weight * jac_loss
            jac_loss_val = jac_loss.item()

        action_anchor_loss_val = 0.0
        if self.config.action_anchor_weight > 0:
            action_anchor_loss = compute_action_anchor_loss(self.actor, self.pi_sym_torch, state)
            policy_loss = policy_loss + self.config.action_anchor_weight * action_anchor_loss
            action_anchor_loss_val = action_anchor_loss.item()

        # NAU regularization
        if hasattr(self.actor, 'regularization_loss') and self.config.nau_reg_weight > 0:
            nau_reg = self.actor.regularization_loss()
            policy_loss = policy_loss + self.config.nau_reg_weight * nau_reg

        l_eff_surrogate_val = 0.0
        l_eff_penalty_val = 0.0
        l_eff_weight = float(getattr(self.config, "nau_l_eff_penalty_weight", 0.0) or 0.0)
        if hasattr(self.actor, 'l_eff_surrogate_loss') and l_eff_weight > 0:
            l_eff_penalty, l_eff_surrogate = self.actor.l_eff_surrogate_loss(
                target=float(getattr(self.config, "nau_l_eff_target", 200.0))
            )
            policy_loss = policy_loss + l_eff_weight * l_eff_penalty
            l_eff_penalty_val = float(l_eff_penalty.detach().item())
            l_eff_surrogate_val = float(l_eff_surrogate.detach().item())

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_clip)
        self.actor_optimizer.step()

        # ===== Step 5: Soft update target network (LAST) =====
        soft_update(self.target_critic, self.critic, self.config.tau)

        self.training_steps += 1
        self._n_train_samples += batch_size

        # L_eff logging (Part XI: composed_deriv_lipschitz_simple)
        L_eff = math.inf
        if self.config.track_L_eff and hasattr(self.actor, 'compute_L_eff'):
            L_eff = self.actor.compute_L_eff()

        return {
            'q_loss': q_loss.item(),
            'policy_loss': policy_loss.item(),
            'ood_loss': ood_loss.item(),
            'q_mean': predicted_q.mean().item(),
            'q_std': q_std.mean().item(),
            'q_std_policy': q_std_for_policy.mean().item(),
            'q_std_guard_loss': q_std_guard_loss_val,
            'alpha': self.alpha,
            'weighted_lambda': w_lambda,
            'weighted_lambda_effective': w_lambda_eff,   # 0 during BAPR warmup, else = w_lambda
            'sym_penalty': sym_p,
            'jac_loss': jac_loss_val,
            'action_anchor_loss': action_anchor_loss_val,
            'surprise': surprise,
            'belief_entropy': self.belief_tracker.entropy,
            'L_eff': L_eff,
            'L_eff_surrogate': l_eff_surrogate_val,
            'L_eff_penalty': l_eff_penalty_val,
        }

    def estimate_ood_bound(self, x_train_boundary: torch.Tensor,
                           x_ood: torch.Tensor, reference=None) -> dict:
        """Evaluate a boundary-point certificate against an action-valued reference."""
        reference = self.pi_sym_torch if reference is None else reference
        if reference is None:
            raise ValueError("An explicit action-valued reference is required")
        result = compute_ood_bound(
            self.actor, reference,
            x_train_boundary.to(self.device), x_ood.to(self.device),
        )
        self._ood_bound_cache = result
        return result

    def _compute_q_loss_bapr_fallback(self, state, action, reward, next_state, done,
                                       new_next_action, next_log_prob, reg_norm, batch_size):
        """BA-PR Q-loss without Γ_sym (when SINDy not yet fitted)."""
        predicted_q = self.critic(state, action)
        target_q_next = self.target_critic(next_state, new_next_action)
        num_critics = self.critic.num_critics

        belief = torch.tensor(self.belief_tracker.belief, dtype=torch.float32, device=self.device)
        penalty_schedule = torch.exp(-0.1 * torch.arange(self.belief_tracker.max_H, dtype=torch.float32, device=self.device))
        weighted_lambda = (belief * penalty_schedule).sum()

        # Group-A1 fix: conservative target via elementwise min over ensemble.
        # Toggle via config.enable_min_q_target. Vanilla SAC ensemble = False.
        if getattr(self.config, "enable_min_q_target", True):
            target_q_min = target_q_next.min(dim=0, keepdim=True)[0]  # [1, batch]
            target_q_min = target_q_min.expand(num_critics, -1)        # [ensemble, batch]
        else:
            target_q_min = target_q_next  # per-critic targets, no min reduction

        next_log_prob_expanded = next_log_prob.unsqueeze(0).repeat(num_critics, 1)
        reg_norm_expanded = reg_norm.unsqueeze(-1).repeat(1, batch_size)

        target_q_min = (target_q_min
                        - self.alpha * next_log_prob_expanded
                        + self.config.weight_reg * reg_norm_expanded)

        target_q_value = reward + (1 - done) * self.config.gamma * target_q_min.unsqueeze(-1)
        ood_loss = predicted_q.std(0).mean()
        q_loss = nn.MSELoss()(predicted_q, target_q_value.squeeze(-1).detach())
        loss = q_loss + self.config.beta_ood * ood_loss

        return loss, predicted_q, ood_loss, weighted_lambda.item(), 0.0

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action using current policy."""
        state_tensor = torch.FloatTensor(self._augment_state_np(state)).to(self.device)
        return self.actor.get_action(state_tensor, deterministic=deterministic)

    def save(self, path: str, extra_state=None):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
            'log_alpha': self.log_alpha.detach(),
            'alpha': self.alpha,
            'training_steps': self.training_steps,
            'n_train_samples': self._n_train_samples,
            'use_context': self.use_context,
            'context_dim': self.context_dim,
            'sindy_report': getattr(self, '_sindy_report', None),
            'irm_report': getattr(self, '_irm_report', None),
            'symbolic_reference_gain': (self.pi_sym_torch.gain
                                        if self.pi_sym_torch is not None else None),
            'config': {k: v for k, v in vars(self.config).items()
                       if isinstance(v, (int, float, str, bool))},
            # Evaluation snapshots stay small; the training runner supplies
            # episode state when it needs a complete resumable checkpoint.
            'training_state': {
                'replay': vars(self.replay_buffer),
                'belief': vars(self.belief_tracker),
                'surprise': vars(self.surprise_computer),
                'context': (vars(self.context_tracker)
                            if self.context_tracker is not None else None),
                'reward_ema_var': getattr(self, '_reward_ema_var', 1.0),
                'numpy_rng': np.random.get_state(),
                'python_rng': random.getstate(),
                'torch_rng': torch.get_rng_state(),
                'cuda_rng': (torch.cuda.get_rng_state_all()
                             if torch.device(self.device).type == 'cuda' else None),
            } if extra_state is not None else None,
            'extra_state': extra_state,
        }, path)

    def load(self, path: str):
        # These are our own full-state checkpoints, including NumPy replay data.
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        if checkpoint.get('config', {}).get('actor_type') != self.config.actor_type:
            raise ValueError("Checkpoint actor type differs from the current architecture")
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.target_critic.load_state_dict(checkpoint['target_critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        with torch.no_grad():
            self.log_alpha.copy_(checkpoint['log_alpha'].to(self.device))
        if 'alpha_optimizer' in checkpoint:
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        self.alpha = checkpoint.get('alpha', self.log_alpha.exp().item())
        self.training_steps = checkpoint['training_steps']
        self._n_train_samples = checkpoint.get('n_train_samples', 0)
        self._sindy_report = checkpoint.get('sindy_report', None)
        self._irm_report = checkpoint.get('irm_report', None)
        gain = checkpoint.get('symbolic_reference_gain')
        self.pi_sym_torch = None
        if gain is not None:
            from csbapr.sindy.feedback import LQRFeedbackReference
            self.pi_sym_torch = LQRFeedbackReference(gain).to(self.device).eval()
            self.pi_sym_torch.requires_grad_(False)
        state = checkpoint.get('training_state')
        if state is not None:
            vars(self.replay_buffer).update(state['replay'])
            vars(self.belief_tracker).update(state['belief'])
            vars(self.surprise_computer).update(state['surprise'])
            if self.context_tracker is not None:
                vars(self.context_tracker).update(state['context'])
            self._reward_ema_var = state['reward_ema_var']
            np.random.set_state(state['numpy_rng'])
            random.setstate(state['python_rng'])
            torch.set_rng_state(state['torch_rng'])
            if torch.device(self.device).type == 'cuda' and state['cuda_rng'] is not None:
                torch.cuda.set_rng_state_all(state['cuda_rng'])
        return checkpoint.get('extra_state')
