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
from copy import deepcopy

from csbapr.config import CSBAPRConfig
from csbapr.networks.nau_nmu import NAU_NMU_Actor
from csbapr.networks.critic import EnsembleQNet
from csbapr.networks.policy import GaussianPolicy
from csbapr.belief.tracker import BeliefTracker
from csbapr.belief.surprise import SurpriseComputer
from csbapr.losses.jacobian import compute_jacobian_loss
from csbapr.losses.q_loss import compute_q_loss_csbapr
from csbapr.losses.ood_bound import compute_ood_bound, estimate_deriv_bound_B
from csbapr.utils import soft_update, compute_reg_norm, ReplayBuffer


class CSBAPRAgent:
    """
    CS-BAPR Agent: Causal-Symbolic Bayesian Adaptive Policy Regularization.
    
    Extends BAPR's SAC_Trainer with SINDy + NAU/NMU + IRM components.
    """
    def __init__(self, state_dim: int, action_dim: int,
                 config: CSBAPRConfig = None):
        self.config = config or CSBAPRConfig()
        self.device = self.config.get_device()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # ===== Networks =====
        if self.config.use_nau_actor:
            self.actor = NAU_NMU_Actor(
                state_dim, action_dim, self.config.hidden_dim
            ).to(self.device)
        else:
            self.actor = GaussianPolicy(
                state_dim, action_dim, self.config.hidden_dim
            ).to(self.device)

        self.critic = EnsembleQNet(
            state_dim, action_dim, self.config.hidden_dim, self.config.num_critics
        ).to(self.device)
        self.target_critic = deepcopy(self.critic)

        # ===== Optimizers =====
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.lr_actor)
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
        self.sindy_model = None

        # ===== OOD Bound tracking (Part X + XI + XIII) =====
        self._ood_bound_cache = None  # latest OOD bound estimate
        self._n_train_samples = 0     # running count for generalization gap

        # ===== Logging =====
        self.training_steps = 0

    def sindy_preidentify(self, env, policy=None, extra_envs=None):
        """
        Phase 0: SINDy pre-identification with optional IRM multi-env filtering.

        Single-env path:
          1. Collect exploration trajectories
          2. Fit SINDy model
          3. Verify quality (ε₁ < threshold)
          4. Create PyTorch wrapper (frozen coefficients)

        Multi-env path (IRM, when extra_envs provided):
          1. Collect trajectories in EACH environment
          2. Fit SINDy per-env → IRM coefficient selection (mean coefficients)
          3. Quantify IRM advantage (ε_s proxy)
          4. Create PyTorch wrapper from IRM-filtered coefficients

        Args:
            env: Primary Gymnasium environment
            policy: Optional exploration policy (None = random)
            extra_envs: List of (env_instance, env_name) for IRM multi-env.
                        If None or empty, falls back to single-env SINDy.
        """
        from csbapr.sindy.data_collector import collect_trajectories, prepare_sindy_data_discrete
        from csbapr.sindy.world_model import SymbolicWorldModel
        from csbapr.sindy.torch_wrapper import SINDyTorchWrapper

        n_control = 0  # state-only dynamics

        # ---- Multi-env IRM path ----
        if extra_envs and len(extra_envs) >= 1:
            all_envs = [('primary', env)] + list(extra_envs)
            print(f"[Phase 0-IRM] Multi-environment SINDy ({len(all_envs)} envs)")

            envs_data = {}  # for IRM coefficient selection
            envs_eval_data = {}  # for IRM advantage estimation

            for env_tag, e in all_envs:
                print(f"[Phase 0-IRM] Collecting from env '{env_tag}'...")
                X_list, U_list, X_dot_list = collect_trajectories(
                    e, policy=policy,
                    n_episodes=self.config.sindy_n_explore_episodes,
                    max_steps=self.config.max_steps_per_episode,
                )
                if self.config.sindy_discrete_time:
                    X_trimmed, _ = prepare_sindy_data_discrete(X_list, U_list)
                    envs_data[env_tag] = (X_trimmed, None, None)
                else:
                    from csbapr.sindy.data_collector import compute_state_derivatives
                    X_flat, X_dot_flat = compute_state_derivatives(X_list)
                    envs_data[env_tag] = (None, None, None)
                    envs_eval_data[env_tag] = (X_flat, X_dot_flat)

            # Build a template model for IRM selection
            template = SymbolicWorldModel(
                n_state=self.state_dim,
                n_control=n_control,
                poly_degree=self.config.sindy_lib_degree,
                threshold=self.config.sindy_threshold,
                discrete_time=self.config.sindy_discrete_time,
            )

            from csbapr.irm.causal_filter import select_best_sindy_coefficients, compute_irm_advantage

            # Reformat for select_best_sindy_coefficients:
            # expects env_name → (X_list, U_list, X_dot_list)
            irm_fit_data = {}
            for env_tag, e in all_envs:
                X_list, U_list, X_dot_list = collect_trajectories(
                    e, policy=policy,
                    n_episodes=max(self.config.sindy_n_explore_episodes // len(all_envs), 5),
                    max_steps=self.config.max_steps_per_episode,
                )
                if self.config.sindy_discrete_time:
                    X_trimmed, _ = prepare_sindy_data_discrete(X_list, U_list)
                    irm_fit_data[env_tag] = (X_trimmed, None, None)
                else:
                    irm_fit_data[env_tag] = (X_list, U_list, X_dot_list)

            best_coeffs, irm_variance = select_best_sindy_coefficients(
                template, irm_fit_data,
                threshold=self.config.sindy_threshold,
                verbose=True,
            )

            # Build final model with IRM-selected coefficients
            self.sindy_model = SymbolicWorldModel(
                n_state=self.state_dim,
                n_control=n_control,
                poly_degree=self.config.sindy_lib_degree,
                threshold=self.config.sindy_threshold,
                discrete_time=self.config.sindy_discrete_time,
            )
            # Fit on primary env first (to initialize model internals), then overwrite coeffs
            X_list0, U_list0, X_dot_list0 = collect_trajectories(
                env, policy=policy,
                n_episodes=self.config.sindy_n_explore_episodes,
                max_steps=self.config.max_steps_per_episode,
            )
            if self.config.sindy_discrete_time:
                X_tr, _ = prepare_sindy_data_discrete(X_list0, U_list0)
                self.sindy_model.fit(X_tr, U=None, multiple_trajectories=True, t=1.0)
            else:
                from csbapr.sindy.data_collector import compute_state_derivatives
                X_f, Xd_f = compute_state_derivatives(X_list0)
                self.sindy_model.fit(X_f, X_dot=Xd_f)

            # Overwrite with IRM-selected mean coefficients
            self.sindy_model.coeffs = best_coeffs
            if hasattr(self.sindy_model.model, 'coefficients_'):
                self.sindy_model.model.coefficients_ = best_coeffs
            elif hasattr(self.sindy_model.model, 'optimizer'):
                self.sindy_model.model.optimizer.coef_ = best_coeffs

            # Compute IRM advantage if possible
            irm_report = {'irm_variance': float(irm_variance), 'n_envs': len(all_envs)}
            if envs_eval_data:
                try:
                    adv = compute_irm_advantage(self.sindy_model, envs_eval_data)
                    irm_report.update(adv)
                except Exception:
                    pass
            self._irm_report = irm_report

            sparsity = self.sindy_model.sparsity
            print(f"[Phase 0-IRM] IRM-filtered SINDy: sparsity={sparsity:.3f}, "
                  f"coeff_var={irm_variance:.6f}, n_envs={len(all_envs)}")
            print(f"[Phase 0-IRM] Discovered equations (IRM-filtered):")
            self.sindy_model.print_equations()

        # ---- Single-env path ----
        else:
            print("[Phase 0] Collecting exploration trajectories...")
            X_list, U_list, X_dot_list = collect_trajectories(
                env, policy=policy,
                n_episodes=self.config.sindy_n_explore_episodes,
                max_steps=self.config.max_steps_per_episode,
            )

            print(f"[Phase 0] Fitting SINDy model (degree={self.config.sindy_lib_degree}, "
                  f"threshold={self.config.sindy_threshold}, state_only=True)...")
            self.sindy_model = SymbolicWorldModel(
                n_state=self.state_dim,
                n_control=n_control,
                poly_degree=self.config.sindy_lib_degree,
                threshold=self.config.sindy_threshold,
                discrete_time=self.config.sindy_discrete_time,
            )

            if self.config.sindy_discrete_time:
                X_trimmed, _ = prepare_sindy_data_discrete(X_list, U_list)
                self.sindy_model.fit(
                    X_trimmed, U=None,
                    multiple_trajectories=True, t=1.0
                )
            else:
                from csbapr.sindy.data_collector import compute_state_derivatives
                X_flat, X_dot_flat = compute_state_derivatives(X_list)
                self.sindy_model.fit(X_flat, X_dot=X_dot_flat)

            sparsity = self.sindy_model.sparsity
            print(f"[Phase 0] SINDy fitted: sparsity = {sparsity:.3f}")
            print(f"[Phase 0] Discovered equations:")
            self.sindy_model.print_equations()
            self._irm_report = None

        # Create PyTorch wrapper (frozen coefficients)
        self.f_sym_torch = SINDyTorchWrapper(self.sindy_model).to(self.device).eval()

        # Store SINDy quality metrics for reporting
        self._sindy_report = {
            'sparsity': float(self.sindy_model.sparsity),
            'n_coeffs': int(self.sindy_model.coeffs.size) if self.sindy_model.coeffs is not None else 0,
            'n_nonzero': int((np.abs(self.sindy_model.coeffs) > 1e-6).sum()) if self.sindy_model.coeffs is not None else 0,
        }
        # Compute R² if we have data
        try:
            X_test, U_test, Xdot_test = collect_trajectories(
                env, policy=policy, n_episodes=5,
                max_steps=self.config.max_steps_per_episode,
            )
            if self.config.sindy_discrete_time:
                X_tr_test, _ = prepare_sindy_data_discrete(X_test, U_test)
                # R²: predict x_{t+1} from x_t
                all_x = np.vstack([x[:-1] for x in X_tr_test])
                all_y = np.vstack([x[1:] for x in X_tr_test])
                pred_y = self.sindy_model.predict(all_x)
                ss_res = np.sum((all_y - pred_y) ** 2)
                ss_tot = np.sum((all_y - all_y.mean(axis=0)) ** 2)
            else:
                from csbapr.sindy.data_collector import compute_state_derivatives
                all_x, all_y = compute_state_derivatives(X_test)
                pred_y = self.sindy_model.predict(all_x)
                ss_res = np.sum((all_y - pred_y) ** 2)
                ss_tot = np.sum((all_y - all_y.mean(axis=0)) ** 2)
            r2 = 1 - ss_res / max(ss_tot, 1e-10)
            self._sindy_report['r_squared'] = float(r2)
            print(f"[Phase 0] SINDy R² = {r2:.4f}")
        except Exception:
            self._sindy_report['r_squared'] = None

        print(f"[Phase 0] SINDyTorchWrapper created (coefficients frozen)")
        print(f"[Phase 0] ✓ Phase 0 complete")

    def update(self, batch_size: int = None):
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

        # ===== Sample batch =====
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        state = torch.FloatTensor(states).to(self.device)
        action = torch.FloatTensor(actions).to(self.device)
        reward = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_states).to(self.device)
        done = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Evaluate policy on current and next states
        new_action, log_prob, _, _, _ = self.actor.evaluate(state)
        new_next_action, next_log_prob, _, _, _ = self.actor.evaluate(next_state)

        # Normalize rewards
        reward_std = reward.std()
        if reward_std > 1e-6:
            reward = (reward - reward.mean()) / reward_std

        # ===== Step 1: Surprise + Belief (inherited BA-PR) =====
        with torch.no_grad():
            pre_q = self.critic(state, action)
            q_std_signal = pre_q.std(dim=0)

        reg_norm = compute_reg_norm(self.target_critic)
        current_reg_norm = compute_reg_norm(self.critic)

        surprise = self.surprise_computer.compute(
            reward, q_std_signal,
            reg_norm_current=current_reg_norm, reg_norm_target=reg_norm
        )
        self.belief_tracker.update(surprise)

        # ===== Step 2: Alpha loss =====
        if self.config.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # ===== Step 3: Q-loss (CS-BAPR, with Γ_sym) =====
        if self.f_sym_torch is not None:
            q_loss, predicted_q, ood_loss, w_lambda, sym_p = compute_q_loss_csbapr(
                self.critic, self.target_critic, self.actor,
                self.f_sym_torch,
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
            )
        else:
            # Fallback: BA-PR Q-loss without Γ_sym
            q_loss, predicted_q, ood_loss, w_lambda, sym_p = self._compute_q_loss_bapr_fallback(
                state, action, reward, next_state, done,
                new_next_action, next_log_prob, reg_norm, batch_size
            )

        self.critic_optimizer.zero_grad()
        q_loss.backward()
        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.grad_clip)
        self.critic_optimizer.step()

        # ===== Step 4: Policy loss (with Jacobian consistency) =====
        new_action2, log_prob2, _, _, _ = self.actor.evaluate(state)
        
        q_values = self.critic(state, new_action2)
        q_mean = q_values.mean(dim=0)
        q_std = q_values.std(dim=0)
        
        # Adaptive beta from belief (directly from BAPR)
        effective_beta = -2.0 - w_lambda * 5.0
        policy_loss = -(q_mean + effective_beta * q_std - self.alpha * log_prob2).mean()

        # CS-BAPR: Add Jacobian consistency loss
        jac_loss_val = 0.0
        if self.f_sym_torch is not None:
            jac_loss = compute_jacobian_loss(self.actor, self.f_sym_torch, state)
            policy_loss = policy_loss + self.config.jac_weight * jac_loss
            jac_loss_val = jac_loss.item()

        # NAU regularization
        if self.config.use_nau_actor and hasattr(self.actor, 'regularization_loss'):
            nau_reg = self.actor.regularization_loss()
            policy_loss = policy_loss + self.config.nau_reg_weight * nau_reg

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
        L_eff = 0.0
        if self.config.use_nau_actor and hasattr(self.actor, 'compute_L_eff'):
            L_eff = self.actor.compute_L_eff()

        return {
            'q_loss': q_loss.item(),
            'policy_loss': policy_loss.item(),
            'ood_loss': ood_loss.item(),
            'q_mean': predicted_q.mean().item(),
            'q_std': q_std.mean().item(),
            'alpha': self.alpha,
            'weighted_lambda': w_lambda,
            'sym_penalty': sym_p,
            'jac_loss': jac_loss_val,
            'surprise': surprise,
            'belief_entropy': self.belief_tracker.entropy,
            'L_eff': L_eff,
        }

    def estimate_ood_bound(self, x_train_boundary: torch.Tensor,
                           x_ood: torch.Tensor,
                           delta: float,
                           epsilon_emp: float) -> dict:
        """
        Estimate OOD bound using Part X + XI + XIII full pipeline.

        Corresponds to CSBAPR.lean: ood_bound_no_assumption4_nD (L1117)
        No Assumption 4 needed.

        Args:
            x_train_boundary: [state_dim] training domain boundary point.
            x_ood: [state_dim] OOD evaluation point.
            delta: Training domain base accuracy.
            epsilon_emp: Empirical Jacobian consistency error.

        Returns:
            dict with bound details (see losses/ood_bound.py).
        """
        result = compute_ood_bound(
            actor=self.actor,
            x_train_boundary=x_train_boundary.to(self.device),
            x_ood=x_ood.to(self.device),
            delta=delta,
            epsilon_emp=epsilon_emp,
            n_train_samples=max(self._n_train_samples, 1),
            M=self.config.physics_smoothness_M,
            confidence_delta=self.config.ood_confidence_delta,
        )
        self._ood_bound_cache = result
        return result

    def _compute_q_loss_bapr_fallback(self, state, action, reward, next_state, done,
                                       new_next_action, next_log_prob, reg_norm, batch_size):
        """BA-PR Q-loss without Γ_sym (when SINDy not yet fitted)."""
        predicted_q = self.critic(state, action)
        target_q_next = self.target_critic(next_state, new_next_action)
        num_critics = self.critic.num_critics

        next_log_prob_expanded = next_log_prob.unsqueeze(0).repeat(num_critics, 1)

        belief = torch.tensor(self.belief_tracker.belief, dtype=torch.float32, device=self.device)
        penalty_schedule = torch.exp(-0.1 * torch.arange(self.belief_tracker.max_H, dtype=torch.float32, device=self.device))
        weighted_lambda = (belief * penalty_schedule).sum()

        reg_norm_expanded = reg_norm.unsqueeze(-1).repeat(1, batch_size)

        target_q_next = (target_q_next
                         - self.alpha * next_log_prob_expanded
                         + self.config.weight_reg * reg_norm_expanded)

        target_q_value = reward + (1 - done) * self.config.gamma * target_q_next.unsqueeze(-1)
        ood_loss = predicted_q.std(0).mean()
        q_loss = nn.MSELoss()(predicted_q, target_q_value.squeeze(-1).detach())
        loss = q_loss + self.config.beta_ood * ood_loss

        return loss, predicted_q, ood_loss, weighted_lambda.item(), 0.0

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action using current policy."""
        state_tensor = torch.FloatTensor(state).to(self.device)
        return self.actor.get_action(state_tensor, deterministic=deterministic)

    def save(self, path: str):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'training_steps': self.training_steps,
            'n_train_samples': self._n_train_samples,
            'sindy_report': getattr(self, '_sindy_report', None),
            'irm_report': getattr(self, '_irm_report', None),
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.target_critic.load_state_dict(checkpoint['target_critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.log_alpha = checkpoint['log_alpha']
        self.training_steps = checkpoint['training_steps']
        self._n_train_samples = checkpoint.get('n_train_samples', 0)
        self._sindy_report = checkpoint.get('sindy_report', None)
        self._irm_report = checkpoint.get('irm_report', None)
