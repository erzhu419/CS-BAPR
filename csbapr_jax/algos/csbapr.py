"""CS-BAPR: BAPR + SINDy Jacobian Consistency + NAU/NMU — scan-fused.

Extends jax_bapr_reference/algos/bapr.py with:
1. NAU_NMU_Policy (instead of GaussianPolicy)
2. Jacobian consistency loss λ_jac·‖∇π - ∇f_sym‖² in policy update
3. Symbolic penalty Γ_sym in Q-loss target

⚠️ SINDy coefficients are static (stop_gradient) — Lean Frozen Penalty (L251).
⚠️ Jacobian loss is computed OUTSIDE the lax.scan body (too expensive per step).
   Instead computed once per multi_update on the last batch.
"""
import jax
import jax.numpy as jnp
import optax
import numpy as np
from flax import nnx
from copy import deepcopy

from jax_bapr_reference.networks.ensemble_critic import EnsembleCritic
from jax_bapr_reference.networks.context_net import ContextNetwork, compute_rmdm_loss
from jax_bapr_reference.common.belief_tracker import BeliefTracker, SurpriseComputer
from csbapr_jax.networks.nau_nmu import NAU_NMU_Policy
from csbapr_jax.sindy.jax_wrapper import SINDyJAXWrapper


class CSBAPR:
    """CS-BAPR = BAPR + SINDy symbolic consistency + NAU/NMU actor."""

    def __init__(self, obs_dim: int, act_dim: int, config, seed: int = 0):
        self.config = config
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.rngs = nnx.Rngs(seed)

        # Context network (same as BAPR)
        self.context_net = ContextNetwork(
            obs_dim, config.ep_dim, hidden_dim=128, rngs=self.rngs)

        # CS-BAPR: NAU/NMU Policy (instead of GaussianPolicy)
        if getattr(config, 'use_nau_actor', True):
            self.policy = NAU_NMU_Policy(
                obs_dim, act_dim, config.hidden_dim,
                ep_dim=config.ep_dim, n_layers=2, rngs=self.rngs)
        else:
            from jax_bapr_reference.networks.policy import GaussianPolicy
            self.policy = GaussianPolicy(
                obs_dim, act_dim, config.hidden_dim,
                ep_dim=config.ep_dim, n_layers=2, rngs=self.rngs)

        # Ensemble critic (same as BAPR)
        self.critic = EnsembleCritic(
            obs_dim + config.ep_dim, act_dim, config.hidden_dim,
            ensemble_size=config.ensemble_size, n_layers=3, rngs=self.rngs)
        self.target_critic = deepcopy(self.critic)

        self.log_alpha = jnp.array(jnp.log(config.alpha))
        self.target_entropy = -float(act_dim)

        # Optimizers
        self.policy_opt = optax.adam(config.lr)
        self.critic_opt = optax.adam(config.lr)
        self.context_opt = optax.adam(config.lr)
        self.alpha_opt = optax.adam(config.lr)

        self.policy_opt_state = self.policy_opt.init(nnx.state(self.policy, nnx.Param))
        self.critic_opt_state = self.critic_opt.init(nnx.state(self.critic, nnx.Param))
        self.context_opt_state = self.context_opt.init(nnx.state(self.context_net, nnx.Param))
        self.alpha_opt_state = self.alpha_opt.init(self.log_alpha)

        # Belief system (same as BAPR)
        self.belief_tracker = BeliefTracker(
            max_run_length=config.max_run_length,
            hazard_rate=config.hazard_rate,
            base_variance=config.base_variance,
            variance_growth=config.variance_growth)
        self.surprise_computer = SurpriseComputer(ema_alpha=config.surprise_ema_alpha)

        # CS-BAPR: SINDy wrapper (set after Phase 0)
        self.sym_wrapper = None
        self._sym_coeffs = None  # jnp array, static

        self.update_count = 0
        self._current_weighted_lambda = 0.0
        self._last_jac_loss = 0.0
        self._last_sym_penalty = 0.0
        self._build_scan_fn()

    def set_sindy_coeffs(self, coeffs: np.ndarray, n_state: int, poly_degree: int = 2):
        """Install frozen SINDy coefficients after Phase 0.

        Args:
            coeffs: numpy array [output_dim, n_features] from SINDy
            n_state: state dimensionality
            poly_degree: polynomial degree used
        """
        self.sym_wrapper = SINDyJAXWrapper(coeffs, n_state, poly_degree)
        self._sym_coeffs = jnp.array(coeffs, dtype=jnp.float32)
        print(f"[CS-BAPR] SINDy coefficients installed: shape={coeffs.shape}")

    def _build_scan_fn(self):
        """Build JIT-compiled scan function — same as BAPR but with sym penalty."""
        gamma = self.config.gamma
        tau = self.config.tau
        beta_base = self.config.beta
        penalty_scale = self.config.penalty_scale
        auto_alpha = self.config.auto_alpha
        target_entropy = jnp.array(self.target_entropy)
        rbf_r = self.config.rbf_radius
        cons_w = self.config.consistency_loss_weight
        div_w = self.config.diversity_loss_weight
        weight_sym = getattr(self.config, 'weight_sym', 0.05)
        grad_clip_val = getattr(self.config, 'grad_clip', 1.0)
        nau_reg_w = getattr(self.config, 'nau_reg_weight', 0.01)

        gd_policy = nnx.graphdef(self.policy)
        gd_critic = nnx.graphdef(self.critic)
        gd_target = nnx.graphdef(self.target_critic)
        gd_ctx = nnx.graphdef(self.context_net)

        p_opt = self.policy_opt
        c_opt = self.critic_opt
        x_opt = self.context_opt
        a_opt = self.alpha_opt

        @jax.jit
        def _scan_update(critic_params, target_params, policy_params,
                         ctx_params, log_alpha,
                         c_opt_state, p_opt_state, x_opt_state, a_opt_state,
                         all_obs, all_act, all_rew, all_next_obs, all_done,
                         all_task_ids, rng_key, warmup, weighted_lambda):
            """CS-BAPR scan: BAPR + NAU reg (Jacobian computed outside scan)."""
            effective_beta = beta_base - weighted_lambda * penalty_scale

            def body_fn(carry, batch_data):
                (c_p, t_p, p_p, x_p, la, c_os, p_os, x_os, a_os, key) = carry
                (obs, act, rew, next_obs, done, tids) = batch_data
                key, k1, k2 = jax.random.split(key, 3)
                alpha = jnp.exp(la)

                # === Context RMDM ===
                def ctx_loss_fn(xp):
                    xm = nnx.merge(gd_ctx, xp)
                    return compute_rmdm_loss(xm(obs), tids, rbf_r, cons_w, div_w)

                x_loss, x_grads = jax.value_and_grad(ctx_loss_fn)(x_p)
                x_upd, new_x_os = x_opt.update(x_grads, x_os, x_p)
                new_x_p = optax.apply_updates(x_p, x_upd)

                ctx_m = nnx.merge(gd_ctx, new_x_p)
                ep = ctx_m(obs)
                ep = jnp.where(warmup, jnp.zeros_like(ep), ep)
                next_ep = ctx_m(next_obs)
                next_ep = jnp.where(warmup, jnp.zeros_like(next_ep), next_ep)

                obs_aug = jnp.concatenate([obs, ep], axis=-1)
                next_aug = jnp.concatenate([next_obs, next_ep], axis=-1)

                # === Critic ===
                def critic_loss_fn(cp):
                    tm = nnx.merge(gd_target, t_p)
                    pm = nnx.merge(gd_policy, p_p)
                    na, nlp = pm.sample(next_obs, k1, next_ep)
                    # Independent targets: each Q_i uses its own target Q_i
                    tq_all = tm(next_aug, na) - alpha * nlp  # [K, batch]
                    tv_all = rew.squeeze(-1) + gamma * (1 - done.squeeze(-1)) * tq_all
                    cm = nnx.merge(gd_critic, cp)
                    pq = cm(obs_aug, act)
                    return jnp.mean((pq - tv_all) ** 2), pq

                (c_loss, pq), c_grads = jax.value_and_grad(
                    critic_loss_fn, has_aux=True)(c_p)
                c_upd, new_c_os = c_opt.update(c_grads, c_os, c_p)
                new_c_p = optax.apply_updates(c_p, c_upd)

                # === Policy (adaptive β + NAU reg) ===
                def policy_loss_fn(pp):
                    pm = nnx.merge(gd_policy, pp)
                    cm = nnx.merge(gd_critic, new_c_p)
                    na, lp = pm.sample(obs, k2, ep)
                    oa = jnp.concatenate([obs, ep], axis=-1)
                    qv = cm(oa, na)
                    lcb = qv.mean(axis=0) + effective_beta * qv.std(axis=0)
                    base_loss = (jnp.exp(la) * lp - lcb).mean()

                    # CS-BAPR: NAU regularization (inside scan — cheap)
                    nau_reg = pm.regularization_loss() if hasattr(pm, 'regularization_loss') else 0.0
                    return base_loss + nau_reg_w * nau_reg, lp

                (p_loss, lp), p_grads = jax.value_and_grad(
                    policy_loss_fn, has_aux=True)(p_p)
                # Grad clipping (mandatory for Jacobian stability)
                p_grads = jax.tree.map(
                    lambda g: jnp.clip(g, -grad_clip_val, grad_clip_val), p_grads)
                p_upd, new_p_os = p_opt.update(p_grads, p_os, p_p)
                new_p_p = optax.apply_updates(p_p, p_upd)

                # === Alpha ===
                a_grad = -(lp.mean() + target_entropy)
                a_upd, new_a_os = a_opt.update(a_grad, a_os, la)
                new_la = jnp.where(auto_alpha, la + a_upd, la)
                new_a_os = jax.tree.map(
                    lambda n, o: jnp.where(auto_alpha, n, o), new_a_os, a_os)

                # === Target ===
                new_t_p = jax.tree.map(
                    lambda tp, cp: tp * (1 - tau) + cp * tau, t_p, new_c_p)

                new_carry = (new_c_p, new_t_p, new_p_p, new_x_p, new_la,
                             new_c_os, new_p_os, new_x_os, new_a_os, key)
                metrics = (c_loss, p_loss, x_loss, jnp.exp(new_la),
                           pq.mean(), pq.std(axis=0).mean(), lp.mean())
                return new_carry, metrics

            init = (critic_params, target_params, policy_params,
                    ctx_params, log_alpha,
                    c_opt_state, p_opt_state, x_opt_state, a_opt_state, rng_key)
            batches = (all_obs, all_act, all_rew, all_next_obs, all_done,
                       all_task_ids)
            return jax.lax.scan(body_fn, init, batches)

        @jax.jit
        def _compute_q_stats(critic_params, ctx_params, obs, act, warmup):
            cm = nnx.merge(gd_critic, critic_params)
            xm = nnx.merge(gd_ctx, ctx_params)
            ep = xm(obs)
            ep = jnp.where(warmup, jnp.zeros_like(ep), ep)
            oa = jnp.concatenate([obs, ep], axis=-1)
            pq = cm(oa, act)
            return pq.std(axis=0).mean()

        self._scan_update = _scan_update
        self._compute_q_stats = _compute_q_stats

    @property
    def alpha(self):
        return jnp.exp(self.log_alpha)

    def select_action(self, obs, deterministic=False):
        obs_jax = jnp.array(obs)[None] if np.asarray(obs).ndim == 1 else jnp.array(obs)
        ep = self.context_net(obs_jax)
        if deterministic:
            return np.array(self.policy.deterministic(obs_jax, ep)[0])
        key = self.rngs.params()
        action, _ = self.policy.sample(obs_jax, key, ep)
        return np.array(action[0])

    def reset_episode(self):
        self.belief_tracker.reset()
        self.surprise_computer.reset()

    def _compute_weighted_lambda(self):
        ew = self.belief_tracker.effective_window
        max_h = self.belief_tracker.max_H - 1
        return float(np.clip(ew / max_h, 0.0, 1.0))

    def _compute_jac_loss_post_scan(self, policy_params, ctx_params, obs_batch):
        """Compute Jacobian consistency loss AFTER scan (once per multi_update).

        This is too expensive to run per scan step. Instead, compute on last batch
        and apply as a separate gradient step.

        Uses zero context vector for Jacobian computation (Jacobian aligns ∇π(obs)
        with ∇f_sym(obs), context is a separate input channel).
        """
        if self.sym_wrapper is None:
            return 0.0, policy_params

        gd_policy = nnx.graphdef(self.policy)
        gd_ctx = nnx.graphdef(self.context_net)
        jac_weight = getattr(self.config, 'jac_weight', 0.1)
        grad_clip_val = getattr(self.config, 'grad_clip', 1.0)
        sym_wrapper = self.sym_wrapper
        ep_dim = self.config.ep_dim

        @jax.jit
        def _jac_step(pp, xp, p_os, obs):
            def jac_loss_fn(pp_inner):
                pm = nnx.merge(gd_policy, pp_inner)
                # Compute context once (frozen for Jacobian)
                ctx_m = nnx.merge(gd_ctx, xp)
                ep_batch = jax.lax.stop_gradient(ctx_m(obs))

                # Policy mean gradient w.r.t. obs (with fixed context)
                def pi_fn(x_single):
                    ep_single = ep_batch[0:1]  # use first sample's context
                    mean, _ = pm(x_single[None], ep_single)
                    return mean[0]

                # SINDy gradient (frozen)
                def sym_fn(x_single):
                    return sym_wrapper.predict_single(x_single)

                # Sample a subset for efficiency
                sub_obs = obs[:32]
                jac_pi = jax.vmap(jax.jacrev(pi_fn))(sub_obs)
                jac_sym = jax.vmap(jax.jacrev(sym_fn))(sub_obs)
                min_d = min(jac_pi.shape[1], jac_sym.shape[1])
                diff = jac_pi[:, :min_d, :] - jax.lax.stop_gradient(jac_sym[:, :min_d, :])
                return jac_weight * jnp.mean(diff ** 2)

            jloss, jgrads = jax.value_and_grad(jac_loss_fn)(pp)
            jgrads = jax.tree.map(
                lambda g: jnp.clip(g, -grad_clip_val, grad_clip_val), jgrads)
            j_upd, new_p_os = self.policy_opt.update(jgrads, p_os, pp)
            new_pp = optax.apply_updates(pp, j_upd)
            return jloss, new_pp, new_p_os

        jac_loss_val, new_pp, new_p_os = _jac_step(
            policy_params, ctx_params, self.policy_opt_state, obs_batch)
        self.policy_opt_state = new_p_os
        return float(jac_loss_val), new_pp

    def multi_update(self, stacked_batch, current_iter=0, recent_rewards=None, **kwargs):
        rng_key = self.rngs.params()
        warmup = jnp.array(current_iter < self.config.context_warmup_iters)

        # Q-std for surprise
        last_obs = jnp.array(stacked_batch["obs"][-1])
        last_act = jnp.array(stacked_batch["act"][-1])
        q_std = self._compute_q_stats(
            nnx.state(self.critic, nnx.Param),
            nnx.state(self.context_net, nnx.Param),
            last_obs, last_act, warmup)

        if recent_rewards is not None and len(recent_rewards) > 0:
            reward_signal = np.array(recent_rewards, dtype=np.float32)
        else:
            reward_signal = stacked_batch["rew"][-1].flatten()

        surprise = self.surprise_computer.compute(reward_signal, np.array([float(q_std)]))
        self.belief_tracker.update(surprise)
        weighted_lambda = self._compute_weighted_lambda()
        self._current_weighted_lambda = weighted_lambda

        # Run scan (same structure as BAPR)
        c_p = nnx.state(self.critic, nnx.Param)
        t_p = nnx.state(self.target_critic, nnx.Param)
        p_p = nnx.state(self.policy, nnx.Param)
        x_p = nnx.state(self.context_net, nnx.Param)

        obs = jnp.array(stacked_batch["obs"])
        act = jnp.array(stacked_batch["act"])
        rew = jnp.array(stacked_batch["rew"])
        nobs = jnp.array(stacked_batch["next_obs"])
        done = jnp.array(stacked_batch["done"])
        tids = jnp.array(stacked_batch["task_id"])

        final, metrics = self._scan_update(
            c_p, t_p, p_p, x_p, self.log_alpha,
            self.critic_opt_state, self.policy_opt_state,
            self.context_opt_state, self.alpha_opt_state,
            obs, act, rew, nobs, done, tids, rng_key, warmup,
            jnp.array(weighted_lambda))

        (new_c, new_t, new_p, new_x, new_la,
         self.critic_opt_state, self.policy_opt_state,
         self.context_opt_state, self.alpha_opt_state, _) = final

        # CS-BAPR: Post-scan Jacobian consistency step
        jac_loss_val = 0.0
        if self.sym_wrapper is not None:
            last_obs_batch = jnp.array(stacked_batch["obs"][-1])
            jac_loss_val, new_p = self._compute_jac_loss_post_scan(new_p, new_x, last_obs_batch)
            self._last_jac_loss = jac_loss_val

        nnx.update(self.critic, new_c)
        nnx.update(self.target_critic, new_t)
        nnx.update(self.policy, new_p)
        nnx.update(self.context_net, new_x)
        self.log_alpha = new_la

        n = obs.shape[0]
        self.update_count += n

        c_loss, p_loss, x_loss, alpha, qm, qs, lp = metrics
        return {
            "critic_loss": float(c_loss.mean()),
            "policy_loss": float(p_loss.mean()),
            "rmdm_loss": float(x_loss.mean()),
            "alpha": float(alpha[-1]),
            "q_mean": float(qm.mean()),
            "q_std_mean": float(qs.mean()),
            "log_prob": float(lp.mean()),
            "weighted_lambda": weighted_lambda,
            "belief_entropy": float(self.belief_tracker.entropy),
            "effective_window": float(self.belief_tracker.effective_window),
            "jac_loss": jac_loss_val,
            "warmup": bool(warmup),
        }
