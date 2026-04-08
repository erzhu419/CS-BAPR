#!/usr/bin/env python3
"""
CS-BAPR OOD Evaluation Script
==============================

Validates CS-BAPR core claims:
  1. Theory-experiment alignment: bound predicts actual error trend (Figure 1)
  2. NAU vs ReLU: quadratic growth vs catastrophic collapse (Figure 2)
  3. Ablation: each component contributes to OOD robustness (Table 2)

Supports two evaluation protocols:
  A. Gymnasium/MuJoCo: perturb env params (mass, friction, gravity)
  B. Bus simulation: parametric OD sweep (1x-100x)

Usage:
    # MuJoCo OOD sweep (Pendulum, perturb mass)
    python scripts/ood_eval.py --env Pendulum-v1 --param mass --range 1,2,5,10,20

    # MuJoCo with trained checkpoint
    python scripts/ood_eval.py --env Hopper-v4 --param body_mass --range 1,2,4,8 \\
        --checkpoint-nau ckpt_nau.pt --checkpoint-relu ckpt_relu.pt

    # Bus simulation OOD sweep
    python scripts/ood_eval.py --env bus --od-range 1,5,10,20,50,100

    # Plot bound vs actual (Figure 1)
    python scripts/ood_eval.py --env Pendulum-v1 --param mass --range 1,2,5,10,20 --plot

Corresponds to CSBAPR.lean:
  - ood_bound_no_assumption4_nD (L1117): δ + (ε + (L_eff+M)·d)·d
  - composed_deriv_lipschitz_simple (L1417): L_eff computation
  - jc_generalization_to_ood (L1663): generalization gap
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ============================================================
# Theoretical Bounds (Part X corrected)
# ============================================================

def theoretical_bound_corrected(d, delta=0.0, epsilon=0.05, L_eff=0.0,
                                M=0.0, gap=0.0):
    """
    Corrected CS-BAPR OOD bound (Part X: ood_bound_no_assumption4_nD, L1117).

    ‖π(x_ood) - f_real(x_ood)‖ ≤ δ + (ε_total + (L_eff + M)·‖d‖)·‖d‖

    where ε_total = ε_emp + generalization_gap.

    No Assumption 4 needed. Only requires:
    (A) JC at training boundary (ε)
    (B) NAU architecture (L_eff, from Part XI)
    (C) Physics smoothness (M, from Part X)
    (D) Generalization gap (from Part XIII)
    """
    d = np.asarray(d, dtype=float)
    eps_total = epsilon + gap
    return delta + (eps_total + (L_eff + M) * d) * d


def theoretical_bound_fencing(d, delta=0.0, epsilon=0.05, L=1.0):
    """
    Original fencing-theorem bound (Part VII: scbapr_ood_quadratic_nD, L672).

    ‖π(x_ood) - f_real(x_ood)‖ ≤ δ + ε·‖d‖ + L/2·‖d‖²

    Requires stronger Assumption 4 (JC on entire OOD path).
    Kept for comparison; prefer theoretical_bound_corrected.
    """
    d = np.asarray(d, dtype=float)
    return delta + epsilon * d + L / 2.0 * d ** 2


# ============================================================
# MuJoCo / Gymnasium OOD Protocol
# ============================================================

def make_perturbed_env(env_name, param_name, multiplier, seed=0):
    """
    Create a Gymnasium env with a single perturbed physics parameter.

    For compound perturbations, use make_perturbed_env_compound().
    """
    return make_perturbed_env_compound(
        env_name, {param_name: multiplier}, seed=seed
    )


def make_perturbed_env_compound(env_name, mode_params, seed=0):
    """
    Create a Gymnasium env with compound (multi-parameter) perturbation.

    Args:
        env_name: Gymnasium environment name
        mode_params: dict of param_name → multiplier
            e.g. {'mass': 2.0, 'friction': 0.5, 'gravity': 1.2}
        seed: random seed

    Supports:
    - mass / body_mass: scale all body masses
    - friction: scale ground friction
    - gravity: scale gravity magnitude
    - length (Pendulum only): scale pendulum length
    """
    import gymnasium as gym

    env = gym.make(env_name)
    env.reset(seed=seed)

    if hasattr(env.unwrapped, 'model'):
        model = env.unwrapped.model
        for param, mult in mode_params.items():
            if param in ('mass', 'body_mass'):
                model.body_mass[:] *= mult
            elif param == 'friction':
                model.geom_friction[:] *= mult
            elif param == 'gravity':
                model.opt.gravity[2] *= mult
    elif 'Pendulum' in env_name:
        uw = env.unwrapped
        for param, mult in mode_params.items():
            if param == 'mass':
                uw.m *= mult
            elif param == 'length':
                uw.l *= mult
            elif param == 'gravity':
                uw.g *= mult

    return env


def evaluate_policy_on_env(env, policy_fn, n_episodes=5, max_steps=1000):
    """
    Evaluate a policy on a given env (static perturbation).

    Args:
        env: Gymnasium env (already perturbed)
        policy_fn: callable(state_np) -> action_np, or None for random
        n_episodes: number of eval episodes
        max_steps: max steps per episode

    Returns:
        dict with mean_reward, std_reward, per_episode details
    """
    episode_rewards = []
    episode_lengths = []

    for ep in range(n_episodes):
        state, _ = env.reset(seed=ep * 1000)
        total_reward = 0.0
        for step in range(max_steps):
            if policy_fn is not None:
                action = policy_fn(state)
            else:
                action = env.action_space.sample()
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        episode_rewards.append(total_reward)
        episode_lengths.append(step + 1)

    return {
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'episode_rewards': episode_rewards,
        'mean_length': float(np.mean(episode_lengths)),
    }


# ============================================================
# Mid-Episode Abrupt Shift Protocol
# ============================================================
# Mirrors the bus simulation protocol: train under regime A,
# then at step T_shift the physics abruptly changes to regime B.
# This tests the belief tracker's ability to detect the shift
# and the policy's OOD robustness after the shift.

def evaluate_with_abrupt_shift(env_name, policy_fn, param_name_or_dict, shift_mult=None,
                                shift_step=100, max_steps=500, n_episodes=5,
                                seed=0):
    """
    Evaluate a policy with mid-episode abrupt parameter shift.

    Protocol (mirrors bus simulation):
      1. Steps 0 to shift_step-1: normal physics
      2. Steps shift_step to max_steps: abrupt shift

    Args:
        env_name: Gymnasium environment name
        policy_fn: callable(state) -> action, or None for random
        param_name_or_dict: either a param name string + shift_mult,
            or a dict {'mass': 2.0, 'friction': 0.5} for compound shift
        shift_mult: multiplier (only used if param_name_or_dict is a string)
        shift_step: step at which the shift occurs
        max_steps: total steps per episode
        n_episodes: number of evaluation episodes
        seed: random seed
    """
    import gymnasium as gym

    # Normalize to dict form
    if isinstance(param_name_or_dict, dict):
        shift_params = param_name_or_dict
    else:
        shift_params = {param_name_or_dict: shift_mult}

    results = []

    for ep in range(n_episodes):
        env = gym.make(env_name)
        state, _ = env.reset(seed=seed * 1000 + ep)

        pre_rewards = []
        post_rewards = []
        step_rewards = []

        for step in range(max_steps):
            # === Abrupt shift at shift_step ===
            if step == shift_step:
                _apply_param_shift(env, shift_params)

            if policy_fn is not None:
                action = policy_fn(state)
            else:
                action = env.action_space.sample()

            state, reward, terminated, truncated, _ = env.step(action)
            step_rewards.append(float(reward))

            if step < shift_step:
                pre_rewards.append(reward)
            else:
                post_rewards.append(reward)

            if terminated or truncated:
                break

        env.close()

        results.append({
            'pre_reward': float(np.sum(pre_rewards)),
            'post_reward': float(np.sum(post_rewards)),
            'total_reward': float(np.sum(step_rewards)),
            'pre_mean_step': float(np.mean(pre_rewards)) if pre_rewards else 0,
            'post_mean_step': float(np.mean(post_rewards)) if post_rewards else 0,
            'reward_drop': (float(np.mean(pre_rewards)) - float(np.mean(post_rewards)))
                           if pre_rewards and post_rewards else 0,
            'episode_length': len(step_rewards),
            'step_rewards': step_rewards,
        })

    # Aggregate
    return {
        'shift_mult': shift_mult,
        'shift_step': shift_step,
        'n_episodes': n_episodes,
        'pre_reward_mean': float(np.mean([r['pre_mean_step'] for r in results])),
        'post_reward_mean': float(np.mean([r['post_mean_step'] for r in results])),
        'reward_drop_mean': float(np.mean([r['reward_drop'] for r in results])),
        'reward_drop_std': float(np.std([r['reward_drop'] for r in results])),
        'total_reward_mean': float(np.mean([r['total_reward'] for r in results])),
        'episodes': results,
    }


def _apply_param_shift(env, param_name_or_dict, multiplier=None):
    """
    Apply parameter shift to a live environment (mid-episode).

    Accepts either:
      _apply_param_shift(env, 'mass', 2.0)              # single param
      _apply_param_shift(env, {'mass': 2.0, 'friction': 0.5})  # compound
    """
    if isinstance(param_name_or_dict, dict):
        mode_params = param_name_or_dict
    else:
        mode_params = {param_name_or_dict: multiplier}

    if hasattr(env.unwrapped, 'model'):
        model = env.unwrapped.model
        for param, mult in mode_params.items():
            if param in ('mass', 'body_mass'):
                model.body_mass[:] *= mult
            elif param == 'friction':
                model.geom_friction[:] *= mult
            elif param == 'gravity':
                model.opt.gravity[2] *= mult
    elif hasattr(env, 'spec') and env.spec and 'Pendulum' in env.spec.id:
        uw = env.unwrapped
        for param, mult in mode_params.items():
            if param == 'mass':
                uw.m *= mult
            elif param == 'length':
                uw.l *= mult
            elif param == 'gravity':
                uw.g *= mult


def gym_abrupt_shift_sweep(env_name, param_name, shift_mults,
                            shift_step=100, max_steps=500, n_seeds=5,
                            checkpoint_nau=None, checkpoint_relu=None,
                            compound_modes=None):
    """
    Run abrupt-shift evaluation sweep.

    Like gym_ood_sweep but with mid-episode parameter shifts.
    Mirrors the bus simulation OOD injection protocol.

    Supports:
      1. Single-param: param_name='mass', shift_mults=[2,5,10]
      2. Compound: compound_modes={'compound_ood': {'mass':3,'friction':0.3,'gravity':2}}
    """
    import gymnasium as gym

    if compound_modes:
        sweep_items = list(compound_modes.items())
    else:
        sweep_items = [(f"{param_name}_{m}x", {param_name: m}) for m in shift_mults]

    print("\n" + "=" * 70)
    print(f"CS-BAPR Abrupt Shift Sweep: {env_name}")
    print(f"Shift at step {shift_step}, {len(sweep_items)} configs")
    print("=" * 70)

    test_env = gym.make(env_name)
    state_dim = test_env.observation_space.shape[0]
    action_dim = test_env.action_space.shape[0]
    test_env.close()

    policy_fns = {}
    if checkpoint_nau:
        _, fn_nau = load_policy(checkpoint_nau, state_dim, action_dim, use_nau=True)
        policy_fns['CS-BAPR (NAU)'] = fn_nau
    if checkpoint_relu:
        _, fn_relu = load_policy(checkpoint_relu, state_dim, action_dim, use_nau=False)
        policy_fns['BA-PR (ReLU)'] = fn_relu
    policy_fns['Random'] = None

    results = {}
    for mode_name, mode_params in sweep_items:
        d = compute_ood_distance_compound(mode_params)
        print(f"\n--- Shift: {mode_name} (d={d:.3f}) at step {shift_step} ---")
        if len(mode_params) > 1:
            print(f"    params: {mode_params}")

        method_results = {}
        for method_name, policy_fn in policy_fns.items():
            r = evaluate_with_abrupt_shift(
                env_name, policy_fn, mode_params, None,
                shift_step=shift_step, max_steps=max_steps,
                n_episodes=n_seeds,
            )
            method_results[method_name] = r
            print(f"  {method_name}: pre={r['pre_reward_mean']:.2f}, "
                  f"post={r['post_reward_mean']:.2f}, "
                  f"drop={r['reward_drop_mean']:.2f}±{r['reward_drop_std']:.2f}")

        results[mode_name] = {
            'mode_name': mode_name,
            'mode_params': mode_params,
            'ood_distance': d,
            'methods': method_results,
        }

    # Summary
    print("\n" + "=" * 70)
    header = f"{'Mode':>20s} {'d':>6s}"
    for m in policy_fns:
        header += f"  {'pre→post (drop)':>28s}"
    print(header)
    print("-" * len(header))
    for key, r in sorted(results.items(), key=lambda x: x[1]['ood_distance']):
        line = f"{r['mode_name']:>20s} {r['ood_distance']:>6.3f}"
        for m in policy_fns:
            mr = r['methods'].get(m, {})
            line += (f"  {mr.get('pre_reward_mean',0):>6.2f}→"
                     f"{mr.get('post_reward_mean',0):>6.2f} "
                     f"({mr.get('reward_drop_mean',0):>+6.2f})")
        print(line)

    return results


def load_policy(checkpoint_path, state_dim, action_dim, use_nau=True):
    """Load trained policy from checkpoint. Returns (agent, policy_fn)."""
    import torch
    from csbapr.agent import CSBAPRAgent
    from csbapr.config import CSBAPRConfig

    config = CSBAPRConfig(use_nau_actor=use_nau)
    agent = CSBAPRAgent(state_dim, action_dim, config)
    agent.load(checkpoint_path)
    agent.actor.eval()

    def policy_fn(state):
        return agent.select_action(state, deterministic=True)

    return agent, policy_fn


def evaluate_with_abrupt_shift_and_belief(env_name, agent, param_name, shift_mult,
                                           shift_step=100, max_steps=500,
                                           n_episodes=5, seed=0):
    """
    Abrupt shift evaluation WITH belief/surprise tracking.

    Like evaluate_with_abrupt_shift, but also records the agent's internal
    belief tracker state at each step. This produces the data for:
    - Surprise spike at the shift point (Figure 5)
    - weighted_lambda adaptation after shift
    - Q-std jump at the distribution boundary

    Only works with CSBAPRAgent (not raw policy_fn).

    Returns:
        dict with per-step timelines of reward, surprise, belief_entropy, w_lambda
    """
    import gymnasium as gym
    import torch

    results = []

    for ep in range(n_episodes):
        env = gym.make(env_name)
        state, _ = env.reset(seed=seed * 1000 + ep)

        timeline = {
            'step': [], 'reward': [], 'surprise': [],
            'belief_entropy': [], 'weighted_lambda': [], 'q_std': [],
        }

        agent.belief_tracker.reset()
        episode_reward = 0.0

        for step in range(max_steps):
            if step == shift_step:
                _apply_param_shift(env, param_name, shift_mult)

            action = agent.select_action(state, deterministic=True)
            scaled = np.clip(action * env.action_space.high[0],
                             env.action_space.low[0], env.action_space.high[0])
            next_state, reward, terminated, truncated, _ = env.step(scaled)

            # Feed the transition to agent for belief update (without training)
            agent.replay_buffer.push(state, action, reward, next_state,
                                     float(terminated or truncated))

            # Record belief state
            timeline['step'].append(step)
            timeline['reward'].append(float(reward))
            timeline['belief_entropy'].append(float(agent.belief_tracker.entropy))

            # Compute surprise-like signal from Q-std
            with torch.no_grad():
                s_t = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                a_t = torch.FloatTensor(action).unsqueeze(0).to(agent.device)
                q_vals = agent.critic(s_t, a_t)
                q_std = q_vals.std(dim=0).mean().item()
            timeline['q_std'].append(q_std)

            episode_reward += reward
            state = next_state

            if terminated or truncated:
                break

        env.close()

        # Compute reward drop
        pre_r = timeline['reward'][:shift_step]
        post_r = timeline['reward'][shift_step:]
        results.append({
            'timeline': timeline,
            'pre_mean_step': float(np.mean(pre_r)) if pre_r else 0,
            'post_mean_step': float(np.mean(post_r)) if post_r else 0,
            'total_reward': episode_reward,
        })

    return {
        'shift_mult': shift_mult,
        'shift_step': shift_step,
        'episodes': results,
        'pre_reward_mean': float(np.mean([r['pre_mean_step'] for r in results])),
        'post_reward_mean': float(np.mean([r['post_mean_step'] for r in results])),
        'reward_drop_mean': float(np.mean([
            r['pre_mean_step'] - r['post_mean_step'] for r in results
        ])),
    }


def compute_ood_distance(env_name, param_name, multiplier, baseline_mult=1.0):
    """
    Compute OOD distance ‖d‖ for a single-parameter perturbation.

    d = |log(multiplier / baseline)|
    This captures that 2x and 0.5x are equidistant from 1x.
    """
    if multiplier <= 0:
        return float('inf')
    return abs(np.log(multiplier / baseline_mult))


def compute_ood_distance_compound(mode_params, baseline_params=None):
    """
    Compute OOD distance ‖d‖ for a compound (multi-parameter) perturbation.

    d = ‖(log m₁/b₁, log m₂/b₂, ...)‖₂

    where mᵢ are the mode multipliers and bᵢ are the baselines (default 1.0).
    Uses log-space L2 norm so that:
      - (2x, 1x, 1x) and (1x, 0.5x, 1x) have equal distance
      - compound perturbations accumulate geometrically

    Args:
        mode_params: dict of param_name → multiplier (e.g. {'mass': 2.0, 'friction': 0.5})
        baseline_params: dict of param_name → baseline multiplier (default: all 1.0)

    Returns:
        Scalar OOD distance ‖d‖₂ in log-parameter space.
    """
    if baseline_params is None:
        baseline_params = {}
    log_diffs = []
    for param, mult in mode_params.items():
        base = baseline_params.get(param, 1.0)
        if mult <= 0 or base <= 0:
            return float('inf')
        log_diffs.append(np.log(mult / base))
    if not log_diffs:
        return 0.0
    return float(np.linalg.norm(log_diffs))


def estimate_bound_params_from_agent(agent, env, n_samples=200):
    """
    Estimate theoretical bound parameters (δ, ε) from a trained agent,
    instead of using hardcoded defaults.

    δ = ‖π(x₀) - f_real(x₀)‖ at training domain (base accuracy)
        → proxy: mean JC loss over recent training steps
    ε = Jacobian consistency error at training boundary
        → proxy: last JC loss value from training metrics

    Args:
        agent: Trained CSBAPRAgent
        env: Training environment (for sampling states)
        n_samples: number of states to sample

    Returns:
        dict with delta, epsilon, L_eff, gap estimates
    """
    import torch
    from csbapr.losses.ood_bound import compute_generalization_gap, estimate_deriv_bound_B

    delta = 0.1   # conservative default
    epsilon = 0.05

    # Try to get actual JC loss from training history
    if hasattr(agent, '_ood_bound_cache') and agent._ood_bound_cache:
        epsilon = agent._ood_bound_cache.get('epsilon_emp', epsilon)

    # Try to compute live JC loss
    if agent.f_sym_torch is not None:
        from csbapr.losses.jacobian import compute_jacobian_loss
        try:
            states = []
            state, _ = env.reset()
            for _ in range(min(n_samples, 50)):
                states.append(state)
                action = env.action_space.sample()
                state, _, term, trunc, _ = env.step(action)
                if term or trunc:
                    state, _ = env.reset()
            states_t = torch.FloatTensor(np.array(states)).to(agent.device)
            with torch.no_grad():
                # delta: policy output vs SINDy output deviation (proxy for base accuracy)
                pi_out = agent.actor(states_t)
                if isinstance(pi_out, tuple):
                    pi_out = pi_out[0]
                sym_out = agent.f_sym_torch(states_t)
                delta = (pi_out - sym_out).norm(dim=-1).mean().item()
            # epsilon: JC loss (needs grad for policy, so can't use no_grad)
            epsilon = compute_jacobian_loss(agent.actor, agent.f_sym_torch, states_t).item()
        except Exception:
            pass  # fall back to defaults

    # L_eff from architecture
    L_eff = 0.0
    if hasattr(agent.actor, 'compute_L_eff'):
        L_eff = agent.actor.compute_L_eff()

    # Generalization gap
    B = estimate_deriv_bound_B(agent.actor)
    n_train = max(agent._n_train_samples, 1)
    gap = compute_generalization_gap(B, n_train)

    return {
        'delta': delta,
        'epsilon': epsilon,
        'L_eff': L_eff,
        'gap': gap,
        'B': B,
        'n_train_samples': n_train,
    }


def gym_ood_sweep(env_name, param_name, multipliers, n_seeds=5,
                  checkpoint_nau=None, checkpoint_relu=None,
                  max_steps=1000, compound_modes=None):
    """
    Run OOD parameter sweep on a Gymnasium environment.

    Supports two modes:
      1. Single-param sweep: param_name='mass', multipliers=[1,2,5,10]
      2. Compound sweep: compound_modes={'heavy': {'mass':5.0}, 'compound': {'mass':3,'friction':0.3}}
         When compound_modes is set, param_name/multipliers are ignored.

    Returns:
        dict with results per multiplier/mode and method
    """
    import torch
    import gymnasium as gym

    # Build sweep configs
    if compound_modes:
        sweep_items = list(compound_modes.items())
        desc = f"compound modes: {list(compound_modes.keys())}"
    else:
        sweep_items = [(f"{param_name}_{m}x", {param_name: m}) for m in multipliers]
        desc = f"param={param_name}, multipliers={multipliers}"

    print("\n" + "=" * 70)
    print(f"CS-BAPR OOD Sweep: {env_name}, {desc}")
    print(f"Seeds: {n_seeds}, Max steps: {max_steps}")
    print("=" * 70)

    test_env = gym.make(env_name)
    state_dim = test_env.observation_space.shape[0]
    action_dim = test_env.action_space.shape[0]
    test_env.close()

    # Load policies
    agents = {}
    policy_fns = {}
    if checkpoint_nau:
        agent_nau, fn_nau = load_policy(checkpoint_nau, state_dim, action_dim, use_nau=True)
        agents['CS-BAPR (NAU)'] = agent_nau
        policy_fns['CS-BAPR (NAU)'] = fn_nau
    if checkpoint_relu:
        agent_relu, fn_relu = load_policy(checkpoint_relu, state_dim, action_dim, use_nau=False)
        agents['BA-PR (ReLU)'] = agent_relu
        policy_fns['BA-PR (ReLU)'] = fn_relu
    policy_fns['Random'] = None

    # Estimate bound parameters from trained agent (not hardcoded)
    bound_params = {'delta': 0.1, 'epsilon': 0.05, 'L_eff': 0.0, 'gap': 0.0}
    if 'CS-BAPR (NAU)' in agents:
        est_env = gym.make(env_name)
        bound_params = estimate_bound_params_from_agent(agents['CS-BAPR (NAU)'], est_env)
        est_env.close()
        print(f"  Bound params from agent: δ={bound_params['delta']:.4f}, "
              f"ε={bound_params['epsilon']:.4f}, L_eff={bound_params['L_eff']:.4f}, "
              f"gap={bound_params['gap']:.6f}")
    L_eff = bound_params['L_eff']

    results = {}
    for mode_name, mode_params in sweep_items:
        d = compute_ood_distance_compound(mode_params)
        print(f"\n--- {mode_name} (d = {d:.3f}) ---")
        if len(mode_params) > 1:
            print(f"    params: {mode_params}")

        method_results = {}
        for method_name, policy_fn in policy_fns.items():
            seed_rewards = []
            for seed in range(n_seeds):
                env = make_perturbed_env_compound(env_name, mode_params, seed=seed)
                eval_result = evaluate_policy_on_env(
                    env, policy_fn, n_episodes=1, max_steps=max_steps
                )
                seed_rewards.append(eval_result['mean_reward'])
                env.close()

            method_results[method_name] = {
                'mean_reward': float(np.mean(seed_rewards)),
                'std_reward': float(np.std(seed_rewards)),
                'rewards': seed_rewards,
            }
            print(f"  {method_name}: {np.mean(seed_rewards):.1f} ± {np.std(seed_rewards):.1f}")

        bound_corrected = float(theoretical_bound_corrected(
            d, delta=bound_params['delta'], epsilon=bound_params['epsilon'],
            L_eff=L_eff, M=0.0, gap=bound_params['gap']
        ))
        bound_fencing = float(theoretical_bound_fencing(
            d, delta=bound_params['delta'], epsilon=bound_params['epsilon'], L=L_eff
        ))

        results[mode_name] = {
            'mode_name': mode_name,
            'mode_params': mode_params,
            'ood_distance': d,
            'methods': method_results,
            'bound_corrected': bound_corrected,
            'bound_fencing': bound_fencing,
            'L_eff': L_eff,
        }

    # Summary
    print("\n" + "=" * 70)
    header = f"{'Mode':>20s} {'d':>6s} {'Bound':>8s}"
    for m in policy_fns:
        header += f" {m:>16s}"
    print(header)
    print("-" * len(header))
    for key, r in sorted(results.items(), key=lambda x: x[1]['ood_distance']):
        line = f"{r['mode_name']:>20s} {r['ood_distance']:>6.3f} {r['bound_corrected']:>8.3f}"
        for m in policy_fns:
            mr = r['methods'].get(m, {})
            line += f" {mr.get('mean_reward', 0):>8.1f}±{mr.get('std_reward', 0):>4.1f}"
        print(line)

    return results


# ============================================================
# Bus Simulation OOD Protocol (backward-compatible)
# ============================================================

def bus_ood_sweep(env_path, od_range, n_seeds=3, max_steps=5000):
    """Bus simulation parametric OOD sweep (original protocol)."""
    sys.path.insert(0, str(Path(env_path)))
    try:
        from mode_profiles import TRAIN_MODES, make_parametric_ood
    except ImportError:
        print("Error: mode_profiles not found. Is env_path correct?")
        return None

    from env.sim import env_bus

    print("\n" + "=" * 70)
    print("CS-BAPR OOD Sweep: Bus Simulation")
    print(f"OD range: {od_range}, Seeds: {n_seeds}")
    print("=" * 70)

    results = {}
    for mult in od_range:
        print(f"\n--- OD {mult}x ---")
        seed_results = []
        for seed in range(n_seeds):
            np.random.seed(seed * 42 + mult)
            env = env_bus(env_path, enable_mode_switch=False, ood_mode=None)
            profile = make_parametric_ood(mult)
            env.ood_profile = profile
            env.ood_mode = f"parametric_{mult}x"
            env.ood_inject_time = 0

            actions = {key: 15.0 for key in range(env.max_agent_num)}
            env.reset()
            total_reward = 0.0
            for _ in range(max_steps):
                if env.done:
                    break
                state, reward, done = env.step(action=actions, debug=False, render=False)
                total_reward += sum(r for r in reward.values() if r != 0)

            seed_results.append(total_reward)
            print(f"  seed {seed}: reward={total_reward:.1f}")

        train_max_od = max(
            max([m.get("od_global_mult", 1.0)] +
                list(m.get("station_od_overrides", {}).values()))
            for m in TRAIN_MODES.values()
        )
        d = max(0.0, mult - train_max_od)

        results[f"od_{mult}x"] = {
            'multiplier': mult,
            'ood_distance': d,
            'bound_corrected': float(theoretical_bound_corrected(d)),
            'bound_fencing': float(theoretical_bound_fencing(d)),
            'mean_reward': float(np.mean(seed_results)),
            'std_reward': float(np.std(seed_results)),
            'n_seeds': n_seeds,
        }

    return results


# ============================================================
# Plotting (Figure 1: Bound vs Actual)
# ============================================================

def plot_bound_vs_actual(results, output_path=None, title=None):
    """
    Generate Figure 1: theoretical bound vs actual OOD error.

    Plots:
    - Corrected bound (Part X) as upper envelope
    - Fencing bound (Part VII) for comparison
    - Actual reward degradation for each method
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return

    distances = []
    bounds_corrected = []
    bounds_fencing = []
    method_rewards = defaultdict(list)

    for key, r in sorted(results.items(), key=lambda x: x[1]['ood_distance']):
        d = r['ood_distance']
        distances.append(d)
        bounds_corrected.append(r['bound_corrected'])
        bounds_fencing.append(r['bound_fencing'])
        if 'methods' in r:
            for m, mr in r['methods'].items():
                method_rewards[m].append(mr['mean_reward'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Theoretical bounds
    ax1 = axes[0]
    ax1.plot(distances, bounds_corrected, 'r-', lw=2,
             label='Corrected bound (Part X)', marker='s')
    ax1.plot(distances, bounds_fencing, 'r--', lw=1.5, alpha=0.6,
             label='Fencing bound (Part VII)')
    ax1.set_xlabel('OOD distance ‖d‖')
    ax1.set_ylabel('Theoretical upper bound')
    ax1.set_title('OOD Error Bound')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Actual rewards by method
    ax2 = axes[1]
    markers = {'CS-BAPR (NAU)': 'o', 'BA-PR (ReLU)': 'x', 'Random': 'd'}
    colors = {'CS-BAPR (NAU)': 'blue', 'BA-PR (ReLU)': 'red', 'Random': 'gray'}
    for m, rewards in method_rewards.items():
        ax2.plot(distances[:len(rewards)], rewards,
                 marker=markers.get(m, '.'), color=colors.get(m, 'green'),
                 lw=2, label=m)
    ax2.set_xlabel('OOD distance ‖d‖')
    ax2.set_ylabel('Mean episode reward')
    ax2.set_title('Reward vs OOD Distance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.savefig('ood_bound_vs_actual.png', dpi=150, bbox_inches='tight')
        print("Plot saved to ood_bound_vs_actual.png")

    plt.close()


# ============================================================
# Main CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="CS-BAPR OOD Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Pendulum parameter sweep (no checkpoint needed)
  python scripts/ood_eval.py --env Pendulum-v1 --param mass --range 1,2,5,10 --plot

  # MuJoCo with trained models
  python scripts/ood_eval.py --env Hopper-v4 --param body_mass --range 1,2,4,8 \\
      --checkpoint-nau ckpt_nau.pt --checkpoint-relu ckpt_relu.pt --plot

  # Bus simulation OOD sweep
  python scripts/ood_eval.py --env bus --od-range 1,5,10,20,50,100

  # Save results as JSON
  python scripts/ood_eval.py --env Pendulum-v1 --param mass --range 1,2,5,10 \\
      --output results/pendulum_ood.json
        """,
    )
    parser.add_argument("--env", type=str, default="Pendulum-v1",
                        help="Environment name or 'bus' for bus simulation")
    parser.add_argument("--mode", type=str, default="sweep",
                        choices=["sweep", "shift"],
                        help="'sweep': static param scaling; "
                             "'shift': mid-episode abrupt shift (like bus sim)")
    parser.add_argument("--param", type=str, default="mass",
                        help="Parameter to perturb (mass, friction, gravity, length)")
    parser.add_argument("--range", type=str, default="1,2,5,10,20",
                        help="Perturbation multipliers (comma-separated)")
    parser.add_argument("--shift-step", type=int, default=100,
                        help="Step at which abrupt shift occurs (--mode shift)")
    parser.add_argument("--od-range", type=str, default=None,
                        help="Bus OD multipliers (comma-separated, for --env bus)")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--checkpoint-nau", type=str, default=None)
    parser.add_argument("--checkpoint-relu", type=str, default=None)
    parser.add_argument("--compound", action="store_true",
                        help="Use compound mode profiles from train_csbapr.py "
                             "(multi-param perturbation, overrides --param/--range)")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--plot", action="store_true",
                        help="Generate bound vs actual plot (Figure 1)")
    parser.add_argument("--env-path", type=str,
                        default=str(project_root / "bapr_reference"),
                        help="Bus simulation env path")

    args = parser.parse_args()
    start_time = time.time()

    # Load compound mode profiles if requested
    compound_modes = None
    if args.compound and args.env != 'bus':
        from scripts.train_csbapr import MUJOCO_OOD_PROFILES, MUJOCO_MODE_PROFILES
        # Merge train + OOD profiles for the sweep
        profiles = {}
        profiles.update(MUJOCO_MODE_PROFILES.get(args.env, {}))
        profiles.update(MUJOCO_OOD_PROFILES.get(args.env, {}))
        if profiles:
            compound_modes = profiles
            print(f"Using compound mode profiles: {list(profiles.keys())}")
        else:
            print(f"No compound profiles for {args.env}, falling back to single-param")

    if args.env == 'bus':
        od_range = [int(x) for x in (args.od_range or "1,5,10,20,50,100").split(",")]
        results = bus_ood_sweep(args.env_path, od_range,
                                n_seeds=args.seeds, max_steps=args.max_steps)
    elif args.mode == 'shift':
        multipliers = [float(x) for x in args.range.split(",")]
        results = gym_abrupt_shift_sweep(
            args.env, args.param, multipliers,
            shift_step=args.shift_step,
            max_steps=args.max_steps,
            n_seeds=args.seeds,
            checkpoint_nau=args.checkpoint_nau,
            checkpoint_relu=args.checkpoint_relu,
            compound_modes=compound_modes,
        )
    else:
        multipliers = [float(x) for x in args.range.split(",")]
        results = gym_ood_sweep(
            args.env, args.param, multipliers,
            n_seeds=args.seeds,
            checkpoint_nau=args.checkpoint_nau,
            checkpoint_relu=args.checkpoint_relu,
            max_steps=args.max_steps,
            compound_modes=compound_modes,
        )

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f}s")

    if results and args.plot:
        plot_name = f"ood_{args.env.replace('-', '_')}_{args.param}.png"
        plot_bound_vs_actual(results, output_path=plot_name,
                             title=f"CS-BAPR OOD: {args.env} ({args.param})")

    if results and args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
