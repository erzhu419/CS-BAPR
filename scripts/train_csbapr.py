#!/usr/bin/env python3
"""
CS-BAPR Training Script — Unified entry point for all experiments.

Supports:
  - Environments: Pendulum-v1, Hopper-v4, HalfCheetah-v4, Walker2d-v4
  - Methods: csbapr (full), csbapr-context, csbapr-relu, csbapr-no-sindy,
             csbapr-no-irm, csbapr-no-sym, csbapr-no-jac, resac, bapr,
             bapr-context, sac
  - Multi-seed: 5 seeds per config

Usage:
    # Full CS-BAPR on Pendulum
    python scripts/train_csbapr.py --env Pendulum-v1 --method csbapr --seed 0

    # BA-PR baseline (ReLU, no SINDy)
    python scripts/train_csbapr.py --env Hopper-v4 --method bapr --seed 0

    # Ablation: CS-BAPR without NAU (ReLU actor)
    python scripts/train_csbapr.py --env Pendulum-v1 --method csbapr-relu --seed 0

    # All seeds for one config
    for s in 0 1 2 3 4; do
        python scripts/train_csbapr.py --env Pendulum-v1 --method csbapr --seed $s &
    done
"""

import argparse
import json
import os
import sys
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from csbapr.agent import CSBAPRAgent
from csbapr.config import CSBAPRConfig
from csbapr.envs.extreme_ood import ExtremeOODConfig, ExtremeOODWrapper


# ============================================================
# Method presets (ablation configurations)
# ============================================================

METHOD_PRESETS = {
    # Full CS-BAPR (NAU + SINDy + IRM + Jac)
    'csbapr': dict(
        use_nau_actor=True,
        weight_sym=0.01,
        jac_weight=0.01,   # reduced from 0.1; curriculum warm-up via jac_curriculum_start
        _use_irm=True,   # multi-env SINDy
    ),
    # Ablation: NAU → ReLU (test Part IX/XI)
    'csbapr-relu': dict(
        use_nau_actor=False,
        weight_sym=0.01,
        jac_weight=0.01,
        _use_irm=True,
    ),
    # Ablation: no SINDy (random f_sym → no Jacobian alignment)
    'csbapr-no-sindy': dict(
        use_nau_actor=True,
        weight_sym=0.0,
        jac_weight=0.0,
    ),
    'csbapr-no-sindy-stable': dict(
        use_nau_actor=True,
        weight_sym=0.0,
        jac_weight=0.0,
        lr_actor=5e-5,
        lr_alpha=1e-4,
        alpha_cap=2.0,
        actor_mean_clip=5.0,
        actor_weight_decay=5e-4,
        nau_reg_weight=0.02,
        nau_l_eff_penalty_weight=0.001,
        nau_l_eff_target=200.0,
    ),
    # Generic-OOD adaptation: NAU actor + online transition context.
    # SINDy/Jacobian are intentionally disabled here because the context-augmented
    # policy has input dimension state+context while f_sym is state-only.
    'csbapr-context': dict(
        use_nau_actor=True,
        use_context=True,
        weight_sym=0.0,
        jac_weight=0.0,
    ),
    # More conservative context variants for coherent latent-load OOD. The raw
    # context methods were unstable on Hopper; these slow the context filter and
    # reduce actor drift while keeping the same no-oracle information setting.
    'csbapr-stable-context': dict(
        use_nau_actor=True,
        use_context=True,
        context_ema_alpha=0.05,
        beta_bc=0.005,
        lr_actor=5e-5,
        lr_alpha=1e-4,
        alpha_cap=2.0,
        actor_mean_clip=5.0,
        actor_weight_decay=5e-4,
        nau_reg_weight=0.02,
        nau_l_eff_penalty_weight=0.001,
        nau_l_eff_target=200.0,
        penalty_scale=1.0,
        weight_sym=0.0,
        jac_weight=0.0,
    ),
    # Ablation: no Γ_sym in Q-loss (test Part III)
    'csbapr-no-sym': dict(
        use_nau_actor=True,
        weight_sym=0.0,
        jac_weight=0.01,
        _use_irm=True,
    ),
    # Ablation: no Jacobian consistency loss (test Part X)
    'csbapr-no-jac': dict(
        use_nau_actor=True,
        weight_sym=0.01,
        jac_weight=0.0,
        _use_irm=True,
    ),
    # Ablation: no IRM (single-env SINDy, tests Pillar III)
    'csbapr-no-irm': dict(
        use_nau_actor=True,
        weight_sym=0.01,
        jac_weight=0.01,
        _use_irm=False,
    ),
    # BA-PR baseline (predecessor, no SINDy/NAU)
    'bapr': dict(
        use_nau_actor=False,
        weight_sym=0.0,
        jac_weight=0.0,
    ),
    # Context ablation: tests whether online regime inference helps independent
    # of NAU/symbolic structure.
    'bapr-context': dict(
        use_nau_actor=False,
        use_context=True,
        weight_sym=0.0,
        jac_weight=0.0,
    ),
    'bapr-stable-context': dict(
        use_nau_actor=False,
        use_context=True,
        context_ema_alpha=0.05,
        beta_bc=0.005,
        lr_actor=1e-4,
        penalty_scale=1.0,
        weight_sym=0.0,
        jac_weight=0.0,
    ),
    # RE-SAC baseline: ensemble risk-sensitive SAC without BOCD adaptation.
    # Keep the fixed LCB-style policy penalty (effective_beta=-2) and OOD
    # Q-std regularization, but disable the BAPR run-length weighted penalty.
    'resac': dict(
        use_nau_actor=False,
        weight_sym=0.0,
        jac_weight=0.0,
        bapr_warmup_iters=10**12,
        penalty_scale=0.0,
    ),
    # Vanilla SAC baseline
    'sac': dict(
        use_nau_actor=False,
        weight_sym=0.0,
        jac_weight=0.0,
        weight_reg=0.0,
        beta_ood=0.0,
    ),
    # Domain Randomization baseline (ReLU + randomized physics each episode)
    'dr': dict(
        use_nau_actor=False,
        weight_sym=0.0,
        jac_weight=0.0,
        _domain_rand=True,
    ),
    # Robust Adversarial RL baseline (SAC + adversarial force perturbation)
    'rarl': dict(
        use_nau_actor=False,
        weight_sym=0.0,
        jac_weight=0.0,
        _rarl=True,
    ),
}

# Environment-specific hyperparameters
ENV_PRESETS = {
    'Pendulum-v1': dict(
        max_episodes=500,
        max_steps_per_episode=200,
        warmup_steps=500,
        batch_size=128,
        hidden_dim=64,
        num_critics=3,
        sindy_n_explore_episodes=10,
        sindy_lib_degree=2,
        sindy_threshold=0.01,  # Pendulum needs lower threshold (small coefficients)
    ),
    'InvertedPendulum-v4': dict(
        max_episodes=300,
        max_steps_per_episode=1000,
        warmup_steps=1000,
        batch_size=128,
        hidden_dim=128,
        num_critics=3,
        sindy_n_explore_episodes=10,
        sindy_lib_degree=2,
        sindy_threshold=0.001,
    ),
    'Hopper-v4': dict(
        max_episodes=2000,
        max_steps_per_episode=1000,
        warmup_steps=5000,
        batch_size=256,
        hidden_dim=256,
        num_critics=5,
        sindy_n_explore_episodes=30,
        sindy_lib_degree=2,
    ),
    'HalfCheetah-v4': dict(
        max_episodes=3000,
        max_steps_per_episode=1000,
        warmup_steps=10000,
        batch_size=256,
        hidden_dim=256,
        num_critics=5,
        sindy_n_explore_episodes=30,
        sindy_lib_degree=2,
    ),
    'Walker2d-v4': dict(
        max_episodes=3000,
        max_steps_per_episode=1000,
        warmup_steps=5000,
        batch_size=256,
        hidden_dim=256,
        num_critics=5,
        sindy_n_explore_episodes=30,
        sindy_lib_degree=2,
    ),
}


# ============================================================
# MuJoCo Mode Profiles (mirrors bus simulation mode_profiles.py)
# ============================================================
# Each mode is a dict of param_name → multiplier.
# During training, the env randomly switches between modes
# every mode_switch_interval episodes, triggering belief tracker surprise.

MUJOCO_MODE_PROFILES = {
    'Pendulum-v1': {
        'normal':      {'mass': 1.0, 'gravity': 1.0, 'length': 1.0},
        'heavy':       {'mass': 2.0, 'gravity': 1.0, 'length': 1.0},
        'light':       {'mass': 0.5, 'gravity': 1.0, 'length': 1.0},
        'high_grav':   {'mass': 1.0, 'gravity': 1.5, 'length': 1.0},
        'long_arm':    {'mass': 1.0, 'gravity': 1.0, 'length': 1.5},
    },
    'InvertedPendulum-v4': {
        'normal':      {'body_mass': 1.0, 'gravity': 1.0},
        'heavy':       {'body_mass': 2.0, 'gravity': 1.0},
        'light':       {'body_mass': 0.5, 'gravity': 1.0},
        'high_grav':   {'body_mass': 1.0, 'gravity': 1.5},
        'compound':    {'body_mass': 1.5, 'gravity': 1.3},
    },
    'Hopper-v4': {
        'normal':      {'body_mass': 1.0, 'friction': 1.0, 'gravity': 1.0},
        'heavy':       {'body_mass': 2.0, 'friction': 1.0, 'gravity': 1.0},
        'slippery':    {'body_mass': 1.0, 'friction': 0.3, 'gravity': 1.0},
        'high_grav':   {'body_mass': 1.0, 'friction': 1.0, 'gravity': 1.5},
        'compound':    {'body_mass': 1.5, 'friction': 0.5, 'gravity': 1.2},
    },
    'HalfCheetah-v4': {
        'normal':      {'body_mass': 1.0, 'friction': 1.0, 'gravity': 1.0},
        'heavy':       {'body_mass': 2.0, 'friction': 1.0, 'gravity': 1.0},
        'slippery':    {'body_mass': 1.0, 'friction': 0.3, 'gravity': 1.0},
        'high_grav':   {'body_mass': 1.0, 'friction': 1.0, 'gravity': 1.5},
        'compound':    {'body_mass': 1.5, 'friction': 0.5, 'gravity': 1.2},
    },
    'Walker2d-v4': {
        'normal':      {'body_mass': 1.0, 'friction': 1.0, 'gravity': 1.0},
        'heavy':       {'body_mass': 2.0, 'friction': 1.0, 'gravity': 1.0},
        'slippery':    {'body_mass': 1.0, 'friction': 0.3, 'gravity': 1.0},
        'high_grav':   {'body_mass': 1.0, 'friction': 1.0, 'gravity': 1.5},
        'compound':    {'body_mass': 1.5, 'friction': 0.5, 'gravity': 1.2},
    },
}

# OOD modes: NEVER seen during training, only at eval time
MUJOCO_OOD_PROFILES = {
    'Pendulum-v1': {
        'extreme_heavy':  {'mass': 10.0, 'gravity': 1.0, 'length': 1.0},
        'extreme_grav':   {'mass': 1.0, 'gravity': 5.0, 'length': 1.0},
        'compound_ood':   {'mass': 5.0, 'gravity': 2.0, 'length': 2.0},
    },
    'InvertedPendulum-v4': {
        'extreme_heavy':  {'body_mass': 10.0, 'gravity': 1.0},
        'extreme_grav':   {'body_mass': 1.0,  'gravity': 5.0},
        'compound_ood':   {'body_mass': 5.0,  'gravity': 2.0},
    },
    'Hopper-v4': {
        'extreme_heavy':  {'body_mass': 5.0, 'friction': 1.0, 'gravity': 1.0},
        'extreme_slip':   {'body_mass': 1.0, 'friction': 0.1, 'gravity': 1.0},
        'compound_ood':   {'body_mass': 3.0, 'friction': 0.3, 'gravity': 2.0},
    },
    'HalfCheetah-v4': {
        'extreme_heavy':  {'body_mass': 5.0, 'friction': 1.0, 'gravity': 1.0},
        'extreme_slip':   {'body_mass': 1.0, 'friction': 0.1, 'gravity': 1.0},
        'compound_ood':   {'body_mass': 3.0, 'friction': 0.3, 'gravity': 2.0},
    },
    'Walker2d-v4': {
        'extreme_heavy':  {'body_mass': 5.0, 'friction': 1.0, 'gravity': 1.0},
        'extreme_slip':   {'body_mass': 1.0, 'friction': 0.1, 'gravity': 1.0},
        'compound_ood':   {'body_mass': 3.0, 'friction': 0.3, 'gravity': 2.0},
    },
}

# How often to switch modes during training (in episodes)
MODE_SWITCH_INTERVAL = (10, 30)  # random interval between 10-30 episodes


def apply_mode_to_env(env, env_name, mode_params):
    """Apply a mode profile (dict of param→mult) to a live environment."""
    if hasattr(env.unwrapped, 'model'):
        model = env.unwrapped.model
        # Store originals on first call
        if not hasattr(env, '_original_body_mass'):
            env._original_body_mass = model.body_mass.copy()
            env._original_friction = model.geom_friction.copy()
            env._original_gravity = model.opt.gravity.copy()
        # Reset to originals then apply multipliers
        model.body_mass[:] = env._original_body_mass * mode_params.get('body_mass', 1.0)
        model.geom_friction[:] = env._original_friction * mode_params.get('friction', 1.0)
        grav_mult = mode_params.get('gravity', 1.0)
        model.opt.gravity[:] = env._original_gravity * grav_mult
    elif 'Pendulum' in env_name:
        uw = env.unwrapped
        if not hasattr(uw, '_original_m'):
            uw._original_m = uw.m
            uw._original_g = uw.g
            uw._original_l = uw.l
        uw.m = uw._original_m * mode_params.get('mass', 1.0)
        uw.g = uw._original_g * mode_params.get('gravity', 1.0)
        uw.l = uw._original_l * mode_params.get('length', 1.0)


def make_config(env_name, method_name):
    """Create config from environment + method presets."""
    config = CSBAPRConfig(env_name=env_name)

    # Apply env preset
    if env_name in ENV_PRESETS:
        for k, v in ENV_PRESETS[env_name].items():
            setattr(config, k, v)

    # Apply method preset (skip internal _keys)
    if method_name in METHOD_PRESETS:
        for k, v in METHOD_PRESETS[method_name].items():
            if not k.startswith('_'):
                setattr(config, k, v)
    else:
        raise ValueError(f"Unknown method: {method_name}. Available: {list(METHOD_PRESETS.keys())}")

    return config


def _get_method_flag(method_name, flag):
    """Get an internal _flag from method preset."""
    return METHOD_PRESETS.get(method_name, {}).get(flag, False)


def _json_safe(value):
    """Convert numpy/torch scalars and containers into JSON-serializable values."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, deque)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    return value


def _recent_rollout_payload(recent_rollout):
    """Materialize recent on-policy transition deques for BAPR surprise."""
    return {k: list(v) for k, v in recent_rollout.items() if len(v) > 0}


def _summarize_metrics(metrics, window=100):
    metrics = list(metrics)[-int(window):]
    if not metrics:
        return {}
    keys = sorted({k for m in metrics for k in m.keys()})
    summary = {}
    for key in keys:
        vals = []
        for metric in metrics:
            value = metric.get(key)
            if value is None:
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                vals.append(value)
        if vals:
            summary[f"{key}_last"] = float(vals[-1])
            summary[f"{key}_mean"] = float(np.mean(vals))
    return summary


def _evaluate_agent(agent, eval_env, max_steps, n_episodes, seed_base):
    rewards = []
    lengths = []
    profile_counts = Counter()

    for eval_ep in range(int(n_episodes)):
        s, info = eval_env.reset(seed=int(seed_base) + eval_ep)
        agent.reset_context()
        er = 0.0
        steps = 0
        profile = info.get("ood_profile") if isinstance(info, dict) else None
        if profile is not None:
            profile_counts[str(profile)] += 1

        for _ in range(max_steps):
            a = agent.select_action(s, deterministic=True)
            sa = np.clip(a * eval_env.action_space.high[0],
                         eval_env.action_space.low[0], eval_env.action_space.high[0])
            ns, r, term, trunc, info = eval_env.step(sa)
            agent.observe_context(s, a, r, ns, done=term or trunc)
            profile = info.get("ood_profile") if isinstance(info, dict) else None
            if profile is not None:
                profile_counts[str(profile)] += 1
            s = ns
            er += r
            steps += 1
            if term or trunc:
                break

        rewards.append(float(er))
        lengths.append(int(steps))

    return {
        "mean": float(np.mean(rewards)) if rewards else 0.0,
        "std": float(np.std(rewards)) if rewards else 0.0,
        "rewards": rewards,
        "lengths": lengths,
        "profile_counts": dict(profile_counts),
    }


# ============================================================
# Domain Randomization helper
# ============================================================

DR_PARAM_RANGES = {
    'Pendulum-v1': {
        'mass': (0.3, 3.0),
        'gravity': (0.5, 2.0),
        'length': (0.5, 2.0),
    },
    'InvertedPendulum-v4': {
        'body_mass': (0.3, 3.0),
        'gravity': (0.5, 2.0),
    },
    'Hopper-v4': {
        'body_mass': (0.5, 3.0),
        'friction': (0.3, 2.0),
        'gravity': (0.7, 1.5),
    },
    'HalfCheetah-v4': {
        'body_mass': (0.5, 3.0),
        'friction': (0.3, 2.0),
        'gravity': (0.7, 1.5),
    },
    'Walker2d-v4': {
        'body_mass': (0.5, 3.0),
        'friction': (0.3, 2.0),
        'gravity': (0.7, 1.5),
    },
}


def make_sindy_exploration_policy(env_name):
    """
    Return a stabilizing exploration policy for SINDy data collection.

    For envs where random actions cause early termination (e.g. InvertedPendulum),
    a simple PD controller keeps the system in a useful state range so SINDy
    can collect informative trajectories.

    Returns None (random policy) for envs that work fine with random exploration.
    """
    if 'InvertedPendulum' in env_name:
        # State: [x, x_dot, theta, theta_dot]
        # PD controller stabilizes around theta=0 (upright).
        # Add small sinusoidal cart perturbation to excite full dynamics.
        import math
        _t = [0]
        def inverted_pendulum_policy(state):
            _t[0] += 1
            x, x_dot, theta, theta_dot = state
            # PD on angle
            u = -(50.0 * theta + 5.0 * theta_dot)
            # Small cart perturbation to excite translational modes
            u += 0.5 * math.sin(0.05 * _t[0])
            return np.clip(np.array([u], dtype=np.float32), -3.0, 3.0)
        return inverted_pendulum_policy
    return None


def randomize_env_params(env, env_name, rng):
    """Apply random physics parameters (Domain Randomization)."""
    ranges = DR_PARAM_RANGES.get(env_name, {})
    mode_params = {}
    for param, (lo, hi) in ranges.items():
        mode_params[param] = rng.uniform(lo, hi)
    apply_mode_to_env(env, env_name, mode_params)
    return mode_params


# ============================================================
# RARL: Adversarial force perturbation
# ============================================================

def apply_rarl_perturbation(action, env, rng, adversary_strength=0.1):
    """
    RARL-style adversarial perturbation: add worst-case force noise.
    Simple implementation: random direction, magnitude proportional to action range.
    """
    noise_scale = adversary_strength * (env.action_space.high - env.action_space.low)
    perturbation = rng.uniform(-1, 1, size=action.shape) * noise_scale
    return np.clip(action + perturbation, env.action_space.low, env.action_space.high)


def train(env_name, method_name, seed, save_dir, max_episodes=None, eval_interval=50,
          extreme_ood_train=False, ood_schedule="episode_random",
          ood_intensity=50.0, ood_abrupt_step=100, ood_switch_steps=100,
          ood_switch_probability=0.35, ood_profile_family="extreme",
          fixed_profile_eval_episodes=1):
    """
    Train CS-BAPR agent and save checkpoints + metrics.

    Supports:
    - IRM multi-env Phase 0 (when _use_irm flag set)
    - Domain Randomization (when _domain_rand flag set)
    - RARL adversarial perturbation (when _rarl flag set)
    - Standard mode-switching training
    - Wall-clock + GPU memory overhead tracking

    Returns:
        dict with training metrics history
    """
    import gymnasium as gym
    import traceback as tb

    # Setup
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    config = make_config(env_name, method_name)
    if max_episodes is not None:
        config.max_episodes = max_episodes

    env = gym.make(env_name)
    if extreme_ood_train:
        env = ExtremeOODWrapper(
            env,
            config=ExtremeOODConfig(
                intensity=ood_intensity,
                schedule=ood_schedule,
                profile_family=ood_profile_family,
                seed=seed,
                abrupt_step=ood_abrupt_step,
                switch_every_steps=ood_switch_steps,
                switch_probability=ood_switch_probability,
            ),
        )
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = CSBAPRAgent(state_dim, action_dim, config)

    use_irm = _get_method_flag(method_name, '_use_irm')
    use_dr = _get_method_flag(method_name, '_domain_rand')
    use_rarl = _get_method_flag(method_name, '_rarl')

    # Phase 0: SINDy pre-identification
    skip_sindy_methods = (
        'sac', 'resac', 'bapr', 'bapr-context',
        'bapr-stable-context',
        'csbapr-no-sindy', 'csbapr-no-sindy-stable',
        'csbapr-context', 'csbapr-stable-context',
        'dr', 'rarl',
    )
    if method_name not in skip_sindy_methods:
        try:
            extra_envs = None
            if use_irm:
                # Create IRM training environments with different physics
                irm_modes = list(MUJOCO_MODE_PROFILES.get(env_name, {}).items())
                # Use non-normal modes as extra environments
                extra_envs = []
                for mode_name, mode_params in irm_modes:
                    if mode_name == 'normal':
                        continue
                    irm_env = gym.make(env_name)
                    apply_mode_to_env(irm_env, env_name, mode_params)
                    extra_envs.append((mode_name, irm_env))
                print(f"[IRM] Created {len(extra_envs)} extra environments for IRM filtering")

            sindy_policy = make_sindy_exploration_policy(env_name)
            agent.sindy_preidentify(env, policy=sindy_policy, extra_envs=extra_envs)

            # Close IRM envs
            if extra_envs:
                for _, e in extra_envs:
                    e.close()
        except Exception as e:
            print(f"[WARN] SINDy Phase 0 failed: {e}. Continuing without SINDy.")
            tb.print_exc()

    # Training loop
    run_name = f"{method_name}_{env_name}_{seed}"
    print(f"\n[TRAIN] {run_name}")
    print(f"  Config: NAU={config.use_nau_actor}, context={config.use_context}, "
          f"weight_sym={config.weight_sym}, jac_weight={config.jac_weight}, "
          f"IRM={use_irm}, DR={use_dr}, RARL={use_rarl}")
    if extreme_ood_train:
        print(f"  Extreme OOD train: schedule={ood_schedule}, "
              f"profile_family={ood_profile_family}, "
              f"intensity={ood_intensity:g}, abrupt_step={ood_abrupt_step}, "
              f"switch_steps={ood_switch_steps}, switch_p={ood_switch_probability:.2f}")

    history = {
        'episode_rewards': [],
        'eval_rewards': [],
        'fixed_profile_eval_rewards': [],
        'training_metrics': deque(maxlen=1000),  # keep only last 1000 update metrics
        'L_eff_history': [],
        'profile_events': [],
        'mode_events': [],
        'overhead': [],  # wall-clock per episode
    }

    fixed_eval_envs = {}
    if extreme_ood_train and fixed_profile_eval_episodes > 0 and hasattr(env, "profiles"):
        for idx, profile in enumerate(env.profiles):
            profile_name = str(profile.get("name", idx))
            fixed_eval_envs[profile_name] = ExtremeOODWrapper(
                gym.make(env_name),
                config=ExtremeOODConfig(
                    intensity=ood_intensity,
                    schedule="static",
                    profile_family=ood_profile_family,
                    seed=seed * 1000 + idx,
                    static_profile=profile_name,
                    abrupt_step=ood_abrupt_step,
                    switch_every_steps=ood_switch_steps,
                    switch_probability=ood_switch_probability,
                ),
            )
        print(f"  Fixed-profile eval: {len(fixed_eval_envs)} profiles × "
              f"{fixed_profile_eval_episodes} episode(s)")

    rollout_window = max(int(getattr(config, "surprise_window", 256)), 8)
    recent_rollout = {
        "state": deque(maxlen=rollout_window),
        "action": deque(maxlen=rollout_window),
        "reward": deque(maxlen=rollout_window),
    }
    if config.use_context:
        recent_rollout["context"] = deque(maxlen=rollout_window)

    best_eval_reward = -float('inf')
    start_time = time.time()
    total_steps = 0

    # Track peak GPU memory
    gpu_mem_peak = 0.0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Mode switching setup (mirrors bus simulation mode_profiles)
    mode_profiles = MUJOCO_MODE_PROFILES.get(env_name, {})
    mode_names = list(mode_profiles.keys()) if mode_profiles else []
    current_mode = 'normal'
    next_switch_ep = rng.randint(*MODE_SWITCH_INTERVAL) if mode_names else config.max_episodes + 1
    last_ood_profile = None

    def record_ood_event(info, episode_idx, episode_step):
        nonlocal last_ood_profile
        if not extreme_ood_train or not isinstance(info, dict):
            return
        profile = info.get("ood_profile")
        if profile is None or profile == last_ood_profile:
            return
        last_ood_profile = profile
        history['profile_events'].append({
            'episode': int(episode_idx + 1),
            'episode_step': int(episode_step),
            'total_steps': int(total_steps + episode_step),
            'profile': str(profile),
            'latent_load': _json_safe(info.get("ood_latent_load")),
            'schedule': str(info.get("ood_schedule", ood_schedule)),
        })

    if mode_names and not use_dr and not extreme_ood_train:
        print(f"  Mode switching: {len(mode_names)} modes, "
              f"switch every {MODE_SWITCH_INTERVAL[0]}-{MODE_SWITCH_INTERVAL[1]} episodes")
        apply_mode_to_env(env, env_name, mode_profiles.get('normal', {}))
        history['mode_events'].append({
            'episode': 1,
            'mode': current_mode,
            'params': mode_profiles.get('normal', {}),
        })

    for episode in range(config.max_episodes):
        ep_start = time.time()

        # === Domain Randomization: randomize every episode ===
        if use_dr:
            dr_params = randomize_env_params(env, env_name, rng)
        # === Standard mode switch ===
        elif mode_names and not extreme_ood_train and episode >= next_switch_ep:
            current_mode = rng.choice(mode_names)
            apply_mode_to_env(env, env_name, mode_profiles[current_mode])
            next_switch_ep = episode + rng.randint(*MODE_SWITCH_INTERVAL)
            history['mode_events'].append({
                'episode': int(episode + 1),
                'mode': str(current_mode),
                'params': mode_profiles[current_mode],
            })

        state, reset_info = env.reset(seed=seed * 10000 + episode)
        record_ood_event(reset_info, episode, 0)
        agent.reset_context()
        episode_reward = 0.0
        done = False
        ep_steps = 0

        while not done:
            context_before = agent._current_context_np() if config.use_context else None
            action = agent.select_action(state)
            scaled_action = np.clip(action * env.action_space.high[0],
                                    env.action_space.low[0], env.action_space.high[0])

            # === RARL: adversarial perturbation on executed action ===
            if use_rarl:
                scaled_action = apply_rarl_perturbation(scaled_action, env, rng)

            next_state, reward, terminated, truncated, step_info = env.step(scaled_action)
            done = terminated or truncated
            recent_rollout["state"].append(np.asarray(state, dtype=np.float32).copy())
            recent_rollout["action"].append(np.asarray(action, dtype=np.float32).copy())
            recent_rollout["reward"].append(float(reward))
            if context_before is not None:
                recent_rollout["context"].append(np.asarray(context_before, dtype=np.float32).copy())
            agent.store_transition(state, action, reward, next_state, float(done))
            record_ood_event(step_info, episode, ep_steps + 1)
            state = next_state
            episode_reward += reward
            ep_steps += 1

            # Update
            if agent.replay_buffer.size >= config.batch_size:
                metrics = agent.update(recent_rollout=_recent_rollout_payload(recent_rollout))
                if metrics:
                    history['training_metrics'].append(metrics)

        total_steps += ep_steps
        history['episode_rewards'].append(episode_reward)
        history['overhead'].append(time.time() - ep_start)

        # Periodic evaluation
        if (episode + 1) % eval_interval == 0:
            context_snapshot = agent.snapshot_context()
            eval_result = _evaluate_agent(
                agent, env,
                max_steps=config.max_steps_per_episode,
                n_episodes=5,
                seed_base=99999,
            )
            agent.restore_context(context_snapshot)

            mean_eval = eval_result['mean']
            history['eval_rewards'].append({
                'episode': episode + 1,
                **eval_result,
                'metrics_summary': _summarize_metrics(history['training_metrics'], window=100),
            })

            if fixed_eval_envs:
                profile_results = {}
                for profile_name, fixed_env in fixed_eval_envs.items():
                    context_snapshot = agent.snapshot_context()
                    profile_results[profile_name] = _evaluate_agent(
                        agent, fixed_env,
                        max_steps=config.max_steps_per_episode,
                        n_episodes=fixed_profile_eval_episodes,
                        seed_base=199999 + (episode + 1) * 100 + len(profile_results) * 10,
                    )
                    agent.restore_context(context_snapshot)
                history['fixed_profile_eval_rewards'].append({
                    'episode': episode + 1,
                    'profiles': profile_results,
                })

            # L_eff tracking
            L_eff = 0.0
            if config.use_nau_actor and hasattr(agent.actor, 'compute_L_eff'):
                L_eff = agent.actor.compute_L_eff()
                history['L_eff_history'].append({
                    'episode': episode + 1,
                    'L_eff': L_eff,
                })

            elapsed = time.time() - start_time
            mode_str = f", mode={current_mode}" if mode_names and not use_dr and not extreme_ood_train else ""
            if use_dr:
                mode_str = ", DR=on"
            if extreme_ood_train:
                active = getattr(env, "active_profile_name", "unknown")
                mode_str = f", extreme_ood={active}"
            if use_rarl:
                mode_str += ", RARL=on"
            print(f"  Ep {episode+1}/{config.max_episodes}: "
                  f"train={np.mean(history['episode_rewards'][-eval_interval:]):.1f}, "
                  f"eval={mean_eval:.1f}, L_eff={L_eff:.3f}{mode_str}, "
                  f"time={elapsed:.0f}s")

            # Save best
            if mean_eval > best_eval_reward:
                best_eval_reward = mean_eval
                os.makedirs(save_dir, exist_ok=True)
                agent.save(os.path.join(save_dir, f"{run_name}_best.pt"))

    # GPU memory tracking
    if torch.cuda.is_available():
        gpu_mem_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB

    # Save final
    total_time = time.time() - start_time
    os.makedirs(save_dir, exist_ok=True)
    agent.save(os.path.join(save_dir, f"{run_name}_final.pt"))

    # Save history
    with open(os.path.join(save_dir, f"{run_name}_history.json"), "w") as f:
        json.dump({
            'env': env_name,
            'method': method_name,
            'seed': seed,
            'episode_rewards': history['episode_rewards'],
            'eval_rewards': history['eval_rewards'],
            'fixed_profile_eval_rewards': history['fixed_profile_eval_rewards'],
            'training_metrics': _json_safe(list(history['training_metrics'])),
            'training_metrics_summary': _summarize_metrics(history['training_metrics'], window=1000),
            'L_eff_history': history['L_eff_history'],
            'profile_events': history['profile_events'],
            'mode_events': history['mode_events'],
            'best_eval_reward': best_eval_reward,
            'total_time': total_time,
            'total_steps': total_steps,
            'mean_episode_time': float(np.mean(history['overhead'])),
            'gpu_memory_peak_mb': gpu_mem_peak,
            'sindy_report': getattr(agent, '_sindy_report', None),
            'irm_report': getattr(agent, '_irm_report', None),
            'extreme_ood_train': extreme_ood_train,
            'extreme_ood_config': {
                'schedule': ood_schedule,
                'profile_family': ood_profile_family,
                'intensity': ood_intensity,
                'abrupt_step': ood_abrupt_step,
                'switch_every_steps': ood_switch_steps,
                'switch_probability': ood_switch_probability,
                'fixed_profile_eval_episodes': fixed_profile_eval_episodes,
            } if extreme_ood_train else None,
            'config': {k: v for k, v in vars(config).items()
                       if isinstance(v, (int, float, str, bool))},
        }, f, indent=2)

    env.close()
    for fixed_env in fixed_eval_envs.values():
        fixed_env.close()
    print(f"[DONE] {run_name}: best_eval={best_eval_reward:.1f}, "
          f"time={total_time:.0f}s, steps={total_steps}, "
          f"gpu_mem={gpu_mem_peak:.0f}MB")
    return history


def main():
    parser = argparse.ArgumentParser(description="CS-BAPR Training")
    parser.add_argument("--env", type=str, default="Pendulum-v1",
                        choices=list(ENV_PRESETS.keys()))
    parser.add_argument("--method", type=str, default="csbapr",
                        choices=list(METHOD_PRESETS.keys()))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--save-dir", type=str, default="results")
    parser.add_argument("--extreme-ood-train", action="store_true",
                        help="Train under bus-like extreme random OOD wrapper")
    parser.add_argument("--ood-schedule", type=str, default="episode_random",
                        choices=["static", "episode_random", "abrupt", "oscillating", "step_random", "markov"],
                        help="Extreme OOD schedule used when --extreme-ood-train is set")
    parser.add_argument("--ood-profile-family", type=str, default="extreme",
                        choices=[
                            "extreme", "structured", "load", "latent_load",
                            "controllable_load", "control_load", "adaptive_load",
                        ],
                        help="OOD profile family: unstructured, structured, or latent-load regimes")
    parser.add_argument("--ood-intensity", type=float, default=50.0,
                        help="Max OOD multiplier used by the generic wrapper")
    parser.add_argument("--ood-abrupt-step", type=int, default=100,
                        help="Step for abrupt OOD shift")
    parser.add_argument("--ood-switch-steps", type=int, default=100,
                        help="Switch period for oscillating/step_random OOD")
    parser.add_argument("--ood-switch-probability", type=float, default=0.35,
                        help="Switch probability at each period tick for markov OOD")
    parser.add_argument("--fixed-profile-eval-episodes", type=int, default=1,
                        help="Episodes per fixed-profile eval when --extreme-ood-train is set")

    args = parser.parse_args()
    train(args.env, args.method, args.seed, args.save_dir,
          args.max_episodes, args.eval_interval,
          extreme_ood_train=args.extreme_ood_train,
          ood_schedule=args.ood_schedule,
          ood_intensity=args.ood_intensity,
          ood_abrupt_step=args.ood_abrupt_step,
          ood_switch_steps=args.ood_switch_steps,
          ood_switch_probability=args.ood_switch_probability,
          ood_profile_family=args.ood_profile_family,
          fixed_profile_eval_episodes=args.fixed_profile_eval_episodes)


if __name__ == "__main__":
    main()
