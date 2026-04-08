#!/usr/bin/env python3
"""
CS-BAPR Training Script — Unified entry point for all experiments.

Supports:
  - Environments: Pendulum-v1, Hopper-v4, HalfCheetah-v4, Walker2d-v4
  - Methods: csbapr (full), csbapr-relu, csbapr-no-sindy, csbapr-no-irm,
             csbapr-no-sym, csbapr-no-jac, bapr, sac
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
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from csbapr.agent import CSBAPRAgent
from csbapr.config import CSBAPRConfig


# ============================================================
# Method presets (ablation configurations)
# ============================================================

METHOD_PRESETS = {
    # Full CS-BAPR (NAU + SINDy + IRM + Jac)
    'csbapr': dict(
        use_nau_actor=True,
        weight_sym=0.01,
        jac_weight=0.1,
        _use_irm=True,   # multi-env SINDy
    ),
    # Ablation: NAU → ReLU (test Part IX/XI)
    'csbapr-relu': dict(
        use_nau_actor=False,
        weight_sym=0.01,
        jac_weight=0.1,
        _use_irm=True,
    ),
    # Ablation: no SINDy (random f_sym → no Jacobian alignment)
    'csbapr-no-sindy': dict(
        use_nau_actor=True,
        weight_sym=0.0,
        jac_weight=0.0,
    ),
    # Ablation: no Γ_sym in Q-loss (test Part III)
    'csbapr-no-sym': dict(
        use_nau_actor=True,
        weight_sym=0.0,
        jac_weight=0.1,
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
        jac_weight=0.1,
        _use_irm=False,
    ),
    # BA-PR baseline (predecessor, no SINDy/NAU)
    'bapr': dict(
        use_nau_actor=False,
        weight_sym=0.0,
        jac_weight=0.0,
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


# ============================================================
# Domain Randomization helper
# ============================================================

DR_PARAM_RANGES = {
    'Pendulum-v1': {
        'mass': (0.3, 3.0),
        'gravity': (0.5, 2.0),
        'length': (0.5, 2.0),
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


def train(env_name, method_name, seed, save_dir, max_episodes=None, eval_interval=50):
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
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = CSBAPRAgent(state_dim, action_dim, config)

    use_irm = _get_method_flag(method_name, '_use_irm')
    use_dr = _get_method_flag(method_name, '_domain_rand')
    use_rarl = _get_method_flag(method_name, '_rarl')

    # Phase 0: SINDy pre-identification
    skip_sindy_methods = ('sac', 'bapr', 'csbapr-no-sindy', 'dr', 'rarl')
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

            agent.sindy_preidentify(env, extra_envs=extra_envs)

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
    print(f"  Config: NAU={config.use_nau_actor}, weight_sym={config.weight_sym}, "
          f"jac_weight={config.jac_weight}, IRM={use_irm}, DR={use_dr}, RARL={use_rarl}")

    history = {
        'episode_rewards': [],
        'eval_rewards': [],
        'training_metrics': [],
        'L_eff_history': [],
        'overhead': [],  # wall-clock per episode
    }

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

    if mode_names and not use_dr:
        print(f"  Mode switching: {len(mode_names)} modes, "
              f"switch every {MODE_SWITCH_INTERVAL[0]}-{MODE_SWITCH_INTERVAL[1]} episodes")
        apply_mode_to_env(env, env_name, mode_profiles.get('normal', {}))

    for episode in range(config.max_episodes):
        ep_start = time.time()

        # === Domain Randomization: randomize every episode ===
        if use_dr:
            dr_params = randomize_env_params(env, env_name, rng)
        # === Standard mode switch ===
        elif mode_names and episode >= next_switch_ep:
            current_mode = rng.choice(mode_names)
            apply_mode_to_env(env, env_name, mode_profiles[current_mode])
            next_switch_ep = episode + rng.randint(*MODE_SWITCH_INTERVAL)

        state, _ = env.reset(seed=seed * 10000 + episode)
        episode_reward = 0.0
        done = False
        ep_steps = 0

        while not done:
            action = agent.select_action(state)
            scaled_action = np.clip(action * env.action_space.high[0],
                                    env.action_space.low[0], env.action_space.high[0])

            # === RARL: adversarial perturbation on executed action ===
            if use_rarl:
                scaled_action = apply_rarl_perturbation(scaled_action, env, rng)

            next_state, reward, terminated, truncated, _ = env.step(scaled_action)
            done = terminated or truncated
            agent.replay_buffer.push(state, action, reward, next_state, float(done))
            state = next_state
            episode_reward += reward
            ep_steps += 1

            # Update
            if agent.replay_buffer.size >= config.batch_size:
                metrics = agent.update()
                if metrics:
                    history['training_metrics'].append(metrics)

        total_steps += ep_steps
        history['episode_rewards'].append(episode_reward)
        history['overhead'].append(time.time() - ep_start)

        # Periodic evaluation
        if (episode + 1) % eval_interval == 0:
            eval_rewards = []
            for eval_ep in range(5):
                s, _ = env.reset(seed=99999 + eval_ep)
                er = 0.0
                for _ in range(config.max_steps_per_episode):
                    a = agent.select_action(s, deterministic=True)
                    sa = np.clip(a * env.action_space.high[0],
                                 env.action_space.low[0], env.action_space.high[0])
                    s, r, term, trunc, _ = env.step(sa)
                    er += r
                    if term or trunc:
                        break
                eval_rewards.append(er)

            mean_eval = np.mean(eval_rewards)
            history['eval_rewards'].append({
                'episode': episode + 1,
                'mean': float(mean_eval),
                'std': float(np.std(eval_rewards)),
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
            mode_str = f", mode={current_mode}" if mode_names and not use_dr else ""
            if use_dr:
                mode_str = ", DR=on"
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
            'L_eff_history': history['L_eff_history'],
            'best_eval_reward': best_eval_reward,
            'total_time': total_time,
            'total_steps': total_steps,
            'mean_episode_time': float(np.mean(history['overhead'])),
            'gpu_memory_peak_mb': gpu_mem_peak,
            'sindy_report': getattr(agent, '_sindy_report', None),
            'irm_report': getattr(agent, '_irm_report', None),
            'config': {k: v for k, v in vars(config).items()
                       if isinstance(v, (int, float, str, bool))},
        }, f, indent=2)

    env.close()
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

    args = parser.parse_args()
    train(args.env, args.method, args.seed, args.save_dir,
          args.max_episodes, args.eval_interval)


if __name__ == "__main__":
    main()
