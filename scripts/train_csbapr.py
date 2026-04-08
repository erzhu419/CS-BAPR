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
    # Full CS-BAPR
    'csbapr': dict(
        use_nau_actor=True,
        weight_sym=0.01,
        jac_weight=0.1,
    ),
    # Ablation: NAU → ReLU (test Part IX/XI)
    'csbapr-relu': dict(
        use_nau_actor=False,
        weight_sym=0.01,
        jac_weight=0.1,
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
    ),
    # Ablation: no Jacobian consistency loss (test Part X)
    'csbapr-no-jac': dict(
        use_nau_actor=True,
        weight_sym=0.01,
        jac_weight=0.0,
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

    # Apply method preset
    if method_name in METHOD_PRESETS:
        for k, v in METHOD_PRESETS[method_name].items():
            setattr(config, k, v)
    else:
        raise ValueError(f"Unknown method: {method_name}. Available: {list(METHOD_PRESETS.keys())}")

    return config


def train(env_name, method_name, seed, save_dir, max_episodes=None, eval_interval=50):
    """
    Train CS-BAPR agent and save checkpoints + metrics.

    Returns:
        dict with training metrics history
    """
    import gymnasium as gym

    # Setup
    torch.manual_seed(seed)
    np.random.seed(seed)

    config = make_config(env_name, method_name)
    if max_episodes is not None:
        config.max_episodes = max_episodes

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = CSBAPRAgent(state_dim, action_dim, config)

    # Phase 0: SINDy pre-identification
    if method_name not in ('sac', 'bapr', 'csbapr-no-sindy'):
        try:
            agent.sindy_preidentify(env)
        except Exception as e:
            print(f"[WARN] SINDy Phase 0 failed: {e}. Continuing without SINDy.")

    # Training loop
    run_name = f"{method_name}_{env_name}_{seed}"
    print(f"\n[TRAIN] {run_name}")
    print(f"  Config: NAU={config.use_nau_actor}, weight_sym={config.weight_sym}, "
          f"jac_weight={config.jac_weight}")

    history = {
        'episode_rewards': [],
        'eval_rewards': [],
        'training_metrics': [],
        'L_eff_history': [],
    }

    best_eval_reward = -float('inf')
    start_time = time.time()

    # Mode switching setup (mirrors bus simulation mode_profiles)
    mode_profiles = MUJOCO_MODE_PROFILES.get(env_name, {})
    mode_names = list(mode_profiles.keys()) if mode_profiles else []
    current_mode = 'normal'
    next_switch_ep = np.random.randint(*MODE_SWITCH_INTERVAL) if mode_names else config.max_episodes + 1

    if mode_names:
        print(f"  Mode switching: {len(mode_names)} modes, "
              f"switch every {MODE_SWITCH_INTERVAL[0]}-{MODE_SWITCH_INTERVAL[1]} episodes")
        apply_mode_to_env(env, env_name, mode_profiles.get('normal', {}))

    for episode in range(config.max_episodes):
        # === Mode switch (like bus simulation's random mode switching) ===
        if mode_names and episode >= next_switch_ep:
            current_mode = np.random.choice(mode_names)
            apply_mode_to_env(env, env_name, mode_profiles[current_mode])
            next_switch_ep = episode + np.random.randint(*MODE_SWITCH_INTERVAL)

        state, _ = env.reset(seed=seed * 10000 + episode)
        episode_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state)
            scaled_action = np.clip(action * env.action_space.high[0],
                                    env.action_space.low[0], env.action_space.high[0])
            next_state, reward, terminated, truncated, _ = env.step(scaled_action)
            done = terminated or truncated
            agent.replay_buffer.push(state, action, reward, next_state, float(done))
            state = next_state
            episode_reward += reward

            # Update
            if agent.replay_buffer.size >= config.batch_size:
                metrics = agent.update()
                if metrics:
                    history['training_metrics'].append(metrics)

        history['episode_rewards'].append(episode_reward)

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
            mode_str = f", mode={current_mode}" if mode_names else ""
            print(f"  Ep {episode+1}/{config.max_episodes}: "
                  f"train={np.mean(history['episode_rewards'][-eval_interval:]):.1f}, "
                  f"eval={mean_eval:.1f}, L_eff={L_eff:.3f}{mode_str}, "
                  f"time={elapsed:.0f}s")

            # Save best
            if mean_eval > best_eval_reward:
                best_eval_reward = mean_eval
                os.makedirs(save_dir, exist_ok=True)
                agent.save(os.path.join(save_dir, f"{run_name}_best.pt"))

    # Save final
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
            'total_time': time.time() - start_time,
            'config': {k: v for k, v in vars(config).items()
                       if isinstance(v, (int, float, str, bool))},
        }, f, indent=2)

    env.close()
    print(f"[DONE] {run_name}: best_eval={best_eval_reward:.1f}, "
          f"time={time.time()-start_time:.0f}s")
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
