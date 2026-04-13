#!/usr/bin/env python3
"""
Quick convergence test: CS-BAPR agent on bus mutation environment.

Tests whether NAU actor can learn bus holding control, and measures
OOD robustness under mega_event / holiday_rush.

Methods:
  bapr       - Standard MLP actor (baseline)
  csbapr     - NAU actor, no JC (NAU-only ablation)
  csbapr-jc  - NAU actor + Jacobian Consistency via linear bus dynamics

Usage:
    cd /home/erzhu419/mine_code/CS-BAPR
    python -u scripts/test_bus_convergence.py --episodes 50 --seed 0 --method csbapr-jc
"""

import argparse
import os
import sys
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')

# Setup paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'bapr_reference'))

from bapr_reference.env.sim import env_bus
from bapr_reference.mode_profiles import TRAIN_MODES, OOD_MODES, make_parametric_ood

# CS-BAPR agent
from csbapr.config import CSBAPRConfig
from csbapr.agent import CSBAPRAgent

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────
# Linear dynamics model for JC on bus env
# ─────────────────────────────────────────────────────────────

class LinearFsym(nn.Module):
    """
    Frozen linear dynamics model: f_sym(s) = A @ s + b

    Fitted from replay buffer transitions (s_t → s_{t+1}) via OLS.
    Registered as buffers (never updated during RL training).

    Used as f_sym in compute_jacobian_loss: ‖∇π - A‖² drives
    policy Jacobian toward actual bus dynamics Jacobian.
    """
    def __init__(self, A: np.ndarray, b: np.ndarray):
        super().__init__()
        self.register_buffer('A', torch.tensor(A, dtype=torch.float32))
        self.register_buffer('b', torch.tensor(b, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.A.T + self.b


def fit_linear_dynamics(replay_buffer, n_samples: int = 2000,
                        state_dim: int = 28) -> LinearFsym:
    """
    Fit s_{t+1} ≈ A @ s_t + b from replay buffer via OLS (ridge, λ=1e-3).

    Uses transitions already collected during warmup — no extra env interaction.
    """
    buf = replay_buffer
    n = min(n_samples, buf.size)
    idx = np.random.choice(buf.size, size=n, replace=False)

    # Extract states and next_states from list-of-tuples buffer
    # buffer entries: (state, action, reward, next_state, done)
    sample = [buf.buffer[i] for i in idx]
    states = np.stack([s[0] for s in sample])      # [n, state_dim]
    next_states = np.stack([s[3] for s in sample]) # [n, state_dim]

    # Ridge OLS: (XᵀX + λI)⁻¹ Xᵀ Y — includes bias via augmented X
    X = np.hstack([states, np.ones((n, 1))])   # [n, state_dim+1]
    Y = next_states                              # [n, state_dim]

    lam = 1e-3
    XtX = X.T @ X + lam * np.eye(state_dim + 1)
    XtY = X.T @ Y
    W = np.linalg.solve(XtX, XtY)  # [state_dim+1, state_dim]

    A = W[:state_dim].T   # [state_dim, state_dim]
    b = W[state_dim]      # [state_dim]

    print(f"  [LinearFsym] Fit from {n} transitions. "
          f"‖A - I‖={np.linalg.norm(A - np.eye(state_dim)):.3f}, "
          f"‖b‖={np.linalg.norm(b):.3f}")
    return LinearFsym(A, b)


# ─────────────────────────────────────────────────────────────
# Environment helpers
# ─────────────────────────────────────────────────────────────

def make_bus_env(enable_mode_switch=False, ood_mode=None, ood_inject_time=None,
                 ood_profile=None):
    """Create bus environment."""
    env_path = os.path.join(project_root, 'bapr_reference', 'env')
    return env_bus(
        env_path,
        debug=False,
        render=False,
        route_sigma=1.5,
        enable_mode_switch=enable_mode_switch,
        ood_mode=ood_mode,
        ood_inject_time=ood_inject_time,
        ood_profile=ood_profile,
    )


# ─────────────────────────────────────────────────────────────
# Episode runner
# ─────────────────────────────────────────────────────────────

def run_episode(env, agent, deterministic=False, train=True, config=None):
    """
    Run one bus episode, collecting transitions and training.

    Returns episode_reward, episode_steps, train_steps.
    """
    env.reset()
    state_dict, reward_dict, _ = env.initialize_state()

    done = False
    from collections import defaultdict
    action_dict = defaultdict(lambda: 0.0)
    episode_reward = 0.0
    episode_steps = 0
    train_steps = 0
    _update_counter = 0

    while not done:
        for key in list(state_dict.keys()):
            slist = state_dict[key]
            if len(slist) == 1:
                if action_dict[key] is None:
                    obs = np.array(slist[0], dtype=np.float32)
                    raw_action = agent.select_action(obs, deterministic=deterministic)
                    holding = float((raw_action[0] + 1.0) / 2.0 * 60.0)
                    action_dict[key] = np.clip(holding, 0.0, 60.0)

            elif len(slist) == 2:
                if slist[0][1] != slist[1][1]:
                    state = np.array(slist[0], dtype=np.float32)
                    next_state = np.array(slist[1], dtype=np.float32)
                    reward = reward_dict[key]
                    held = action_dict[key]
                    action_norm = np.array([held / 30.0 - 1.0], dtype=np.float32)

                    if train:
                        agent.replay_buffer.push(state, action_norm, reward, next_state, 0.0)

                    episode_steps += 1
                    episode_reward += reward
                    _update_counter += 1

                state_dict[key] = slist[1:]
                obs = np.array(state_dict[key][0], dtype=np.float32)
                raw_action = agent.select_action(obs, deterministic=deterministic)
                holding = float((raw_action[0] + 1.0) / 2.0 * 60.0)
                action_dict[key] = np.clip(holding, 0.0, 60.0)

        state_dict, reward_dict, done = env.step(action_dict)

        # Train every 20 transitions
        if train and _update_counter >= 20:
            _update_counter = 0
            if agent.replay_buffer.size >= config.batch_size:
                agent.update()
                train_steps += 1

    return episode_reward, episode_steps, train_steps


# ─────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────

def evaluate(agent, ood_mode=None, ood_profile=None, n_eval=3):
    """Evaluate agent on bus env (optionally with OOD injection)."""
    rewards = []
    for _ in range(n_eval):
        env = make_bus_env(
            ood_mode=ood_mode,
            ood_inject_time=0 if ood_profile or ood_mode else None,
            ood_profile=ood_profile,
        )
        r, steps, _ = run_episode(env, agent, deterministic=True, train=False, config=None)
        rewards.append(r)
    return np.mean(rewards), np.std(rewards)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--method', type=str, default='csbapr',
                        choices=['csbapr', 'csbapr-jc', 'bapr'])
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Config ──
    config = CSBAPRConfig()
    config.state_dim = 28
    config.action_dim = 1
    config.hidden_dim = 64
    config.num_critics = 3
    config.batch_size = 256
    config.warmup_steps = 500
    config.lr_actor = 3e-4
    config.lr_critic = 3e-4
    config.max_steps_per_episode = 50000

    if args.method == 'csbapr-jc':
        config.use_nau_actor = True
        config.jac_weight = 0.0        # starts at 0; enabled after linear dynamics fit
        config.weight_sym = 0.0
        config.nau_reg_weight = 0.01
        config.actor_weight_decay = 1e-4
        config.jac_curriculum_start = 5   # enable JC from ep 5 onward
        print(f"[CS-BAPR+JC] NAU actor + Jacobian Consistency (linear bus dynamics)")

    elif args.method == 'csbapr':
        config.use_nau_actor = True
        config.jac_weight = 0.0
        config.weight_sym = 0.0
        config.nau_reg_weight = 0.01
        config.actor_weight_decay = 1e-4
        print(f"[CS-BAPR] NAU actor, no JC/SINDy (NAU-only ablation)")

    else:  # bapr
        config.use_nau_actor = False
        config.jac_weight = 0.0
        config.weight_sym = 0.0
        config.nau_reg_weight = 0.0
        config.actor_weight_decay = 0.0
        print(f"[BAPR] Standard MLP actor")

    agent = CSBAPRAgent(config.state_dim, config.action_dim, config)
    print(f"Config: state={config.state_dim}, action={config.action_dim}, "
          f"hidden={config.hidden_dim}, NAU={config.use_nau_actor}")

    # JC params (csbapr-jc only)
    _jac_weight_final = 0.01 if args.method == 'csbapr-jc' else 0.0
    _jac_curriculum = getattr(config, 'jac_curriculum_start', 5)
    _linear_fsym_fitted = False

    # ── Training ──
    env = make_bus_env(enable_mode_switch=False)
    best_reward = -float('inf')
    history = []
    start = time.time()

    print(f"\n{'='*60}")
    print(f"Training: {args.method} on bus env, {args.episodes} episodes")
    print(f"{'='*60}")

    for ep in range(args.episodes):
        # Curriculum JC: fit linear dynamics once warmup is done, then enable JC
        if args.method == 'csbapr-jc':
            if not _linear_fsym_fitted and agent.replay_buffer.size >= 2000:
                print(f"\n  [ep{ep+1}] Fitting linear bus dynamics for JC...")
                agent.f_sym_torch = fit_linear_dynamics(
                    agent.replay_buffer, n_samples=2000,
                    state_dim=config.state_dim
                ).to(agent.device)
                _linear_fsym_fitted = True

            if _linear_fsym_fitted and ep >= _jac_curriculum:
                agent.config.jac_weight = _jac_weight_final
            else:
                agent.config.jac_weight = 0.0

        ep_reward, ep_steps, train_steps = run_episode(
            env, agent, deterministic=False, train=True, config=config
        )
        history.append(ep_reward)

        if ep_reward > best_reward:
            best_reward = ep_reward
            os.makedirs('/tmp/bus_ckpt', exist_ok=True)
            agent.save(f'/tmp/bus_ckpt/{args.method}_best.pt')

        if (ep + 1) % 5 == 0:
            elapsed = time.time() - start
            recent = np.mean(history[-5:]) if len(history) >= 5 else np.mean(history)
            jac_str = (f", jac={'ON' if agent.config.jac_weight > 0 else 'off'}"
                       if args.method == 'csbapr-jc' else "")
            print(f"  ep{ep+1:4d}: reward={ep_reward:8.1f}, recent_5={recent:8.1f}, "
                  f"best={best_reward:8.1f}, steps={ep_steps}, "
                  f"buffer={agent.replay_buffer.size}, time={elapsed:.0f}s{jac_str}")

    env.close() if hasattr(env, 'close') else None

    # ── OOD Evaluation ──
    print(f"\n{'='*60}")
    print(f"OOD Evaluation")
    print(f"{'='*60}")

    mean_r, std_r = evaluate(agent, n_eval=3)
    print(f"  normal:              {mean_r:8.1f} ± {std_r:.1f}")

    for ood_name in ['mega_event', 'holiday_rush', 'emergency_evacuation']:
        mean_r, std_r = evaluate(agent, ood_mode=ood_name, n_eval=3)
        print(f"  {ood_name:20s}: {mean_r:8.1f} ± {std_r:.1f}")

    print(f"\nParametric OD sweep:")
    for mult in [1, 5, 10, 20, 50]:
        profile = make_parametric_ood(mult)
        mean_r, std_r = evaluate(agent, ood_profile=profile, n_eval=3)
        print(f"  od_{mult}x:{'':>14s} {mean_r:8.1f} ± {std_r:.1f}")

    total = time.time() - start
    print(f"\nTotal time: {total:.0f}s")


if __name__ == '__main__':
    main()
