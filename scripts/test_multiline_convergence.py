#!/usr/bin/env python3
"""
test_multiline_convergence.py

CS-BAPR on BusSimEnv (7X SUMO-calibrated) + cross-line OOD evaluation.

Train on 7X (BusSimEnv, ~3-5s/ep), evaluate on:
  - 7X     (in-distribution)
  - 102S   (OOD: 40-station, different demand)
  - 311X   (OOD: 43-station, longer route)
  - 705X   (OOD: 48-station, 21 buses)

All lines share state_dim=15:
  [0] line_idx    [1] bus_id    [2] station_id  [3] time_period  [4] direction
  [5] fwd_hw      [6] bwd_hw   [7] waiting_pax  [8] target_hw   [9] base_stop_dur
  [10] sim_time   [11] gap      [12] co_fwd_hw  [13] co_bwd_hw  [14] seg_speed

This is the MultiLineEnv / SUMO-calibrated version of test_bus_convergence.py.
NAU-only (no JC) is the method being validated.

Usage:
    cd /home/erzhu419/mine_code/CS-BAPR
    python -u scripts/test_multiline_convergence.py --episodes 50 --seed 0 --method csbapr
"""

import argparse
import os
import sys
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')

# ── Paths ──
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

OFFLINE_SUMO_ENV = '/home/erzhu419/mine_code/offline-sumo/env'
sys.path.insert(0, OFFLINE_SUMO_ENV)

from envs.bus_sim_env import BusSimEnv
from sim_core.sim import MultiLineEnv, env_bus

from csbapr.config import CSBAPRConfig
from csbapr.agent import CSBAPRAgent

import torch

# ─────────────────────────────────────────────────────────────
# Obs normalization
# ─────────────────────────────────────────────────────────────

# Normalisation constants for each of the 15 obs dims.
# Categorical dims (0-4): divide by cardinality upper bound.
# Continuous dims (5-14): divide by typical max value (soft scale).
_OBS_SCALE = np.array([
    12.0,     # [0] line_idx  (0-11)
    25.0,     # [1] bus_id    (0-24)
    55.0,     # [2] station_id (max 48 across all lines, use 55 for margin)
    24.0,     # [3] time_period (0-23 hours)
    1.0,      # [4] direction (0/1)
    600.0,    # [5] fwd_headway (s)   — target is 360s, burst up to ~600
    600.0,    # [6] bwd_headway (s)
    80.0,     # [7] waiting_pax
    600.0,    # [8] target_hw = 360 constant
    120.0,    # [9] base_stop_dur (s)
    72000.0,  # [10] sim_time (s, max 20h)
    600.0,    # [11] gap = target - fwd_hw
    600.0,    # [12] co_fwd_hw
    600.0,    # [13] co_bwd_hw
    20.0,     # [14] seg_speed (m/s)
], dtype=np.float32)


def normalize_obs(obs_raw):
    """Normalise raw 15-dim obs to roughly [-1, 1] / [0, 1]."""
    arr = np.array(obs_raw, dtype=np.float32)
    return arr / _OBS_SCALE


# ─────────────────────────────────────────────────────────────
# Episode runner — BusSimEnv (step_to_event event-driven API)
# ─────────────────────────────────────────────────────────────

def run_episode_sim(env, agent, deterministic=False, train=True, config=None,
                    train_freq=20):
    """
    Run one episode on BusSimEnv (or any env_bus with step_to_event).

    Uses the event-driven loop from H2Oplus/bus_h2o/train_sim.py:
      - step_to_event() skips idle ticks → much faster than step-by-step
      - pending dict tracks (state, action) pairs; settles when station changes

    Returns: (episode_reward, episode_decisions, train_steps)
    """
    env.reset()
    state_dict, _, _ = env.initialize_state()

    action_dict = {k: None for k in range(env.max_agent_num)}
    pending = {}       # bus_id → (norm_state, action_arr)
    ep_reward = 0.0
    ep_decisions = 0
    train_steps = 0
    _update_counter = 0

    # Seed first actions from initial state
    for bus_id, obs_list in state_dict.items():
        if not obs_list:
            continue
        sv = normalize_obs(obs_list[-1])
        raw = agent.select_action(sv, deterministic=deterministic)
        hold = float((raw[0] + 1.0) / 2.0 * 60.0)
        action_dict[bus_id] = np.clip(hold, 0.0, 60.0)
        pending[bus_id] = (sv, np.array([raw[0]], dtype=np.float32))

    done = False
    while not done:
        cur_state, reward_dict, done = env.step_to_event(action_dict)

        # Reset pending actions
        for k in action_dict:
            action_dict[k] = None

        for bus_id, obs_list in cur_state.items():
            if not obs_list:
                continue
            sv_new = normalize_obs(obs_list[-1])
            r_raw = float(reward_dict.get(bus_id, 0.0))

            if bus_id in pending:
                sv_old, a_old = pending[bus_id]
                # Settle transition when station changes (obs[2] = station_id)
                if int(sv_old[2] * 55) != int(sv_new[2] * 55):
                    pending.pop(bus_id)
                    if train:
                        agent.replay_buffer.push(sv_old, a_old, r_raw, sv_new, 0.0)
                    ep_reward += r_raw
                    ep_decisions += 1
                    _update_counter += 1

                    raw = agent.select_action(sv_new, deterministic=deterministic)
                    hold = float((raw[0] + 1.0) / 2.0 * 60.0)
                    action_dict[bus_id] = np.clip(hold, 0.0, 60.0)
                    pending[bus_id] = (sv_new, np.array([raw[0]], dtype=np.float32))
            else:
                raw = agent.select_action(sv_new, deterministic=deterministic)
                hold = float((raw[0] + 1.0) / 2.0 * 60.0)
                action_dict[bus_id] = np.clip(hold, 0.0, 60.0)
                pending[bus_id] = (sv_new, np.array([raw[0]], dtype=np.float32))

        # Train every train_freq decisions
        if train and _update_counter >= train_freq:
            _update_counter = 0
            if config and agent.replay_buffer.size >= config.batch_size:
                agent.update()
                train_steps += 1

    return ep_reward, ep_decisions, train_steps


# ─────────────────────────────────────────────────────────────
# Evaluation on a raw env_bus line (no step_to_event)
# ─────────────────────────────────────────────────────────────

def _step_to_event_raw(env, action_dict):
    """
    Loop env.step() until at least one bus emits an obs, or episode ends.
    Equivalent to BusSimEnv.step_to_event() for raw env_bus instances.
    """
    while True:
        state, reward, done = env.step(action_dict)
        if done or any(v for v in state.values()):
            return state, reward, done


def run_episode_envbus(env, agent, deterministic=False, train=False, config=None,
                       train_freq=20):
    """
    Run one episode on a raw env_bus instance (no BusSimEnv wrapper).

    Uses pending-dict transition detection (same as train_sim.py):
      - _step_to_event_raw() skips idle ticks
      - Transition is settled when station_id (obs[2]) changes

    Works for both in-distribution (7X) and OOD (102S, 311X, 705X) eval.
    """
    env.reset()
    state_dict, _, _ = env.initialize_state()

    from collections import defaultdict
    action_dict = defaultdict(lambda: 0.0)
    pending = {}    # bus_id → (sv_old, action_arr)
    ep_reward = 0.0
    ep_decisions = 0
    _update_counter = 0

    # Seed initial actions from initialize_state() output
    for bus_id, obs_list in state_dict.items():
        if not obs_list:
            continue
        sv = normalize_obs(obs_list[-1])
        raw = agent.select_action(sv, deterministic=deterministic)
        hold = float((raw[0] + 1.0) / 2.0 * 60.0)
        action_dict[bus_id] = np.clip(hold, 0.0, 60.0)
        pending[bus_id] = (sv, np.array([raw[0]], dtype=np.float32))

    done = False
    while not done:
        cur_state, reward_dict, done = _step_to_event_raw(env, action_dict)

        # Reset all actions after each step
        for k in list(action_dict.keys()):
            action_dict[k] = 0.0

        for bus_id, obs_list in cur_state.items():
            if not obs_list:
                continue
            sv_new = normalize_obs(obs_list[-1])
            r_raw = float(reward_dict.get(bus_id, 0.0))

            if bus_id in pending:
                sv_old, a_old = pending[bus_id]
                # Settle transition when station_id changes (obs[2])
                if int(sv_old[2] * 55) != int(sv_new[2] * 55):
                    pending.pop(bus_id)
                    if train:
                        agent.replay_buffer.push(sv_old, a_old, r_raw, sv_new, 0.0)
                    ep_reward += r_raw
                    ep_decisions += 1
                    _update_counter += 1

            # Issue next action
            raw = agent.select_action(sv_new, deterministic=deterministic)
            hold = float((raw[0] + 1.0) / 2.0 * 60.0)
            action_dict[bus_id] = np.clip(hold, 0.0, 60.0)
            pending[bus_id] = (sv_new, np.array([raw[0]], dtype=np.float32))

        if train and _update_counter >= train_freq:
            _update_counter = 0
            if config and agent.replay_buffer.size >= config.batch_size:
                agent.update()

    return ep_reward, ep_decisions


def evaluate_line(line_env, agent, n_eval=3):
    """Evaluate agent on a specific line (env_bus instance)."""
    rewards = []
    for _ in range(n_eval):
        r, _ = run_episode_envbus(line_env, agent, deterministic=True)
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
                        choices=['csbapr', 'bapr'])
    parser.add_argument('--env_path', type=str,
                        default='/home/erzhu419/mine_code/offline-sumo/env/calibrated_env')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Config ──
    config = CSBAPRConfig()
    config.state_dim = 15
    config.action_dim = 1
    config.hidden_dim = 64
    config.num_critics = 3
    config.batch_size = 256
    config.warmup_steps = 500
    config.lr_actor = 3e-4
    config.lr_critic = 3e-4

    if args.method == 'csbapr':
        config.use_nau_actor = True
        config.jac_weight = 0.0
        config.weight_sym = 0.0
        config.nau_reg_weight = 0.01
        config.actor_weight_decay = 1e-4
        print(f"[CS-BAPR] NAU actor (Lipschitz-constrained)")
    else:
        config.use_nau_actor = False
        config.jac_weight = 0.0
        config.weight_sym = 0.0
        config.nau_reg_weight = 0.0
        config.actor_weight_decay = 0.0
        print(f"[BAPR] Standard MLP actor")

    agent = CSBAPRAgent(config.state_dim, config.action_dim, config)
    print(f"Config: state={config.state_dim}, action={config.action_dim}, "
          f"hidden={config.hidden_dim}, NAU={config.use_nau_actor}")

    # ── Load BusSimEnv (7X) for training ──
    print(f"\nLoading BusSimEnv (7X) from {args.env_path}")
    env_bus._DATA_CACHE.clear()
    train_env = BusSimEnv(path=args.env_path)
    print(f"7X: max_agent={train_env.max_agent_num}, stations={len(train_env.stations)}")

    # ── Load all lines for OOD eval ──
    print("\nLoading MultiLineEnv for cross-line OOD eval...")
    env_bus._DATA_CACHE.clear()
    multi_env = MultiLineEnv(args.env_path)
    eval_lines = ['7X', '102S', '311X', '705X']
    print(f"OOD eval lines: {eval_lines}")

    # ── Training ──
    best_reward = -float('inf')
    history = []
    start = time.time()

    print(f"\n{'='*60}")
    print(f"Training: {args.method} on 7X, {args.episodes} episodes")
    print(f"{'='*60}")

    for ep in range(args.episodes):
        ep_reward, ep_decisions, train_steps = run_episode_sim(
            train_env, agent,
            deterministic=False, train=True, config=config,
            train_freq=20
        )
        history.append(ep_reward)

        if ep_reward > best_reward:
            best_reward = ep_reward
            os.makedirs('/tmp/multiline_ckpt', exist_ok=True)
            agent.save(f'/tmp/multiline_ckpt/{args.method}_best.pt')

        if (ep + 1) % 5 == 0:
            elapsed = time.time() - start
            recent = np.mean(history[-5:]) if len(history) >= 5 else np.mean(history)
            print(f"  ep{ep+1:4d}: reward={ep_reward:8.1f}, recent_5={recent:8.1f}, "
                  f"best={best_reward:8.1f}, decisions={ep_decisions}, "
                  f"buffer={agent.replay_buffer.size}, time={elapsed:.0f}s")

    # ── Cross-line OOD Evaluation ──
    print(f"\n{'='*60}")
    print(f"Cross-line OOD Evaluation (train: 7X)")
    print(f"{'='*60}")

    for line_id in eval_lines:
        if line_id not in multi_env.line_map:
            print(f"  {line_id}: not found")
            continue
        line_env = multi_env.line_map[line_id]
        mean_r, std_r = evaluate_line(line_env, agent, n_eval=3)
        tag = "(ID)" if line_id == '7X' else "(OOD)"
        print(f"  {line_id:6s} {tag}: {mean_r:8.1f} ± {std_r:.1f}")

    total = time.time() - start
    print(f"\nTotal time: {total:.0f}s")


if __name__ == '__main__':
    main()
