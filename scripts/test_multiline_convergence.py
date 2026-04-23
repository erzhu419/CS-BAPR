#!/usr/bin/env python3
"""
test_multiline_convergence.py

CS-BAPR on MultiLineEnv (12 SUMO-calibrated lines simultaneously).

Train and eval both use MultiLineEnv.step_to_event() — all 12 lines
step together each tick, sharing the same co_line_buses context.
Single policy handles all lines (state_dim=15 is shared).

OOD meaning here: the agent trains on ALL lines at once. Per-line
eval reward shows how well the shared policy handles each route's
different demand pattern, stop spacing, and timetable.

Usage:
    cd /home/erzhu419/mine_code/CS-BAPR
    python -u scripts/test_multiline_convergence.py --episodes 50 --method csbapr

Timing estimate: ~120s/ep (12 lines) → 50 ep ≈ 100 min
"""

import argparse
import os
import sys
import time
import warnings
from collections import defaultdict
import numpy as np
import torch

warnings.filterwarnings('ignore')

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

OFFLINE_SUMO_ENV = os.environ.get(
    'OFFLINE_SUMO_ENV',
    os.path.join(os.path.dirname(project_root), 'offline-sumo', 'env'),
)
if not os.path.isdir(OFFLINE_SUMO_ENV):
    # Fallback to author's local path
    OFFLINE_SUMO_ENV = '/home/erzhu419/mine_code/offline-sumo/env'
sys.path.insert(0, OFFLINE_SUMO_ENV)

from sim_core.sim import MultiLineEnv, env_bus

from csbapr.config import CSBAPRConfig
from csbapr.agent import CSBAPRAgent

# ─────────────────────────────────────────────────────────────
# Obs normalisation (15-dim, shared across all lines)
# ─────────────────────────────────────────────────────────────

_OBS_SCALE = np.array([
    12.0,     # [0] line_idx  (0–11)
    25.0,     # [1] bus_id    (0–24)
    55.0,     # [2] station_id (max 48 + margin)
    24.0,     # [3] time_period (hour)
    1.0,      # [4] direction
    600.0,    # [5] fwd_headway (s)
    600.0,    # [6] bwd_headway (s)
    80.0,     # [7] waiting_pax
    600.0,    # [8] target_hw
    120.0,    # [9] base_stop_dur (s)
    72000.0,  # [10] sim_time (s)
    600.0,    # [11] gap
    600.0,    # [12] co_fwd_hw
    600.0,    # [13] co_bwd_hw
    20.0,     # [14] seg_speed (m/s)
], dtype=np.float32)


def normalize_obs(obs_raw):
    return np.array(obs_raw, dtype=np.float32) / _OBS_SCALE


# ─────────────────────────────────────────────────────────────
# Episode runner — MultiLineEnv (all 12 lines simultaneously)
# ─────────────────────────────────────────────────────────────

def run_episode_multiline(env, agent, deterministic=False, train=True,
                          config=None, train_freq=20):
    """
    Run one episode on MultiLineEnv.

    All 12 lines step together each tick via step_to_event().
    A single shared policy handles all bus decisions across lines.

    Transitions are collected per (line_id, bus_id) using the pending-dict
    pattern: a transition is settled when station_id (obs[2]) changes.

    Returns:
        reward_by_line : dict[line_id → float]   cumulative reward per line
        ep_decisions   : int                      total transitions collected
        train_steps    : int
    """
    env.reset()
    state_dict, _, _ = env.initialize_state()

    # action_dict: {line_id: {bus_id: float}}
    action_dict = {lid: {i: 0.0 for i in range(le.max_agent_num)}
                   for lid, le in env.line_map.items()}

    pending = {}          # (line_id, bus_id) → (sv_old, action_arr)
    reward_by_line = defaultdict(float)
    ep_decisions = 0
    train_steps = 0
    _update_counter = 0

    # Seed initial actions from initialize_state() output
    for line_id, bus_dict in state_dict.items():
        for bus_id, obs_list in bus_dict.items():
            if not obs_list:
                continue
            sv = normalize_obs(obs_list[-1])
            raw = agent.select_action(sv, deterministic=deterministic)
            hold = float((raw[0] + 1.0) / 2.0 * 60.0)
            action_dict[line_id][bus_id] = np.clip(hold, 0.0, 60.0)
            pending[(line_id, bus_id)] = (sv, np.array([raw[0]], dtype=np.float32))

    done = False
    while not done:
        cur_state, reward_dict, done = env.step_to_event(action_dict)

        # Reset actions after each event tick
        for lid in action_dict:
            for k in action_dict[lid]:
                action_dict[lid][k] = 0.0

        for line_id, bus_dict in cur_state.items():
            line_reward = reward_dict.get(line_id, {})
            for bus_id, obs_list in bus_dict.items():
                if not obs_list:
                    continue
                sv_new = normalize_obs(obs_list[-1])
                r_raw = float(line_reward.get(bus_id, 0.0))

                key = (line_id, bus_id)
                if key in pending:
                    sv_old, a_old = pending[key]
                    # Settle when station_id changes
                    if int(sv_old[2] * 55) != int(sv_new[2] * 55):
                        pending.pop(key)
                        if train:
                            agent.replay_buffer.push(sv_old, a_old, r_raw, sv_new, 0.0)
                        reward_by_line[line_id] += r_raw
                        ep_decisions += 1
                        _update_counter += 1

                raw = agent.select_action(sv_new, deterministic=deterministic)
                hold = float((raw[0] + 1.0) / 2.0 * 60.0)
                action_dict[line_id][bus_id] = np.clip(hold, 0.0, 60.0)
                pending[key] = (sv_new, np.array([raw[0]], dtype=np.float32))

        if train and _update_counter >= train_freq:
            _update_counter = 0
            if config and agent.replay_buffer.size >= config.batch_size:
                agent.update()
                train_steps += 1

    return reward_by_line, ep_decisions, train_steps


# ─────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────

def evaluate(env, agent, n_eval=3):
    """Run n_eval episodes, return mean/std reward per line."""
    all_rewards = defaultdict(list)
    for _ in range(n_eval):
        r_by_line, _, _ = run_episode_multiline(
            env, agent, deterministic=True, train=False, config=None
        )
        for lid, r in r_by_line.items():
            all_rewards[lid].append(r)
    return {lid: (np.mean(v), np.std(v)) for lid, v in all_rewards.items()}


def evaluate_ood(agent, env_path, od_mults, n_eval=3):
    """OOD eval: sweep od_mult values, report total reward mean/std."""
    results = {}
    for mult in od_mults:
        env_bus._DATA_CACHE.clear()
        env_ood = MultiLineEnv(env_path, od_mult=mult)
        rewards = []
        for _ in range(n_eval):
            r_by_line, _, _ = run_episode_multiline(
                env_ood, agent, deterministic=True, train=False, config=None
            )
            rewards.append(sum(r_by_line.values()))
        results[mult] = (np.mean(rewards), np.std(rewards))
        print(f"  od_{mult:4.0f}x: {np.mean(rewards):9.1f} ± {np.std(rewards):.1f}")
    return results


def evaluate_abrupt(agent, env_path, bursts, inject_time=3600, n_eval=3):
    """Abrupt shift eval: normal OD → sudden burst at inject_time seconds."""
    results = {}
    for burst in bursts:
        env_bus._DATA_CACHE.clear()
        env_ab = MultiLineEnv(env_path, od_mult=1.0)
        env_ab.set_ood_burst(inject_time=inject_time, burst_mult=burst)
        rewards = []
        for _ in range(n_eval):
            r_by_line, _, _ = run_episode_multiline(
                env_ab, agent, deterministic=True, train=False, config=None
            )
            rewards.append(sum(r_by_line.values()))
        results[burst] = (np.mean(rewards), np.std(rewards))
        print(f"  burst_{burst:4.0f}x @ t={inject_time}s: {np.mean(rewards):9.1f} ± {np.std(rewards):.1f}")
    return results


# Oscillating within-episode OOD schedules. Each schedule is a list of
# (t_seconds, multiplier) pairs applied in order; between entries the
# multiplier stays constant. Episodes are ~18000 s on SUMO-calibrated data.
OSCILLATING_SCHEDULES = {
    # Commuter day: morning peak → off-peak → noon surge → evening → dinner event
    'commuter_day': [
        (0,     1.0),
        (3600,  10.0),   # +1h: morning peak
        (7200,  2.0),    # +2h: off-peak
        (10800, 20.0),   # +3h: noon surge
        (14400, 5.0),    # +4h: evening
        (16200, 50.0),   # +4.5h: extreme dinner event
    ],
    # Square wave: rapid on/off switching to stress the belief tracker
    'square_wave_20x': [
        (0,     1.0),
        (1800,  20.0),
        (3600,  1.0),
        (5400,  20.0),
        (7200,  1.0),
        (9000,  20.0),
        (10800, 1.0),
        (12600, 20.0),
        (14400, 1.0),
        (16200, 20.0),
    ],
    # Escalating: each step 10x bigger than the last (4 jumps in 18000s)
    'escalating_10x': [
        (0,     1.0),
        (3600,  2.0),
        (7200,  5.0),
        (10800, 15.0),
        (14400, 50.0),
    ],
}


def evaluate_oscillating(agent, env_path, schedules=None, n_eval=3):
    """Multi-burst within-episode oscillation eval.

    Tests the policy under piecewise-constant demand schedules that change
    multiple times within a single episode. Mirrors real operational
    non-stationarity (peak vs off-peak vs events).
    """
    if schedules is None:
        schedules = OSCILLATING_SCHEDULES
    results = {}
    for name, schedule in schedules.items():
        env_bus._DATA_CACHE.clear()
        env_osc = MultiLineEnv(env_path, od_mult=1.0)
        env_osc.set_ood_schedule(schedule)
        rewards = []
        for _ in range(n_eval):
            r_by_line, _, _ = run_episode_multiline(
                env_osc, agent, deterministic=True, train=False, config=None
            )
            rewards.append(sum(r_by_line.values()))
        results[name] = (np.mean(rewards), np.std(rewards))
        peak_mult = max(m for _, m in schedule)
        n_switches = len(schedule) - 1
        print(f"  osc_{name:<18} ({n_switches:2d} switches, peak {peak_mult:5.1f}x): "
              f"{np.mean(rewards):9.1f} ± {np.std(rewards):.1f}")
    return results


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--method', type=str, default='csbapr',
                        choices=['csbapr', 'bapr', 'csbapr-kan', 'csbapr-no-nau'])
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

    if args.method == 'csbapr-kan':
        config.actor_type = 'kan'
        config.use_nau_actor = False
        config.jac_weight = 0.0
        config.weight_sym = 0.0
        config.nau_reg_weight = 0.001  # KAN 内部 spline L1+entropy 正则权重
        config.actor_weight_decay = 1e-4
        config.beta_bc = 0.0  # KAN spline 平滑性已防 drift，无需 bc_loss
        print(f"[CS-BAPR-KAN] KAN actor (spline edges, smooth extrapolation)")
    elif args.method == 'csbapr':
        config.use_nau_actor = True
        config.jac_weight = 0.0
        config.weight_sym = 0.0
        config.nau_reg_weight = 0.01
        config.actor_weight_decay = 1e-4
        config.beta_bc = 0.0  # NAU 架构约束已足够防 drift，bc_loss 会过正则化
        print(f"[CS-BAPR] NAU actor (Lipschitz-constrained)")
    elif args.method == 'csbapr-no-nau':
        # Ablation: remove NAU, test whether fixes alone (min-Q+reward scaling+alpha floor)
        # explain the CS-BAPR advantage, or NAU actually contributes
        config.use_nau_actor = False
        config.jac_weight = 0.0
        config.weight_sym = 0.0
        config.nau_reg_weight = 0.0
        config.actor_weight_decay = 1e-4
        config.beta_bc = 0.0
        print(f"[CS-BAPR-no-NAU] MLP actor + training fixes (isolate NAU contribution)")
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

    # ── Load MultiLineEnv (all 12 lines) ──
    print(f"\nLoading MultiLineEnv from {args.env_path}")
    env_bus._DATA_CACHE.clear()
    env = MultiLineEnv(args.env_path)
    n_lines = len(env.line_map)
    total_buses = sum(le.max_agent_num for le in env.line_map.values())
    print(f"Lines: {list(env.line_map.keys())}")
    print(f"Total buses: {total_buses}, ~120s/ep estimated")

    # ── Checkpoint paths (per method × seed) ──
    ckpt_dir = os.environ.get('CKPT_DIR', '/tmp/multiline_ckpt')
    os.makedirs(ckpt_dir, exist_ok=True)
    tag = f"{args.method}_seed{args.seed}"
    last_ckpt = os.path.join(ckpt_dir, f"{tag}_last.pt")
    best_ckpt = os.path.join(ckpt_dir, f"{tag}_best.pt")
    meta_path = os.path.join(ckpt_dir, f"{tag}_meta.json")
    SAVE_EVERY = int(os.environ.get('CKPT_EVERY', '10'))   # episodes

    # ── Resume from last checkpoint if present ──
    import json as _json
    start_ep = 0
    best_total = -float('inf')
    history = []
    if os.path.exists(last_ckpt) and os.path.exists(meta_path):
        try:
            agent.load(last_ckpt)
            with open(meta_path) as _f:
                meta = _json.load(_f)
            start_ep = int(meta.get('next_ep', 0))
            best_total = float(meta.get('best_total', -float('inf')))
            history = list(meta.get('history', []))
            # Reward-EMA state (if the agent tracks it)
            ema = meta.get('reward_ema_var', None)
            if ema is not None:
                agent._reward_ema_var = float(ema)
            print(f"[RESUME] {tag}: loaded {last_ckpt}, starting at ep {start_ep}, "
                  f"best_so_far={best_total:.0f}, history_len={len(history)}")
            if start_ep >= args.episodes:
                print(f"[RESUME] already completed {start_ep}/{args.episodes} episodes, skipping training")
        except Exception as e:
            print(f"[RESUME] failed to load {last_ckpt}: {e}")
            print(f"[RESUME] starting from scratch")
            start_ep = 0
            best_total = -float('inf')
            history = []

    def _atomic_save_meta():
        """Write meta JSON via tmp file to survive crashes during save."""
        tmp = meta_path + '.tmp'
        with open(tmp, 'w') as _f:
            _json.dump({
                'next_ep': ep + 1,
                'best_total': best_total,
                'history': history,
                'reward_ema_var': getattr(agent, '_reward_ema_var', None),
                'method': args.method,
                'seed': args.seed,
            }, _f)
        os.replace(tmp, meta_path)

    start = time.time()

    print(f"\n{'='*60}")
    if start_ep > 0:
        print(f"Training: {args.method} on MultiLineEnv ({n_lines} lines), "
              f"resume ep {start_ep}/{args.episodes}")
    else:
        print(f"Training: {args.method} on MultiLineEnv ({n_lines} lines), {args.episodes} ep")
    print(f"{'='*60}")

    ep = start_ep - 1  # so _atomic_save_meta's `ep + 1` is correct at first save
    for ep in range(start_ep, args.episodes):
        r_by_line, ep_decisions, train_steps = run_episode_multiline(
            env, agent, deterministic=False, train=True, config=config,
            train_freq=20
        )
        ep_total = sum(r_by_line.values())
        history.append(ep_total)

        if ep_total > best_total:
            best_total = ep_total
            agent.save(best_ckpt + '.tmp')
            os.replace(best_ckpt + '.tmp', best_ckpt)

        # Periodic last-checkpoint save (for resume); every SAVE_EVERY eps
        if (ep + 1) % SAVE_EVERY == 0 or (ep + 1) == args.episodes:
            agent.save(last_ckpt + '.tmp')
            os.replace(last_ckpt + '.tmp', last_ckpt)
            _atomic_save_meta()

        if (ep + 1) % 5 == 0:
            elapsed = time.time() - start
            recent = np.mean(history[-5:]) if len(history) >= 5 else np.mean(history)
            per_line = ', '.join(f'{lid}:{r:.0f}' for lid, r in sorted(r_by_line.items()))
            print(f"  ep{ep+1:4d}: total={ep_total:9.0f}, recent_5={recent:9.0f}, "
                  f"best={best_total:9.0f}, dec={ep_decisions}, "
                  f"buf={agent.replay_buffer.size}, t={elapsed:.0f}s")
            print(f"          {per_line}")

    # ── In-distribution Evaluation ──
    print(f"\n{'='*60}")
    print(f"Per-line Evaluation (ID, od_mult=1x)")
    print(f"{'='*60}")

    results = evaluate(env, agent, n_eval=3)
    total_mean = sum(m for m, _ in results.values())
    for lid in sorted(results.keys()):
        m, s = results[lid]
        print(f"  {lid:6s}: {m:8.1f} ± {s:.1f}")
    print(f"  {'TOTAL':6s}: {total_mean:8.1f}")

    # ── OOD Evaluation (parametric OD sweep) ──
    print(f"\n{'='*60}")
    print(f"OOD Evaluation (parametric OD sweep)")
    print(f"{'='*60}")
    evaluate_ood(agent, args.env_path, od_mults=[1, 2, 5, 10, 20, 50], n_eval=3)

    # ── Abrupt Shift Evaluation (mega_event-style burst mid-episode) ──
    print(f"\n{'='*60}")
    print(f"Abrupt Shift Evaluation (burst at t=3600s)")
    print(f"{'='*60}")
    evaluate_abrupt(agent, args.env_path, bursts=[5, 10, 20, 50],
                    inject_time=3600, n_eval=3)

    # ── Oscillating OOD (multi-burst within-episode schedule) ──
    print(f"\n{'='*60}")
    print(f"Oscillating OOD Evaluation (within-episode demand schedules)")
    print(f"{'='*60}")
    evaluate_oscillating(agent, args.env_path, n_eval=3)

    total = time.time() - start
    print(f"\nTotal time: {total:.0f}s")


if __name__ == '__main__':
    main()
