#!/usr/bin/env python3
"""
CS-BAPR vs BAPR: Convergence + Extreme OOD Validation

Validates the core paper claim:
  "CS-BAPR (NAU + SINDy) maintains bounded error under extreme OOD
   while BAPR (ReLU) degrades catastrophically"

Like the bus simulation 10x passenger demand spike, we test:
  - Pendulum mass 10x (extreme heavy)
  - Pendulum gravity 5x (extreme high-g)
  - Compound: mass 5x + gravity 2x + length 2x

Usage:
    python scripts/validate_ood_advantage.py --episodes 200 --seeds 3
    python scripts/validate_ood_advantage.py --quick   # 80 eps, 2 seeds
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
from scripts.train_csbapr import (
    apply_mode_to_env, MUJOCO_MODE_PROFILES, METHOD_PRESETS,
    make_config, _get_method_flag, make_sindy_exploration_policy
)
from scripts.ood_eval import (
    make_perturbed_env_compound, evaluate_policy_on_env,
    compute_ood_distance_compound, theoretical_bound_corrected
)


# ---- OOD test cases (mirror bus 10x spike) ----
OOD_CASES = {
    'normal':          {'mass': 1.0, 'gravity': 1.0, 'length': 1.0},
    '2x_mass':         {'mass': 2.0, 'gravity': 1.0, 'length': 1.0},
    '5x_mass':         {'mass': 5.0, 'gravity': 1.0, 'length': 1.0},
    '10x_mass':        {'mass': 10.0, 'gravity': 1.0, 'length': 1.0},   # extreme
    '5x_gravity':      {'mass': 1.0, 'gravity': 5.0, 'length': 1.0},    # extreme
    'compound_ood':    {'mass': 5.0, 'gravity': 2.0, 'length': 2.0},    # compound extreme
}


def train_agent(method_name, env_name, seed, n_episodes, save_path):
    """Train a single agent and return (agent, history)."""
    import gymnasium as gym
    import traceback

    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    config = make_config(env_name, method_name)
    config.max_episodes = n_episodes
    config.sindy_n_explore_episodes = 5  # fast for validation

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = CSBAPRAgent(state_dim, action_dim, config)

    # Phase 0: SINDy (if applicable)
    use_irm = _get_method_flag(method_name, '_use_irm')
    skip_sindy = method_name in ('sac', 'bapr', 'csbapr-no-sindy', 'dr', 'rarl')
    sindy_ok = False
    if not skip_sindy:
        try:
            extra_envs = None
            if use_irm:
                irm_modes = [(n, p) for n, p in MUJOCO_MODE_PROFILES[env_name].items()
                             if n != 'normal'][:2]  # 2 extra envs for speed
                extra_envs = []
                for mode_name, mode_params in irm_modes:
                    e = gym.make(env_name)
                    apply_mode_to_env(e, env_name, mode_params)
                    extra_envs.append((mode_name, e))
            sindy_policy = make_sindy_exploration_policy(env_name)
            agent.sindy_preidentify(env, policy=sindy_policy, extra_envs=extra_envs)
            if extra_envs:
                for _, e in extra_envs:
                    e.close()
            sindy_ok = True
        except Exception as ex:
            print(f"    [WARN] SINDy failed: {ex}")

    # Training
    mode_profiles = MUJOCO_MODE_PROFILES.get(env_name, {})
    # Disable mode switching so both methods train in normal physics.
    # OOD advantage is tested at INFERENCE, not during training.
    # (Multi-mode training destabilizes CS-BAPR due to JC loss anchored to normal SINDy)
    mode_names = []  # empty = no switching
    apply_mode_to_env(env, env_name, mode_profiles.get('normal', {}))

    history = {'eval_rewards': [], 'episode_rewards': []}
    best_eval = -float('inf')
    start = time.time()

    # JC curriculum: disable JC for first jac_curriculum_start episodes so SAC base converges,
    # then enable at reduced weight. Only applies to methods that use JC (jac_weight > 0).
    _jac_weight_final = config.jac_weight
    _jac_curriculum = config.jac_curriculum_start  # default 100

    for ep in range(n_episodes):
        if mode_names and ep >= next_switch:
            current_mode = rng.choice(mode_names)
            apply_mode_to_env(env, env_name, mode_profiles[current_mode])
            next_switch = ep + rng.randint(10, 30)

        # Curriculum: enable JC loss only after warm-up phase
        if _jac_weight_final > 0:
            agent.config.jac_weight = _jac_weight_final if ep >= _jac_curriculum else 0.0

        state, _ = env.reset(seed=seed * 10000 + ep)
        ep_reward = 0.0
        done = False
        while not done:
            action = agent.select_action(state)
            scaled = np.clip(action * env.action_space.high[0],
                             env.action_space.low[0], env.action_space.high[0])
            next_state, reward, term, trunc, _ = env.step(scaled)
            done = term or trunc
            agent.replay_buffer.push(state, action, reward, next_state, float(done))
            state = next_state
            ep_reward += reward
            if agent.replay_buffer.size >= config.batch_size:
                agent.update()

        history['episode_rewards'].append(ep_reward)

        if (ep + 1) % 20 == 0:
            eval_r = []
            for ei in range(3):
                s, _ = env.reset(seed=99999 + ei)
                er = 0.0
                for _ in range(config.max_steps_per_episode):
                    a = agent.select_action(s, deterministic=True)
                    sa = np.clip(a * env.action_space.high[0],
                                 env.action_space.low[0], env.action_space.high[0])
                    s, r, t2, tr2, _ = env.step(sa)
                    er += r
                    if t2 or tr2:
                        break
                eval_r.append(er)
            mean_eval = np.mean(eval_r)
            history['eval_rewards'].append({'ep': ep + 1, 'mean': float(mean_eval)})
            if mean_eval > best_eval:
                best_eval = mean_eval
                # Save best checkpoint immediately
                best_path = save_path.replace('.pt', '_best.pt')
                os.makedirs(os.path.dirname(best_path), exist_ok=True)
                agent.save(best_path)
            elapsed = time.time() - start
            L_eff = agent.actor.compute_L_eff() if hasattr(agent.actor, 'compute_L_eff') else 0
            jac_active = agent.config.jac_weight > 0
            print(f"    ep{ep+1:4d}: eval={mean_eval:8.1f} (best={best_eval:8.1f}), "
                  f"L_eff={L_eff:.3f}, jac={'ON' if jac_active else 'off'}, "
                  f"time={elapsed:.0f}s, sindy={sindy_ok}")

    env.close()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    agent.save(save_path)
    return agent, history, best_eval


def eval_ood(agent, env_name, ood_cases, n_episodes=5):
    """Evaluate agent across all OOD cases. Returns dict of case → mean_reward."""
    results = {}
    for case_name, params in ood_cases.items():
        env = make_perturbed_env_compound(env_name, params, seed=0)
        policy_fn = lambda s: agent.select_action(s, deterministic=True)
        r = evaluate_policy_on_env(env, policy_fn, n_episodes=n_episodes, max_steps=200)
        env.close()
        d = compute_ood_distance_compound(params)
        results[case_name] = {
            'mean_reward': r['mean_reward'],
            'std_reward': r['std_reward'],
            'ood_distance': d,
            'params': params,
        }
    return results


def print_comparison(method_results, ood_cases):
    """Print side-by-side OOD comparison table."""
    methods = list(method_results.keys())
    print("\n" + "=" * 80)
    print("OOD COMPARISON: CS-BAPR vs BAPR (higher reward = better)")
    print("(Mirrors bus 10x passenger demand: extreme OOD parameter change)")
    print("=" * 80)

    header = f"{'Case':>18s}  {'dist':>5s}"
    for m in methods:
        header += f"  {m:>16s}"
    header += "  {'CS>BAPR?':>10s}"
    print(header)
    print("-" * len(header))

    for case_name in ood_cases.keys():
        d = method_results[methods[0]][case_name]['ood_distance']
        line = f"{case_name:>18s}  {d:>5.2f}"
        rewards = {}
        for m in methods:
            r = method_results[m][case_name]['mean_reward']
            rewards[m] = r
            line += f"  {r:>13.1f}±{method_results[m][case_name]['std_reward']:>3.0f}"
        cs_r = rewards.get('csbapr', -9999)
        bapr_r = rewards.get('bapr', -9999)
        better = "✓ YES" if cs_r > bapr_r else "✗ NO"
        line += f"  {better:>10s}"
        print(line)

    print("=" * 80)

    # Summary
    if 'csbapr' in methods and 'bapr' in methods:
        wins = sum(
            1 for c in ood_cases
            if method_results['csbapr'][c]['mean_reward'] >
               method_results['bapr'][c]['mean_reward']
        )
        print(f"\nCS-BAPR wins in {wins}/{len(ood_cases)} OOD cases")

        # Extreme cases only
        extreme = [c for c in ood_cases if any(
            x in c for x in ('10x', '5x', 'compound')
        )]
        ex_wins = sum(
            1 for c in extreme
            if method_results['csbapr'][c]['mean_reward'] >
               method_results['bapr'][c]['mean_reward']
        )
        print(f"CS-BAPR wins in {ex_wins}/{len(extreme)} EXTREME OOD cases")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--seeds', type=int, default=2)
    parser.add_argument('--quick', action='store_true',
                        help='80 episodes, 2 seeds (faster validation)')
    parser.add_argument('--env', default='Pendulum-v1')
    parser.add_argument('--save-dir', default='/tmp/csbapr_ood_validate')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training and load existing checkpoints')
    args = parser.parse_args()

    if args.quick:
        args.episodes = 80
        args.seeds = 2

    methods_to_test = ['csbapr', 'bapr']
    seeds = list(range(args.seeds))
    env_name = args.env

    print(f"\n{'='*60}")
    print(f"CS-BAPR OOD Advantage Validation")
    print(f"Env: {env_name} | Episodes: {args.episodes} | Seeds: {seeds}")
    print(f"OOD cases: {list(OOD_CASES.keys())}")
    print(f"{'='*60}\n")

    # ---- Phase 1: Training ----
    all_agents = {}  # method → list of (agent, best_eval)
    all_histories = {}

    for method in methods_to_test:
        all_agents[method] = []
        all_histories[method] = []
        print(f"\n[Training] {method}")
        for seed in seeds:
            ckpt_path = f"{args.save_dir}/{method}_seed{seed}.pt"
            if args.skip_training and os.path.exists(ckpt_path):
                print(f"  Seed {seed}: loading existing checkpoint")
                config = make_config(env_name, method)
                import gymnasium as gym
                env = gym.make(env_name)
                agent = CSBAPRAgent(env.observation_space.shape[0],
                                    env.action_space.shape[0], config)
                agent.load(ckpt_path)
                env.close()
                all_agents[method].append((agent, 0))
                continue

            print(f"  Seed {seed}:")
            agent, hist, best_eval = train_agent(
                method, env_name, seed, args.episodes, ckpt_path
            )
            # Load best checkpoint for OOD evaluation (not final)
            best_path = ckpt_path.replace('.pt', '_best.pt')
            if os.path.exists(best_path):
                agent.load(best_path)
                print(f"  Seed {seed}: loaded best checkpoint for OOD eval")
            all_agents[method].append((agent, best_eval))
            all_histories[method].append(hist)
            print(f"  Seed {seed} done: best_eval = {best_eval:.1f}")

    # ---- Phase 2: OOD Evaluation ----
    print(f"\n\n{'='*60}")
    print("Phase 2: OOD Evaluation")
    print(f"{'='*60}")

    method_ood_results = {}  # method → aggregated OOD results

    for method in methods_to_test:
        print(f"\n[OOD Eval] {method}")
        seed_results = []

        for agent, best_eval in all_agents[method]:
            r = eval_ood(agent, env_name, OOD_CASES, n_episodes=5)
            seed_results.append(r)

        # Aggregate across seeds
        agg = {}
        for case_name in OOD_CASES:
            rewards = [sr[case_name]['mean_reward'] for sr in seed_results]
            d = seed_results[0][case_name]['ood_distance']
            agg[case_name] = {
                'mean_reward': float(np.mean(rewards)),
                'std_reward': float(np.std(rewards)),
                'ood_distance': d,
                'seed_rewards': rewards,
            }
            print(f"  {case_name:>16s} (d={d:.2f}): "
                  f"{np.mean(rewards):8.1f} ± {np.std(rewards):5.1f}")

        method_ood_results[method] = agg

    # ---- Phase 3: Print Comparison ----
    print_comparison(method_ood_results, OOD_CASES)

    # ---- Save results ----
    os.makedirs(args.save_dir, exist_ok=True)
    out = f"{args.save_dir}/ood_comparison_{env_name}.json"
    with open(out, 'w') as f:
        json.dump({
            'env': env_name,
            'episodes': args.episodes,
            'seeds': seeds,
            'results': method_ood_results,
        }, f, indent=2)
    print(f"\nResults saved to {out}")

    # ---- Training convergence check ----
    if all_histories:
        print(f"\n{'='*60}")
        print("Convergence check (eval reward over training):")
        for method, histories in all_histories.items():
            for i, hist in enumerate(histories):
                evals = hist.get('eval_rewards', [])
                if evals:
                    first = evals[0]['mean']
                    last = evals[-1]['mean']
                    best_h = max(e['mean'] for e in evals)
                    converging = last > first
                    print(f"  {method} seed{i}: {first:.1f} → {last:.1f} "
                          f"(best={best_h:.1f}) {'↑ converging' if converging else '↓ not converging'}")


if __name__ == '__main__':
    main()
