#!/usr/bin/env python3
"""
CS-BAPR Hyperparameter Sensitivity Analysis
=============================================

Sweeps key hyperparameters to validate robustness:
  1. λ_sym  (weight_sym): symbolic consistency penalty weight
  2. λ_jac  (jac_weight): Jacobian consistency weight
  3. α_mix  (NAU/NMU mix ratio — evaluated via nau_reg_weight proxy)

For each sweep, trains on a single env with 3 seeds and reports
eval reward ± std as a function of the hyperparameter value.

Usage:
    # Quick sweep on Pendulum (50 episodes per run)
    python scripts/sensitivity_analysis.py --env Pendulum-v1 --max-episodes 50

    # Full sweep on Hopper
    python scripts/sensitivity_analysis.py --env Hopper-v4

    # Single hyperparameter
    python scripts/sensitivity_analysis.py --env Pendulum-v1 --param weight_sym
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# Sweep configurations: param_name → list of values to try
SWEEP_CONFIGS = {
    'weight_sym': [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
    'jac_weight': [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
    'nau_reg_weight': [0.0, 0.001, 0.005, 0.01, 0.05, 0.1],
    'beta_ood': [0.0, 0.01, 0.05, 0.1, 0.2, 0.5],
}


def run_sensitivity_sweep(env_name, param_name, values, seeds=(0, 1, 2),
                          max_episodes=None, save_dir='results/sensitivity'):
    """Run a single hyperparameter sweep."""
    from scripts.train_csbapr import train, make_config

    results = []

    for val in values:
        seed_rewards = []
        seed_times = []

        for seed in seeds:
            run_save = os.path.join(save_dir, f"{param_name}_{val}", f"seed_{seed}")

            # Check if already done
            history_file = os.path.join(run_save, f"csbapr_{env_name}_{seed}_history.json")
            if os.path.exists(history_file):
                with open(history_file) as f:
                    data = json.load(f)
                seed_rewards.append(data['best_eval_reward'])
                seed_times.append(data.get('total_time', 0))
                print(f"  [SKIP] {param_name}={val}, seed={seed} "
                      f"(cached: {data['best_eval_reward']:.1f})")
                continue

            print(f"\n--- {param_name}={val}, seed={seed} ---")

            # Override the specific parameter
            import scripts.train_csbapr as tmod
            original_preset = tmod.METHOD_PRESETS['csbapr'].copy()
            tmod.METHOD_PRESETS['csbapr'][param_name] = val

            try:
                history = train(env_name, 'csbapr', seed, run_save,
                                max_episodes=max_episodes, eval_interval=50)
                best_r = max(e['mean'] for e in history['eval_rewards']) if history['eval_rewards'] else 0
                seed_rewards.append(best_r)
            except Exception as e:
                print(f"  [ERROR] {e}")
                seed_rewards.append(float('nan'))
            finally:
                tmod.METHOD_PRESETS['csbapr'] = original_preset

        results.append({
            'param': param_name,
            'value': val,
            'mean_reward': float(np.nanmean(seed_rewards)),
            'std_reward': float(np.nanstd(seed_rewards)),
            'seed_rewards': seed_rewards,
        })
        print(f"  {param_name}={val}: {np.nanmean(seed_rewards):.1f} ± {np.nanstd(seed_rewards):.1f}")

    return results


def plot_sensitivity(all_results, output_dir):
    """Generate sensitivity analysis plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plots")
        return

    for param_name, results in all_results.items():
        if not results:
            continue

        fig, ax = plt.subplots(figsize=(7, 4.5))

        values = [r['value'] for r in results]
        means = [r['mean_reward'] for r in results]
        stds = [r['std_reward'] for r in results]

        ax.errorbar(values, means, yerr=stds, fmt='o-', lw=2, capsize=4,
                     color='blue', markersize=8)
        ax.fill_between(values,
                         [m - s for m, s in zip(means, stds)],
                         [m + s for m, s in zip(means, stds)],
                         alpha=0.15, color='blue')

        # Mark default value
        default_vals = {'weight_sym': 0.01, 'jac_weight': 0.1,
                        'nau_reg_weight': 0.01, 'beta_ood': 0.1}
        if param_name in default_vals:
            ax.axvline(default_vals[param_name], color='red', ls='--',
                       alpha=0.5, label=f'default={default_vals[param_name]}')

        ax.set_xlabel(param_name)
        ax.set_ylabel('Best Eval Reward')
        ax.set_title(f'Sensitivity: {param_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Use log scale if values span orders of magnitude
        if max(values) / max(min(v for v in values if v > 0), 1e-10) > 20:
            ax.set_xscale('log')

        os.makedirs(output_dir, exist_ok=True)
        out = os.path.join(output_dir, f'sensitivity_{param_name}.pdf')
        fig.savefig(out, bbox_inches='tight')
        fig.savefig(out.replace('.pdf', '.png'), bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out}")


def main():
    parser = argparse.ArgumentParser(description="CS-BAPR Sensitivity Analysis")
    parser.add_argument("--env", type=str, default="Pendulum-v1")
    parser.add_argument("--param", type=str, default=None,
                        help="Specific param to sweep (default: all)")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--save-dir", type=str, default="results/sensitivity")
    parser.add_argument("--output-dir", type=str, default="paper/figures")

    args = parser.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(','))

    if args.param:
        sweep_params = {args.param: SWEEP_CONFIGS[args.param]}
    else:
        sweep_params = SWEEP_CONFIGS

    all_results = {}
    for param_name, values in sweep_params.items():
        print(f"\n{'='*60}")
        print(f"Sweeping {param_name}: {values}")
        print(f"{'='*60}")
        results = run_sensitivity_sweep(
            args.env, param_name, values,
            seeds=seeds, max_episodes=args.max_episodes,
            save_dir=os.path.join(args.save_dir, args.env),
        )
        all_results[param_name] = results

    # Save combined results
    os.makedirs(args.save_dir, exist_ok=True)
    out_json = os.path.join(args.save_dir, f"sensitivity_{args.env}.json")
    with open(out_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_json}")

    # Plot
    plot_sensitivity(all_results, args.output_dir)


if __name__ == "__main__":
    main()
