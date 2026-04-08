#!/usr/bin/env python3
"""
CS-BAPR Paper Figure Generation
================================

Generates all figures and tables for the CS-BAPR paper from experiment results.

Figures:
  1. OOD bound vs actual error (theory-experiment alignment)
  2. NAU vs ReLU OOD error growth curves
  3. Ablation study bar chart
  4. L_eff evolution during training
  5. Training curves comparison

Tables:
  1. Main results (reward ± std across methods and envs)
  2. Ablation study
  3. Computational cost comparison

Usage:
    python scripts/plot_results.py --results-dir results/
    python scripts/plot_results.py --results-dir results/ --output-dir paper/figures/
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 10,
        'figure.dpi': 150,
    })
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not installed, generating tables only")


def load_all_results(results_dir):
    """Load all experiment history JSON files from results directory."""
    results_dir = Path(results_dir)
    all_data = defaultdict(lambda: defaultdict(list))

    for json_file in results_dir.rglob("*_history.json"):
        with open(json_file) as f:
            data = json.load(f)
        env = data.get('env', 'unknown')
        method = data.get('method', 'unknown')
        all_data[env][method].append(data)

    return all_data


def load_ood_results(results_dir):
    """Load OOD evaluation results."""
    results_dir = Path(results_dir)
    ood_data = {}
    for json_file in results_dir.rglob("ood_results.json"):
        env = json_file.parent.name
        with open(json_file) as f:
            ood_data[env] = json.load(f)
    return ood_data


# ============================================================
# Figure 1: Bound vs Actual OOD Error
# ============================================================

def plot_figure1_bound_vs_actual(ood_data, output_dir):
    """
    Figure 1: Theory-experiment alignment.
    Shows corrected bound (Part X) as upper envelope over actual error.
    """
    if not HAS_MPL:
        return

    for env, results in ood_data.items():
        fig, ax = plt.subplots(figsize=(7, 5))

        distances = []
        bounds = []
        method_data = defaultdict(lambda: {'d': [], 'reward': [], 'std': []})

        for key, r in sorted(results.items(), key=lambda x: x[1].get('ood_distance', 0)):
            d = r.get('ood_distance', 0)
            distances.append(d)
            bounds.append(r.get('bound_corrected', 0))

            if 'methods' in r:
                for m, mr in r['methods'].items():
                    method_data[m]['d'].append(d)
                    method_data[m]['reward'].append(mr['mean_reward'])
                    method_data[m]['std'].append(mr.get('std_reward', 0))

        # Plot bound
        ax.fill_between(distances, 0, bounds, alpha=0.15, color='red',
                        label='Theoretical bound (Part X)')
        ax.plot(distances, bounds, 'r--', lw=2, alpha=0.8)

        # Plot methods
        styles = {
            'CS-BAPR (NAU)': ('o-', 'blue'),
            'BA-PR (ReLU)': ('x--', 'red'),
            'Random': ('d:', 'gray'),
        }
        for m, md in method_data.items():
            marker, color = styles.get(m, ('.-', 'green'))
            ax.errorbar(md['d'], md['reward'], yerr=md['std'],
                        fmt=marker, color=color, lw=2, capsize=3, label=m)

        ax.set_xlabel('OOD distance ‖d‖')
        ax.set_ylabel('Performance metric')
        ax.set_title(f'{env}: Theoretical Bound vs Actual OOD Performance')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        out = os.path.join(output_dir, f'figure1_bound_{env}.pdf')
        fig.savefig(out, bbox_inches='tight')
        fig.savefig(out.replace('.pdf', '.png'), bbox_inches='tight')
        plt.close()
        print(f"  Figure 1 saved: {out}")


# ============================================================
# Figure 2: NAU vs ReLU OOD Error Growth
# ============================================================

def plot_figure2_nau_vs_relu(ood_data, output_dir):
    """
    Figure 2: NAU quadratic growth vs ReLU catastrophic collapse.
    """
    if not HAS_MPL:
        return

    for env, results in ood_data.items():
        fig, ax = plt.subplots(figsize=(7, 5))

        nau_d, nau_r, relu_d, relu_r = [], [], [], []

        for key, r in sorted(results.items(), key=lambda x: x[1].get('ood_distance', 0)):
            d = r.get('ood_distance', 0)
            if 'methods' in r:
                if 'CS-BAPR (NAU)' in r['methods']:
                    nau_d.append(d)
                    nau_r.append(r['methods']['CS-BAPR (NAU)']['mean_reward'])
                if 'BA-PR (ReLU)' in r['methods']:
                    relu_d.append(d)
                    relu_r.append(r['methods']['BA-PR (ReLU)']['mean_reward'])

        if not nau_d and not relu_d:
            continue

        if nau_d:
            # Normalize: reward drop from baseline
            nau_baseline = nau_r[0] if nau_r else 1
            nau_drop = [1 - r / nau_baseline if nau_baseline != 0 else 0 for r in nau_r]
            ax.plot(nau_d, nau_drop, 'bo-', lw=2, markersize=8,
                    label='CS-BAPR (NAU): quadratic growth')
        if relu_d:
            relu_baseline = relu_r[0] if relu_r else 1
            relu_drop = [1 - r / relu_baseline if relu_baseline != 0 else 0 for r in relu_r]
            ax.plot(relu_d, relu_drop, 'rx--', lw=2, markersize=8,
                    label='BA-PR (ReLU): catastrophic collapse')

        # Overlay theoretical quadratic curve
        d_smooth = np.linspace(0, max(nau_d + relu_d), 100)
        ax.plot(d_smooth, 0.1 * d_smooth ** 2, 'b:', alpha=0.4,
                label='$O(d^2)$ reference')

        ax.set_xlabel('OOD distance ‖d‖')
        ax.set_ylabel('Reward degradation (fraction)')
        ax.set_title(f'{env}: NAU vs ReLU OOD Error Growth')
        ax.legend()
        ax.grid(True, alpha=0.3)

        out = os.path.join(output_dir, f'figure2_nau_relu_{env}.pdf')
        fig.savefig(out, bbox_inches='tight')
        fig.savefig(out.replace('.pdf', '.png'), bbox_inches='tight')
        plt.close()
        print(f"  Figure 2 saved: {out}")


# ============================================================
# Figure 3: Training Curves
# ============================================================

def plot_figure3_training_curves(all_data, output_dir):
    """
    Figure 3: Training curves comparison across methods.
    """
    if not HAS_MPL:
        return

    for env, methods in all_data.items():
        fig, ax = plt.subplots(figsize=(8, 5))

        colors = {
            'csbapr': 'blue', 'csbapr-relu': 'orange', 'csbapr-no-sindy': 'green',
            'csbapr-no-sym': 'purple', 'csbapr-no-jac': 'brown',
            'bapr': 'red', 'sac': 'gray',
        }

        for method, runs in methods.items():
            # Aggregate eval rewards across seeds
            all_evals = []
            for run in runs:
                evals = run.get('eval_rewards', [])
                if evals:
                    eps = [e['episode'] for e in evals]
                    means = [e['mean'] for e in evals]
                    all_evals.append((eps, means))

            if not all_evals:
                continue

            # Align to common episodes
            min_len = min(len(e[1]) for e in all_evals)
            episodes = all_evals[0][0][:min_len]
            rewards_matrix = np.array([e[1][:min_len] for e in all_evals])

            mean_r = rewards_matrix.mean(axis=0)
            std_r = rewards_matrix.std(axis=0)

            color = colors.get(method, 'black')
            ax.plot(episodes, mean_r, color=color, lw=2, label=method)
            ax.fill_between(episodes, mean_r - std_r, mean_r + std_r,
                            color=color, alpha=0.15)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Evaluation Reward')
        ax.set_title(f'{env}: Training Curves')
        ax.legend(loc='lower right', ncol=2)
        ax.grid(True, alpha=0.3)

        out = os.path.join(output_dir, f'figure3_training_{env}.pdf')
        fig.savefig(out, bbox_inches='tight')
        fig.savefig(out.replace('.pdf', '.png'), bbox_inches='tight')
        plt.close()
        print(f"  Figure 3 saved: {out}")


# ============================================================
# Figure 4: L_eff Evolution
# ============================================================

def plot_figure4_leff(all_data, output_dir):
    """
    Figure 4: L_eff (Part XI) evolution during training.
    Shows NAU weights converging toward {-1,0,1} → L_eff decreasing.
    """
    if not HAS_MPL:
        return

    for env, methods in all_data.items():
        csbapr_runs = methods.get('csbapr', [])
        if not csbapr_runs:
            continue

        fig, ax = plt.subplots(figsize=(7, 4))

        for run in csbapr_runs:
            leff_hist = run.get('L_eff_history', [])
            if leff_hist:
                eps = [e['episode'] for e in leff_hist]
                leffs = [e['L_eff'] for e in leff_hist]
                ax.plot(eps, leffs, alpha=0.4, color='blue')

        # Mean
        all_leff = [run.get('L_eff_history', []) for run in csbapr_runs]
        if all_leff and all_leff[0]:
            min_len = min(len(h) for h in all_leff if h)
            if min_len > 0:
                eps = [all_leff[0][i]['episode'] for i in range(min_len)]
                mean_leff = np.mean([[h[i]['L_eff'] for i in range(min_len)]
                                     for h in all_leff if len(h) >= min_len], axis=0)
                ax.plot(eps, mean_leff, 'b-', lw=2.5, label='Mean L_eff')

        ax.set_xlabel('Episode')
        ax.set_ylabel('L_eff (composed derivative Lipschitz)')
        ax.set_title(f'{env}: L_eff Evolution (Part XI)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        out = os.path.join(output_dir, f'figure4_leff_{env}.pdf')
        fig.savefig(out, bbox_inches='tight')
        fig.savefig(out.replace('.pdf', '.png'), bbox_inches='tight')
        plt.close()
        print(f"  Figure 4 saved: {out}")


# ============================================================
# Table 1: Main Results
# ============================================================

def generate_table1_main_results(all_data, output_dir):
    """
    Table 1: Main results — best eval reward ± std across methods and envs.
    """
    envs = sorted(all_data.keys())
    all_methods = set()
    for env_methods in all_data.values():
        all_methods.update(env_methods.keys())
    methods = sorted(all_methods)

    lines = []
    lines.append("% Table 1: Main Results")
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{Best evaluation reward (mean $\\pm$ std over 5 seeds).}")
    lines.append("\\label{tab:main}")
    col_spec = "l" + "c" * len(envs)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    lines.append("Method & " + " & ".join(envs) + " \\\\")
    lines.append("\\midrule")

    for method in methods:
        row = [method.replace('-', '\\text{-}')]
        for env in envs:
            runs = all_data.get(env, {}).get(method, [])
            if runs:
                best_rewards = [r.get('best_eval_reward', 0) for r in runs]
                mean = np.mean(best_rewards)
                std = np.std(best_rewards)
                row.append(f"${mean:.1f} \\pm {std:.1f}$")
            else:
                row.append("---")
        lines.append(" & ".join(row) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    table_str = "\n".join(lines)
    out = os.path.join(output_dir, "table1_main.tex")
    with open(out, "w") as f:
        f.write(table_str)
    print(f"  Table 1 saved: {out}")

    # Also print to console
    print("\n" + table_str + "\n")


# ============================================================
# Table 2: Computational Cost
# ============================================================

def generate_table_cost(all_data, output_dir):
    """Table: wall-clock time comparison."""
    lines = []
    lines.append("% Table: Computational Cost")
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{Wall-clock training time (seconds, single seed).}")
    lines.append("\\begin{tabular}{lrr}")
    lines.append("\\toprule")
    lines.append("Method & Mean Time (s) & Overhead vs SAC \\\\")
    lines.append("\\midrule")

    for env, methods in all_data.items():
        lines.append(f"\\multicolumn{{3}}{{l}}{{\\textbf{{{env}}}}} \\\\")
        sac_time = None
        for method in ['sac', 'bapr', 'csbapr', 'csbapr-relu']:
            runs = methods.get(method, [])
            if runs:
                times = [r.get('total_time', 0) for r in runs]
                mean_t = np.mean(times)
                if method == 'sac':
                    sac_time = mean_t
                overhead = f"{mean_t/sac_time:.2f}x" if sac_time and sac_time > 0 else "---"
                lines.append(f"  {method} & {mean_t:.0f} & {overhead} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    table_str = "\n".join(lines)
    out = os.path.join(output_dir, "table_cost.tex")
    with open(out, "w") as f:
        f.write(table_str)
    print(f"  Cost table saved: {out}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="CS-BAPR Paper Figure/Table Generator")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default="paper/figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading experiment results...")
    all_data = load_all_results(args.results_dir)
    ood_data = load_ood_results(args.results_dir)

    if not all_data and not ood_data:
        print("No results found. Run experiments first:")
        print("  bash scripts/run_experiments.sh --quick")
        return

    print(f"Found data for {len(all_data)} envs, {len(ood_data)} OOD evaluations")

    # Figures
    if ood_data:
        print("\nGenerating Figure 1 (Bound vs Actual)...")
        plot_figure1_bound_vs_actual(ood_data, args.output_dir)
        print("Generating Figure 2 (NAU vs ReLU)...")
        plot_figure2_nau_vs_relu(ood_data, args.output_dir)

    if all_data:
        print("Generating Figure 3 (Training Curves)...")
        plot_figure3_training_curves(all_data, args.output_dir)
        print("Generating Figure 4 (L_eff Evolution)...")
        plot_figure4_leff(all_data, args.output_dir)

        # Tables
        print("\nGenerating tables...")
        generate_table1_main_results(all_data, args.output_dir)
        generate_table_cost(all_data, args.output_dir)

    print(f"\nAll outputs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
