#!/usr/bin/env python3
"""
CS-BAPR Paper Figure & Table Generation
=========================================

Figures:
  1. OOD bound vs actual error (theory-experiment alignment, Q3)
  2. NAU vs ReLU OOD error growth curves (Q4)
  3. Training curves comparison (Q1)
  4. L_eff evolution during training (Part XI)
  5. Abrupt shift: reward + belief/surprise timeline (Q1 extension)
  6. Ablation bar chart (Q2)
  7. Sensitivity analysis heatmap

Tables:
  1. Main results (reward ± std across methods and envs, with significance)
  2. Ablation study
  3. Computational cost comparison (wall-clock, GPU memory, overhead)
  4. SINDy identification quality (R², sparsity, IRM variance)

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
from scipy import stats as sp_stats

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
        'legend.fontsize': 9,
        'figure.dpi': 150,
        'font.family': 'serif',
    })
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not installed, generating tables only")


# ============================================================
# Data Loading
# ============================================================

def load_all_results(results_dir):
    """Load all experiment history JSON files."""
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
    for json_file in results_dir.rglob("ood_*_results.json"):
        env = json_file.parent.name
        mode = 'sweep' if 'sweep' in json_file.name else 'shift'
        key = f"{env}_{mode}"
        with open(json_file) as f:
            ood_data[key] = json.load(f)
    return ood_data


def load_sensitivity_results(results_dir):
    """Load sensitivity analysis results."""
    results_dir = Path(results_dir)
    sens_data = {}
    for json_file in results_dir.rglob("sensitivity_*.json"):
        with open(json_file) as f:
            sens_data[json_file.stem] = json.load(f)
    return sens_data


# ============================================================
# Statistical Testing
# ============================================================

def welch_ttest(a, b):
    """Welch's t-test (unequal variance). Returns (t_stat, p_value)."""
    a, b = np.array(a), np.array(b)
    if len(a) < 2 or len(b) < 2:
        return 0.0, 1.0
    return sp_stats.ttest_ind(a, b, equal_var=False)


def bootstrap_ci(data, n_boot=10000, ci=0.95, seed=42):
    """Bootstrap confidence interval for the mean."""
    rng = np.random.RandomState(seed)
    data = np.array(data)
    n = len(data)
    if n < 2:
        return float(data.mean()), float(data.mean()), float(data.mean())
    boot_means = np.array([rng.choice(data, n, replace=True).mean()
                           for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    lo = np.percentile(boot_means, 100 * alpha)
    hi = np.percentile(boot_means, 100 * (1 - alpha))
    return float(data.mean()), float(lo), float(hi)


def significance_marker(p):
    """Return significance marker for p-value."""
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    return ''


# ============================================================
# Color / Style Definitions
# ============================================================

METHOD_COLORS = {
    'csbapr': '#2166ac',
    'csbapr-relu': '#f4a582',
    'csbapr-no-sindy': '#92c5de',
    'csbapr-no-sym': '#d6604d',
    'csbapr-no-jac': '#b2182b',
    'csbapr-no-irm': '#4393c3',
    'bapr': '#e66101',
    'sac': '#878787',
    'dr': '#5e3c99',
    'rarl': '#008837',
}

METHOD_LABELS = {
    'csbapr': 'CS-BAPR (ours)',
    'csbapr-relu': 'CS-BAPR w/o NAU',
    'csbapr-no-sindy': 'CS-BAPR w/o SINDy',
    'csbapr-no-sym': r'CS-BAPR w/o $\Gamma_{\rm sym}$',
    'csbapr-no-jac': r'CS-BAPR w/o $\mathcal{L}_{\rm AC}$',
    'csbapr-no-irm': 'CS-BAPR w/o IRM',
    'bapr': 'BA-PR',
    'sac': 'SAC',
    'dr': 'Domain Rand.',
    'rarl': 'RARL',
}

MAIN_METHODS = ['csbapr', 'bapr', 'sac', 'dr', 'rarl']
ABLATION_METHODS = ['csbapr', 'csbapr-relu', 'csbapr-no-sindy',
                    'csbapr-no-sym', 'csbapr-no-jac', 'csbapr-no-irm']


# ============================================================
# Figure 1: Bound vs Actual OOD Error (Q3)
# ============================================================

def plot_figure1_bound_vs_actual(ood_data, output_dir):
    if not HAS_MPL:
        return
    for key, results in ood_data.items():
        if 'sweep' not in key:
            continue
        env = key.replace('_sweep', '')
        fig, ax = plt.subplots(figsize=(7, 5))

        distances, bounds = [], []
        method_data = defaultdict(lambda: {'d': [], 'reward': [], 'std': []})

        for rkey, r in sorted(results.items(), key=lambda x: x[1].get('ood_distance', 0)):
            d = r.get('ood_distance', 0)
            distances.append(d)
            bounds.append(r.get('bound_corrected', 0))
            if 'methods' in r:
                for m, mr in r['methods'].items():
                    method_data[m]['d'].append(d)
                    method_data[m]['reward'].append(mr['mean_reward'])
                    method_data[m]['std'].append(mr.get('std_reward', 0))

        ax.fill_between(distances, 0, bounds, alpha=0.15, color='red',
                        label='Theoretical bound (Part X)')
        ax.plot(distances, bounds, 'r--', lw=2, alpha=0.8)

        styles = {
            'CS-BAPR (NAU)': ('o-', '#2166ac'),
            'BA-PR (ReLU)': ('x--', '#e66101'),
            'Random': ('d:', '#878787'),
        }
        for m, md in method_data.items():
            marker, color = styles.get(m, ('.-', 'green'))
            ax.errorbar(md['d'], md['reward'], yerr=md['std'],
                        fmt=marker, color=color, lw=2, capsize=3, label=m)

        ax.set_xlabel(r'OOD distance $\|d\|$')
        ax.set_ylabel('Mean episode reward')
        ax.set_title(f'{env}: Theoretical Bound vs Actual OOD Performance')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        _save_fig(fig, output_dir, f'figure1_bound_{env}')


# ============================================================
# Figure 2: NAU vs ReLU OOD Error Growth (Q4)
# ============================================================

def plot_figure2_nau_vs_relu(ood_data, output_dir):
    if not HAS_MPL:
        return
    for key, results in ood_data.items():
        if 'sweep' not in key:
            continue
        env = key.replace('_sweep', '')
        fig, ax = plt.subplots(figsize=(7, 5))

        nau_d, nau_r, relu_d, relu_r = [], [], [], []
        for rkey, r in sorted(results.items(), key=lambda x: x[1].get('ood_distance', 0)):
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
            nau_base = nau_r[0] if nau_r else 1
            nau_drop = [1 - r / nau_base if nau_base != 0 else 0 for r in nau_r]
            ax.plot(nau_d, nau_drop, 'o-', color='#2166ac', lw=2, ms=8,
                    label='CS-BAPR (NAU): quadratic growth')
        if relu_d:
            relu_base = relu_r[0] if relu_r else 1
            relu_drop = [1 - r / relu_base if relu_base != 0 else 0 for r in relu_r]
            ax.plot(relu_d, relu_drop, 'x--', color='#e66101', lw=2, ms=8,
                    label='BA-PR (ReLU): catastrophic collapse')

        d_smooth = np.linspace(0, max(nau_d + relu_d), 100)
        ax.plot(d_smooth, 0.1 * d_smooth ** 2, ':', color='#2166ac', alpha=0.4,
                label=r'$O(d^2)$ reference')

        ax.set_xlabel(r'OOD distance $\|d\|$')
        ax.set_ylabel('Reward degradation (fraction)')
        ax.set_title(f'{env}: NAU vs ReLU OOD Error Growth')
        ax.legend()
        ax.grid(True, alpha=0.3)

        _save_fig(fig, output_dir, f'figure2_nau_relu_{env}')


# ============================================================
# Figure 3: Training Curves (Q1)
# ============================================================

def plot_figure3_training_curves(all_data, output_dir):
    if not HAS_MPL:
        return
    for env, methods in all_data.items():
        fig, ax = plt.subplots(figsize=(8, 5))

        for method in MAIN_METHODS + ['csbapr-no-irm']:
            runs = methods.get(method, [])
            all_evals = []
            for run in runs:
                evals = run.get('eval_rewards', [])
                if evals:
                    all_evals.append(([e['episode'] for e in evals],
                                      [e['mean'] for e in evals]))
            if not all_evals:
                continue

            min_len = min(len(e[1]) for e in all_evals)
            episodes = all_evals[0][0][:min_len]
            rewards_matrix = np.array([e[1][:min_len] for e in all_evals])
            mean_r = rewards_matrix.mean(axis=0)
            std_r = rewards_matrix.std(axis=0)

            color = METHOD_COLORS.get(method, 'black')
            label = METHOD_LABELS.get(method, method)
            ax.plot(episodes, mean_r, color=color, lw=2, label=label)
            ax.fill_between(episodes, mean_r - std_r, mean_r + std_r,
                            color=color, alpha=0.12)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Evaluation Reward')
        ax.set_title(f'{env}: Training Curves')
        ax.legend(loc='lower right', ncol=2, fontsize=8)
        ax.grid(True, alpha=0.3)

        _save_fig(fig, output_dir, f'figure3_training_{env}')


# ============================================================
# Figure 4: L_eff Evolution (Part XI)
# ============================================================

def plot_figure4_leff(all_data, output_dir):
    if not HAS_MPL:
        return
    for env, methods in all_data.items():
        csbapr_runs = methods.get('csbapr', [])
        relu_runs = methods.get('csbapr-relu', [])
        if not csbapr_runs:
            continue

        fig, ax = plt.subplots(figsize=(7, 4))

        for runs, label, color in [(csbapr_runs, 'CS-BAPR (NAU)', '#2166ac'),
                                   (relu_runs, 'CS-BAPR-ReLU', '#e66101')]:
            all_leff = [run.get('L_eff_history', []) for run in runs]
            all_leff = [h for h in all_leff if h]
            if not all_leff:
                continue
            for h in all_leff:
                eps = [e['episode'] for e in h]
                leffs = [e['L_eff'] for e in h]
                ax.plot(eps, leffs, alpha=0.2, color=color)

            min_len = min(len(h) for h in all_leff)
            if min_len > 0:
                eps = [all_leff[0][i]['episode'] for i in range(min_len)]
                mean_leff = np.mean([[h[i]['L_eff'] for i in range(min_len)]
                                     for h in all_leff], axis=0)
                ax.plot(eps, mean_leff, '-', color=color, lw=2.5, label=label)

        ax.set_xlabel('Episode')
        ax.set_ylabel(r'$L_{\rm eff}$ (composed derivative Lipschitz)')
        ax.set_title(f'{env}: $L_{{\\rm eff}}$ Evolution (Part XI)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        _save_fig(fig, output_dir, f'figure4_leff_{env}')


# ============================================================
# Figure 5: Abrupt Shift Timeline (belief/surprise/reward)
# ============================================================

def plot_figure5_abrupt_shift(ood_data, output_dir):
    """Plot reward + Q-std timeline around abrupt shift point."""
    if not HAS_MPL:
        return
    for key, results in ood_data.items():
        if 'shift' not in key:
            continue
        env = key.replace('_shift', '')

        # Pick the most dramatic shift mode
        best_key = max(results.keys(),
                       key=lambda k: abs(results[k].get('ood_distance', 0)))
        r = results[best_key]

        for method_name, mr in r.get('methods', {}).items():
            episodes = mr.get('episodes', [])
            if not episodes or 'step_rewards' not in episodes[0]:
                continue

            fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

            # Average step rewards across episodes
            max_len = max(len(ep['step_rewards']) for ep in episodes)
            reward_matrix = np.full((len(episodes), max_len), np.nan)
            for i, ep in enumerate(episodes):
                sr = ep['step_rewards']
                reward_matrix[i, :len(sr)] = sr

            mean_rew = np.nanmean(reward_matrix, axis=0)
            steps = np.arange(max_len)
            shift_step = mr.get('shift_step', r.get('shift_step', 100))

            # Panel 1: Step reward
            axes[0].plot(steps, mean_rew, color='#2166ac', lw=1.5, alpha=0.8)
            axes[0].axvline(shift_step, color='red', ls='--', lw=1.5,
                           label=f'Shift at step {shift_step}')
            axes[0].set_ylabel('Step Reward')
            axes[0].set_title(f'{env}: Abrupt Shift ({best_key}, {method_name})')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Panel 2: Smoothed reward (rolling window)
            window = min(20, max_len // 5)
            if window > 1:
                smooth = np.convolve(mean_rew, np.ones(window)/window, mode='same')
                axes[1].plot(steps, smooth, color='#2166ac', lw=2)
            axes[1].axvline(shift_step, color='red', ls='--', lw=1.5)
            axes[1].set_xlabel('Step')
            axes[1].set_ylabel('Smoothed Reward')
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            safe_method = method_name.replace(' ', '_').replace('(', '').replace(')', '')
            _save_fig(fig, output_dir, f'figure5_shift_{env}_{safe_method}')


# ============================================================
# Figure 6: Ablation Bar Chart (Q2)
# ============================================================

def plot_figure6_ablation(all_data, output_dir):
    if not HAS_MPL:
        return
    for env, methods in all_data.items():
        present = [m for m in ABLATION_METHODS if m in methods]
        if len(present) < 3:
            continue

        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(present))
        width = 0.6

        means, stds, colors, labels = [], [], [], []
        for m in present:
            runs = methods[m]
            rewards = [r.get('best_eval_reward', 0) for r in runs]
            means.append(np.mean(rewards))
            stds.append(np.std(rewards))
            colors.append(METHOD_COLORS.get(m, 'gray'))
            labels.append(METHOD_LABELS.get(m, m))

        bars = ax.bar(x, means, width, yerr=stds, color=colors, capsize=4,
                      edgecolor='white', linewidth=0.5)

        # Significance vs csbapr
        csbapr_rewards = [r.get('best_eval_reward', 0)
                          for r in methods.get('csbapr', [])]
        for i, m in enumerate(present):
            if m == 'csbapr':
                continue
            m_rewards = [r.get('best_eval_reward', 0) for r in methods[m]]
            if len(m_rewards) >= 2 and len(csbapr_rewards) >= 2:
                _, p = welch_ttest(csbapr_rewards, m_rewards)
                marker = significance_marker(p)
                if marker:
                    ax.text(i, means[i] + stds[i] + 0.02 * abs(max(means)),
                            marker, ha='center', fontsize=12, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=9)
        ax.set_ylabel('Best Eval Reward')
        ax.set_title(f'{env}: Ablation Study (Q2)')
        ax.grid(True, alpha=0.3, axis='y')

        _save_fig(fig, output_dir, f'figure6_ablation_{env}')


# ============================================================
# Figure 7: Sensitivity Analysis
# ============================================================

def plot_figure7_sensitivity(sens_data, output_dir):
    if not HAS_MPL or not sens_data:
        return
    for key, param_results in sens_data.items():
        env = key.replace('sensitivity_', '')
        n_params = len(param_results)
        if n_params == 0:
            continue

        fig, axes = plt.subplots(1, min(n_params, 4), figsize=(4 * min(n_params, 4), 4),
                                  squeeze=False)
        axes = axes[0]

        defaults = {'weight_sym': 0.01, 'jac_weight': 0.1,
                     'nau_reg_weight': 0.01, 'beta_ood': 0.1}

        for i, (param_name, results) in enumerate(param_results.items()):
            if i >= 4:
                break
            ax = axes[i]
            vals = [r['value'] for r in results]
            means = [r['mean_reward'] for r in results]
            stds = [r['std_reward'] for r in results]

            ax.errorbar(vals, means, yerr=stds, fmt='o-', lw=2, capsize=3,
                        color='#2166ac', ms=6)
            if param_name in defaults:
                ax.axvline(defaults[param_name], color='red', ls='--', alpha=0.5)
            ax.set_xlabel(param_name)
            ax.set_ylabel('Reward')
            ax.set_title(param_name)
            ax.grid(True, alpha=0.3)
            if max(vals) / max(min(v for v in vals if v > 0), 1e-10) > 20:
                ax.set_xscale('log')

        fig.suptitle(f'{env}: Hyperparameter Sensitivity', fontsize=13)
        plt.tight_layout()
        _save_fig(fig, output_dir, f'figure7_sensitivity_{env}')


# ============================================================
# Table 1: Main Results (with statistical significance)
# ============================================================

def generate_table1_main_results(all_data, output_dir):
    envs = sorted(all_data.keys())
    display_methods = MAIN_METHODS + ['csbapr-no-irm']

    lines = []
    lines.append("% Table 1: Main Results")
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Best evaluation reward (mean $\\pm$ std over 5 seeds). "
                 "Statistical significance vs.~CS-BAPR: "
                 "$^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$ (Welch's $t$-test).}")
    lines.append("\\label{tab:main}")
    col_spec = "l" + "c" * len(envs)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    lines.append("Method & " + " & ".join(envs) + " \\\\")
    lines.append("\\midrule")

    for method in display_methods:
        label = METHOD_LABELS.get(method, method)
        label_tex = label.replace('_', '\\_')
        row = [label_tex]
        for env in envs:
            runs = all_data.get(env, {}).get(method, [])
            csbapr_runs = all_data.get(env, {}).get('csbapr', [])
            if runs:
                rewards = [r.get('best_eval_reward', 0) for r in runs]
                mean = np.mean(rewards)
                std = np.std(rewards)
                cell = f"${mean:.1f} \\pm {std:.1f}$"

                # Bold best, significance for non-csbapr
                if method != 'csbapr' and csbapr_runs and len(rewards) >= 2:
                    cs_rewards = [r.get('best_eval_reward', 0) for r in csbapr_runs]
                    _, p = welch_ttest(cs_rewards, rewards)
                    sig = significance_marker(p)
                    if sig:
                        cell = f"${mean:.1f} \\pm {std:.1f}${sig}"
                elif method == 'csbapr':
                    cell = f"$\\mathbf{{{mean:.1f} \\pm {std:.1f}}}$"
                row.append(cell)
            else:
                row.append("---")
        lines.append(" & ".join(row) + " \\\\")
        if method == 'csbapr':
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")

    table_str = "\n".join(lines)
    out = os.path.join(output_dir, "table1_main.tex")
    with open(out, "w") as f:
        f.write(table_str)
    print(f"  Table 1 saved: {out}")
    print("\n" + table_str + "\n")


# ============================================================
# Table 2: Ablation Study
# ============================================================

def generate_table2_ablation(all_data, output_dir):
    envs = sorted(all_data.keys())

    lines = []
    lines.append("% Table 2: Ablation Study")
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Ablation study: each CS-BAPR component's contribution. "
                 "Removing any pillar degrades OOD performance.}")
    lines.append("\\label{tab:ablation}")
    col_spec = "ll" + "c" * len(envs)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    lines.append("Variant & Removed & " + " & ".join(envs) + " \\\\")
    lines.append("\\midrule")

    ablation_desc = {
        'csbapr': ('Full', 'None'),
        'csbapr-relu': ('w/o NAU', 'NAU $\\to$ ReLU'),
        'csbapr-no-sindy': ('w/o SINDy', '$\\Gamma_{\\rm sym}$, $\\mathcal{L}_{\\rm AC}$'),
        'csbapr-no-sym': ('w/o $\\Gamma_{\\rm sym}$', 'Sym.~Q-penalty'),
        'csbapr-no-jac': ('w/o $\\mathcal{L}_{\\rm AC}$', 'Jac.~consistency'),
        'csbapr-no-irm': ('w/o IRM', 'Causal filtering'),
    }

    for method in ABLATION_METHODS:
        variant, removed = ablation_desc.get(method, (method, ''))
        row = [variant, removed]
        for env in envs:
            runs = all_data.get(env, {}).get(method, [])
            if runs:
                rewards = [r.get('best_eval_reward', 0) for r in runs]
                mean, std = np.mean(rewards), np.std(rewards)
                cell = f"${mean:.1f} \\pm {std:.1f}$"
                if method == 'csbapr':
                    cell = f"$\\mathbf{{{mean:.1f} \\pm {std:.1f}}}$"
                row.append(cell)
            else:
                row.append("---")
        lines.append(" & ".join(row) + " \\\\")
        if method == 'csbapr':
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")

    table_str = "\n".join(lines)
    out = os.path.join(output_dir, "table2_ablation.tex")
    with open(out, "w") as f:
        f.write(table_str)
    print(f"  Table 2 saved: {out}")


# ============================================================
# Table 3: Computational Cost
# ============================================================

def generate_table3_cost(all_data, output_dir):
    lines = []
    lines.append("% Table 3: Computational Cost")
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{Computational cost comparison (single seed).}")
    lines.append("\\label{tab:cost}")
    lines.append("\\begin{tabular}{llrrr}")
    lines.append("\\toprule")
    lines.append("Env & Method & Time (s) & Overhead & GPU (MB) \\\\")
    lines.append("\\midrule")

    for env, methods in sorted(all_data.items()):
        sac_time = None
        for method in ['sac', 'bapr', 'dr', 'rarl', 'csbapr-no-irm', 'csbapr']:
            runs = methods.get(method, [])
            if not runs:
                continue
            times = [r.get('total_time', 0) for r in runs]
            gpus = [r.get('gpu_memory_peak_mb', 0) for r in runs]
            mean_t = np.mean(times)
            mean_g = np.mean(gpus)
            if method == 'sac':
                sac_time = mean_t
            overhead = f"{mean_t/sac_time:.2f}$\\times$" if sac_time and sac_time > 0 else "---"
            label = METHOD_LABELS.get(method, method)
            lines.append(f"  {env} & {label} & {mean_t:.0f} & {overhead} & {mean_g:.0f} \\\\")
        lines.append("\\midrule")

    # Remove last \\midrule
    if lines[-1] == "\\midrule":
        lines[-1] = "\\bottomrule"
    else:
        lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    table_str = "\n".join(lines)
    out = os.path.join(output_dir, "table3_cost.tex")
    with open(out, "w") as f:
        f.write(table_str)
    print(f"  Table 3 saved: {out}")


# ============================================================
# Table 4: SINDy Quality + IRM Report
# ============================================================

def generate_table4_sindy(all_data, output_dir):
    lines = []
    lines.append("% Table 4: SINDy Identification Quality")
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{SINDy identification quality and IRM filtering diagnostics.}")
    lines.append("\\label{tab:sindy}")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("Environment & $R^2$ & Sparsity & IRM Var. & $n_{\\rm envs}$ \\\\")
    lines.append("\\midrule")

    for env, methods in sorted(all_data.items()):
        runs = methods.get('csbapr', [])
        r2_vals, sp_vals, irm_var_vals, n_envs_vals = [], [], [], []
        for run in runs:
            sr = run.get('sindy_report')
            if sr:
                if sr.get('r_squared') is not None:
                    r2_vals.append(sr['r_squared'])
                sp_vals.append(sr.get('sparsity', 0))
            ir = run.get('irm_report')
            if ir:
                irm_var_vals.append(ir.get('irm_variance', 0))
                n_envs_vals.append(ir.get('n_envs', 0))

        r2 = f"${np.mean(r2_vals):.3f}$" if r2_vals else "---"
        sp = f"${np.mean(sp_vals):.3f}$" if sp_vals else "---"
        iv = f"${np.mean(irm_var_vals):.6f}$" if irm_var_vals else "---"
        ne = f"${int(np.mean(n_envs_vals))}$" if n_envs_vals else "---"
        lines.append(f"  {env} & {r2} & {sp} & {iv} & {ne} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    table_str = "\n".join(lines)
    out = os.path.join(output_dir, "table4_sindy.tex")
    with open(out, "w") as f:
        f.write(table_str)
    print(f"  Table 4 saved: {out}")


# ============================================================
# Helper
# ============================================================

def _save_fig(fig, output_dir, name):
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, f'{name}.pdf')
    png_path = os.path.join(output_dir, f'{name}.png')
    fig.savefig(pdf_path, bbox_inches='tight')
    fig.savefig(png_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  {name} saved: {pdf_path}")


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
    sens_data = load_sensitivity_results(args.results_dir)

    if not all_data and not ood_data:
        print("No results found. Run experiments first:")
        print("  bash scripts/run_experiments.sh --quick")
        return

    print(f"Found: {len(all_data)} envs, {len(ood_data)} OOD evals, "
          f"{len(sens_data)} sensitivity sweeps")

    # Figures
    if ood_data:
        print("\nFigure 1 (Bound vs Actual)...")
        plot_figure1_bound_vs_actual(ood_data, args.output_dir)
        print("Figure 2 (NAU vs ReLU)...")
        plot_figure2_nau_vs_relu(ood_data, args.output_dir)
        print("Figure 5 (Abrupt Shift)...")
        plot_figure5_abrupt_shift(ood_data, args.output_dir)

    if all_data:
        print("Figure 3 (Training Curves)...")
        plot_figure3_training_curves(all_data, args.output_dir)
        print("Figure 4 (L_eff)...")
        plot_figure4_leff(all_data, args.output_dir)
        print("Figure 6 (Ablation)...")
        plot_figure6_ablation(all_data, args.output_dir)

    if sens_data:
        print("Figure 7 (Sensitivity)...")
        plot_figure7_sensitivity(sens_data, args.output_dir)

    # Tables
    if all_data:
        print("\nGenerating tables...")
        generate_table1_main_results(all_data, args.output_dir)
        generate_table2_ablation(all_data, args.output_dir)
        generate_table3_cost(all_data, args.output_dir)
        generate_table4_sindy(all_data, args.output_dir)

    print(f"\nAll outputs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
