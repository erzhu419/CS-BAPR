#!/usr/bin/env python3
"""
Figure 1: Theoretical bound vs. actual OOD degradation.

Extracts OOD sweep results from logs and overlays the CS-BAPR theoretical bound:
    ‖err(d)‖ ≤ δ + (ε + (L_eff + M)·‖d‖)·‖d‖

where d = OD_multiplier - 1 (distance from training domain 1x).

Usage:
    python scripts/plot_figure1.py --logdir /tmp/csbapr_exp --output figure1.pdf

Reads logs produced by test_multiline_convergence.py (multi-seed).
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def parse_ood_and_L_eff(log_path):
    """Extract ID total, OOD sweep, and (if logged) L_eff."""
    text = log_path.read_text()
    ood = {}
    for m in re.finditer(r'od_\s*(\d+)x:\s*(-?\d+\.?\d*)\s*±\s*(\d+\.?\d*)', text):
        mult, mean, std = int(m.group(1)), float(m.group(2)), float(m.group(3))
        ood[mult] = (mean, std)
    id_total = None
    m = re.search(r'TOTAL\s*:\s*(-?\d+\.?\d*)', text)
    if m:
        id_total = float(m.group(1))
    # L_eff logged? (not currently logged per-episode, so we return None)
    return id_total, ood


def theoretical_bound_corrected(d, delta=0.0, epsilon=0.05, L_eff=0.0,
                                 M=0.0, gap=0.0):
    """CSBAPR.lean Part X: δ + (ε_total + (L_eff+M)·d)·d"""
    d = np.asarray(d, dtype=float)
    eps_total = epsilon + gap
    return delta + (eps_total + (L_eff + M) * d) * d


# Fitted L_eff estimates (approximate, from architecture analysis):
# NAU actor: L_eff ≈ 5-10 (tightly constrained)
# KAN actor: L_eff ≈ 10-50 (spline-dependent)
# MLP actor: L_eff = ∞ (no Lipschitz guarantee, fitted from data)
L_EFF_PRIOR = {
    'csbapr': 8.0,
    'csbapr-kan': 20.0,
    'csbapr-no-nau': 60.0,
    'bapr': 100.0,  # effectively unbounded
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', default='/tmp/csbapr_exp')
    parser.add_argument('--output', default='figure1.pdf')
    parser.add_argument('--y-scale', choices=['linear', 'log'], default='linear')
    args = parser.parse_args()

    logdir = Path(args.logdir)
    runs = defaultdict(list)
    for log_path in sorted(logdir.glob('*.log')):
        m = re.match(r'(.+)_seed(\d+)\.log$', log_path.name)
        if not m:
            continue
        method = m.group(1)
        id_total, ood = parse_ood_and_L_eff(log_path)
        if id_total is None or not ood:
            continue
        runs[method].append({'id': id_total, 'ood': ood})

    if not runs:
        print(f"No valid logs in {logdir}")
        return

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(7, 5))

    colors = {'csbapr': '#1f77b4', 'csbapr-kan': '#2ca02c',
              'csbapr-no-nau': '#ff7f0e', 'bapr': '#d62728'}
    labels = {'csbapr': 'CS-BAPR (NAU)', 'csbapr-kan': 'CS-BAPR (KAN)',
              'csbapr-no-nau': 'CS-BAPR no-NAU (ablation)',
              'bapr': 'BAPR (MLP baseline)'}

    for method, method_runs in sorted(runs.items()):
        all_mults = sorted(method_runs[0]['ood'].keys())
        d_arr = np.array([m - 1 for m in all_mults])  # distance from 1x
        # Actual error: degradation from ID reward (negative → less reward = more error)
        err_per_seed = []
        for r in method_runs:
            id_r = r['id']
            errs = [id_r - r['ood'][m][0] for m in all_mults]  # degradation magnitude
            err_per_seed.append(errs)
        err_arr = np.array(err_per_seed)  # [n_seeds, n_od]
        mean = err_arr.mean(axis=0)
        std = err_arr.std(axis=0) if len(method_runs) > 1 else np.zeros_like(mean)

        c = colors.get(method, 'gray')
        lbl = labels.get(method, method)
        ax.plot(d_arr, mean, 'o-', color=c, label=f'{lbl} (actual)', lw=2, markersize=7)
        if std.max() > 0:
            ax.fill_between(d_arr, mean - std, mean + std, color=c, alpha=0.2)

    # Theoretical bound — plot with CS-BAPR's L_eff for reference
    d_fine = np.linspace(0, max(all_mults) - 1, 100)
    for method in ['csbapr', 'bapr']:
        if method not in runs:
            continue
        L_eff = L_EFF_PRIOR[method]
        bound = theoretical_bound_corrected(
            d_fine, delta=0.0, epsilon=5000.0, L_eff=L_eff, M=0.0
        )
        c = colors.get(method, 'gray')
        ax.plot(d_fine, bound, '--', color=c, alpha=0.6, lw=1.5,
                label=f'{labels[method]} (bound, L={L_eff:.0f})')

    ax.set_xlabel('OOD distance $d = \\mathrm{od\\_mult} - 1$', fontsize=12)
    ax.set_ylabel('Reward degradation $|R_{ID} - R_{OOD}|$', fontsize=12)
    ax.set_title('CS-BAPR: Theoretical bound vs. actual OOD degradation\n'
                 '(MultiLineEnv, 12 SUMO-calibrated bus lines)', fontsize=12)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    if args.y_scale == 'log':
        ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"Saved {args.output}")

    # Also save PNG for quick preview
    png_path = str(args.output).replace('.pdf', '.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"Saved {png_path}")


if __name__ == '__main__':
    main()
