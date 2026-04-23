#!/usr/bin/env python3
"""
Collect multi-seed experiment results from logs into a JSON/markdown table.

Usage:
    python scripts/collect_results.py --logdir /tmp/csbapr_exp
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
import numpy as np


def parse_log(path):
    """Extract ID eval, OOD sweep, abrupt burst, and oscillating eval from a run log."""
    text = path.read_text()
    result = {'id_total': None, 'ood': {}, 'abrupt': {}, 'oscillating': {}}

    # ID total: "TOTAL : -619368.7"
    m = re.search(r'TOTAL\s*:\s*(-?\d+\.?\d*)', text)
    if m:
        result['id_total'] = float(m.group(1))

    # OOD sweep: "od_   5x: -645404.6 ± 10733.1"
    for m in re.finditer(r'od_\s*(\d+)x:\s*(-?\d+\.?\d*)\s*±\s*(\d+\.?\d*)', text):
        mult, mean, std = int(m.group(1)), float(m.group(2)), float(m.group(3))
        result['ood'][mult] = (mean, std)

    # Abrupt: "burst_   5x @ t=3600s: -645404.6 ± 10733.1"
    for m in re.finditer(r'burst_\s*(\d+)x @ t=\d+s:\s*(-?\d+\.?\d*)\s*±\s*(\d+\.?\d*)', text):
        burst, mean, std = int(m.group(1)), float(m.group(2)), float(m.group(3))
        result['abrupt'][burst] = (mean, std)

    # Oscillating: "osc_commuter_day       ( 5 switches, peak  50.0x): -645404.6 ± 10733.1"
    for m in re.finditer(
        r'osc_([a-zA-Z0-9_]+?)\s*\(\s*\d+ switches,\s*peak\s*[\d.]+x\):\s*(-?\d+\.?\d*)\s*±\s*(\d+\.?\d*)',
        text,
    ):
        name, mean, std = m.group(1), float(m.group(2)), float(m.group(3))
        result['oscillating'][name] = (mean, std)

    return result


def aggregate(results_per_method):
    """Aggregate across seeds: mean ± std of means."""
    agg = {}
    for method, runs in results_per_method.items():
        id_totals = [r['id_total'] for r in runs if r['id_total'] is not None]
        agg[method] = {
            'id_mean': float(np.mean(id_totals)) if id_totals else None,
            'id_std': float(np.std(id_totals)) if id_totals else None,
            'n_seeds': len(id_totals),
            'ood': {},
            'abrupt': {},
            'oscillating': {},
        }
        all_od = set()
        for r in runs:
            all_od.update(r['ood'].keys())
        for mult in sorted(all_od):
            vals = [r['ood'][mult][0] for r in runs if mult in r['ood']]
            if vals:
                agg[method]['ood'][mult] = (float(np.mean(vals)), float(np.std(vals)))

        all_burst = set()
        for r in runs:
            all_burst.update(r['abrupt'].keys())
        for burst in sorted(all_burst):
            vals = [r['abrupt'][burst][0] for r in runs if burst in r['abrupt']]
            if vals:
                agg[method]['abrupt'][burst] = (float(np.mean(vals)), float(np.std(vals)))

        all_osc = set()
        for r in runs:
            all_osc.update(r['oscillating'].keys())
        for name in sorted(all_osc):
            vals = [r['oscillating'][name][0] for r in runs if name in r['oscillating']]
            if vals:
                agg[method]['oscillating'][name] = (float(np.mean(vals)), float(np.std(vals)))
    return agg


def print_markdown(agg):
    """Print results as a markdown table."""
    methods = sorted(agg.keys())
    print("\n## ID Evaluation (od_mult=1x)\n")
    print("| Method | n_seeds | Total reward (mean ± std) |")
    print("|---|---|---|")
    for m in methods:
        a = agg[m]
        if a['id_mean'] is not None:
            print(f"| {m} | {a['n_seeds']} | {a['id_mean']:,.0f} ± {a['id_std']:,.0f} |")

    # OOD sweep
    all_od = set()
    for a in agg.values():
        all_od.update(a['ood'].keys())
    if all_od:
        print("\n## OOD Evaluation (parametric OD sweep)\n")
        header = "| Method | " + " | ".join(f"{m}x" for m in sorted(all_od)) + " |"
        print(header)
        print("|" + "---|" * (len(all_od) + 1))
        for m in methods:
            row = [m]
            for mult in sorted(all_od):
                if mult in agg[m]['ood']:
                    mean, std = agg[m]['ood'][mult]
                    row.append(f"{mean:,.0f}±{std:,.0f}")
                else:
                    row.append('—')
            print("| " + " | ".join(row) + " |")

    # Abrupt shift
    all_burst = set()
    for a in agg.values():
        all_burst.update(a['abrupt'].keys())
    if all_burst:
        print("\n## Abrupt Shift Evaluation (burst at t=3600s)\n")
        header = "| Method | " + " | ".join(f"burst_{b}x" for b in sorted(all_burst)) + " |"
        print(header)
        print("|" + "---|" * (len(all_burst) + 1))
        for m in methods:
            row = [m]
            for burst in sorted(all_burst):
                if burst in agg[m]['abrupt']:
                    mean, std = agg[m]['abrupt'][burst]
                    row.append(f"{mean:,.0f}±{std:,.0f}")
                else:
                    row.append('—')
            print("| " + " | ".join(row) + " |")

    # Oscillating within-episode schedules
    all_osc = set()
    for a in agg.values():
        all_osc.update(a['oscillating'].keys())
    if all_osc:
        print("\n## Oscillating OOD Evaluation (within-episode demand schedules)\n")
        header = "| Method | " + " | ".join(sorted(all_osc)) + " |"
        print(header)
        print("|" + "---|" * (len(all_osc) + 1))
        for m in methods:
            row = [m]
            for name in sorted(all_osc):
                if name in agg[m]['oscillating']:
                    mean, std = agg[m]['oscillating'][name]
                    row.append(f"{mean:,.0f}±{std:,.0f}")
                else:
                    row.append('—')
            print("| " + " | ".join(row) + " |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', default='/tmp/csbapr_exp')
    parser.add_argument('--output', default=None, help='JSON output path')
    args = parser.parse_args()

    logdir = Path(args.logdir)
    # Parse filenames like "csbapr_seed0.log"
    runs_by_method = defaultdict(list)
    for log_path in sorted(logdir.glob('*.log')):
        m = re.match(r'(.+)_seed(\d+)\.log$', log_path.name)
        if not m:
            continue
        method, seed = m.group(1), int(m.group(2))
        parsed = parse_log(log_path)
        parsed['seed'] = seed
        runs_by_method[method].append(parsed)

    if not runs_by_method:
        print(f"No logs found in {logdir}")
        return

    agg = aggregate(runs_by_method)
    print_markdown(agg)

    if args.output:
        Path(args.output).write_text(json.dumps(agg, indent=2))
        print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
