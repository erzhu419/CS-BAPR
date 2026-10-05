#!/usr/bin/env python3
"""Plot demand-sweep return degradation from the reported table aggregates."""
import argparse
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

ROOT = Path(__file__).resolve().parents[1]
MULTIPLIERS = (1, 2, 5, 10, 20, 50)
METHODS = {
    "bapr-pristine": ("Pristine ReLU", "#777777", "o", "--"),
    "bapr": ("Legacy BAPR", "#c44e52", "s", "--"),
    "csbapr-no-nau": ("Stabilized ReLU", "#0072b2", "^", "-"),
    "csbapr": ("NAU + quadratic", "#009e73", "D", "-"),
    "csbapr-tanh": ("Stabilized tanh", "#cc79a7", "v", "-"),
    "csbapr-kan": ("Stabilized KAN", "#b87500", "P", "-"),
}


def read_sweep(path):
    sweep = {method: {} for method in METHODS}
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            if row["label"] != "tab:ood_sweep":
                continue
            match = re.fullmatch(r"m=(\d+) return median", row["metric"])
            if match is None or row["unit"] != "K":
                raise ValueError("Unexpected demand-sweep metric or return unit")
            sweep[row["method"]][int(match.group(1))] = float(row["value"])
    for method, values in sweep.items():
        if set(values) != set(MULTIPLIERS):
            raise ValueError(f"Incomplete demand sweep for {method}")
    return sweep


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path,
        default=ROOT / "paper/source_data/reported_tables.csv",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "paper/figures/figure1.pdf",
    )
    args = parser.parse_args()
    sweep = read_sweep(args.source)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 12,
        "legend.fontsize": 10.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for method, (label, color, marker, line) in METHODS.items():
        values = sweep[method]
        degradation = [values[1] - values[m] for m in MULTIPLIERS]
        ax.plot(
            MULTIPLIERS, degradation, label=label, color=color,
            marker=marker, linestyle=line, linewidth=1.6, markersize=5,
        )
    ax.set_xscale("log")
    ax.set_xticks(MULTIPLIERS, labels=[str(m) for m in MULTIPLIERS])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel(r"Demand multiplier $m$")
    ax.set_ylabel(r"Return degradation ($10^3$)")
    ax.set_ylim(-15, 710)
    ax.set_xlim(0.9, 56)
    ax.grid(axis="y", color="#dedede", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.legend(
        *ax.get_legend_handles_labels(), loc="lower center",
        bbox_to_anchor=(0.5, 0.01), ncol=2, frameon=False,
        handlelength=2, columnspacing=1.5,
    )
    fig.subplots_adjust(left=0.15, right=0.98, top=0.98, bottom=0.34)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    fig.savefig(args.output.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"Saved {args.output} and {args.output.with_suffix('.png')}")
    print("Source: Table 2 aggregate medians; no seed-level uncertainty or theory overlay.")


if __name__ == "__main__":
    main()
