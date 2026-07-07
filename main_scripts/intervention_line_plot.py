"""
Line plot: absolute AUC vs. % concepts intervened, all categories overlaid.
One line per category, round markers at each measured point.

Run from covad/ directory:
    uv run main_scripts/intervention_line_plot.py --pkl results/intervention_mvtec_sag_real.pkl
"""

import argparse
import os
import pickle
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", type=str, default="results/intervention_mvtec_sag_real.pkl")
    parser.add_argument("--metric", type=str, default="auc", choices=["auc", "f1"])
    parser.add_argument("--save_path", type=str, default="plots/intervention_line_plot.png")
    args = parser.parse_args()

    with open(args.pkl, "rb") as f:
        results = pickle.load(f)

    valid = {cat: v for cat, v in results.items() if "error" not in v}
    categories = sorted(valid.keys())

    cmap = plt.get_cmap("tab20")
    colors = {cat: cmap(i / max(len(categories) - 1, 1)) for i, cat in enumerate(categories)}

    fig, ax = plt.subplots(figsize=(9, 3), dpi=300)

    for cat in categories:
        v = valid[cat]
        scores = [float(x) for x in v[args.metric]]
        x_perc = np.linspace(0, 100, len(scores))
        ax.plot(
            x_perc, scores,
            # marker="o", 
            label=cat, color=colors[cat],
            markerfacecolor="none", markeredgewidth=1.2,
            linewidth=1.8, markersize=4,
        )

    ax.set_xlabel("Concepts Intervened (%)", fontsize=10)
    ax.set_ylabel(args.metric.upper(), fontsize=10)
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # place legend outside the plot on the right
    ax.legend(title="Category", fontsize=7, title_fontsize=8,
              loc="center left", bbox_to_anchor=(1.05, 0.43), borderaxespad=0, ncol=1)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
    fig.savefig(args.save_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"Saved to {args.save_path}")


if __name__ == "__main__":
    main()
