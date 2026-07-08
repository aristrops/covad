"""
Bar plot of AUC Δ from baseline at fixed intervention thresholds.

For each category: grouped bars at 0%, 20%, 40%, 60%, 80%, 100% concepts intervened.
Y-axis = absolute pp gain vs. zero-intervention baseline.

Run from covad/ directory:
    uv run main_scripts/intervention_bar_plot.py --pkl results/intervention_mvtec_sag_real.pkl
"""

import argparse
import os
import pickle
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

THRESHOLDS = [0, 20, 40, 60, 80, 100]
THRESHOLD_COLORS = {
    0: "#cccccc",
    20: "#4e8ef7",
    40: "#2ec4b6",
    60: "#4caf50",
    80: "#ff9800",
    100: "#e63946",
}


def auc_at_threshold(scores, pct):
    """Linearly interpolate scores (len = num_concepts+1) at a given percentage."""
    x = np.linspace(0, 100, len(scores))
    return float(np.interp(pct, x, scores))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pkl", type=str, default="results/intervention_mvtec_sag_real.pkl"
    )
    parser.add_argument("--metric", type=str, default="auc", choices=["auc", "f1"])
    parser.add_argument(
        "--save_path", type=str, default="plots/intervention_bar_plot.png"
    )
    args = parser.parse_args()

    with open(args.pkl, "rb") as f:
        results = pickle.load(f)

    valid = {cat: v for cat, v in results.items() if "error" not in v}
    categories = sorted(valid.keys())
    n_cats = len(categories)
    n_thr = len(THRESHOLDS)

    # compute Δ pp from baseline for each (category, threshold)
    delta = np.zeros((n_cats, n_thr))
    for i, cat in enumerate(categories):
        scores = [float(x) for x in valid[cat][args.metric]]
        baseline = scores[0]
        for j, pct in enumerate(THRESHOLDS):
            val = auc_at_threshold(scores, pct)
            delta[i, j] = (val - baseline) * 100  # percentage points

    # layout
    bar_width = 0.25
    group_gap = 0.6
    x_centers = np.arange(n_cats) * (n_thr * bar_width + group_gap)

    fig, ax = plt.subplots(figsize=0.6 * np.array([max(14, n_cats * 1.2), 5]), dpi=400)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    for j, pct in enumerate(THRESHOLDS):
        offsets = x_centers + (j - n_thr / 2 + 0.5) * bar_width
        ax.bar(
            offsets,
            delta[:, j],
            width=bar_width,
            color=THRESHOLD_COLORS[pct],
            label=f"{pct}%",
            zorder=3,
        )

    ax.axhline(0, color="black", linewidth=0.8, zorder=2)
    ax.set_xticks(x_centers)
    # slight tilt for readability
    ax.set_xticklabels(categories, fontsize=9, rotation=25, ha="right")
    ax.set_ylabel(f"Δ from baseline ({args.metric.upper()}, pp)", fontsize=10)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    legend_patches = [
        mpatches.Patch(color=THRESHOLD_COLORS[p], label=f"{p}%") for p in THRESHOLDS
    ]
    # ax.legend(
    #     handles=legend_patches,
    #     title="Perc. of intervened concepts",
    #     fontsize=7,
    #     title_fontsize=8,
    #     loc="upper center",
    #     ncol=len(THRESHOLDS),
    #     bbox_to_anchor=(0.25, 0.3),
    # )
    
    # put it vertically on the right side of the plot
    ax.legend(
        handles=legend_patches,
        title="Perc. of\nintervened\nconcepts",
        fontsize=7,
        title_fontsize=8,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )

    # remove all figure margins so saved image has literally zero white margin
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
    fig.savefig(args.save_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"Saved to {args.save_path}")


if __name__ == "__main__":
    main()
