"""
Plot intervention results as percentage of concepts intervened.
Allows comparing categories with different numbers of concepts on a shared x-axis.

Run from covad/ directory:
    uv run main_scripts/plot_intervention_perc.py --pkl results/intervention_mvtec_sag.pkl
"""

import argparse
import os
import sys
import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

COMMON_X = np.linspace(0, 100, 200)


def interpolate_to_common(scores, num_concepts):
    # scores has num_concepts+1 points: 0%, ..., 100%
    x = np.linspace(0, 100, len(scores))
    return np.interp(COMMON_X, x, scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", type=str, default="results/intervention_mvtec_sag.pkl")
    parser.add_argument("--metric", type=str, default="auc", choices=["f1", "auc"])
    parser.add_argument("--save_path", type=str, default="plots/interventions_perc.png")
    args = parser.parse_args()

    with open(args.pkl, "rb") as f:
        results = pickle.load(f)

    # filter out errored categories
    valid = {cat: v for cat, v in results.items() if "error" not in v}
    if not valid:
        print("No valid results found.")
        return

    categories = sorted(valid.keys())
    print(f"Plotting {len(categories)} categories: {categories}")

    # interpolate each category to common percentage x-axis
    interp_curves = {}
    for cat in categories:
        v = valid[cat]
        scores = [float(x) for x in v[args.metric]]
        interp_curves[cat] = interpolate_to_common(scores, v["num_concepts"])

    # --- markdown table: baseline (0 interventions) per category ---
    table_lines = [
        f"# Intervention Baseline ({args.metric.upper()}, 0 concepts intervened)\n",
        "| Category | Baseline |",
        "|----------|----------|",
    ]
    baseline_vals = []
    for cat in categories:
        v = valid[cat]
        baseline = float(v[args.metric][0])
        baseline_vals.append(baseline)
        table_lines.append(f"| {cat} | {baseline:.4f} |")
    avg_baseline = np.mean(baseline_vals)
    table_lines.append(f"| **Average** | **{avg_baseline:.4f}** |")

    table_path = args.save_path.replace(".png", "_baseline_table.md")
    os.makedirs(os.path.dirname(os.path.abspath(table_path)), exist_ok=True)
    with open(table_path, "w") as f:
        f.write("\n".join(table_lines) + "\n")
    print(f"Saved table to {table_path}")
    print("\n".join(table_lines))

    stacked = np.stack(list(interp_curves.values()), axis=0)  # (n_cats, 200)
    mean_curve = stacked.mean(axis=0)
    std_curve = stacked.std(axis=0)

    cmap = plt.get_cmap("tab20")
    colors = {cat: cmap(i / max(len(categories) - 1, 1)) for i, cat in enumerate(categories)}

    # --- figure 1: overlay + average (2 panels) ---
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    ax = axes1[0]
    for cat in categories:
        v = valid[cat]
        scores = [float(x) for x in v[args.metric]]
        x_perc = np.linspace(0, 100, len(scores))
        ax.plot(x_perc, scores, marker="o", label=cat, color=colors[cat],
                markerfacecolor="none", markeredgewidth=1.2, linewidth=1.2, markersize=4)
    ax.set_xlabel("Concepts Intervened (%)")
    ax.set_ylabel(args.metric.upper())
    ax.set_title("All Categories")
    ax.legend(title="Category", fontsize=7, title_fontsize=8, ncol=2)
    ax.grid(True, alpha=0.4)
    ax.set_xlim(0, 100)

    ax = axes1[1]
    ax.plot(COMMON_X, mean_curve, color="steelblue", linewidth=2, label="Mean")
    ax.fill_between(COMMON_X,
                    mean_curve - std_curve,
                    mean_curve + std_curve,
                    alpha=0.25, color="steelblue", label="±1 std")
    ax.set_xlabel("Concepts Intervened (%)")
    ax.set_ylabel(args.metric.upper())
    ax.set_title("Average Across All Categories")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)
    ax.set_xlim(0, 100)

    fig1.suptitle(f"SAG Joint Model — {args.metric.upper()} vs. Concept Intervention", fontsize=12)
    fig1.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
    fig1.savefig(args.save_path)
    plt.close(fig1)
    print(f"Saved to {args.save_path}")

    # --- figure 1b: delta from baseline (absolute pp and relative %) ---
    delta_abs = {}   # absolute percentage points
    delta_rel = {}   # relative % increase
    for cat in categories:
        curve = interp_curves[cat]
        baseline = curve[0]
        delta_abs[cat] = (curve - baseline) * 100
        delta_rel[cat] = ((curve - baseline) / (baseline + 1e-8)) * 100

    for delta_curves, label, fname_suffix in [
        (delta_abs, "Absolute gain (pp)", "pp"),
        (delta_rel, "Relative gain (%)", "rel"),
    ]:
        stacked_d = np.stack(list(delta_curves.values()), axis=0)
        mean_d = stacked_d.mean(axis=0)
        std_d = stacked_d.std(axis=0)

        fig_d, axes_d = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

        ax = axes_d[0]
        for cat in categories:
            ax.plot(COMMON_X, delta_curves[cat], label=cat, color=colors[cat],
                    linewidth=1.2)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Concepts Intervened (%)")
        ax.set_ylabel(label)
        ax.set_title("Per-Category")
        ax.legend(title="Category", fontsize=7, title_fontsize=8, ncol=2)
        ax.grid(True, alpha=0.4)
        ax.set_xlim(0, 100)

        ax = axes_d[1]
        ax.plot(COMMON_X, mean_d, color="steelblue", linewidth=2, label="Mean")
        ax.fill_between(COMMON_X, mean_d - std_d, mean_d + std_d,
                        alpha=0.25, color="steelblue", label="±1 std")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Concepts Intervened (%)")
        ax.set_ylabel(label)
        ax.set_title("Average Across All Categories")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.4)
        ax.set_xlim(0, 100)

        fig_d.suptitle(f"SAG Joint — {label} vs. Concept Intervention", fontsize=12)
        fig_d.tight_layout()
        delta_path = args.save_path.replace(".png", f"_delta_{fname_suffix}.png")
        fig_d.savefig(delta_path)
        plt.close(fig_d)
        print(f"Saved delta plot to {delta_path}")

    # --- figure 2: per-category subplots grid ---
    n_cats = len(categories)
    ncols = 4
    nrows = int(np.ceil((n_cats + 1) / ncols))  # +1 for the average panel

    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3), dpi=300)
    axes2_flat = axes2.flatten()

    for i, cat in enumerate(categories):
        ax = axes2_flat[i]
        v = valid[cat]
        scores = [float(x) for x in v[args.metric]]
        x_perc = np.linspace(0, 100, len(scores))
        ax.plot(x_perc, scores, marker="o", color=colors[cat],
                markerfacecolor="none", markeredgewidth=1.5, linewidth=1.5, markersize=5)
        ax.set_title(cat, fontsize=10)
        ax.set_xlim(0, 100)
        ax.set_ylim(bottom=min(0.4, min(scores) - 0.05))
        ax.grid(True, alpha=0.4)
        ax.set_xlabel("% Concepts", fontsize=8)
        ax.set_ylabel(args.metric.upper(), fontsize=8)
        ax.tick_params(labelsize=7)

    # last panel: average
    ax = axes2_flat[n_cats]
    ax.plot(COMMON_X, mean_curve, color="steelblue", linewidth=2, label="Mean")
    ax.fill_between(COMMON_X,
                    mean_curve - std_curve,
                    mean_curve + std_curve,
                    alpha=0.25, color="steelblue", label="±1 std")
    ax.set_title("Average", fontsize=10)
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.4)
    ax.set_xlabel("% Concepts", fontsize=8)
    ax.set_ylabel(args.metric.upper(), fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)

    # hide unused axes
    for j in range(n_cats + 1, len(axes2_flat)):
        axes2_flat[j].set_visible(False)

    fig2.suptitle(f"SAG Joint Model — {args.metric.upper()} vs. Concept Intervention", fontsize=13)
    fig2.tight_layout()

    grid_save = args.save_path.replace(".png", "_grid.png")
    fig2.savefig(grid_save)
    plt.close(fig2)
    print(f"Saved grid to {grid_save}")


if __name__ == "__main__":
    main()
