import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_anomaly_ratios(anomaly_ratios, results_main, results_attr, expand_dim):

    colors = {
    "hazelnut": "darkviolet",
    "carpet": "dodgerblue",
    "screw": "plum"
    }

    # --- Plot Main AUROC ---
    plt.figure(figsize=(8, 6))
    for category, aucs in results_main.items():
        plt.plot(anomaly_ratios, aucs, marker='o', label=category, color = colors.get(category))
    plt.xlabel("Anomaly Ratio")
    plt.ylabel("AD AUROC")
    plt.title("AD AUROC vs Anomaly Ratio")
    plt.legend(title="Category")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/auroc_vs_anomaly_ratio_main_{expand_dim}MLP.png")

    # --- Plot Attr AUROC ---
    plt.figure(figsize=(8, 6))
    for category, aucs in results_attr.items():
        plt.plot(anomaly_ratios, aucs, marker='s', label=category, color = colors.get(category))
    plt.xlabel("Anomaly Ratio")
    plt.ylabel("Concept AUROC")
    plt.title("Concept AUROC vs Anomaly Ratio")
    plt.legend(title="Category")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/auroc_vs_anomaly_ratio_attr_{expand_dim}MLP.png")


def plot_concept_vs_main(main_scores, attr_scores, categories, model_type):

    plt.figure(figsize=(7, 7))
    plt.scatter(main_scores, attr_scores, color="steelblue", s=60)
    for i, cat in enumerate(categories):
        plt.text(main_scores[i] + 0.005, attr_scores[i] + 0.005, cat, fontsize=9)

    plt.xlabel("Main AUROC", fontsize=12)
    plt.ylabel("Concept AUROC", fontsize=12)
    plt.title(f"Concept AUROC vs Main AUROC across categories for {model_type} model", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"plots/concepts_vs_main_{model_type}.png", dpi=300)
    plt.close()


def auc_heatmap(main_aucs, attr_aucs, mean_auc_main, mean_auc_attr, categories, model_type):

    main_aucs_plot = main_aucs.copy()
    attr_aucs_plot = attr_aucs.copy()
    categories_plot = categories.copy()

    # add "All" column
    main_aucs_plot.append(mean_auc_main)
    attr_aucs_plot.append(mean_auc_attr)
    categories_plot.append("All")

    data = np.array([main_aucs_plot, attr_aucs_plot])

    # Create labels for rows
    row_labels = ['AD', 'Concepts']

    # Create the heatmap
    plt.figure(figsize=(10, 3))
    ax = sns.heatmap(
        data,
        annot=True,          
        fmt=".2f",           
        xticklabels=categories_plot,
        yticklabels=row_labels,
        cmap="YlGnBu"
    )

    # Add title and adjust layout
    plt.title(f"AUROC Scores for {model_type} model")
    plt.xlabel("Category")
    plt.tight_layout()
    plt.savefig(f"plots/aucs_heatmap_{model_type}.png", dpi=300)