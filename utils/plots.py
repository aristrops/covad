import matplotlib.pyplot as plt

def plot_anomaly_ratios(anomaly_ratios, results_main, results_attr, expand_dim):

    colors = {
    "hazelnut": "darkviolet",
    "carpet": "dodgerblue",
    "screw": "plum"
    }

    # --- Plot Main F1 ---
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

    # --- Plot Attr F1 ---
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


def plot_concept_vs_main(results_main, results_attr, model_type):
    """
    Plots Concept AUROC (Attribute AUROC) vs Main AUROC for all categories.
    """
    categories = list(results_main.keys())
    main_scores = []
    attr_scores = []

    for cat in categories:
        # Handle case where multiple anomaly ratios were evaluated
        main = results_main[cat]
        attr = results_attr[cat]
        if isinstance(main, list):
            main = main[0]  # take first value if single ratio per category
            attr = attr[0]
        main_scores.append(main)
        attr_scores.append(attr)

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