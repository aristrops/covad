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