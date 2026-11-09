from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "xai" / "report_charts"
OUT.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(ROOT / "results" / "xai" / "comparison_summary.csv")

labels = [f"{r.dataset[:3].upper()}-{r.model.upper()}-{r.classification[:3]}" for r in df[df.method == "SHAP"].itertuples()]
x = np.arange(len(labels))
for column, ylabel, name in [
    ("faithfulness_mean", "Mean top-k probability drop", "faithfulness_comparison.png"),
    ("stability_mean", "Same-input cosine similarity", "stability_comparison.png"),
    ("time_mean", "Mean seconds per explanation", "time_comparison.png"),
    ("sparsity_fraction_mean", "Active feature fraction", "sparsity_comparison.png"),
]:
    shap = df[df.method == "SHAP"][column].to_numpy()
    lime = df[df.method == "LIME"][column].to_numpy()
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(x - 0.19, shap, 0.38, label="SHAP", color="#1f4e79")
    ax.bar(x + 0.19, lime, 0.38, label="LIME", color="#c55a11")
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)
