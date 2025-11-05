import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import os

# === 1. Paths and setup ===
base_dir = "../data/phase3"
targets = ["GDP_growth_yoy", "Inflation_yoy", "Unemployment_rate"]
os.makedirs(os.path.join(base_dir, "residual_plots"), exist_ok=True)
output_dir = os.path.join(base_dir, "residual_plots")

for name in targets:
    file_path = f"{base_dir}/residuals_train_{name}.csv"
    if not os.path.exists(file_path):
        print(f"ERROR: Missing file for {name}, skipping.")
        continue

    residuals = pd.read_csv(file_path, index_col=0)
    if isinstance(residuals, pd.DataFrame):
        residuals = residuals.iloc[:, 0]
    print(f"\n=== Residual analysis for {name} ===")
    print(f"Observations: {len(residuals)}, Mean={residuals.mean():.3f}, Std={residuals.std():.3f}")

    # === Histogram ===
    plt.figure(figsize=(6,4))
    sns.histplot(residuals, bins=20, kde=True)
    plt.title(f"{name}: Residual Distribution")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/residuals_hist_{name}.png")
    plt.close()

    # === Q-Q Plot ===
    plt.figure(figsize=(6,4))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title(f"{name}: Q-Q Plot (Normality Check)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/residuals_qq_{name}.png")
    plt.close()

    # === Print normality test ===
    shapiro_test = stats.shapiro(residuals)
    print(f"Shapiro-Wilk test p-value: {shapiro_test.pvalue:.4f}")
    if shapiro_test.pvalue < 0.05:
        print("→ Residuals deviate from normality.")
    else:
        print("→ Residuals approximately normal.")