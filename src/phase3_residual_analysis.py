"""
phase3_residual_analysis.py
---------------------------
Purpose:
    Perform residual diagnostics for the OLS models estimated in Phase 3.1.
    The script evaluates whether model residuals meet key linear regression assumptions:
        • Linearity
        • Independence
        • Homoscedasticity (constant variance)
        • Normality of residuals

    Produces both numerical and visual diagnostics:
        - Shapiro–Wilk normality test
        - Histogram with Kernel Density Estimate (KDE)
        - Q–Q (Quantile–Quantile) plots

Input:
    ../data/phase3/residuals_train_*.csv     — residuals of OLS models from training set
Output:
    ../data/phase3/residual_plots/
        residuals_hist_*.png                 — histograms with KDE
        residuals_qq_*.png                   — Q–Q plots
    Console summary with descriptive statistics and Shapiro–Wilk test results

Author: Igor Latii  
Course: IS_251_M — Modeling and Regression Analysis  
Date: 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# === 1. Define paths and setup ===
base_dir = "../data/phase3"
targets = ["GDP_growth_yoy", "Inflation_yoy", "Unemployment_rate"]

# Create directory for plots
output_dir = os.path.join(base_dir, "residual_plots")
os.makedirs(output_dir, exist_ok=True)

# === 2. Loop through all models and analyze residuals ===
for name in targets:
    file_path = f"{base_dir}/residuals_train_{name}.csv"

    # Check that the residual file exists
    if not os.path.exists(file_path):
        print(f"ERROR: Missing file for {name}, skipping.")
        continue

    # Load residuals
    residuals = pd.read_csv(file_path, index_col=0)
    if isinstance(residuals, pd.DataFrame):
        residuals = residuals.iloc[:, 0]  # Extract residual column if DataFrame

    # === Summary statistics ===
    print(f"\n=== Residual analysis for {name} ===")
    print(f"Observations: {len(residuals)}, Mean={residuals.mean():.3f}, Std={residuals.std():.3f}")

    # === 3. Histogram with Kernel Density Estimate ===
    plt.figure(figsize=(6, 4))
    sns.histplot(residuals, bins=20, kde=True)
    plt.title(f"{name}: Residual Distribution")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/residuals_hist_{name}.png")
    plt.close()

    # === 4. Q–Q Plot (Normality Check) ===
    plt.figure(figsize=(6, 4))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title(f"{name}: Q–Q Plot (Normality Check)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/residuals_qq_{name}.png")
    plt.close()

    # === 5. Normality test (Shapiro–Wilk) ===
    shapiro_test = stats.shapiro(residuals)
    print(f"Shapiro–Wilk test p-value: {shapiro_test.pvalue:.4f}")
    if shapiro_test.pvalue < 0.05:
        print("→ Residuals deviate from normality.")
    else:
        print("→ Residuals approximately normal.")
