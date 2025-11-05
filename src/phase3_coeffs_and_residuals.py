"""
phase3_coeffs_and_residuals.py
------------------------------
Purpose:
    Estimate and interpret OLS (Ordinary Least Squares) regression models 
    for three macroeconomic targets and extract:
        • Coefficients with significance tests (p-values)
        • Confidence intervals
        • Model fit summaries
        • Residuals (train/test) for diagnostic analysis

    This script corresponds to **Phase 3.1: Coefficient Interpretation** 
    and partially to **Phase 3.2: Residual Analysis**.

Input:
    ../data/processed_features.csv     — engineered quarterly dataset with lags
Output:
    ../data/phase3/
        coeffs_*.csv                   — table of coefficients and p-values
        summary_*.txt                  — full statistical model summary
        residuals_train_*.csv          — residuals on train sample
        residuals_test_*.csv           — residuals on test sample

Author: Igor Latii  
Course: IS_251_M — Modeling and Regression Analysis  
Date: 2025
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from pathlib import Path

# === 0. Paths and directories ===
IN_PATH = "../data/processed_features.csv"
OUT_DIR = Path("../data/phase3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# === 1. Load prepared dataset ===
df = pd.read_csv(IN_PATH, parse_dates=["TIME_PERIOD"], index_col="TIME_PERIOD")

# === 2. Define target variables and predictor set ===
targets = {
    "GDP_growth_yoy": "GDP Growth (YoY)",
    "Inflation_yoy": "Inflation (YoY, HICP)",
    "Unemployment_rate": "Unemployment Rate",
}

# Selected features: only lagged variables with acceptable collinearity (based on VIF)
base_features = [
    "GDP_growth_yoy_l1",
    "Inflation_yoy_l1",
    "Unemployment_rate_l1",
    "Exports (National Accounts)_l1",
    "Industrial Production Index_l1",
    "Retail Trade Turnover_l1",
    "Energy Prices_l1",
    "Air Passenger Transport_l1",
    "Freight Transport_l1",
    "Emigration of Citizens_l1",
    "Net Migration (World Bank)_l1",
]

# Keep only existing columns (for compatibility)
features = [c for c in base_features if c in df.columns]

# === 3. Split data into train/test sets (chronologically) ===
train_df = df[df.index < "2021-01-01"]
test_df  = df[df.index >= "2021-01-01"]

print(f"Train: {train_df.index.min().date()} → {train_df.index.max().date()}  (n={len(train_df)})")
print(f"Test : {test_df.index.min().date()} → {test_df.index.max().date()}   (n={len(test_df)})")
print("Features used:", features)

# === 4. Define reusable OLS fitting function ===
def fit_ols_and_export(target_col: str, target_label: str):
    """
    Fits an OLS model for the given target variable using standardized predictors.
    Saves coefficients, p-values, confidence intervals, and residuals to files.
    """

    # --- Prepare training data ---
    X_train = train_df[features].dropna()
    y_train = train_df.loc[X_train.index, target_col].dropna()

    # Synchronize indices to handle any missing values
    common_idx = X_train.index.intersection(y_train.index)
    X_train = X_train.loc[common_idx]
    y_train = y_train.loc[common_idx]

    # --- Standardize predictors (train only) ---
    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(
        scaler.fit_transform(X_train),
        index=X_train.index,
        columns=X_train.columns
    )

    # Add constant term (intercept)
    X_train_sc_const = sm.add_constant(X_train_sc, has_constant="add")

    # --- Fit OLS model ---
    model = sm.OLS(y_train, X_train_sc_const).fit()

    # --- Export coefficients, p-values, and confidence intervals ---
    params = model.params.rename("coef")
    pvals = model.pvalues.rename("pvalue")
    conf  = model.conf_int()
    conf.columns = ["conf_low", "conf_high"]

    coeffs = pd.concat([params, pvals, conf], axis=1).reset_index().rename(columns={"index": "feature"})
    coeffs.to_csv(OUT_DIR / f"coeffs_{target_col}.csv", index=False)

    # --- Save full summary report (R², F-statistic, Durbin–Watson, JB test, etc.) ---
    with open(OUT_DIR / f"summary_{target_col}.txt", "w", encoding="utf-8") as f:
        f.write(model.summary().as_text())

    # --- Compute residuals for training set ---
    y_pred_train = model.predict(X_train_sc_const)
    resid_train = (y_train - y_pred_train).rename("residual")

    train_out = pd.concat(
        [y_train.rename("y_true"), y_pred_train.rename("y_pred"), resid_train],
        axis=1
    )
    train_out.to_csv(OUT_DIR / f"residuals_train_{target_col}.csv", index_label="TIME_PERIOD")

    # --- Prepare and evaluate on test set ---
    X_test = test_df[features].dropna()
    y_test = test_df.loc[X_test.index, target_col].dropna()

    # Align indices
    common_idx_test = X_test.index.intersection(y_test.index)
    X_test = X_test.loc[common_idx_test]
    y_test = y_test.loc[common_idx_test]

    if len(X_test) > 0:
        X_test_sc = pd.DataFrame(
            scaler.transform(X_test),
            index=X_test.index,
            columns=X_test.columns
        )
        X_test_sc_const = sm.add_constant(X_test_sc, has_constant="add")

        y_pred_test = model.predict(X_test_sc_const)
        resid_test = (y_test - y_pred_test).rename("residual")

        test_out = pd.concat(
            [y_test.rename("y_true"), y_pred_test.rename("y_pred"), resid_test],
            axis=1
        )
        test_out.to_csv(OUT_DIR / f"residuals_test_{target_col}.csv", index_label="TIME_PERIOD")

    # --- Print significant predictors (p ≤ 0.10) ---
    sig = coeffs[(coeffs["feature"] != "const") & (coeffs["pvalue"] <= 0.1)].sort_values("pvalue")
    print(f"\n[{target_label}] significant (p<=0.10):")
    if sig.empty:
        print("  none at 10% level")
    else:
        for _, r in sig.iterrows():
            print(f"  {r['feature']}: coef={r['coef']:.3f}, p={r['pvalue']:.3f}")

    return model


# === 5. Fit models for all target variables ===
models = {}
for tcol, tlabel in targets.items():
    if tcol not in df.columns:
        print(f"Skip {tlabel} — column not found.")
        continue
    print(f"\n=== OLS for {tlabel} ===")
    models[tcol] = fit_ols_and_export(tcol, tlabel)

print(f"\n✅ SUCCESS: All models estimated and outputs saved to {OUT_DIR.resolve()}")
