"""
split_vif_check.py
-------------------
Purpose:
    Split the prepared macroeconomic dataset into training and testing subsets 
    based on time periods, and evaluate multicollinearity among predictors 
    using the Variance Inflation Factor (VIF).

    This step ensures that highly correlated variables are identified and 
    can be removed or adjusted before model training.

Input:
    ../data/processed_features.csv   (output from prepare_features.py)

Output:
    ../data/vif_analysis.csv         (sorted VIF table for all features)

Author: Igor Latii
Course: IS_251_M — Modeling and Regression Analysis
Date: 2025
"""

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

# === 1. Load processed dataset ===
# Dataset includes lagged variables and moving averages from the previous step
file_path = "../data/processed_features.csv"
df = pd.read_csv(file_path, parse_dates=["TIME_PERIOD"], index_col="TIME_PERIOD")
print("Loaded data:", df.shape)

# === 2. Define target variables ===
targets = ["GDP_growth_yoy", "Inflation_yoy", "Unemployment_rate"]

# === 3. Define feature set (remove direct target variables and duplicates) ===
# Remove raw GDP, HICP, and Unemployment Rate to avoid redundancy
drop_cols = targets + [
    "GDP (Quarterly)", 
    "Inflation (HICP Manufacturing)", 
    "Unemployment Rate"
]
X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()

# === 4. Split data into training and testing sets by time ===
# Time-based split ensures chronological integrity (no data leakage)
train_df = df[df.index < "2021-01-01"]
test_df = df[df.index >= "2021-01-01"]

print(f"Train period: {train_df.index.min().strftime('%Y-%m')} → {train_df.index.max().strftime('%Y-%m')}")
print(f"Test period:  {test_df.index.min().strftime('%Y-%m')} → {test_df.index.max().strftime('%Y-%m')}")
print(f"Train size: {train_df.shape[0]}, Test size: {test_df.shape[0]}")

# === 5. Prepare train set for VIF computation ===
# Use only training data for multicollinearity diagnostics
X_train = X.loc[train_df.index].dropna().copy()

# Standardize features to stabilize VIF computation
scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)

# === 6. Compute Variance Inflation Factor (VIF) for each predictor ===
# VIF quantifies how much a variable is explained by other predictors
vif_data = pd.DataFrame()
vif_data["Feature"] = X_scaled.columns
vif_data["VIF"] = [
    variance_inflation_factor(X_scaled.values, i)
    for i in range(X_scaled.shape[1])
]

# Sort features by descending VIF to identify most correlated variables
vif_data = vif_data.sort_values(by="VIF", ascending=False)

# === 7. Display top correlated variables ===
print("\nTop 10 features by VIF:")
print(vif_data.head(10))

# === 8. Save VIF results ===
vif_data.to_csv("../data/vif_analysis.csv", index=False)
print("\n✅ SUCCESS: VIF analysis saved to '../data/vif_analysis.csv'")
