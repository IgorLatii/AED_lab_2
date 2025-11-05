"""
prepare_features.py
-------------------
Purpose:
    Generate target and predictor features for regression modeling
    based on the harmonized quarterly macroeconomic dataset.

    This script constructs:
      - Year-over-year growth rates (GDP, Inflation)
      - Lagged predictors (t–1)
      - Moving-average (MA(4)) smoothed features for cyclical indicators

Input:
    ../data/merged_quarterly.csv  (output from make_quarterly.py)

Output:
    ../data/processed_features.csv

Author: Igor Latii
Course: IS_251_M — Modeling and Regression Analysis
Date: 2025
"""

import pandas as pd

# === 1. Load the harmonized quarterly dataset ===
# Data already resampled to quarterly frequency in the previous step
file_path = "../data/merged_quarterly.csv"
df = pd.read_csv(file_path, parse_dates=["TIME_PERIOD"], index_col="TIME_PERIOD")
print("Loaded data shape:", df.shape)

# === 2. Construct target variables ===
# GDP Growth (YoY) — annualized percentage change over 4 quarters
df["GDP_growth_yoy"] = 100 * (df["GDP (Quarterly)"] / df["GDP (Quarterly)"].shift(4) - 1)

# Inflation (YoY, HICP) — annual percentage change in consumer prices
df["Inflation_yoy"] = 100 * (
    df["Inflation (HICP Manufacturing)"] / df["Inflation (HICP Manufacturing)"].shift(4) - 1
)

# Unemployment rate — level variable (already quarterly)
df["Unemployment_rate"] = df["Unemployment Rate"]

# === 3. Generate first-order lags (t−1) ===
# Lagging predictors introduces temporal dependence between quarters
lag_features = [
    "GDP_growth_yoy",
    "Inflation_yoy",
    "Unemployment_rate",
    "Exports (National Accounts)",
    "Industrial Production Index",
    "Retail Trade Turnover",
    "Energy Prices",
    "Air Passenger Transport",
    "Freight Transport",
    "Net Migration (World Bank)",
    "Emigration of Citizens",
]

for col in lag_features:
    if col in df.columns:
        df[f"{col}_l1"] = df[col].shift(1)  # one-period lag

# === 4. Add moving-average (MA4) filters for cyclical variables ===
# MA(4) = average of the current and previous 3 quarters
# Reduces high-frequency noise and captures underlying trends
ma_features = [
    "Energy Prices",
    "Industrial Production Index",
    "Retail Trade Turnover",
    "Exports (National Accounts)",
]

for col in ma_features:
    if col in df.columns:
        df[f"{col}_ma4"] = df[col].rolling(window=4, min_periods=1).mean()

# === 5. Drop initial rows with missing values due to lagging ===
# The first few quarters will have NaN values from shift(1) and shift(4)
df = df.dropna().copy()

# === 6. Save prepared dataset ===
df.to_csv("../data/processed_features.csv", index_label="TIME_PERIOD")

# === 7. Output summary ===
print("SUCCESS: Feature engineering complete and saved to '../data/processed_features.csv'")
print("Final shape:", df.shape)

# Display covered time period
start_period = df.index.min().strftime("%Y-%m")
end_period = df.index.max().strftime("%Y-%m")
print(f"Period: {start_period} → {end_period}")