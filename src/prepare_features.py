import pandas as pd

# === 1. Load the harmonized quarterly dataset ===
file_path = "../data/merged_quarterly.csv"
df = pd.read_csv(file_path, parse_dates=["TIME_PERIOD"], index_col="TIME_PERIOD")
print("Loaded data shape:", df.shape)

# === 2. Construct target variables ===
# GDP YoY growth (annualized % change)
df["GDP_growth_yoy"] = 100 * (df["GDP (Quarterly)"] / df["GDP (Quarterly)"].shift(4) - 1)

# Inflation YoY (annual % change in HICP)
df["Inflation_yoy"] = 100 * (df["Inflation (HICP Manufacturing)"] / df["Inflation (HICP Manufacturing)"].shift(4) - 1)

# Unemployment rate (already level, not differenced)
df["Unemployment_rate"] = df["Unemployment Rate"]

# === 3. Generate first-order lags (t−1) for key predictors and targets ===
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
    "Emigration of Citizens"
]

for col in lag_features:
    if col in df.columns:
        df[f"{col}_l1"] = df[col].shift(1)

# === 4. Add moving-average (MA4) filters for cyclical variables ===
ma_features = [
    "Energy Prices",
    "Industrial Production Index",
    "Retail Trade Turnover",
    "Exports (National Accounts)"
]

for col in ma_features:
    if col in df.columns:
        df[f"{col}_ma4"] = df[col].rolling(window=4, min_periods=1).mean()

# === 5. Drop initial rows with missing lags (first few quarters) ===
df = df.dropna().copy()

# === 6. Save prepared dataset ===
df.to_csv("../data/processed_features.csv", index_label="TIME_PERIOD")

print("SUCCESS: Feature engineering complete and saved to '../data/processed_features.csv'")
print("Final shape:", df.shape)
# Clean output for period
start_period = df.index.min().strftime("%Y-%m")
end_period = df.index.max().strftime("%Y-%m")
print(f"Period: {start_period} → {end_period}")
