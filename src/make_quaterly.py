"""
make_quarterly.py
-----------------
Purpose:
    Convert a mixed-frequency macroeconomic dataset (monthly, quarterly,
    semiannual, annual) into a unified quarterly time series suitable
    for regression analysis.

Dataset:
    Input:  ../data/merged_df_readable.csv
    Output: ../data/merged_quarterly.csv

Author: Igor Latii
Course: IS_251_M — Modeling and Regression Analysis
Date: 2025
"""

import pandas as pd

# === 1. Load source dataset ===
# File created in Assignment 1 (EDA phase)
file_path = "../data/merged_df_readable.csv"
df = pd.read_csv(file_path)

# === 2. Parse TIME_PERIOD column as datetime ===
# Handle potential invalid date formats safely
df['TIME_PERIOD'] = pd.to_datetime(df['TIME_PERIOD'], errors='coerce')
print("\nNumber of incorrect dates:", df['TIME_PERIOD'].isna().sum())

# Sort by time and set as index for resampling
df = df.set_index('TIME_PERIOD').sort_index()
print("\nIndex type:", type(df.index))

# === 3. Define variable frequency categories ===
# These categories follow the publication frequency of Eurostat & World Bank data
monthly_cols = [
    'Inflation (HICP Manufacturing)',
    'Unemployment Rate',
    'Air Passenger Transport',
    'Industrial Production Index',
    'Retail Trade Turnover',
    'Tourist Overnight Stays'
]

quarterly_cols = [
    'GDP (Quarterly)',
    'Employment'
]

semiannual_cols = ['Energy Prices']

annual_cols = [
    'Exports (National Accounts)',
    'Emigration of Citizens',
    'Road Passenger Transport',
    'Freight Transport',
    'Net Migration (World Bank)',
    'Population'
]

# === 4. Adjust annual indicators to year-end ===
# Ensures that annual values align with the last quarter of each year
for col in annual_cols:
    if col in df.columns:
        df[col] = df[col].copy()
        df[col].index = df.index + pd.offsets.YearEnd(0)

# === 5. Create a quarterly date range index ===
# Covers the full time period of the dataset
df_quarterly = pd.DataFrame(index=pd.date_range(
    start=df.index.min(), end=df.index.max(), freq='QE'))

# === 6. Resample by frequency type ===

# Monthly → quarterly mean (average within quarter)
for col in monthly_cols:
    if col in df.columns:
        temp = df[[col]].resample('QE').mean()
        df_quarterly[col] = temp[col]

# Quarterly → last observation (official quarterly series)
for col in quarterly_cols:
    if col in df.columns:
        temp = df[[col]].resample('QE').last()
        df_quarterly[col] = temp[col]

# Semiannual → forward-fill (same value across next quarters)
for col in semiannual_cols:
    if col in df.columns:
        temp = df[[col]].resample('QE').ffill()
        df_quarterly[col] = temp[col]

# Annual → fill forward to all quarters in that year
for col in annual_cols:
    if col in df.columns:
        temp = df[[col]].resample('YE').last()   # align to year-end
        temp = temp.resample('QE').ffill()       # fill across quarters
        df_quarterly[col] = temp[col]

# === 7. Keep only data starting from year 2000 ===
# Focus on modern economic period (consistent coverage)
df_quarterly = df_quarterly[df_quarterly.index >= '2000-01-01']

# === 8. Remove empty columns (if no data available) ===
df_quarterly = df_quarterly.dropna(axis=1, how='all')

# === 9. Final cleaning and export ===
# Fill small gaps between quarters
df_quarterly = df_quarterly.sort_index().ffill().bfill()

# Save harmonized quarterly dataset
df_quarterly.to_csv("../data/merged_quarterly.csv", index_label="TIME_PERIOD")

# === 10. Summary output ===
print("\nSUCCESS: The data is converted to quarterly frequency and stored in '../data/merged_quarterly.csv'")
print("Dimension:", df_quarterly.shape)
print("Period:", df_quarterly.index.min(), "→", df_quarterly.index.max())