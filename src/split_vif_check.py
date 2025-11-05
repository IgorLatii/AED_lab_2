import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

# === 1. Load processed dataset ===
file_path = "../data/processed_features.csv"
df = pd.read_csv(file_path, parse_dates=["TIME_PERIOD"], index_col="TIME_PERIOD")
print("Loaded data:", df.shape)

# === 2. Define targets ===
targets = ["GDP_growth_yoy", "Inflation_yoy", "Unemployment_rate"]

# === 3. Define feature set (exclude targets and raw quarterly GDP/HICP duplicates) ===
drop_cols = targets + [
    "GDP (Quarterly)", "Inflation (HICP Manufacturing)", "Unemployment Rate"
]
X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()

# === 4. Train/Test split by time ===
train_df = df[df.index < "2021-01-01"]
test_df = df[df.index >= "2021-01-01"]

print(f"Train period: {train_df.index.min().strftime('%Y-%m')} → {train_df.index.max().strftime('%Y-%m')}")
print(f"Test period:  {test_df.index.min().strftime('%Y-%m')} → {test_df.index.max().strftime('%Y-%m')}")
print(f"Train size: {train_df.shape[0]}, Test size: {test_df.shape[0]}")

# === 5. Prepare data for VIF computation (use train set only) ===
X_train = X.loc[train_df.index].dropna().copy()

# Standardize features to stabilize VIF calculation
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)

# === 6. Compute VIF for each feature ===
vif_data = pd.DataFrame()
vif_data["Feature"] = X_scaled.columns
vif_data["VIF"] = [variance_inflation_factor(X_scaled.values, i) for i in range(X_scaled.shape[1])]

# Sort by descending VIF
vif_data = vif_data.sort_values(by="VIF", ascending=False)

# === 7. Display top correlated variables ===
print("\nTop 10 features by VIF:")
print(vif_data.head(10))

# === 8. Save VIF results ===
vif_data.to_csv("../data/vif_analysis.csv", index=False)
print("\nSUCCESS: VIF analysis saved to '../data/vif_analysis.csv'")