import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# === 1. Load processed dataset ===
file_path = "../data/processed_features.csv"
df = pd.read_csv(file_path, parse_dates=["TIME_PERIOD"], index_col="TIME_PERIOD")

# === 2. Define targets ===
targets = {
    "GDP_growth_yoy": "GDP Growth (YoY)",
    "Inflation_yoy": "Inflation (YoY, HICP)",
    "Unemployment_rate": "Unemployment Rate"
}

# === 3. Define reduced feature set based on VIF analysis ===
selected_features = [
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
    "Net Migration (World Bank)_l1"
]

# Keep only existing columns
selected_features = [f for f in selected_features if f in df.columns]

# === 4. Split train/test by time ===
train_df = df[df.index < "2021-01-01"]
test_df = df[df.index >= "2021-01-01"]

# === 5. Prepare function for evaluation ===
def evaluate_model(model, X_train, y_train, X_test, y_test):
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    return {
        "R2_train": r2_score(y_train, y_pred_train),
        "R2_test": r2_score(y_test, y_pred_test),
        "RMSE_test": np.sqrt(mean_squared_error(y_test, y_pred_test)),
        "MAE_test": mean_absolute_error(y_test, y_pred_test)
    }

# === 6. Train and evaluate for each target ===
results = []
for target, label in targets.items():
    if target not in df.columns:
        print(f"⚠️ Skipping {target} (not found in dataset)")
        continue

    X_train = train_df[selected_features].dropna()
    y_train = train_df.loc[X_train.index, target]
    X_test = test_df[selected_features].dropna()
    y_test = test_df.loc[X_test.index, target]

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- OLS ---
    ols = LinearRegression()
    ols.fit(X_train_scaled, y_train)
    res_ols = evaluate_model(ols, X_train_scaled, y_train, X_test_scaled, y_test)
    res_ols["Model"] = "OLS"
    res_ols["Target"] = label
    results.append(res_ols)

    # --- Ridge ---
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    res_ridge = evaluate_model(ridge, X_train_scaled, y_train, X_test_scaled, y_test)
    res_ridge["Model"] = "Ridge"
    res_ridge["Target"] = label
    results.append(res_ridge)

    # --- Lasso ---
    lasso = Lasso(alpha=0.05, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)
    res_lasso = evaluate_model(lasso, X_train_scaled, y_train, X_test_scaled, y_test)
    res_lasso["Model"] = "Lasso"
    res_lasso["Target"] = label
    results.append(res_lasso)

# === 7. Combine results ===
results_df = pd.DataFrame(results)
results_df = results_df[["Target", "Model", "R2_train", "R2_test", "RMSE_test", "MAE_test"]]
results_df.to_csv("../data/model_results.csv", index=False)

print("\n✅ Model training & evaluation complete.")
print("Results saved to '../data/model_results.csv'")
print(results_df)
