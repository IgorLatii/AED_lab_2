import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from pathlib import Path

# === 0. Paths ===
IN_PATH = "../data/processed_features.csv"
OUT_DIR = Path("../data/phase3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# === 1. Load data ===
df = pd.read_csv(IN_PATH, parse_dates=["TIME_PERIOD"], index_col="TIME_PERIOD")

# === 2. Targets and selected features (без мультиколлинеарных дублей) ===
targets = {
    "GDP_growth_yoy": "GDP Growth (YoY)",
    "Inflation_yoy": "Inflation (YoY, HICP)",
    "Unemployment_rate": "Unemployment Rate",
}

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

# Оставляем только реально существующие
features = [c for c in base_features if c in df.columns]

# === 3. Train/Test split по времени ===
train_df = df[df.index < "2021-01-01"]
test_df  = df[df.index >= "2021-01-01"]

print(f"Train: {train_df.index.min().date()} → {train_df.index.max().date()}  (n={len(train_df)})")
print(f"Test : {test_df.index.min().date()} → {test_df.index.max().date()}   (n={len(test_df)})")
print("Features used:", features)

# === 4. Вспомогательные функции ===
def fit_ols_and_export(target_col: str, target_label: str):
    # Синхронизация X/y и удаление NaN (по train)
    X_train = train_df[features].dropna()
    y_train = train_df.loc[X_train.index, target_col].dropna()
    # На всякий случай пересекаем индексы (если у y_train были NaN)
    common_idx = X_train.index.intersection(y_train.index)
    X_train = X_train.loc[common_idx]
    y_train = y_train.loc[common_idx]

    # Стандартизация предикторов (только по train!)
    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(
        scaler.fit_transform(X_train),
        index=X_train.index,
        columns=X_train.columns
    )

    # Добавляем константу
    X_train_sc_const = sm.add_constant(X_train_sc, has_constant="add")

    # OLS
    model = sm.OLS(y_train, X_train_sc_const).fit()

    # --- Коэффициенты, p-values, интервалы ---
    params = model.params.rename("coef")
    pvals = model.pvalues.rename("pvalue")
    conf  = model.conf_int()
    conf.columns = ["conf_low", "conf_high"]

    coeffs = pd.concat([params, pvals, conf], axis=1).reset_index().rename(columns={"index": "feature"})
    coeffs.to_csv(OUT_DIR / f"coeffs_{target_col}.csv", index=False)

    # Сохраняем полный summary в txt
    with open(OUT_DIR / f"summary_{target_col}.txt", "w", encoding="utf-8") as f:
        f.write(model.summary().as_text())

    # --- Предсказания и резидуалы на train/test ---
    # Train residuals
    y_pred_train = model.predict(X_train_sc_const)
    resid_train = (y_train - y_pred_train).rename("residual")

    train_out = pd.concat(
        [y_train.rename("y_true"), y_pred_train.rename("y_pred"), resid_train], axis=1
    )
    train_out.to_csv(OUT_DIR / f"residuals_train_{target_col}.csv", index_label="TIME_PERIOD")

    # Test: готовим X по тем же фичам и тем же scaler
    X_test = test_df[features].dropna()
    # Совместим индексы test X и y
    y_test = test_df.loc[X_test.index, target_col].dropna()
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
            [y_test.rename("y_true"), y_pred_test.rename("y_pred"), resid_test], axis=1
        )
        test_out.to_csv(OUT_DIR / f"residuals_test_{target_col}.csv", index_label="TIME_PERIOD")

    # Краткий вывод по значимым предикторам
    sig = coeffs[(coeffs["feature"]!="const") & (coeffs["pvalue"]<=0.1)].sort_values("pvalue")
    print(f"\n[{target_label}] significant (p<=0.10):")
    if sig.empty:
        print("  none at 10% level")
    else:
        for _, r in sig.iterrows():
            print(f"  {r['feature']}: coef={r['coef']:.3f}, p={r['pvalue']:.3f}")

    return model

# === 5. Запуск по всем таргетам ===
models = {}
for tcol, tlabel in targets.items():
    if tcol not in df.columns:
        print(f"Skip {tlabel} — column not found.")
        continue
    print(f"\n=== OLS for {tlabel} ===")
    models[tcol] = fit_ols_and_export(tcol, tlabel)

print(f"\nSUCCES: Done. Outputs saved to: {OUT_DIR.resolve()}")