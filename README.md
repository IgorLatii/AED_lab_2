# Modeling and Regression Analysis — Lab 2  
**Author:** Igor Latii  
**Course:** IS_251_M 
**Year:** 2025  

---

## 📘 Project Overview
This project continues the previous *Exploratory Data Analysis (EDA)* assignment by developing regression models to forecast Latvia’s key macroeconomic indicators:  
- **GDP Growth (YoY)**  
- **Inflation (YoY, HICP)**  
- **Unemployment Rate**

The analysis includes:  
- Data harmonization across mixed frequencies (monthly → quarterly)  
- Feature engineering and lag creation  
- Model training and evaluation (OLS, Ridge, Lasso)  
- Coefficient interpretation, residual analysis, and refinement  

---

## 📂 Folder Structure
```
/src → All Python scripts
/data → Input and output CSV files
/data/phase3 → Coefficients, residuals, plots for Phase 3
/reports → Final report (DOCX / PDF)
```


---

## ⚙️ Execution Order
```bash
git pull
python -m venv .venv
source .venv/bin/activate    # (Linux/Mac)
.venv\Scripts\activate       # (Windows)
pip install -r requirements.txt
python make_quarterly.py
python prepare_features.py.py
python split_vif_check.py
python train_models.py
python phase3_coeffs_and_residuals.py
python phase3_residual_analysis.py
```
Run the scripts **in the following order** to reproduce the analysis:

1. **make_quarterly.py**  
   Converts mixed-frequency data (monthly, quarterly, annual) into a unified **quarterly dataset** (`merged_quarterly.csv`).

2. **prepare_features.py**  
   Creates derived variables:  
   - Year-over-year growth rates (GDP, Inflation)  
   - Lagged predictors and moving averages (MA(4))  
   → Output: `processed_features.csv`

3. **split_vif_check.py**  
   Splits the dataset into **train (2001–2020)** and **test (2021–2025)** sets.  
   Calculates **Variance Inflation Factor (VIF)** to detect multicollinearity.  
   → Output: `vif_analysis.csv`

4. **train_models.py**  
   Trains three regression models (OLS, Ridge, Lasso) for each target variable.  
   Evaluates performance (R², RMSE, MAE).  
   → Output: `model_results.csv`

5. **phase3_coeffs_and_residuals.py**  
   Estimates OLS models, exports coefficients and p-values.  
   → Output: `summary_*.txt`, `coeffs_*.csv`, and residual files in `/data/phase3/`

6. **phase3_residual_analysis.py**  
   Tests residuals for **normality and independence**.  
   Generates histograms and Q–Q plots.  
   → Output: residual plots (`residuals_hist_*.png`, `residuals_qq_*.png`)

---

## 🧾 Results Summary
- **GDP Growth Model:** High explanatory power (R² = 0.857) — persistence and industrial output are key.  
- **Inflation Model:** Low R² (0.176) — volatility dominated by demographic and energy factors.  
- **Unemployment Model:** Very strong fit (R² = 0.966) — GDP, exports, and inflation drive dynamics.

---

## 🧠 Notes
- All scripts require `pandas`, `numpy`, `matplotlib`, and `statsmodels`.  
- Python 3.10+ recommended.  
- The dataset and code are fully reproducible — run scripts sequentially from `/src`.

---

**Final Report:** `IS_251_M_Igor_Latii_Lucrarea_de_Laborator_nr.2.docx` 
