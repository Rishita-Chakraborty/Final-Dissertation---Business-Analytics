# Valencia Short-Stay Revenue Optimisation (Airbnb) — Dash & Models

**Snapshot:** Inside Airbnb, Valencia, **15-Mar-2025** (entire homes).  
**KPIs (listing-level medians):** ADR ≈ **€110**, Occupancy ≈ **0.51**, RevPAR ≈ **€53.7**.  
**Models:** Log-ADR OLS (adj. R² ≈ **0.125**, RMSE ≈ **€81.83**); Binomial GLM for occupancy with peer-relative price (CV Brier ≈ **0.080**, AUC ≈ **0.56**).  
**App:** Interactive Dash dashboard for “what-if” pricing and neighbourhood insights.

---

## What’s inside
- **Data pipeline** and cleaned table: `valencia_clean.parquet` (plus CSV).
- **Models:** `models/price_ols.pkl`, `models/occ_glm.pkl` (+ columns/medians JSONs).
- **Dashboard:** `Valencia Airbnb Project V1.py` (city map, top neighbourhoods, Top-10 ADR, scenario panel).
- **Validation & robustness:** `02_validation_and_robustness.ipynb` (Elastic Net, grouped-CV, winsor sensitivity, calibration, CIs).
- **Outputs:** figures & CSVs in `outputs/` (seasonality, calibration, playbook, positioning, metrics).

> ⚠️ **Data**: large/raw files from Inside Airbnb are not included. See *Data* section to download and place them locally.

---

## Quickstart

```bash
# 1) Create env (conda or venv)
conda create -n valencia python=3.11 -y
conda activate valencia

# 2) Install
pip install -r requirements.txt

# 3) Place data (see paths below), then run the dashboard:
python "Valencia Airbnb Project V1.py"
# App will serve at http://127.0.0.1:8055/
