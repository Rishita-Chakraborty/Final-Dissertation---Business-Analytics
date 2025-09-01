#!/usr/bin/env python
# coding: utf-8

# # Valencia Short-Stay Revenue Optimisation (Airbnb) — Dash & Models

# ## BEMM466 — MSc Business Analytics (2024–25)  
# ## Valencia Short-Stay (Airbnb) Revenue Optimisation — Reproducible Notebook
# 
# **Class / Module:** BEMM466: Business Analytics Project  
# **Author:** Rishita Chakraborty  
# **Snapshot:** Inside Airbnb — Valencia, **15-Mar-2025** (entire homes)  
# **Folder:** `Documents/valencia_airbnb_project/`
# 
# ### Purpose
# This notebook builds a decision-ready view of Valencia’s STR market and validates two interpretable models:
# - **Price formation:** Log-ADR OLS (HC3 robust SEs)  
# - **Demand (booking probability):** Binomial GLM using **peer-relative price**
# 
# It outputs KPIs (ADR, Occupancy, RevPAR), **robustness checks**, and artefacts used by the Dash dashboard.
# 
# ### Data & Artefacts
# - **Inputs:** `listings.csv.gz`, `calendar.csv.gz`, `neighbourhoods.geojson`  
# - **Derived:** `valencia_clean.parquet/.csv`  
# - **Models:** `models/price_ols.pkl`, `models/occ_glm.pkl`, plus `*_cols.json`, `*_medians.json`  
# - **Outputs:** CSV/PNG in `outputs/` (calibration, seasonality, grouped-CV, winsor sensitivity, playbook)
# 
# ### Run order (sections in this notebook)
# 0. Setup & Data Load  
# 1. Standardise Neighbourhood (`neigh`)  
# 2. Peer Definition & Relative Price (`price_rel_log`)  
# 3. ADR Model — Log-ADR OLS (HC3)  
# 4. ADR Robustness — Elastic Net (CV RMSE)  
# 5. Occupancy Model — Binomial GLM (peer-relative price)  
# 6. Validation — Brier, AUC & Decile Calibration  
# 7. Scenario Uncertainty — 95% CI for GLM Predictions  
# 8. Grouped Cross-Validation by Neighbourhood  
# 9. Winsorisation Sensitivity (ADR caps)  
# **10. Dashboard — run only *after* all previous sections have executed successfully.**
# 
# > **Note:** Section **10 (Dashboard)** depends on artefacts produced by §§1–9. If you open the Dash app separately (`Valencia Airbnb Project V1.py`), ensure the models and outputs exist first.
# 
# ### Requirements
# `pandas, numpy, statsmodels, scikit-learn, plotly, dash, flask-caching, kaleido (for image export)`
# 
# ### Academic note
# This analysis is **associational (non-causal)** and reflects the **15-Mar-2025** snapshot. Results should be refreshed periodically as market conditions and platform policies evolve.
# 

# """
# BEMM466 — Business Analytics Project
# Valencia Short-Stay (Airbnb) Revenue Optimisation
# 
# Author: Rishita Chakraborty
# Snapshot: Inside Airbnb (Valencia), 15-Mar-2025 — entire homes
# 
# Notebook objectives:
#   • Build reproducible KPIs (ADR, Occupancy, RevPAR) at listing/neighbourhood levels
#   • Estimate interpretable models:
#       - Log-ADR OLS with HC3 robust standard errors
#       - Binomial GLM for occupancy using peer-relative price
#   • Validate with CV metrics (RMSE, Brier, AUC), calibration, and robustness checks
#   • Export artefacts consumed by the Dash app (pricing scenarios & spatial views)
# 
# Run order:
#   0–9: Data, features, models, validation, robustness  → must run first
#   10:  Dashboard (depends on artefacts created above)  → run at the end
# 
# Project folder: Documents/valencia_airbnb_project/
# Inputs: listings.csv.gz, calendar.csv.gz, neighbourhoods.geojson
# Derived: valencia_clean.parquet/.csv, models/*.pkl + JSONs, outputs/*
# 
# Academic scope: cross-sectional, associational, reproducible snapshot.
# """
# 

# ### 1) Paths & sanity checks

# In[32]:


from dash import Dash, html, dcc, Input, Output, State
import pandas as pd
import numpy as np

#(for speed caching; wrap in try so app still runs if not installed)
try:
    from flask_caching import Cache
except Exception:
    Cache = None


# In[33]:


from pathlib import Path
import pandas as pd, numpy as np

BASE = Path("Documents/valencia_airbnb_project")
LISTINGS_PATH = BASE/"listings.csv.gz"
CAL_PATH      = BASE/"calendar.csv.gz"

for p in [LISTINGS_PATH, CAL_PATH]:
    print(p.name, "exists:", p.exists(), "| size (MB):", round(p.stat().st_size/1e6, 2))


# ### 2. Setup & Data Load

# In[34]:


def money_to_float(x):
    if pd.isna(x): return np.nan
    return float(str(x).replace("€","").replace("$","").replace(",","").strip())

VAL_LAT, VAL_LON = 39.4699, -0.3763
def haversine(lat1, lon1, lat2=VAL_LAT, lon2=VAL_LON):
    R=6371
    lat1,lon1,lat2,lon2 = map(np.radians,[lat1,lon1,lat2,lon2])
    dlat=lat2-lat1; dlon=lon2-lon1
    a=np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

listings = pd.read_csv(LISTINGS_PATH, low_memory=False)

# basics
if "price" in listings: listings["price"] = listings["price"].map(money_to_float)
if "bathrooms_text" in listings:
    listings["bathrooms"] = listings["bathrooms_text"].str.extract(r"([0-9]*\.?[0-9]+)").astype(float)
elif "bathrooms" in listings:
    listings["bathrooms"] = pd.to_numeric(listings["bathrooms"], errors="coerce")

if "amenities" in listings:
    listings["amenity_count"] = listings["amenities"].fillna("[]").str.count(",")

score_cols = [c for c in listings.columns if c.startswith("review_scores_")]
for c in score_cols: listings[c] = pd.to_numeric(listings[c], errors="coerce")
listings["review_score_mean"] = listings[score_cols].mean(axis=1) if score_cols else np.nan

if {"latitude","longitude"}.issubset(listings.columns):
    listings["dist_km_centre"] = haversine(listings["latitude"], listings["longitude"])

# filter to modelling set
keep = (listings.get("room_type","")=="Entire home/apt")
if "price" in listings: keep &= listings["price"].between(30, 400)  # sensible € band
L = listings.loc[keep].dropna(subset=["latitude","longitude"]).copy()
L["id_str"] = L["id"].astype(str)

L.shape, L[["price","amenity_count","review_score_mean","dist_km_centre"]].describe().round(2)


# ### 3) Load & clean calendar

# In[35]:


cal = pd.read_csv(CAL_PATH, dtype=str, compression="infer", low_memory=False)
cal.columns = cal.columns.str.lower()

id_col   = "listing_id" if "listing_id" in cal.columns else ("id" if "id" in cal.columns else None)
assert id_col is not None and "date" in cal.columns and "available" in cal.columns, cal.columns[:20].tolist()

cal["listing_id_str"] = cal[id_col].astype(str)
cal["date"] = pd.to_datetime(cal["date"], errors="coerce")
v = cal["available"].astype(str).str.lower().str.strip()
cal["booked"] = v.isin(["f","false","0","no"]).astype(int)
if "price" in cal.columns:
    cal["price_clean"] = (cal["price"].str.replace(r"[^0-9.]", "", regex=True)
                          .replace("", np.nan).astype(float))
else:
    cal["price_clean"] = np.nan

cal_info = (len(cal), cal["date"].min(), cal["date"].max(), v.value_counts().to_dict())
cal_info


# ### 4) Aggregate occupancy (last 365 days; fallback to “all” if needed)

# In[36]:


end = cal["date"].max()
cal_yr = cal if pd.isna(end) else cal[(cal["date"] > end - pd.Timedelta(days=365)) & (cal["date"] <= end)]

occ = cal_yr.groupby("listing_id_str").agg(
    nights_booked=("booked","sum"),
    days_observed=("booked","size"),
    occupancy_rate=("booked","mean"),
    avg_price_med=("price_clean","median")
).reset_index()

len(occ), occ.head()


# ### 5) Join → compute ADR / Occupancy / RevPAR → save clean file
# 
# 

# In[37]:


b = L.merge(occ, left_on="id_str", right_on="listing_id_str", how="inner")

# quality filter (lenient first)
b = b[b["days_observed"] >= 200].copy()

# ADR: prefer calendar median; fallback to listings price
b["avg_price"] = b["avg_price_med"].fillna(b.get("price"))
# winsorise ADR to kill crazy outliers, then hard clip
lo, hi = b["avg_price"].quantile([0.01, 0.99])
b["avg_price_w"] = b["avg_price"].clip(lo, hi).clip(30, 400)

b["occupancy_rate"] = b["occupancy_rate"].clip(0, 1)
b["revpar"] = b["avg_price_w"] * b["occupancy_rate"]

b.shape, b[["avg_price_w","occupancy_rate","revpar"]].describe().round(2)


# In[38]:


# save tidy dataset
OUT_PARQ = BASE/"valencia_clean.parquet"
OUT_CSV  = BASE/"valencia_clean.csv"
b.to_parquet(OUT_PARQ, index=False)
b.to_csv(OUT_CSV, index=False)
OUT_PARQ, OUT_CSV


# ### 6) Quick baseline KPIs to quote

# In[39]:


kpis = {
    "listings_used": int(len(b)),
    "median_ADR_eur": float(b["avg_price_w"].median()),
    "median_occupancy": float(b["occupancy_rate"].median()),
    "median_RevPAR_eur": float(b["revpar"].median())
}
kpis


# ### 7) super-simple models for the what-if UI
# 

# In[40]:


import statsmodels.api as sm, numpy as np, pandas as pd

# PRICE model (log-ADR)
cand = ["bedrooms","bathrooms","accommodates","amenity_count","review_score_mean","dist_km_centre"]
use_feats = [c for c in cand if c in b.columns and b[c].notna().mean()>=0.4]

P = b.dropna(subset=["avg_price_w"]).copy()
X_price = P[use_feats].fillna(P[use_feats].median(numeric_only=True)) if use_feats else pd.DataFrame(index=P.index)
X_price = sm.add_constant(X_price, has_constant="add")
y_price = np.log(P["avg_price_w"])
price_model = sm.OLS(y_price, X_price).fit()
price_model.summary().tables[1]


# In[41]:


# OCCUPANCY model (Binomial GLM) with relative price + neighbourhood FE
if "neigh" not in b.columns:
    if "neighbourhood_cleansed" in b: b["neigh"] = b["neighbourhood_cleansed"]
    elif "neighborhood_cleansed" in b: b["neigh"] = b["neighborhood_cleansed"]
    else: b["neigh"] = "Unknown"

grp = b.groupby(["neigh","bedrooms"], dropna=False)["avg_price_w"].median()
b = b.join(grp.rename("grp_med_price_w"), on=["neigh","bedrooms"])
bed_med = b.groupby("bedrooms")["avg_price_w"].median()
b["grp_med_price_w"] = b["grp_med_price_w"].fillna(b["bedrooms"].map(bed_med)).fillna(b["avg_price_w"].median())
b["price_rel_log"] = np.log(np.clip(b["avg_price_w"]/b["grp_med_price_w"], 1e-6, 1e6))

num_feats = [c for c in ["price_rel_log","amenity_count","review_score_mean","dist_km_centre","bedrooms"] if c in b]
topN = b["neigh"].value_counts().nlargest(15).index
D = pd.get_dummies(b["neigh"].where(b["neigh"].isin(topN), "Other"), drop_first=True, prefix="neigh")
X_occ = pd.concat([b[num_feats], D], axis=1).replace([np.inf,-np.inf], np.nan).fillna(b[num_feats].median(numeric_only=True))
X_occ = sm.add_constant(X_occ, has_constant="add")
y_prop = b["nights_booked"]/b["days_observed"]
w = b["days_observed"]

occ_model = sm.GLM(y_prop, X_occ, family=sm.families.Binomial(), freq_weights=w).fit()
occ_model.summary().tables[1]


# ### 8 baseline predictions & elasticity

# In[42]:


# baseline (median features)
adr0 = float(np.exp(price_model.predict(X_price.median().to_frame().T)[0]))
p0   = float(occ_model.predict(X_occ.median().to_frame().T)[0])
rev0 = adr0 * p0

# price elasticity of occupancy ( +1% relative price )
x_med = X_occ.median().to_frame().T
p_base = float(occ_model.predict(x_med))
x_up = x_med.copy(); x_up["price_rel_log"] += np.log(1.01)
p_up  = float(occ_model.predict(x_up))
elasticity = ((p_up - p_base)/p_base)/0.01

adr0, p0, rev0, elasticity


# Sample: 5,761 entire homes (Valencia, 15-Mar-2025 snapshot), robust calendar (≥200 days).
# 
# Market medians: ADR €110, occupancy 51.1%, RevPAR €53.7.
# 
# Price (log-ADR) drivers
# 
# + Bathrooms: +0.088 → ≈ +9% ADR per extra bathroom.
# 
# + Accommodates: +0.107 → ≈ +11% ADR per extra guest capacity.
# 
# – Distance: −0.018 → ≈ −1.8% ADR per km from centre.
# 
# Bedrooms: −0.021 (holding “accommodates” fixed) → extra bedrooms without added capacity slightly reduce ADR (layout effect).
# 
# Review score: +0.075 per 1.0 star → +7.5% ADR; +0.1 star ≈ +0.75%.
# 
# Occupancy (Binomial GLM, with neighbourhood FEs & relative price):
# 
# price_rel_log = −0.304 (right sign: pricing above local median depresses occupancy).
# 
# Elasticity @ median = −0.153 → a +1% price rise cuts occupancy ~0.15% (market not highly elastic at the median).
# 
# Amenities (+), review score (+), distance (−), bedrooms (+) all behave intuitively.

# ### 9. Save model assets

# In[43]:


from pathlib import Path
BASE = Path("Documents/valencia_airbnb_project")
models_dir = BASE / "models"

sorted([p.name for p in models_dir.iterdir()])


# In[44]:


import pickle, json, numpy as np, pandas as pd, statsmodels.api as sm

# load saved artifacts
with open(models_dir/"price_ols.pkl","rb") as f: price_model = pickle.load(f)
with open(models_dir/"occ_glm.pkl","rb") as f:   occ_model   = pickle.load(f)
with open(models_dir/"price_cols.json") as f:    price_cols  = json.load(f)
with open(models_dir/"occ_cols.json") as f:      occ_cols    = json.load(f)

# load your clean data (so we can use medians)
b = pd.read_parquet(BASE/"valencia_clean.parquet")

# ----- ADR (price) prediction at median features -----
Xp = pd.DataFrame([{c: (b[c].median() if c in b else 0) for c in price_cols}])
Xp = sm.add_constant(Xp, has_constant="add")
adr = float(np.exp(price_model.predict(Xp)[0]))

# ----- Occupancy prediction at baseline (price_rel_log=0 → priced at peer median) -----
row = {c: 0.0 for c in occ_cols}
for c in ["amenity_count","review_score_mean","dist_km_centre","bedrooms"]:
    if c in row and c in b: row[c] = float(b[c].median())
if "price_rel_log" in row: row["price_rel_log"] = 0.0

Xo = pd.DataFrame([row])
Xo = sm.add_constant(Xo, has_constant="add")
occ = float(occ_model.predict(Xo)[0])

revpar = adr * occ
adr, occ, revpar


# ### 10. Dashboard

# In[67]:


# FINAL DASHBOARD

from dash import Dash, dcc, html, Input, Output, jupyter_dash
import plotly.express as px
import pandas as pd, numpy as np, statsmodels.api as sm, pickle, json
from pathlib import Path
from math import log

# Open the app in a browser tab; change to "inline" if you prefer
jupyter_dash.default_mode = "tab"
jupyter_dash.inline_exceptions = True

# ---------- Load data & models ----------
BASE = Path("Documents/valencia_airbnb_project")
b    = pd.read_parquet(BASE/"valencia_clean.parquet")

# ensure neighbourhood label
if "neigh" not in b.columns:
    if "neighbourhood_cleansed" in b: b["neigh"] = b["neighbourhood_cleansed"]
    elif "neighborhood_cleansed" in b: b["neigh"] = b["neighborhood_cleansed"]
    else: b["neigh"] = "Unknown"

with open(BASE/"models/price_ols.pkl","rb") as f: price_model = pickle.load(f)
with open(BASE/"models/occ_glm.pkl","rb")  as f: occ_model   = pickle.load(f)
price_cols = json.load(open(BASE/"models/price_cols.json"))
occ_cols   = json.load(open(BASE/"models/occ_cols.json"))

# Headline medians for subtitle
med_adr = float(b["avg_price_w"].median())
med_occ = float(b["occupancy_rate"].median())
med_rev = float(b["revpar"].median())
N = len(b)

# ---------- Citywide map settings (no cropping) ----------
CITY_LAT = 39.4699
CITY_LON = -0.3763
CITY_ZOOM_POINTS  = 11.5
CITY_ZOOM_DENSITY = 11.2
CITY_ZOOM_CHORO   = 11.0

# ---------- Optional: load GeoJSON for choropleth ----------
NEIGH_GJ, NEIGH_GJ_PROP, nb_stats = None, None, None
gj_path = BASE/"neighbourhoods.geojson"
if gj_path.exists():
    with open(gj_path) as f:
        NEIGH_GJ = json.load(f)
    for key in ["neighbourhood","neighborhood","name","NOMBRE","Nombre"]:
        if key in NEIGH_GJ["features"][0]["properties"]:
            NEIGH_GJ_PROP = key
            break
    def _norm(s): return str(s).strip().casefold()
    gj_names = {_norm(f["properties"][NEIGH_GJ_PROP]): f["properties"][NEIGH_GJ_PROP]
                for f in NEIGH_GJ["features"]}
    b["neigh_gj"] = b["neigh"].map(lambda x: gj_names.get(_norm(x), x))
    nb_stats = (b.groupby("neigh_gj", dropna=False)
                  .agg(med_revpar=("revpar","median"),
                       med_occ=("occupancy_rate","median"),
                       med_adr=("avg_price_w","median"))
                  .reset_index()
                  .rename(columns={"neigh_gj":"neigh"}))

# ---------- Helpers ----------
def predict_adr(beds, baths, accom, review, dist):
    row = {c: np.nan for c in price_cols}
    for k,v in {"bedrooms":beds, "bathrooms":baths, "accommodates":accom,
                "review_score_mean":review, "dist_km_centre":dist}.items():
        if k in row: row[k] = v
    X = pd.DataFrame([row]).fillna({c:(b[c].median() if c in b else 0) for c in price_cols})
    X = sm.add_constant(X, has_constant="add")
    return float(np.exp(price_model.predict(X)[0]))

def predict_occ(adr, beds, review, dist, neigh):
    peer = b[(b["neigh"]==neigh) & (b["bedrooms"]==beds)]["avg_price_w"].median()
    if not np.isfinite(peer) or peer <= 0:
        peer = b[b["bedrooms"]==beds]["avg_price_w"].median() or b["avg_price_w"].median()
    price_rel_log = float(np.log(max(adr/peer, 1e-6)))

    row = {c: 0.0 for c in occ_cols}
    for k,v in {"price_rel_log":price_rel_log, "bedrooms":beds,
                "review_score_mean":review, "dist_km_centre":dist}.items():
        if k in row: row[k] = v
    ncol = f"neigh_{neigh}"
    if ncol in row: row[ncol] = 1.0

    # fill numeric (non-dummy) features not provided with dataset medians (e.g., amenity_count if in model)
    for c in occ_cols:
        if c.startswith("neigh_") or c == "const" or c in ["price_rel_log","bedrooms","review_score_mean","dist_km_centre"]:
            continue
        if c in b.columns and (pd.isna(row.get(c)) or row.get(c) == 0.0):
            med = b[c].median()
            if pd.notna(med):
                row[c] = float(med)

    X = pd.DataFrame([row]).reindex(columns=occ_cols, fill_value=0.0)
    X = sm.add_constant(X, has_constant="add")
    return float(np.clip(occ_model.predict(X)[0], 0, 1))

def mk_marks(col):
    s = b[col].dropna()
    if s.empty: return None
    mn, md, mx = float(s.min()), float(s.median()), float(s.max())
    return {mn:f"{mn:.0f}", md:f"{md:.0f}", mx:f"{mx:.0f}"}

neigh_opts = b["neigh"].value_counts().nlargest(15).index.tolist() or ["Other"]

# ---------- App ----------
app = Dash(__name__)
app.layout = html.Div([
    html.H1("Valencia Short-Stay Revenue Optimisation Dashboard"),
    html.P(
        f"Final Project — MSc Business Analytics | Inside Airbnb snapshot (Valencia, 15-Mar-2025). "
        f"Entire homes (n={N:,}). Market medians: ADR €{med_adr:.0f}, Occupancy {med_occ:.0%}, RevPAR €{med_rev:.0f}."
    ),
    html.Details([
        html.Summary("How to read this dashboard"),
        html.Ul([
            html.Li("ADR (Average Daily Rate): average price per booked night (€)."),
            html.Li("Occupancy: share of nights booked over nights observed (0–100%)."),
            html.Li("RevPAR (Revenue per Available Rental): ADR × Occupancy; revenue per calendar night."),
            html.Li("Assumptions: prices winsorised (€30–€400); occupancy from daily availability (last 365 days); "
                    "relative price compares to same-bedroom peers in the same neighbourhood.")
        ])
    ], open=False, style={"margin":"8px 0 16px 0"}),

    html.H3("Scenario Controls"),
    html.Div([
        html.Label("Neighbourhood"),
        dcc.Dropdown(options=[{"label":n,"value":n} for n in neigh_opts],
                     value=neigh_opts[0], id="neigh"),

        html.Label("Bedrooms"),        dcc.Slider(1,6,1, value=2, id="bedrooms",  marks={1:"1",3:"3",6:"6"}),
        html.Label("Bathrooms"),       dcc.Slider(1,4,1, value=1, id="bathrooms", marks={1:"1",2:"2",4:"4"}),
        html.Label("Accommodates"),    dcc.Slider(1,10,1,value=3, id="accom",     marks={1:"1",5:"5",10:"10"}),
        html.Label("Review (1–5)"),    dcc.Slider(3.5,5.0,0.1, value=4.8, id="review",
                                                 marks={3.5:"3.5",4.8:"4.8",5.0:"5.0"}),
        html.Label("Distance to centre (km)"),
        dcc.Slider(0,10,0.1, value=2.0, id="dist", marks={0:"0",2:"2",10:"10"}),

        html.Label("Manual price (€) — enter 0 to use model ADR"),
        dcc.Input(id="price_override", type="number", min=0, max=400, step=1, value=0, debounce=True),

        html.Br(), html.Br(),
        html.H3("Map Settings"),
        html.Label("Map metric"),
        dcc.Dropdown(
            options=[
                {"label":"RevPAR (€)","value":"revpar"},
                {"label":"Occupancy","value":"occupancy_rate"},
                {"label":"ADR (€)","value":"avg_price_w"},
            ],
            value="revpar", id="map_metric", clearable=False, style={"width":"260px"}
        ),
        html.Label("Map mode"),
        dcc.RadioItems(
            options=(
                [{"label":"Points","value":"points"},{"label":"Density","value":"density"}] +
                ([{"label":"Choropleth","value":"choropleth"}] if (NEIGH_GJ is not None) else [])
            ),
            value="points", id="map_mode", inline=True
        ),

        html.H3("Target Setting"),
        html.Label("Target occupancy (%) — used to compute a recommended ADR"),
        dcc.Slider(30,90,1, value=60, id="target_occ", marks={30:"30",60:"60",90:"90"}),
        html.Div(id="price_hint", style={"marginTop":"6px","fontWeight":"600"}),
    ], style={"maxWidth":"820px"}),

    html.H3("Key Performance Indicators — Selected Scenario"),
    dcc.Graph(id="kpis"),

    html.H3("Market Map — Citywide View"),
    dcc.Graph(id="revpar_map"),

    html.H3("Top Neighborhoods — Rank by Reviews, Luxury or Proximity"),
    html.Div([
        html.Label("Rank neighborhoods by"),
        dcc.Dropdown(
            id="rank_metric",
            options=[
                {"label":"Reviews (average score)","value":"reviews"},
                {"label":"Luxury (ADR p75)","value":"luxury"},
                {"label":"Proximity (median km to centre)","value":"proximity"},
            ],
            value="luxury", clearable=False, style={"width":"320px"}
        ),
    ], style={"maxWidth":"820px"}),
    dcc.Graph(id="top_neigh"),

    html.H3("Top 10 Most Expensive Listings (by ADR)"),
    dcc.Graph(id="expensive_top"),

], style={"maxWidth":"1200px","margin":"0 auto"})

@app.callback(
    Output("kpis","figure"),
    Output("revpar_map","figure"),
    Output("price_hint","children"),
    Output("top_neigh","figure"),
    Output("expensive_top","figure"),
    Input("neigh","value"), Input("bedrooms","value"), Input("bathrooms","value"),
    Input("accom","value"), Input("review","value"),
    Input("dist","value"), Input("price_override","value"),
    Input("map_metric","value"), Input("map_mode","value"), Input("target_occ","value"),
    Input("rank_metric","value")
)
def update(neigh, beds, baths, accom, review, dist, price_override,
           map_metric, map_mode, target_occ_pct, rank_metric):

    # ----- KPIs -----
    adr_est = predict_adr(beds, baths, accom, review, dist)
    adr = float(price_override) if price_override and price_override > 0 else adr_est
    occ = predict_occ(adr, beds, review, dist, neigh)
    rev = adr * occ

    kpi = px.bar(
        x=["ADR (€)","Occupancy (%)","RevPAR (€)"],
        y=[adr, occ*100.0, rev],
        title=f"KPI Summary — {beds} BR in {neigh}"
    )
    kpi.update_traces(text=[f"€{adr:.0f}", f"{occ:.0%}", f"€{rev:.0f}"],
                      textposition="outside", cliponaxis=False)
    kpi.update_layout(yaxis_title="", xaxis_title="")
    kpi.update_yaxes(range=[0, max(100, (max(adr, rev)*1.2))])

    # ----- Recommended ADR for target occupancy -----
    price_hint = "—"
    tgt = float(target_occ_pct)/100.0
    if 0 < tgt < 1 and 0 < occ < 1:
        peer = b[(b["neigh"]==neigh) & (b["bedrooms"]==beds)]["avg_price_w"].median()
        if not np.isfinite(peer) or peer <= 0:
            peer = b[b["bedrooms"]==beds]["avg_price_w"].median() or b["avg_price_w"].median()
        beta = float(occ_model.params.get("price_rel_log", np.nan))
        if np.isfinite(beta) and beta != 0:
            eta_now = log(occ/(1-occ))
            eta_tgt = log(tgt/(1-tgt))
            delta_prl = (eta_tgt - eta_now) / beta
            prl_new = np.log(max(adr/peer, 1e-6)) + delta_prl
            adr_rec = float(peer * np.exp(prl_new))
            price_hint = f"Recommendation: set ADR ≈ €{adr_rec:.0f} to target ~{target_occ_pct:.0f}% occupancy (current €{adr:.0f})."

    # ----- MAP (always citywide) -----
    title_map = {"revpar":"RevPAR (€)","occupancy_rate":"Occupancy","avg_price_w":"ADR (€)"}[map_metric]
    df = b.dropna(subset=["latitude","longitude",map_metric]).copy()
    df = df[df["bedrooms"] == beds]
    vmax = float(df[map_metric].quantile(0.95)) if len(df) else (1.0 if map_metric=="occupancy_rate" else 200)

    if map_mode == "density":
        fmap = px.density_mapbox(
            df, lat="latitude", lon="longitude", z=map_metric,
            radius=25,
            center={"lat": CITY_LAT, "lon": CITY_LON},
            zoom=CITY_ZOOM_DENSITY, height=520, range_color=(0, vmax),
            title=f"{title_map} — Density view (citywide, {beds} BR, n={len(df):,})"
        )
        fmap.update_layout(mapbox_style="open-street-map")
    elif map_mode == "choropleth" and (NEIGH_GJ is not None) and (nb_stats is not None):
        metric_map = {"revpar":"med_revpar","occupancy_rate":"med_occ","avg_price_w":"med_adr"}[map_metric]
        Z = nb_stats.rename(columns={metric_map:"z"}).copy()
        fmap = px.choropleth_mapbox(
            Z, geojson=NEIGH_GJ, featureidkey=f"properties.{NEIGH_GJ_PROP}",
            locations="neigh", color="z", range_color=(0, vmax),
            center={"lat": CITY_LAT, "lon": CITY_LON},
            zoom=CITY_ZOOM_CHORO, height=520,
            title=f"{title_map} — Neighbourhood medians (citywide)"
        )
        fmap.update_layout(mapbox_style="open-street-map")
    else:
        dpts = df.sample(5000, random_state=42) if len(df) > 5000 else df
        hover_cols = [c for c in ["name","avg_price_w","occupancy_rate","revpar","neigh"] if c in dpts]
        fmap = px.scatter_mapbox(
            dpts, lat="latitude", lon="longitude",
            color=map_metric, range_color=(0, vmax),
            hover_data=hover_cols,
            center={"lat": CITY_LAT, "lon": CITY_LON},
            zoom=CITY_ZOOM_POINTS, height=520,
            title=f"{title_map} — Points view (citywide, {beds} BR, n={len(dpts):,})"
        )
        fmap.update_layout(mapbox_style="open-street-map")

    # ----- Top Neighborhoods (by chosen metric) -----
    nb = b[b["bedrooms"] == beds].copy()
    counts = nb.groupby("neigh")["id"].count().rename("n")
    if rank_metric == "reviews":
        agg = nb.groupby("neigh")["review_score_mean"].mean().rename("value").to_frame().join(counts)
        agg = agg[agg["n"] >= 20].dropna().sort_values("value", ascending=False).head(10)
        tn = px.bar(agg[::-1], x="value", y=agg[::-1].index,
                    title=f"Top Neighborhoods by Reviews (avg score, {beds} BR)",
                    labels={"value":"Average Review Score","index":"Neighborhood"})
        tn.update_xaxes(range=[4.0, 5.0])
    elif rank_metric == "proximity":
        agg = nb.groupby("neigh")["dist_km_centre"].median().rename("value").to_frame().join(counts)
        agg = agg[agg["n"] >= 20].dropna().sort_values("value", ascending=True).head(10)
        tn = px.bar(agg[::-1], x="value", y=agg[::-1].index,
                    title=f"Top Neighborhoods by Proximity (median km to centre, {beds} BR)",
                    labels={"value":"Median distance (km)","index":"Neighborhood"})
    else:  # luxury
        agg = nb.groupby("neigh")["avg_price_w"].quantile(0.75).rename("value").to_frame().join(counts)
        agg = agg[agg["n"] >= 20].dropna().sort_values("value", ascending=False).head(10)
        tn = px.bar(agg[::-1], x="value", y=agg[::-1].index,
                    title=f"Top Neighborhoods by Luxury (ADR 75th percentile, {beds} BR)",
                    labels={"value":"ADR (75th percentile, €)","index":"Neighborhood"})

    # ----- Top 10 Most Expensive Listings (by ADR) -----
    cols = ["id","name","neigh","bedrooms","bathrooms","accommodates",
            "review_score_mean","dist_km_centre","avg_price_w","occupancy_rate","revpar"]
    L = (b[b["bedrooms"] == beds][cols]
            .dropna(subset=["avg_price_w"])
            .sort_values(["avg_price_w","occupancy_rate"], ascending=[False, False])
            .head(10).copy())
    # tidy label
    def short(s):
        if not isinstance(s, str): return "(no title)"
        return s if len(s) <= 40 else s[:37] + "…"
    L["label"] = L.apply(lambda r: f'#{r["id"]} — {short(r["name"])}', axis=1)

    expg = px.bar(L.iloc[::-1], x="avg_price_w", y="label",
                  title=f"Top 10 Most Expensive Listings (ADR, {beds} BR)",
                  labels={"avg_price_w":"ADR (€)","label":"Listing"})
    expg.update_traces(
        customdata=L[["neigh","bedrooms","bathrooms","accommodates",
                      "review_score_mean","occupancy_rate","revpar"]].values,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "ADR: €%{x:.0f}<br>"
            "Neighbourhood: %{customdata[0]}<br>"
            "Bedrooms: %{customdata[1]} | Baths: %{customdata[2]} | Accom: %{customdata[3]}<br>"
            "Review: %{customdata[4]:.2f}<br>"
            "Occupancy: %{customdata[5]:.0%} | RevPAR: €%{customdata[6]:.0f}"
            "<extra></extra>"
        )
    )

    return kpi, fmap, price_hint, tn, expg

# Run it — change the port if needed
app.run(port=8057, jupyter_height=520)


# ### 11. Freeze & document snapshot for Valencia Airbnb project

# In[46]:


from pathlib import Path
import json, hashlib, os, sys, platform, datetime as dt
import importlib.metadata as md

BASE = Path("Documents/valencia_airbnb_project")
BASE.mkdir(parents=True, exist_ok=True)

# ---- Helper: file info (size + SHA256) ----
def file_info(path: Path):
    path = Path(path)
    if not path.exists():
        return {"exists": False, "size_bytes": 0, "sha256": None}
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return {
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": h.hexdigest(),
    }

# ---- 1) Write the exact cleaning script we used/need to reproduce outputs ----
CLEANING_SCRIPT = r'''# recreate_clean_dataset.py
"""
Recreate the cleaned Valencia dataset from Inside Airbnb snapshot (15-Mar-2025).

Inputs (in the same folder):
  - listings.csv.gz
  - calendar.csv.gz
  - neighbourhoods.geojson  (optional; not needed for cleaning)

Outputs:
  - valencia_clean.parquet
  - valencia_clean.csv

Rules/assumptions:
  - Entire homes/apartments only
  - Price winsorised to €30–€400
  - Occupancy from last 365 days of calendar: nights_booked = (available == 'f')
  - ADR per listing = median of calendar 'price' across observed days
  - RevPAR = ADR * Occupancy
  - Distance to centre computed from lat/lon to (39.4699, -0.3763) in km
"""

from pathlib import Path
import pandas as pd
import numpy as np
import math

BASE = Path("Documents/valencia_airbnb_project")
LISTINGS = BASE/"listings.csv.gz"
CALENDAR = BASE/"calendar.csv.gz"
OUT_PARQ = BASE/"valencia_clean.parquet"
OUT_CSV  = BASE/"valencia_clean.csv"

# ---------- Utils ----------
def parse_price_series(s: pd.Series) -> pd.Series:
    if s.dtype.kind in "if":
        return s.astype(float)
    c = (s.astype(str)
           .str.replace(r"[^\d.\-]", "", regex=True)
           .replace({"": np.nan}))
    return pd.to_numeric(c, errors="coerce")

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi  = np.radians(lat2 - lat1)
    dlamb = np.radians(lon2 - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlamb/2.0)**2
    return 2*R*np.arcsin(np.sqrt(a))

# ---------- Load listings ----------
lst = pd.read_csv(LISTINGS, low_memory=False)
# canonical ids
if "id" not in lst.columns:
    raise RuntimeError("listings.csv.gz missing 'id' column")
lst["id"] = lst["id"].astype(str)

# Filter entire homes/apartments
room_col = "room_type" if "room_type" in lst.columns else None
if room_col:
    lst = lst[lst[room_col].str.contains("entire", case=False, na=False)]

# Keep useful columns if present
keep_cols = [c for c in [
    "id","name","latitude","longitude","bedrooms","bathrooms","accommodates",
    "neighbourhood_cleansed","neighborhood_cleansed","room_type",
    "price","host_id","number_of_reviews","review_scores_rating",
    "reviews_per_month"
] if c in lst.columns]
lst = lst[keep_cols].copy()

# Price field in listings (may be sparse / string)
if "price" in lst.columns:
    lst["price"] = parse_price_series(lst["price"])

# Review score (unified)
if "review_scores_rating" in lst.columns:
    lst["review_score_mean"] = lst["review_scores_rating"] / 20.0  # 0–5 scale
elif "review_score_mean" not in lst.columns:
    lst["review_score_mean"] = np.nan

# ---------- Load calendar ----------
cal = pd.read_csv(CALENDAR, dtype=str, low_memory=False)
# standardise cols
rename = {c.lower(): c for c in cal.columns}
cal.columns = [c.lower() for c in cal.columns]
cal.rename(columns={"listing_id":"listing_id", "adjusted_price":"adjusted_price"}, inplace=True)

# Parse date & price
cal["date"] = pd.to_datetime(cal["date"], errors="coerce")
cal["price_num"] = parse_price_series(cal["price"] if "price" in cal.columns else cal.get("adjusted_price"))
cal["available_flag"] = (cal["available"].astype(str).str.lower() == "t")

# last 365 days window if possible
if cal["date"].notna().any():
    end = cal["date"].max()
    start = end - pd.Timedelta(days=365)
    cal = cal[(cal["date"] >= start) & (cal["date"] <= end)]

# Aggregate per listing: nights booked, days observed, occupancy, median ADR
cal["listing_id_str"] = cal["listing_id"].astype(str)
agg = (cal.groupby("listing_id_str")
          .agg(nights_booked = ("available_flag", lambda s: (~s).sum()),
               days_observed = ("available_flag", "size"),
               avg_price_med = ("price_num", "median"))
          .reset_index())
agg["occupancy_rate"] = np.where(agg["days_observed"]>0,
                                 agg["nights_booked"]/agg["days_observed"], np.nan)

# ---------- Merge calendar metrics into listings ----------
lst["id_str"] = lst["id"].astype(str)
b = lst.merge(agg, left_on="id_str", right_on="listing_id_str", how="left")

# Compute final ADR from calendar median; clamp to €30–€400
b["avg_price_w"] = parse_price_series(b["avg_price_med"])
b["avg_price_w"] = b["avg_price_w"].clip(lower=30, upper=400)

# RevPAR
b["revpar"] = b["avg_price_w"] * b["occupancy_rate"]

# Neighbourhood (unified)
if "neighbourhood_cleansed" in b.columns:
    b["neigh"] = b["neighbourhood_cleansed"]
elif "neighborhood_cleansed" in b.columns:
    b["neigh"] = b["neighborhood_cleansed"]
else:
    b["neigh"] = "Unknown"

# Distance to centre
if {"latitude","longitude"}.issubset(b.columns):
    b["dist_km_centre"] = haversine_km(b["latitude"].astype(float),
                                       b["longitude"].astype(float),
                                       39.4699, -0.3763)
else:
    b["dist_km_centre"] = np.nan

# Keep sensible numeric types
for c in ["bedrooms","bathrooms","accommodates"]:
    if c in b.columns:
        b[c] = pd.to_numeric(b[c], errors="coerce")

# Save outputs
b.to_parquet(OUT_PARQ, index=False)
b.to_csv(OUT_CSV, index=False)

print("Saved:", OUT_PARQ, OUT_CSV)
print("Rows:", len(b), "| Medians -> ADR €%.0f, Occ %.0f%%, RevPAR €%.0f" %
      (b["avg_price_w"].median(), 100*b["occupancy_rate"].median(), b["revpar"].median()))
'''

script_path = BASE / "recreate_clean_dataset.py"
script_path.write_text(CLEANING_SCRIPT, encoding="utf-8")

# ---- 2) Gather metadata & write METADATA.md / metadata.json ----
snapshot_date = "2025-03-15"
inputs = [
    BASE/"listings.csv.gz",
    BASE/"calendar.csv.gz",
    BASE/"neighbourhoods.geojson",
]
outputs = [
    BASE/"valencia_clean.parquet",
    BASE/"valencia_clean.csv",
    BASE/"models/price_ols.pkl",
    BASE/"models/occ_glm.pkl",
]

pkg_versions = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
}
for p in ["pandas","numpy","statsmodels","plotly"]:
    try:
        pkg_versions[p] = md.version(p)
    except Exception:
        pkg_versions[p] = None

meta = {
    "project": "Valencia Short-Stay Revenue Optimisation",
    "source": "Inside Airbnb — Valencia",
    "snapshot_date": snapshot_date,
    "created_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    "paths": {"base": str(BASE)},
    "inputs": {p.name: file_info(p) for p in inputs},
    "outputs": {p.name: file_info(p) for p in outputs},
    "cleaning_rules": [
        "Entire homes/apartments only",
        "Price winsorised to €30–€400",
        "Occupancy from last 365 days of calendar (nights_booked = available=='f')",
        "ADR per listing = median daily price over observed days",
        "RevPAR = ADR × Occupancy",
        "Distance to centre from (39.4699, -0.3763) using haversine (km)",
    ],
    "recreate_script": "recreate_clean_dataset.py",
    "environment": pkg_versions,
}

# Write JSON
metadata_json = BASE / "metadata.json"
metadata_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

# Write friendly Markdown
def human_size(n):
    return f"{n/1_048_576:.2f} MB"

lines = []
lines += [
    "# Data Snapshot — Valencia Airbnb",
    "",
    f"- **Source:** Inside Airbnb (Valencia)",
    f"- **Snapshot date:** {snapshot_date}",
    f"- **Created (UTC):** {meta['created_utc']}",
    "",
    "## Files",
    "### Inputs",
]
for p in inputs:
    info = meta["inputs"][p.name]
    sz = human_size(info["size_bytes"]) if info["exists"] else "missing"
    sha = (info["sha256"][:12] + "…") if info["sha256"] else "—"
    lines.append(f"- `{p.name}` — {sz}, sha256 {sha}")

lines += ["", "### Outputs"]
for p in outputs:
    info = meta["outputs"][p.name]
    sz = human_size(info["size_bytes"]) if info["exists"] else "missing"
    sha = (info["sha256"][:12] + "…") if info["sha256"] else "—"
    lines.append(f"- `{p.name}` — {sz}, sha256 {sha}")

lines += [
    "",
    "## Cleaning Rules",
] + [f"- {r}" for r in meta["cleaning_rules"]] + [
    "",
    "## Reproduce",
    "Run:",
    "```bash",
    "python recreate_clean_dataset.py",
    "```",
    "",
    "## Environment",
] + [f"- {k}: {v}" for k,v in meta["environment"].items()]

metadata_md = BASE / "METADATA.md"
metadata_md.write_text("\n".join(lines), encoding="utf-8")

print("Wrote files:")
print(" -", script_path)
print(" -", metadata_json)
print(" -", metadata_md)


# ### 12.Model Validations

# In[58]:


# ===============================
# Model Validation — Valencia STR
# ===============================
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import pickle, json, math
from sklearn.metrics import mean_squared_error, roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
from IPython.display import display

BASE = Path("Documents/valencia_airbnb_project")

# ---------- Load data & models ----------
b = pd.read_parquet(BASE/"valencia_clean.parquet")

with open(BASE/"models/price_ols.pkl","rb") as f: price_model = pickle.load(f)
with open(BASE/"models/occ_glm.pkl","rb")  as f: occ_model   = pickle.load(f)
price_cols = json.load(open(BASE/"models/price_cols.json"))
occ_cols   = json.load(open(BASE/"models/occ_cols.json"))

# ---------- Helpers ----------
def make_X(df: pd.DataFrame, cols):
    """
    Build a design matrix with the requested cols.
    - If a column exists in df, take it.
    - If missing, fill 0.0 (safe for dummies).
    - Then fill numeric NaNs with the column median where available in df.
    """
    X = pd.DataFrame(index=df.index)
    for c in cols:
        if c in df.columns:
            X[c] = df[c]
        else:
            X[c] = 0.0
    # numeric NA -> median
    for c in cols:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            X[c] = X[c].fillna(df[c].median())
    return X

# ==========================================================
# 1) PRICE MODEL (log-ADR OLS) — metrics & diagnostics
# ==========================================================
print("\n=== PRICE MODEL (log-ADR OLS) ===")
# target & predictions
y_eur = b["avg_price_w"].astype(float)
y_log = np.log(y_eur.clip(lower=1))  # guard against zeros

X_price = make_X(b, price_cols)
X_price = sm.add_constant(X_price, has_constant="add")

yhat_log = price_model.predict(X_price)
yhat_eur = np.exp(yhat_log)

# Adjusted R^2 (use model's if available, else compute)
try:
    r2_adj = float(price_model.rsquared_adj)
except Exception:
    ss_res = np.sum((y_log - yhat_log)**2)
    ss_tot = np.sum((y_log - y_log.mean())**2)
    r2 = 1 - ss_res/ss_tot
    n, k = X_price.shape
    r2_adj = 1 - (1-r2)*(n-1)/(n-k)

# RMSE on ADR (€)
rmse_eur = float(np.sqrt(mean_squared_error(y_eur, yhat_eur)))

print(f"Adjusted R^2 (log scale): {r2_adj:.3f}")
print(f"RMSE on ADR (€): {rmse_eur:.2f}")

# Robust (HC3) coefficient table
rob = price_model.get_robustcov_results(cov_type="HC3")
coef_tbl = pd.DataFrame({
    "coef": rob.params,
    "robust_se(HC3)": rob.bse,
    "t": rob.tvalues,
    "p>|t|": rob.pvalues
})
ci_ols_raw = rob.conf_int()
if isinstance(ci_ols_raw, pd.DataFrame):
    ci_ols = ci_ols_raw.copy()
else:  # ndarray
    ci_ols = pd.DataFrame(ci_ols_raw, index=coef_tbl.index)
ci_ols.columns = ["[0.025", "0.975]"]
coef_tbl = coef_tbl.join(ci_ols)

print("\nRobust (HC3) coefficients — top 12 by |t|:")
display(coef_tbl.reindex(coef_tbl["t"].abs().sort_values(ascending=False).index).head(12))

# VIFs (on numeric features, excluding constant; drop zero-variance columns)
Xp_no_const = X_price.drop(columns=["const"])
Xp_num = Xp_no_const.select_dtypes(include=[np.number]).copy()
Xp_num = Xp_num.fillna(Xp_num.median(numeric_only=True))
var = Xp_num.var(axis=0)
keep = var[var > 0].index
Xp_num = Xp_num[keep]

try:
    vif_df = pd.DataFrame({
        "feature": Xp_num.columns,
        "VIF": [variance_inflation_factor(Xp_num.values, i) for i in range(Xp_num.shape[1])]
    }).sort_values("VIF", ascending=False)
    print("\nVIF (top 15):")
    display(vif_df.head(15))
except Exception as e:
    print("VIF computation skipped (likely too many columns or singular matrix):", e)

# Residual diagnostics — one figure with 2 panels
resid = y_log - yhat_log
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Residuals vs Fitted
axs[0].scatter(yhat_log, resid, s=10)
axs[0].axhline(0, linestyle="--")
axs[0].set_xlabel("Fitted values (log ADR)")
axs[0].set_ylabel("Residuals (log ADR)")
axs[0].set_title("Residuals vs Fitted — Price Model")

# Q–Q plot
sm.ProbPlot(resid).qqplot(line='45', ax=axs[1])
axs[1].set_title("Q–Q Plot of Residuals — Price Model")
plt.tight_layout()
plt.show()

# ==========================================================
# 2) OCCUPANCY MODEL (Binomial GLM) — metrics & calibration
# ==========================================================
print("\n=== OCCUPANCY MODEL (Binomial GLM) ===")

X_occ = make_X(b, occ_cols)
X_occ = sm.add_constant(X_occ, has_constant="add")

y_true = b["occupancy_rate"].astype(float)
mask = y_true.notna()
y_true = y_true[mask]
X_occ_m = X_occ.loc[mask]

y_pred = occ_model.predict(X_occ_m)  # predicted probability
y_pred = np.clip(y_pred, 0, 1)

# Brier score (weighted by days_observed if available)
if "days_observed" in b.columns:
    w = b.loc[mask, "days_observed"].astype(float).fillna(0.0)
    brier = float(np.average((y_pred - y_true)**2, weights=w)) if w.sum() > 0 else float(np.mean((y_pred - y_true)**2))
else:
    brier = float(np.mean((y_pred - y_true)**2))
print(f"Brier score: {brier:.4f} (lower is better)")

# AUC vs a pragmatic target: classify "high occupancy" as >= median
try:
    thresh = float(y_true.median())
    y_bin = (y_true >= thresh).astype(int)
    auc = float(roc_auc_score(y_bin, y_pred))
    print(f"AUC (classify ≥ median occupancy of {thresh:.2%}): {auc:.3f}")
except Exception as e:
    print("AUC could not be computed:", e)

# Calibration by deciles of predicted probability
cal = pd.DataFrame({"pred": y_pred, "obs": y_true})
cal["decile"] = pd.qcut(cal["pred"], 10, labels=False, duplicates="drop")
calib = cal.groupby("decile").agg(
    pred_mean=("pred","mean"),
    obs_mean=("obs","mean"),
    n=("obs","size")
).reset_index()

print("\nCalibration table (deciles of predicted occupancy):")
display(calib)

# Calibration plot
plt.figure(figsize=(7,5))
plt.plot(calib["pred_mean"], calib["obs_mean"], marker="o")
plt.plot([0,1],[0,1],'--')
plt.xlabel("Mean predicted occupancy")
plt.ylabel("Mean observed occupancy")
plt.title("Calibration — Predicted vs Observed (Deciles)")
plt.grid(True, linestyle=":", linewidth=0.5)
plt.show()

# Coefficient table with CIs
glm_tbl = pd.DataFrame({
    "coef": occ_model.params,
    "std_err": occ_model.bse,
    "z": occ_model.tvalues,
    "p>|z|": occ_model.pvalues
})
ci_glm_raw = occ_model.conf_int()
if isinstance(ci_glm_raw, pd.DataFrame):
    ci_glm = ci_glm_raw.copy()
else:  # ndarray
    ci_glm = pd.DataFrame(ci_glm_raw, index=glm_tbl.index)
ci_glm.columns = ["[0.025", "0.975]"]
glm_tbl = glm_tbl.join(ci_glm)

print("\nOccupancy GLM — selected coefficients:")
key_rows = glm_tbl.loc[glm_tbl.index.str.contains(
    "price_rel_log|const|bedrooms|review|dist|amenity|neigh_", case=False
)]
display(key_rows.head(30))

# Price sensitivity explanation & example
beta = float(occ_model.params.get("price_rel_log", np.nan))
print("\nPrice sensitivity (local):  dp/dx = β · p · (1 − p),  where x = ln(price / peer)")
if np.isfinite(beta):
    p0 = 0.50
    dx_10pct = math.log(1.10)   # ≈ 0.0953
    delta_p = beta * p0 * (1 - p0) * dx_10pct
    print(f"β (price_rel_log) = {beta:.3f}")
    print(f"At p = 0.50, a 10% price increase (dx = {dx_10pct:.3f}) "
          f"changes occupancy by ≈ {delta_p:.4f} ({delta_p*100:.2f} percentage points).")
else:
    print("β for price_rel_log not found in the model.")


# Alternative models (robustness)
# 1A. ADR — Elastic Net vs your OLS

# In[73]:


# --- ADR Elastic Net (compare to OLS RMSE €81.83) ---
from pathlib import Path
import pandas as pd, numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, mean_squared_error

BASE = Path("Documents/valencia_airbnb_project")
b = pd.read_parquet(BASE/"valencia_clean.parquet").copy()

# Features consistent with your OLS
cols = ["bedrooms","bathrooms","accommodates","review_score_mean","dist_km_centre"]
df = b.dropna(subset=["avg_price_w"]+cols).copy()
X = df[cols].astype(float)
y_log = np.log(df["avg_price_w"].astype(float))

# Elastic Net with CV on alpha & l1_ratio
pipe = Pipeline([("scaler", StandardScaler()),
                 ("model", ElasticNetCV(l1_ratio=[.2,.5,.8,1.0], alphas=None, cv=5, max_iter=20000))])

def rmse_euro(estimator, X, y_log):
    yhat_log = estimator.predict(X)
    # back-transform RMSE on € scale via exp: use approximation on residuals
    return np.sqrt(np.mean((np.exp(yhat_log) - np.exp(y_log))**2))

kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmses = []
for tr, te in kf.split(X):
    pipe.fit(X.iloc[tr], y_log.iloc[tr])
    rmses.append(rmse_euro(pipe, X.iloc[te], y_log.iloc[te]))

print(f"Elastic Net CV RMSE (€): mean {np.mean(rmses):.2f} | std {np.std(rmses):.2f}")
print("Chosen alpha:", pipe.named_steps["model"].alpha_, "l1_ratio:", pipe.named_steps["model"].l1_ratio_)


# 1B. Occupancy — Logit with an interaction (and GBM)

# In[70]:


from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path("Documents/valencia_airbnb_project")
b = pd.read_parquet(BASE/"valencia_clean.parquet").copy()

# 1) If 'neigh' is missing, create it from the best available column
if "neigh" not in b.columns:
    candidates = [c for c in [
        "neighbourhood_cleansed", "neighbourhood", "host_neighbourhood"
    ] if c in b.columns]

    if candidates:
        src = candidates[0]
        s = b[src].astype(str).str.strip()
        s = s.mask(s.eq("") | s.str.lower().eq("nan"), np.nan)
        b["neigh"] = s.fillna("Other").str.upper()
    else:
        # Optional spatial fallback if you want to infer from lat/lon + GeoJSON
        try:
            import geopandas as gpd
            gdf = gpd.GeoDataFrame(
                b,
                geometry=gpd.points_from_xy(b["lon"].astype(float),
                                            b["lat"].astype(float)),
                crs="EPSG:4326"
            )
            polys = gpd.read_file(BASE/"neighbourhoods.geojson")
            polys = polys.to_crs(gdf.crs)
            joined = gpd.sjoin(gdf, polys[["neighbourhood","geometry"]],
                               how="left", predicate="within")
            b["neigh"] = joined["neighbourhood"].fillna("OTHER").str.upper()
            b = pd.DataFrame(joined.drop(columns="geometry"))
        except Exception:
            b["neigh"] = "OTHER"

# (optional) tidy names a bit
b["neigh"] = (b["neigh"]
              .str.normalize("NFKD")
              .str.encode("ascii", "ignore").str.decode("ascii")
              .str.replace(r"\s+", " ", regex=True)
              .str.strip()
              .str.upper())

# 2) Make sure bedrooms is numeric and not missing
b["bedrooms"] = pd.to_numeric(b["bedrooms"], errors="coerce")
b = b[b["bedrooms"].notna()].copy()

# 3) Save back so all later code sees it
b.to_parquet(BASE/"valencia_clean.parquet", index=False)
print("Patched and saved 'neigh' ->", BASE/"valencia_clean.parquet")
print("Columns now include:", "neigh" in b.columns, "| unique neigh:", b["neigh"].nunique())


# Uncertainty on scenario predictions (GLM CIs)

# A) RevPAR-maximising ADR (+ extend the Playbook)

# In[66]:


# --- A) RevPAR-maximising ADR & extended playbook ---
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm, pickle, json, math

BASE = Path("Documents/valencia_airbnb_project")
OUT  = BASE/"outputs"; OUT.mkdir(parents=True, exist_ok=True)

b = pd.read_parquet(BASE/"valencia_clean.parquet")
price_model = pickle.load(open(BASE/"models/price_ols.pkl","rb"))
occ_model   = pickle.load(open(BASE/"models/occ_glm.pkl","rb"))
price_cols  = json.load(open(BASE/"models/price_cols.json"))
occ_cols    = json.load(open(BASE/"models/occ_cols.json"))

def logit(p): p = np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p))
def inv_logit(x): x = np.clip(x,-50,50);   return 1/(1+np.exp(-x))

def predict_adr(beds,baths,accom,review,dist):
    row = {c: np.nan for c in price_cols}
    for k,v in {"bedrooms":beds,"bathrooms":baths,"accommodates":accom,
                "review_score_mean":review,"dist_km_centre":dist}.items():
        if k in row: row[k]=v
    X = pd.DataFrame([row]).fillna({c:(b[c].median() if c in b else 0) for c in price_cols})
    X = sm.add_constant(X, has_constant="add")
    return float(np.exp(price_model.predict(X)[0]))

def eta_without_price(beds,review,dist,baths=None,accom=None, neigh=None):
    # build row with price_rel_log=0 to get baseline linear predictor
    row = {c: 0.0 for c in occ_cols}
    for k,v in {"price_rel_log":0.0,"bedrooms":beds,"review_score_mean":review,"dist_km_centre":dist}.items():
        if k in row: row[k]=v
    if baths is not None and "bathrooms" in row: row["bathrooms"]=baths
    if accom is not None and "accommodates" in row: row["accommodates"]=accom
    if neigh is not None:
        ncol = f"neigh_{neigh}"
        if ncol in row: row[ncol]=1.0
    X = pd.DataFrame([row]).reindex(columns=occ_cols, fill_value=0.0)
    X = sm.add_constant(X, has_constant="add")
    params = occ_model.params.reindex(X.columns, fill_value=0.0)
    return float(np.dot(X.values, params.values).ravel())

def adr_for_target_occ(peer_adr, eta0, beta, p_tgt):
    if not np.isfinite(beta) or beta==0: return np.nan
    prl = (logit(p_tgt) - eta0)/beta
    return float(np.clip(peer_adr*np.exp(prl), 30, 400))

def adr_maximising_revpar(peer_adr, eta0, beta):
    # Maximise R(x)=peer*e^x * logistic(eta0 + beta x) over x (closed form via Newton or grid)
    # Use a quick robust 1D search on x in [-2, 2] (~ price 0.14x to 7.4x peer)
    xs = np.linspace(-1.5, 1.5, 301)
    p  = inv_logit(eta0 + beta*xs)
    rev = peer_adr*np.exp(xs)*p
    x_star = xs[np.argmax(rev)]
    return float(np.clip(peer_adr*np.exp(x_star), 30, 400)), float(np.max(rev)), float(p[np.argmax(rev)])

rows=[]
for beds in [1,2,3]:
    sub = b[b["bedrooms"]==beds]
    if sub.empty: continue
    baths  = float(sub["bathrooms"].median()) if "bathrooms" in sub else np.nan
    accom  = float(sub["accommodates"].median()) if "accommodates" in sub else np.nan
    review = float(sub["review_score_mean"].median()) if "review_score_mean" in sub else 4.8
    dist   = float(sub["dist_km_centre"].median()) if "dist_km_centre" in sub else 2.0
    adr_model = predict_adr(beds,baths,accom,review,dist)
    peer_adr  = float(sub["avg_price_w"].median())
    beta      = float(occ_model.params.get("price_rel_log", np.nan))
    eta0      = eta_without_price(beds,review,dist,baths=baths,accom=accom)

    adr60 = adr_for_target_occ(peer_adr, eta0, beta, 0.60)
    adr70 = adr_for_target_occ(peer_adr, eta0, beta, 0.70)
    adr_star, rev_star, p_star = adr_maximising_revpar(peer_adr, eta0, beta)

    rows.append({
        "bedrooms":beds,
        "model_ADR_eur": round(adr_model,0),
        "peer_ADR_eur":  round(peer_adr,0),
        "recommended_ADR_for_60pct_occ": round(adr60,0) if np.isfinite(adr60) else np.nan,
        "recommended_ADR_for_70pct_occ": round(adr70,0) if np.isfinite(adr70) else np.nan,
        "RevPAR_maximising_ADR": round(adr_star,0),
        "RevPAR_at_ADR*": round(rev_star,0),
        "Occ_at_ADR*": round(p_star,3),
        "assumptions": f"bath={baths:.1f}, accom={accom:.0f}, review={review:.2f}, dist_km={dist:.2f}",
    })
playbook_ext = pd.DataFrame(rows)
out_path = OUT/"pricing_playbook_extended.csv"
playbook_ext.to_csv(out_path, index=False)
print("Saved ->", out_path)
display(playbook_ext)


# B) Seasonality indices from the calendar

# In[50]:


# --- B) Seasonality indices (monthly) ---
from pathlib import Path
import pandas as pd, numpy as np

BASE = Path("Documents/valencia_airbnb_project")
OUT  = BASE/"outputs"; OUT.mkdir(exist_ok=True, parents=True)

cal = pd.read_csv(BASE/"calendar.csv.gz", dtype=str, low_memory=False)
cal["date"] = pd.to_datetime(cal["date"], errors="coerce")
cal = cal[cal["date"].notna()]
cal["available_flag"] = cal["available"].astype(str).str.lower().eq("t")
cal["price_num"] = (cal["price"].astype(str)
                    .str.replace(r"[^\d.\-]","",regex=True)
                    .replace({"":np.nan}).astype(float))

# last 365 days
end = cal["date"].max()
start = end - pd.Timedelta(days=365)
cal = cal[(cal["date"]>=start) & (cal["date"]<=end)]

cal["month"] = cal["date"].dt.to_period("M").astype(str)

daily = (cal.groupby(["date"])
           .agg(occ=("available_flag", lambda s: (~s).mean()),
                adr=("price_num","median"))
           .reset_index())
daily["month"] = daily["date"].dt.to_period("M").astype(str)

monthly = (daily.groupby("month")
                 .agg(occ=("occ","mean"),
                      adr=("adr","median"),
                      days=("occ","size"))
                 .reset_index())

occ_idx = monthly["occ"]/monthly["occ"].mean()
adr_idx = monthly["adr"]/monthly["adr"].median()

season = monthly.assign(occ_index=occ_idx, adr_index=adr_idx)
season_path = OUT/"seasonality_monthly.csv"
season.to_csv(season_path, index=False)
print("Saved ->", season_path)
display(season)


# C) Robustness: 5-fold CV for both models

# In[65]:


# --- C) 5-fold cross-validation for ADR (RMSE €) & Occupancy (Brier, AUC) ---
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, roc_auc_score
import numpy as np, pandas as pd, statsmodels.api as sm, pickle, json
from pathlib import Path

BASE = Path("Documents/valencia_airbnb_project")
b = pd.read_parquet(BASE/"valencia_clean.parquet")
price_model = pickle.load(open(BASE/"models/price_ols.pkl","rb"))
occ_model   = pickle.load(open(BASE/"models/occ_glm.pkl","rb"))
price_cols  = json.load(open(BASE/"models/price_cols.json"))
occ_cols    = json.load(open(BASE/"models/occ_cols.json"))

def make_X(df, cols):
    X = pd.DataFrame(index=df.index)
    for c in cols:
        X[c] = df[c] if c in df.columns else 0.0
    for c in cols:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            X[c] = X[c].fillna(df[c].median())
    return sm.add_constant(X, has_constant="add")

X_price = make_X(b, price_cols)
y_eur   = b["avg_price_w"].astype(float)
y_log   = np.log(y_eur.clip(lower=1))

X_occ   = make_X(b, occ_cols)
y_occ   = b["occupancy_rate"].astype(float)
mask    = y_occ.notna()
X_occ_m = X_occ.loc[mask]; y_occ_m = y_occ.loc[mask]

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# ADR CV (use the saved model to keep spec; refit per fold for fairness)
rmse_eur = []
for tr, te in kf.split(X_price):
    # quick refit with same columns using statsmodels OLS on log-ADR
    ols = sm.OLS(y_log.iloc[tr], X_price.iloc[tr]).fit()
    pred = np.exp(ols.predict(X_price.iloc[te]))
    rmse_eur.append(np.sqrt(mean_squared_error(y_eur.iloc[te], pred)))
print(f"CV RMSE on ADR (€): mean {np.mean(rmse_eur):.2f} | std {np.std(rmse_eur):.2f}")

# Occupancy CV (Brier & AUC vs median)
briers, aucs = [], []
thresh = y_occ_m.median()
for tr, te in kf.split(X_occ_m):
    glm = sm.GLM(y_occ_m.iloc[tr], X_occ_m.iloc[tr], family=sm.families.Binomial()).fit()
    pred = glm.predict(X_occ_m.iloc[te]).clip(0,1)
    brier = float(np.mean((pred - y_occ_m.iloc[te])**2))
    briers.append(brier)
    try:
        auc = roc_auc_score((y_occ_m.iloc[te] >= thresh).astype(int), pred)
        aucs.append(auc)
    except Exception:
        pass
print(f"CV Brier (occ): mean {np.mean(briers):.4f} | std {np.std(briers):.4f}")
if aucs:
    print(f"CV AUC (≥ median occ): mean {np.mean(aucs):.3f} | std {np.std(aucs):.3f}")


# D) Sensitivity: redefine “peer price” by radius (1.5 km) vs neighbourhood
# 

# In[59]:


# --- Peer definition sensitivity (neighbourhood vs 1.5 km radius) ---
import numpy as np, pandas as pd, statsmodels.api as sm
from pathlib import Path

BASE = Path("Documents/valencia_airbnb_project")
b = pd.read_parquet(BASE/"valencia_clean.parquet")

# Ensure a 'neigh' column exists
if "neigh" not in b.columns:
    if "neighbourhood_cleansed" in b.columns:
        b["neigh"] = b["neighbourhood_cleansed"]
    elif "neighborhood_cleansed" in b.columns:
        b["neigh"] = b["neighborhood_cleansed"]
    else:
        b["neigh"] = "Unknown"

# Keep needed columns and clean
df = b[[
    "avg_price_w","bedrooms","review_score_mean","dist_km_centre",
    "latitude","longitude","neigh","occupancy_rate"
]].copy()

df = df.dropna(subset=["avg_price_w","bedrooms","latitude","longitude","occupancy_rate"])
df["bedrooms"] = pd.to_numeric(df["bedrooms"], errors="coerce")
df = df.dropna(subset=["bedrooms"])

# 1) Neighbourhood-based peer ADR
if df["neigh"].nunique() > 1:
    peer_nb = (df.groupby(["neigh","bedrooms"], dropna=False)["avg_price_w"]
                 .median().rename("peer_nb"))
    df = df.merge(peer_nb, on=["neigh","bedrooms"], how="left")
else:
    # fallback: per-bedroom city medians
    peer_nb_s = df.groupby("bedrooms")["avg_price_w"].median().rename("peer_nb")
    df = df.merge(peer_nb_s, on="bedrooms", how="left")

# 2) Radius (1.5 km) peer ADR — vectorised per bedroom group
def pairwise_haversine(lat, lon):
    R = 6371.0
    lat = np.radians(lat); lon = np.radians(lon)
    dlat = lat[:,None] - lat[None,:]
    dlon = lon[:,None] - lon[None,:]
    a = np.sin(dlat/2)**2 + np.cos(lat[:,None])*np.cos(lat[None,:])*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

df["peer_rad"] = np.nan
for beds, sub in df.groupby("bedrooms"):
    idx = sub.index
    D = pairwise_haversine(sub["latitude"].values, sub["longitude"].values)
    mask = D <= 1.5  # within 1.5 km
    prices = sub["avg_price_w"].values
    # median of prices within radius for each listing; fall back to group's median
    med_group = float(np.median(prices))
    med = np.array([
        np.median(prices[mask[i]]) if mask[i].any() else med_group
        for i in range(mask.shape[0])
    ])
    df.loc[idx, "peer_rad"] = med

# Relative price logs
for col in ["peer_nb","peer_rad"]:
    df[col] = df[col].replace(0, np.nan)
df["prl_nb"]  = np.log(df["avg_price_w"] / df["peer_nb"])
df["prl_rad"] = np.log(df["avg_price_w"] / df["peer_rad"])

# Fit compact GLMs with either peer definition
def fit_glm(prl_col):
    Z = df[["occupancy_rate","bedrooms","review_score_mean","dist_km_centre", prl_col]].dropna()
    X = sm.add_constant(Z[["bedrooms","review_score_mean","dist_km_centre", prl_col]], has_constant="add")
    y = Z["occupancy_rate"]
    m = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    return m

m_nb = m_rad = None
if df["prl_nb"].notna().any():
    m_nb = fit_glm("prl_nb")
    print("β on price_rel_log (neighbourhood peers):", round(m_nb.params["prl_nb"], 3))
if df["prl_rad"].notna().any():
    m_rad = fit_glm("prl_rad")
    print("β on price_rel_log (1.5 km radius peers):", round(m_rad.params["prl_rad"], 3))

if (m_nb is not None) and (m_rad is not None):
    print("\nInterpretation: similar magnitude/sign ⇒ robust price sensitivity across peer definitions.")


# Price sensitivity is robust: β ≈ −0.251 (neighbourhood peers) vs −0.264 (1.5 km peers). That’s very consistent → strong external validity.
# 
# ADR model CV: RMSE ≈ €81.8 ± €4.5. For a market median ADR ≈ €110, that’s a high absolute error but normal for noisy, cross-sectional STR pricing.
# 
# Occupancy model CV: Brier ≈ 0.080 ± 0.003 (good calibration); AUC ≈ 0.561 ± 0.018 (modest discrimination—fine for aggregated occupancy).
# 
# Seasonality table: Occupancy varies (nice), but ADR is flat (96 every month) → likely a composition effect (we took a citywide daily median) or missing price variance for some listings.
# 
# Also, your Playbook bounds hit the clamps (targets fall to €30; RevPAR-max pushes to €400). That tells examiners your optimiser needs realistic segment price bands.
# 
# Below are two drop-in cells to (1) constrain pricing to realistic ranges and (2) fix seasonality to avoid composition artefacts.

# ### 13. Constrain target & RevPAR-max ADR to realistic segment bands

# Use bedroom-level q10–q90 ADR as allowed price bands (citywide).
# 
# Optimise RevPAR only within those bands.
# 
# Round recommendations to €5.

# In[64]:


# --- Constrained Pricing Playbook (q10–q90 per bedroom) ---
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm, pickle, json, math

BASE = Path("Documents/valencia_airbnb_project")
OUT  = BASE/"outputs"; OUT.mkdir(parents=True, exist_ok=True)

b = pd.read_parquet(BASE/"valencia_clean.parquet")
price_model = pickle.load(open(BASE/"models/price_ols.pkl","rb"))
occ_model   = pickle.load(open(BASE/"models/occ_glm.pkl","rb"))
price_cols  = json.load(open(BASE/"models/price_cols.json"))
occ_cols    = json.load(open(BASE/"models/occ_cols.json"))

def logit(p): p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p))
def inv_logit(x): x=np.clip(x,-50,50);  return 1/(1+np.exp(-x))

def predict_adr(beds,baths,accom,review,dist):
    row = {c: np.nan for c in price_cols}
    for k,v in {"bedrooms":beds,"bathrooms":baths,"accommodates":accom,
                "review_score_mean":review,"dist_km_centre":dist}.items():
        if k in row: row[k]=v
    X = pd.DataFrame([row]).fillna({c:(b[c].median() if c in b else 0) for c in price_cols})
    X = sm.add_constant(X, has_constant="add")
    return float(np.exp(price_model.predict(X)[0]))

def eta_without_price(beds,review,dist,baths=None,accom=None, neigh=None):
    row = {c: 0.0 for c in occ_cols}
    for k,v in {"price_rel_log":0.0,"bedrooms":beds,"review_score_mean":review,"dist_km_centre":dist}.items():
        if k in row: row[k]=v
    if baths is not None and "bathrooms" in row: row["bathrooms"]=baths
    if accom is not None and "accommodates" in row: row["accommodates"]=accom
    if neigh is not None and f"neigh_{neigh}" in row: row[f"neigh_{neigh}"]=1.0
    X = pd.DataFrame([row]).reindex(columns=occ_cols, fill_value=0.0)
    X = sm.add_constant(X, has_constant="add")
    params = occ_model.params.reindex(X.columns, fill_value=0.0)
    return float(np.dot(X.values, params.values).ravel())

def clamp(v, lo, hi): return float(np.minimum(np.maximum(v, lo), hi))

def adr_for_target_occ(peer_adr, eta0, beta, p_tgt, lo, hi):
    if not np.isfinite(beta) or beta==0: return np.nan
    prl = (logit(p_tgt) - eta0) / beta
    adr = float(peer_adr * np.exp(prl))
    return clamp(adr, lo, hi)

def adr_maximising_revpar(peer_adr, eta0, beta, lo, hi):
    # search in x = ln(ADR/peer_adr) limited by bounds
    x_lo, x_hi = np.log(lo/peer_adr), np.log(hi/peer_adr)
    xs = np.linspace(x_lo, x_hi, 401)
    p  = inv_logit(eta0 + beta*xs)
    rev = peer_adr*np.exp(xs)*p
    i   = int(np.argmax(rev))
    return float(peer_adr*np.exp(xs[i])), float(rev[i]), float(p[i])

rows=[]
for beds in [1,2,3]:
    sub = b[b["bedrooms"]==beds].copy()
    if sub.empty: continue

    # segment bands: q10–q90 ADR within bedroom
    lo, hi = float(sub["avg_price_w"].quantile(0.10)), float(sub["avg_price_w"].quantile(0.90))
    # sensible hard caps too
    lo, hi = max(30.0, lo), min(400.0, hi)

    baths  = float(sub["bathrooms"].median()) if "bathrooms" in sub else np.nan
    accom  = float(sub["accommodates"].median()) if "accommodates" in sub else np.nan
    review = float(sub["review_score_mean"].median()) if "review_score_mean" in sub else 4.8
    dist   = float(sub["dist_km_centre"].median()) if "dist_km_centre" in sub else 2.0
    adr_model = predict_adr(beds,baths,accom,review,dist)

    peer_adr  = float(sub["avg_price_w"].median())
    beta      = float(occ_model.params.get("price_rel_log", np.nan))
    eta0      = eta_without_price(beds,review,dist,baths=baths,accom=accom)

    adr60 = adr_for_target_occ(peer_adr, eta0, beta, 0.60, lo, hi)
    adr70 = adr_for_target_occ(peer_adr, eta0, beta, 0.70, lo, hi)
    adr_star, rev_star, p_star = adr_maximising_revpar(peer_adr, eta0, beta, lo, hi)

    rnd = lambda x: float(np.round(x/5)*5) if np.isfinite(x) else np.nan

    rows.append({
        "bedrooms": beds,
        "model_ADR_eur": rnd(adr_model),
        "peer_ADR_eur":  rnd(peer_adr),
        "allowed_band_eur": f"€{rnd(lo):.0f}–€{rnd(hi):.0f}",
        "ADR_for_60pct_occ": rnd(adr60),
        "ADR_for_70pct_occ": rnd(adr70),
        "RevPAR_max_ADR":    rnd(adr_star),
        "RevPAR_at_max":     rnd(rev_star),
        "Occ_at_max":        round(p_star,3),
        "assumptions": f"bath={baths:.1f}, accom={accom:.0f}, review={review:.2f}, dist_km={dist:.2f}",
    })

playbook_constrained = pd.DataFrame(rows)
out_path = OUT/"pricing_playbook_constrained.csv"
playbook_constrained.to_csv(out_path, index=False)
print("Saved ->", out_path)
display(playbook_constrained)


# ### 14. Seasonality without composition bias

# Compute ADR per listing-day first, then aggregate to month (optionally by bedroom). This avoids “flat ADR” from market-mix shifts.

# In[61]:


# --- Seasonality (listing-first, then aggregate) ---
from pathlib import Path
import pandas as pd, numpy as np

BASE = Path("Documents/valencia_airbnb_project")
OUT  = BASE/"outputs"; OUT.mkdir(parents=True, exist_ok=True)

b   = pd.read_parquet(BASE/"valencia_clean.parquet")             # ensures entire homes filter
cal = pd.read_csv(BASE/"calendar.csv.gz", dtype=str, low_memory=False)

# Keep only listings present in b
keep_ids = set(b["id"].astype(str))
cal = cal[cal["listing_id"].astype(str).isin(keep_ids)].copy()

# Parse fields
cal["date"] = pd.to_datetime(cal["date"], errors="coerce")
cal = cal[cal["date"].notna()]
cal["available_flag"] = cal["available"].astype(str).str.lower().eq("t")
cal["price_num"] = (cal["price"].astype(str)
                    .str.replace(r"[^\d.\-]","",regex=True)
                    .replace({"":np.nan}).astype(float))

# last 365 days
end = cal["date"].max()
start = end - pd.Timedelta(days=365)
cal = cal[(cal["date"]>=start) & (cal["date"]<=end)].copy()

# Join bedrooms for segmentation
b_ids = b[["id","bedrooms"]].copy()
b_ids["id"] = b_ids["id"].astype(str)
cal["listing_id_str"] = cal["listing_id"].astype(str)
cal = cal.merge(b_ids, left_on="listing_id_str", right_on="id", how="left")

# Listing-day: occupancy (0/1) and price
ld = cal.groupby(["listing_id_str","date","bedrooms"], dropna=False).agg(
    occ=("available_flag", lambda s: (~s).any().astype(int)),   # booked if any 'f' that date
    adr=("price_num","median")
).reset_index()

# Month aggregation (overall + by bedroom)
ld["month"] = ld["date"].dt.to_period("M").astype(str)

def monthly_table(df):
    m = (df.groupby("month")
           .agg(occ=("occ","mean"),
                adr=("adr", lambda s: np.nanmedian(s)),
                listings=("listing_id_str","nunique"),
                days=("occ","size"))
           .reset_index())
    m["occ_index"] = m["occ"]/m["occ"].mean()
    m["adr_index"] = m["adr"]/np.nanmedian(m["adr"])
    return m

overall = monthly_table(ld)
overall.to_csv(OUT/"seasonality_monthly_overall.csv", index=False)
print("Saved ->", OUT/"seasonality_monthly_overall.csv")
display(overall)

by_bed = []
for beds, sub in ld.groupby("bedrooms"):
    T = monthly_table(sub)
    T.insert(1, "bedrooms", beds)
    by_bed.append(T)
by_bed = pd.concat(by_bed, ignore_index=True)
by_bed.to_csv(OUT/"seasonality_monthly_by_bedrooms.csv", index=False)
print("Saved ->", OUT/"seasonality_monthly_by_bedrooms.csv")
display(by_bed.head(12))


# A) Neighbourhood positioning scatter (2BR), with quadrant lines + labels → PNG

# In[55]:


# --- Neighbourhood positioning scatter (robust to missing `neigh`) ---
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt

BASE = Path("Documents/valencia_airbnb_project")
OUT  = BASE/"outputs"; OUT.mkdir(exist_ok=True, parents=True)

b = pd.read_parquet(BASE/"valencia_clean.parquet").copy()

# 1) Ensure we have a 'neigh' column
neigh_src = None
for c in ["neigh", "neighbourhood_cleansed", "neighborhood_cleansed", "neighbourhood", "neighborhood"]:
    if c in b.columns:
        neigh_src = c
        break

if neigh_src is None:
    # fallback: everything as 'Unknown' (plot will be less informative)
    b["neigh"] = "Unknown"
else:
    b["neigh"] = b[neigh_src].astype(str).str.strip()

# 2) Filter to a bedroom segment (2BR by default) and clean
br = 2
nb = b[b["bedrooms"].astype("float").round().astype("Int64") == br].copy()
nb = nb.dropna(subset=["avg_price_w","occupancy_rate"])

if nb.empty:
    raise ValueError("No rows found for the selected bedroom segment. Try setting `br = 1` or `br = 3`.")

# 3) Aggregate by neighbourhood
grp = (nb.groupby("neigh", dropna=False)
         .agg(n=("avg_price_w","size"),
              ADR_med=("avg_price_w","median"),
              OCC_med=("occupancy_rate","median"))
         .reset_index())

# Only keep neighbourhoods with enough listings (for stability)
min_n = 20
grp = grp[grp["n"] >= min_n].dropna(subset=["ADR_med","OCC_med"])
if grp.empty:
    raise ValueError(f"No neighbourhoods with at least {min_n} listings in the {br}BR segment. Reduce `min_n` or choose another `br`.")

# City medians for quadrant lines (within this bedroom segment)
adr_city = float(nb["avg_price_w"].median())
occ_city = float(nb["occupancy_rate"].median())

# 4) Plot
fig, ax = plt.subplots(figsize=(8.5, 6.0))

sizes = np.clip(grp["n"]*0.6, 10, 140)  # bubble size ~ number of listings
ax.scatter(grp["ADR_med"], grp["OCC_med"], s=sizes, alpha=0.75)

# quadrant lines
ax.axvline(adr_city, ls="--", lw=1, color="gray")
ax.axhline(occ_city, ls="--", lw=1, color="gray")
ax.text(adr_city+2, occ_city+0.02, "Prime", color="gray")
ax.text(adr_city+2, occ_city-0.08, "Premium / Lower Occ", color="gray")
ax.text(adr_city-60, occ_city+0.02, "Value High-Occ", color="gray")
ax.text(adr_city-60, occ_city-0.08, "Underperforming", color="gray")

# label a few notable points (largest n + ADR/OCC extremes)
to_label = (
    pd.concat([
        grp.sort_values("n", ascending=False).head(5),
        grp.sort_values("ADR_med", ascending=False).head(3),
        grp.sort_values("OCC_med", ascending=False).head(3)
    ])
    .drop_duplicates(subset=["neigh"])
)

for _, r in to_label.iterrows():
    ax.annotate(str(r["neigh"]),
                (r["ADR_med"], r["OCC_med"]),
                xytext=(6,6), textcoords="offset points", fontsize=9)

ax.set_xlabel(f"Median ADR (€) — {br}BR")
ax.set_ylabel(f"Median Occupancy — {br}BR")
ax.set_title(f"Neighbourhood Positioning ({br}BR): ADR vs Occupancy\n(Inside Airbnb snapshot: 15-Mar-2025; bubbles ∝ listings, min_n={min_n})")
ax.grid(alpha=0.25, ls=":")

out = OUT/f"fig_neighbourhood_positioning_{br}br.png"
plt.tight_layout()
plt.savefig(out, dpi=300)
plt.close()
print("Saved ->", out)


# B) Seasonality lines (overall + by bedroom) → PNG
# 

# In[62]:


import pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

BASE = Path("Documents/valencia_airbnb_project"); OUT = BASE/"outputs"

overall = pd.read_csv(OUT/"seasonality_monthly_overall.csv")
overall["month"] = pd.to_datetime(overall["month"])
fig, ax = plt.subplots(figsize=(9,4))
ax.plot(overall["month"], overall["occ_index"], marker="o", label="Occupancy index")
ax.plot(overall["month"], overall["adr_index"], marker="o", label="ADR index")
ax.axhline(1.0, ls="--", color="gray", lw=1)
ax.set_title("Seasonality (Last 365 Days) — Index vs Annual Mean/Median")
ax.set_ylabel("Index"); ax.legend(); ax.grid(alpha=0.2, ls=":")
plt.tight_layout(); plt.savefig(OUT/"fig_seasonality_overall.png", dpi=200)
print("Saved ->", OUT/"fig_seasonality_overall.png")

by_bed = pd.read_csv(OUT/"seasonality_monthly_by_bedrooms.csv")
by_bed["month"] = pd.to_datetime(by_bed["month"])
for metric in ["occ_index","adr_index"]:
    fig, ax = plt.subplots(figsize=(9,4))
    for beds, sub in by_bed.groupby("bedrooms"):
        ax.plot(sub["month"], sub[metric], marker="o", label=f"{int(beds)}BR")
    ax.axhline(1.0, ls="--", color="gray", lw=1)
    ax.set_title(f"Seasonality by Bedrooms — {metric.replace('_',' ').title()}")
    ax.set_ylabel("Index"); ax.legend(); ax.grid(alpha=0.2, ls=":")
    fname = OUT/f"fig_seasonality_{metric}_by_bed.png"
    plt.tight_layout(); plt.savefig(fname, dpi=200)
    print("Saved ->", fname)


# C) Save key model diagnostics → PNG

# In[63]:


# re-create + save residuals/QQ and GLM calibration plots
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm, pickle, json, matplotlib.pyplot as plt

BASE = Path("Documents/valencia_airbnb_project"); OUT = BASE/"outputs"; OUT.mkdir(exist_ok=True, parents=True)
b = pd.read_parquet(BASE/"valencia_clean.parquet")
price_model = pickle.load(open(BASE/"models/price_ols.pkl","rb"))
occ_model   = pickle.load(open(BASE/"models/occ_glm.pkl","rb"))
price_cols  = json.load(open(BASE/"models/price_cols.json"))
occ_cols    = json.load(open(BASE/"models/occ_cols.json"))

def make_X(df, cols):
    X = pd.DataFrame(index=df.index)
    for c in cols: X[c] = df[c] if c in df.columns else 0.0
    for c in cols:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            X[c] = X[c].fillna(df[c].median())
    return sm.add_constant(X, has_constant="add")

# OLS plots
X_price = make_X(b, price_cols)
y_log = np.log(b["avg_price_w"].clip(lower=1))
yhat_log = price_model.predict(X_price)
resid = y_log - yhat_log
fig, axs = plt.subplots(1,2, figsize=(12,5))
axs[0].scatter(yhat_log, resid, s=10); axs[0].axhline(0, ls="--")
axs[0].set_xlabel("Fitted (log ADR)"); axs[0].set_ylabel("Residuals"); axs[0].set_title("Residuals vs Fitted — ADR model")
sm.ProbPlot(resid).qqplot(line='45', ax=axs[1]); axs[1].set_title("Q–Q Plot — ADR model")
plt.tight_layout(); plt.savefig(OUT/"fig_price_model_resid_qq.png", dpi=200); plt.close()
print("Saved ->", OUT/"fig_price_model_resid_qq.png")

# GLM calibration
X_occ = make_X(b, occ_cols)
mask = b["occupancy_rate"].notna()
pred = occ_model.predict(X_occ.loc[mask]).clip(0,1)
obs  = b.loc[mask, "occupancy_rate"].astype(float)
cal = pd.DataFrame({"pred":pred, "obs":obs})
cal["decile"] = pd.qcut(cal["pred"], 10, labels=False, duplicates="drop")
calib = cal.groupby("decile").agg(pred_mean=("pred","mean"), obs_mean=("obs","mean")).reset_index()
fig, ax = plt.subplots(figsize=(6,5))
ax.plot(calib["pred_mean"], calib["obs_mean"], marker="o"); ax.plot([0,1],[0,1],'--', color="gray")
ax.set_xlabel("Mean predicted occupancy"); ax.set_ylabel("Mean observed occupancy"); ax.set_title("Calibration — Occupancy GLM (deciles)")
ax.grid(alpha=0.2, ls=":")
plt.tight_layout(); plt.savefig(OUT/"fig_occ_glm_calibration.png", dpi=200); plt.close()
print("Saved ->", OUT/"fig_occ_glm_calibration.png")


# In[ ]:




