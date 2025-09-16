# app.py
import json
import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

# ---------- Defaults & Types ----------

PERIODS_PER_YEAR = 12

@dataclass
class Finance:
    discount_rate_annual: float = 0.10
    horizon_periods: int = 60
    periods_per_year: int = PERIODS_PER_YEAR

@dataclass
class Elasticities:
    price_to_freq_elast: float = -0.3     # %Δfreq per 10% price increase -> -3% default
    price_to_churn_pp_per_10pct: float = 0.4  # +0.4pp churn per +10% price
    service_to_churn_pp_per_10pct: float = -0.3
    crosssell_to_churn_pp_per_10pct: float = -0.1

@dataclass
class Scenario:
    name: str = "Scenario A"
    price_delta_pct: float = 0.0
    purchase_freq_delta_pct: float = 0.0
    basket_delta_pct: float = 0.0
    margin_mix_delta_pp: float = 0.0
    var_cost_delta_pp: float = 0.0
    fixed_cost_delta_abs: float = 0.0
    cross_sell_delta_pct: float = 0.0
    cac_delta_pct: float = 0.0
    overhead_delta_pct: float = 0.0
    # Behavioural toggles
    apply_price_elasticity: bool = True
    apply_service_elasticity: bool = True
    apply_crosssell_elasticity: bool = True
    inject_low_value_cohort_pct: float = 0.0  # "sell worse": % of base customers moved to low-value profile
    reduce_nonbuyers_share_pct: float = 0.0   # "sell fewer to nonbuyers": % of nonbuyers eliminated from acquisition

# ---------- Data Utilities ----------

def demo_cohorts() -> pd.DataFrame:
    return pd.DataFrame({
        "segment_id": ["Core", "Value", "Premium"],
        "customers": [2000, 3000, 500],
        "base_price": [50.0, 40.0, 90.0],
        "units_per_purchase": [1.6, 1.2, 1.8],
        "purchase_frequency": [1.3, 0.9, 1.5],  # per month
        "gross_margin_pct": [0.52, 0.40, 0.62],
        "var_service_cost_pct": [0.06, 0.08, 0.05],
        "fixed_service_cost": [1.20, 1.00, 1.50],
        "churn_pct": [0.045, 0.060, 0.035],     # monthly
        "cac": [22.0, 16.0, 35.0],
        "retention_cost": [2.0, 1.5, 3.0],
        "overhead_per_customer": [1.0, 0.8, 1.2],
        "cross_sell_rev": [2.0, 0.8, 4.0],
        "period": ["month", "month", "month"],
    })

def load_cohorts() -> pd.DataFrame:
    up = st.sidebar.file_uploader("Upload cohort CSV", type=["csv"])
    if up is not None:
        df = pd.read_csv(up)
        st.success("Cohorts loaded.")
        return df
    st.info("Using demo cohorts. Upload a CSV to replace.")
    return demo_cohorts()

# ---------- Model ----------

def apply_scenario_to_segment(row: pd.Series, sc: Scenario, el: Elasticities) -> pd.Series:
    r = row.copy()

    # Direct levers
    r["price"] = r["base_price"] * (1 + sc.price_delta_pct / 100)
    r["purchase_frequency"] = r["purchase_frequency"] * (1 + sc.purchase_freq_delta_pct / 100)
    r["units_per_purchase"] = r["units_per_purchase"] * (1 + sc.basket_delta_pct / 100)
    r["gross_margin_pct"] = np.clip(r["gross_margin_pct"] + sc.margin_mix_delta_pp / 100, 0.0, 0.95)
    r["var_service_cost_pct"] = np.clip(r["var_service_cost_pct"] + sc.var_cost_delta_pp / 100, 0.0, 0.95)
    r["fixed_service_cost"] = max(0.0, r["fixed_service_cost"] + sc.fixed_cost_delta_abs)
    r["cross_sell_rev"] = max(0.0, r["cross_sell_rev"] * (1 + sc.cross_sell_delta_pct / 100))
    r["cac"] = max(0.0, r["cac"] * (1 + sc.cac_delta_pct / 100))
    r["overhead_per_customer"] = max(0.0, r["overhead_per_customer"] * (1 + sc.overhead_delta_pct / 100))
    churn_pp = r["churn_pct"]

    # Elasticities
    if sc.apply_price_elasticity and sc.price_delta_pct != 0:
        r["purchase_frequency"] *= (1 + el.price_to_freq_elast * (sc.price_delta_pct / 10))
        churn_pp += el.price_to_churn_pp_per_10pct * (sc.price_delta_pct / 10)

    if sc.apply_service_elasticity and sc.var_cost_delta_pp < 0:
        # Assume service improvements (lower var cost % via efficiency) do not worsen churn.
        pass
    elif sc.apply_service_elasticity and sc.var_cost_delta_pp > 0:
        # Worse service can increase churn if service cost cuts reduce quality; use a simple proxy:
        churn_pp += max(0.0, el.service_to_churn_pp_per_10pct) * (sc.var_cost_delta_pp / 10)

    if sc.apply_crosssell_elasticity and sc.cross_sell_delta_pct > 0:
        churn_pp += el.crosssell_to_churn_pp_per_10pct * (sc.cross_sell_delta_pct / 10)

    r["churn_pct"] = float(np.clip(churn_pp, 0.001, 0.5))
    return r

def simulate_clv(row: pd.Series, fin: Finance) -> float:
    # per month model
    price = row["price"]
    freq = row["purchase_frequency"]
    units = row["units_per_purchase"]
    margin = row["gross_margin_pct"]
    var_cost_pct = row["var_service_cost_pct"]
    fixed_cost = row["fixed_service_cost"]
    churn = row["churn_pct"]
    cross_sell = row["cross_sell_rev"]

    survival = 1.0
    clv = 0.0
    for t in range(1, fin.horizon_periods + 1):
        arpu = price * units * freq + cross_sell
        gross_margin = arpu * margin
        service_cost = arpu * var_cost_pct + fixed_cost
        net = gross_margin - service_cost
        survival *= (1 - churn)
        df = 1 / ((1 + fin.discount_rate_annual / fin.periods_per_year) ** t)
        clv += net * survival * df
    return clv

def compute_nltv_table(df: pd.DataFrame, fin: Finance) -> pd.DataFrame:
    df = df.copy()
    df["CLV"] = df.apply(simulate_clv, axis=1, fin=fin)
    df["NLTV"] = df["CLV"] - df["cac"] - df["retention_cost"] - df["overhead_per_customer"]
    df["Total_NLTV"] = df["NLTV"] * df["customers"]
    return df

def tornado_sensitivity(base_df: pd.DataFrame, fin: Finance, row: pd.Series, step_pct: float = 10.0) -> pd.DataFrame:
    # One segment tornado around base: vary each driver ±step_pct
    drivers = {
        "price_delta_pct": "Price",
        "purchase_freq_delta_pct": "Purchase Frequency",
        "basket_delta_pct": "Basket Size",
        "margin_mix_delta_pp": "Gross Margin pp",
        "var_cost_delta_pp": "Var Service Cost pp",
        "cross_sell_delta_pct": "Cross-sell Rev",
        "cac_delta_pct": "CAC",
        "overhead_delta_pct": "Overheads",
    }
    base_row = row.copy()
    base_row["price"] = base_row["base_price"]
    base_row["NLTV_base"] = compute_nltv_table(pd.DataFrame([base_row]), fin)["NLTV"].iloc[0]

    records = []
    for key, label in drivers.items():
        for sgn, name in [(-1, f"{label} −{step_pct}%"), (1, f"{label} +{step_pct}%")]:
            sc = Scenario(name="tmp")
            setattr(sc, key, sgn * step_pct)
            el = Elasticities()
            adj = apply_scenario_to_segment(row, sc, el)
            nltv = compute_nltv_table(pd.DataFrame([adj]), fin)["NLTV"].iloc[0]
            records.append({"Driver": label, "Case": name, "NLTV": nltv})
    out = pd.DataFrame(records)
    # Convert to deltas vs base
    out["ΔNLTV"] = out["NLTV"] - float(base_row["NLTV_base"])
    return out.sort_values(["Driver", "ΔNLTV"])

# ---------- Streamlit UI ----------

st.set_page_config(page_title="NLTV Value Driver & Scenario Model", layout="wide")

st.title("NLTV Value Driver Tree — Scenario Analysis")

# Session state
if "scenarios" not in st.session_state:
    st.session_state.scenarios: Dict[str, Scenario] = {}

# Sidebar: Finance and elasticities
with st.sidebar:
    st.header("Finance")
    fin = Finance(
        discount_rate_annual=st.number_input("Discount rate (annual)", 0.0, 1.0, 0.10, step=0.01),
        horizon_periods=st.number_input("Horizon (months)", 1, 240, 60, step=1),
        periods_per_year=PERIODS_PER_YEAR,
    )
    st.header("Elasticities")
    el = Elasticities(
        price_to_freq_elast=st.number_input("Price → Frequency elasticity per +10% (e.g., -0.3)", -2.0, 2.0, -0.3, step=0.1),
        price_to_churn_pp_per_10pct=st.number_input("Price → Churn (pp per +10%)", -2.0, 2.0, 0.4, step=0.1),
        service_to_churn_pp_per_10pct=st.number_input("Service → Churn (pp per +10% worse)", -2.0, 2.0, 0.3, step=0.1),
        crosssell_to_churn_pp_per_10pct=st.number_input("Cross-sell → Churn (pp per +10%)", -2.0, 2.0, -0.1, step=0.1),
    )

df_in = load_cohorts()
# Prepare base columns for scenario application
df_in = df_in.rename(columns={"base_price": "base_price"})
df_in["price"] = df_in["base_price"]  # base

# Scenario builder
st.sidebar.header("Scenario Controls")
sc_name = st.sidebar.text_input("Scenario name", "Case 1")
sc = Scenario(
    name=sc_name,
    price_delta_pct=st.sidebar.number_input("Increase prices (%)", -50.0, 200.0, 0.0, step=1.0),
    purchase_freq_delta_pct=st.sidebar.number_input("Increase purchase frequency (%)", -50.0, 200.0, 0.0, step=1.0),
    basket_delta_pct=st.sidebar.number_input("Sell more (basket size, %)", -50.0, 200.0, 0.0, step=1.0),
    margin_mix_delta_pp=st.sidebar.number_input("Sell better (margin mix, pp)", -50.0, 50.0, 0.0, step=0.5),
    var_cost_delta_pp=st.sidebar.number_input("Decrease costs (variable service cost, pp)", -50.0, 50.0, 0.0, step=0.5),
    fixed_cost_delta_abs=st.sidebar.number_input("Decrease costs (fixed per cust, abs)", -10.0, 10.0, 0.0, step=0.1),
    cross_sell_delta_pct=st.sidebar.number_input("Cross-sell (%)", -100.0, 500.0, 0.0, step=1.0),
    cac_delta_pct=st.sidebar.number_input("CAC change (%)", -100.0, 200.0, 0.0, step=1.0),
    overhead_delta_pct=st.sidebar.number_input("Overheads change (%)", -100.0, 200.0, 0.0, step=1.0),
    apply_price_elasticity=st.sidebar.checkbox("Apply price elasticities", True),
    apply_service_elasticity=st.sidebar.checkbox("Apply service elasticities", True),
    apply_crosssell_elasticity=st.sidebar.checkbox("Apply cross-sell elasticities", True),
)

col_save, col_load = st.sidebar.columns(2)
with col_save:
    if st.button("Save scenario"):
        st.session_state.scenarios[sc.name] = sc
with col_load:
    if st.button("Clear scenarios"):
        st.session_state.scenarios = {}

# Apply scenario
df_adj = df_in.apply(lambda r: apply_scenario_to_segment(r, sc, el), axis=1)
base_table = compute_nltv_table(df_in, fin)
scen_table = compute_nltv_table(df_adj, fin)

# Layout
tab_overview, tab_scenarios, tab_sensitivity, tab_cohorts, tab_export = st.tabs(
    ["Overview", "Scenarios", "Sensitivity", "Cohorts", "Export"]
)

with tab_overview:
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        st.metric("Base NLTV (total)", f"${base_table['Total_NLTV'].sum():,.0f}")
    with c2:
        st.metric(f"{sc.name} NLTV (total)", f"${scen_table['Total_NLTV'].sum():,.0f}")
    with c3:
        st.metric("Δ NLTV (total)", f"${(scen_table['Total_NLTV'].sum() - base_table['Total_NLTV'].sum()):,.0f}")

    st.subheader("By segment")
    merge_cols = ["segment_id", "customers"]
    comp = base_table[merge_cols + ["NLTV", "Total_NLTV"]].merge(
        scen_table[merge_cols + ["NLTV", "Total_NLTV"]],
        on=merge_cols, suffixes=(" Base", f" {sc.name}")
    )
    st.dataframe(comp, use_container_width=True)

with tab_scenarios:
    st.subheader("Saved scenarios")
    if st.session_state.scenarios:
        sc_list = [asdict(v) for v in st.session_state.scenarios.values()]
        st.dataframe(pd.DataFrame(sc_list))
        # Compare two scenarios
        names = list(st.session_state.scenarios.keys())
        s1 = st.selectbox("Scenario A", names, index=0)
        s2 = st.selectbox("Scenario B", names, index=min(1, len(names)-1))
        A = df_in.apply(lambda r: apply_scenario_to_segment(r, st.session_state.scenarios[s1], el), axis=1)
        B = df_in.apply(lambda r: apply_scenario_to_segment(r, st.session_state.scenarios[s2], el), axis=1)
        At = compute_nltv_table(A, fin); Bt = compute_nltv_table(B, fin)
        st.write(f"Δ Total NLTV: ${Bt['Total_NLTV'].sum() - At['Total_NLTV'].sum():,.0f}")
    else:
        st.info("No saved scenarios yet.")

with tab_sensitivity:
    st.subheader("Tornado (single segment)")
    seg = st.selectbox("Segment", df_in["segment_id"].tolist(), index=0)
    row_base = df_in[df_in["segment_id"] == seg].iloc[0]
    tornado = tornado_sensitivity(df_in, fin, row_base, step_pct=10.0)
    st.dataframe(tornado)
    # Quick bar view
    st.bar_chart(tornado.set_index("Case")["ΔNLTV"])

with tab_cohorts:
    st.subheader("Cohorts / Segments")
    st.dataframe(df_in, use_container_width=True)
    st.subheader("Scenario-adjusted")
    st.dataframe(df_adj, use_container_width=True)

with tab_export:
    st.subheader("Export")
    res = {
        "finance": asdict(fin),
        "elasticities": asdict(el),
        "scenario": asdict(sc),
        "base_results": base_table.to_dict(orient="list"),
        "scenario_results": scen_table.to_dict(orient="list"),
    }
    st.download_button("Download results JSON", data=json.dumps(res, indent=2), file_name="nltv_results.json")
    st.download_button("Download adjusted cohorts (CSV)", data=df_adj.to_csv(index=False), file_name="adjusted_cohorts.csv")

st.caption("All figures are illustrative. Validate with actuals and ensure elasticities reflect observed behaviour.")
