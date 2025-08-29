import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Project Ganesha – Unlocking Free Cash Flow", layout="wide")

# -----------------------------
# Utilities
# -----------------------------
def mock_monthly_calendar(start="2023-01-01", periods=24):
    dt = pd.date_range(start=start, periods=periods, freq="MS")
    return pd.DataFrame({"month": dt})

def generate_mock_drivers(start="2023-01-01", periods=24, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cal = mock_monthly_calendar(start, periods)
    customers = 10000 * (1 + 0.01*np.arange(periods)) + rng.normal(0, 50, periods)
    arpu = 50 * (1 + 0.002*np.arange(periods)) + rng.normal(0, 0.2, periods)
    ebitda_margin = 0.25 + 0.01*np.sin(np.linspace(0, 3*np.pi, periods))
    capex_pct_revenue = 0.06 + 0.005*np.cos(np.linspace(0, 2*np.pi, periods))
    wc_intensity = 0.12 + 0.01*np.sin(np.linspace(0, 2*np.pi, periods) + 0.5)
    tax_rate = np.full(periods, 0.25)

    df = cal.copy()
    df["customers"] = customers
    df["arpu"] = arpu
    df["ebitda_margin"] = ebitda_margin
    df["capex_pct_revenue"] = capex_pct_revenue
    df["wc_intensity"] = wc_intensity
    df["tax_rate"] = tax_rate
    return df

def compute_financials(drivers: pd.DataFrame) -> pd.DataFrame:
    df = drivers.copy().sort_values("month").reset_index(drop=True)
    df["revenue"] = df["customers"] * df["arpu"]
    df["ebitda"] = df["revenue"] * df["ebitda_margin"]
    df["taxes"] = np.where(df["ebitda"]>0, df["ebitda"] * df["tax_rate"], 0.0)
    df["capex"] = df["capex_pct_revenue"] * df["revenue"]
    df["nwc"] = df["wc_intensity"] * df["revenue"]
    df["delta_nwc"] = df["nwc"].diff().fillna(df["nwc"])
    df["fcf"] = df["ebitda"] - df["taxes"] - df["capex"] - df["delta_nwc"]
    df["year"] = df["month"].dt.year
    return df

def apply_overrides(drivers: pd.DataFrame, overrides: dict) -> pd.DataFrame:
    df = drivers.copy()
    for col, transform in overrides.items():
        if callable(transform):
            df[col] = transform(df[col])
        else:
            if isinstance(transform, (int, float)):
                df[col] = df[col] * transform
            else:
                df[col] = transform
    return df

def run_scenario(base_drivers: pd.DataFrame, name: str, overrides: dict) -> pd.DataFrame:
    scen_drivers = apply_overrides(base_drivers, overrides)
    scen_fin = compute_financials(scen_drivers)
    scen_fin["scenario"] = name
    return scen_fin

def aggregate_fcf(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(["scenario","year"], as_index=False)["fcf"].sum()
              .rename(columns={"fcf":"fcf_yearly"}))

def total_fcf(df: pd.DataFrame) -> float:
    return float(df["fcf"].sum())

# -----------------------------
# Load or generate data
# -----------------------------
st.sidebar.header("Data & Scenario Controls")
use_mock = st.sidebar.checkbox("Use built-in mock data", value=True)

if use_mock:
    base_drivers = generate_mock_drivers()
else:
    st.sidebar.info("Upload a drivers CSV with columns: month, customers, arpu, ebitda_margin, capex_pct_revenue, wc_intensity, tax_rate")
    uploaded = st.sidebar.file_uploader("Upload drivers CSV", type=["csv"])
    if uploaded is not None:
        base_drivers = pd.read_csv(uploaded, parse_dates=["month"])
    else:
        st.stop()

# -----------------------------
# Scenario Parameters
# -----------------------------
st.sidebar.subheader("Scenario Adjustments")
cust_mult = st.sidebar.slider("Customers multiplier", 0.5, 1.5, 1.00, 0.01)
arpu_mult = st.sidebar.slider("ARPU multiplier", 0.5, 1.5, 1.00, 0.01)
ebitda_pp = st.sidebar.slider("EBITDA margin Δ (pp)", -0.10, 0.10, 0.00, 0.005)
capex_pp = st.sidebar.slider("Capex intensity Δ (pp)", -0.10, 0.10, 0.00, 0.005)
wc_pp = st.sidebar.slider("Working capital intensity Δ (pp)", -0.10, 0.10, 0.00, 0.005)

overrides = {
    "customers": cust_mult,
    "arpu": arpu_mult,
    "ebitda_margin": (lambda s: s + ebitda_pp),
    "capex_pct_revenue": (lambda s: s + capex_pp),
    "wc_intensity": (lambda s: s + wc_pp),
}

# -----------------------------
# Compute scenarios
# -----------------------------
base_fin = run_scenario(base_drivers, "Base", {})
scen_fin = run_scenario(base_drivers, "Scenario", overrides)
all_fin = pd.concat([base_fin, scen_fin], ignore_index=True)

# -----------------------------
# Layout
# -----------------------------
st.title("Project Ganesha – Unlocking Free Cash Flow")
st.caption("Identify and remove constraints to cash flow by linking value drivers to financial outcomes.")

col1, col2, col3 = st.columns(3)
col1.metric("Total FCF (Base)", f"{total_fcf(base_fin):,.0f}")
col2.metric("Total FCF (Scenario)", f"{total_fcf(scen_fin):,.0f}", delta=f"{(total_fcf(scen_fin)-total_fcf(base_fin)):,.0f}")
col3.metric("Scenario FCF vs Base (%)", f"{(total_fcf(scen_fin)/max(1,total_fcf(base_fin))-1)*100:,.1f}%")

st.subheader("Monthly FCF")
pivot = all_fin.pivot_table(index="month", columns="scenario", values="fcf", aggfunc="sum").reset_index()

fig = plt.figure(figsize=(10,4))
plt.plot(pivot["month"], pivot["Base"], label="Base")
plt.plot(pivot["month"], pivot["Scenario"], label="Scenario")
plt.title("Monthly Free Cash Flow")
plt.xlabel("Month")
plt.ylabel("FCF")
plt.legend()
st.pyplot(fig)

st.subheader("Yearly FCF by Scenario")
st.dataframe(aggregate_fcf(all_fin))

# -----------------------------
# Elasticity (local sensitivity around Base)
# -----------------------------
st.subheader("FCF Elasticity (local sensitivity around Base)")
ELASTICITY_BUMP = 0.01
multiplicative = ["customers", "arpu"]
absolute_pp = ["ebitda_margin", "capex_pct_revenue", "wc_intensity"]

elasticity_rows = []
base_total = total_fcf(base_fin)
for d in multiplicative + absolute_pp:
    if d in multiplicative:
        bump_override = {d: 1 + ELASTICITY_BUMP}
    else:
        bump_override = {d: (lambda s, d=d: s + ELASTICITY_BUMP)}
    bumped = run_scenario(base_drivers, f"Base + bump {d}", bump_override)
    delta = total_fcf(bumped) - base_total
    elasticity_rows.append({
        "driver": d,
        "bump": "+1%" if d in multiplicative else "+1pp",
        "delta_total_fcf": delta,
        "elasticity_fcf_per_unit": delta / ELASTICITY_BUMP
    })

elasticity_df = pd.DataFrame(elasticity_rows).sort_values("delta_total_fcf", ascending=True)
st.dataframe(elasticity_df)

st.download_button("Download monthly results (CSV)", data=all_fin.to_csv(index=False), file_name="ganesha_results_monthly.csv", mime="text/csv")
st.download_button("Download yearly results (CSV)", data=aggregate_fcf(all_fin).to_csv(index=False), file_name="ganesha_results_yearly.csv", mime="text/csv")
st.download_button("Download elasticity (CSV)", data=elasticity_df.to_csv(index=False), file_name="ganesha_elasticity.csv", mime="text/csv")

st.markdown("""
**Notes**
- Change the scenario sliders in the sidebar to see how FCF responds.
- Elasticity table shows first-order sensitivities. For interactions or larger moves, consider multi-factor perturbations.
- Replace the mock drivers with your own by unticking *Use built-in mock data* and uploading a CSV with the required columns.
""")
