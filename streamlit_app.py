import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import networkx as nx

st.set_page_config(page_title="Project Ganesha – ASA FCF Modeller", layout="wide")

# -----------------------------
# Data utilities
# -----------------------------
def mock_monthly_calendar(start="2023-01-01", periods=24):
    dt = pd.date_range(start=start, periods=periods, freq="MS")
    return pd.DataFrame({"month": dt})

def generate_mock_drivers_asa(start="2023-01-01", periods=24, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cal = mock_monthly_calendar(start, periods)

    # Core ASA drivers with light trends/seasonality
    units = 2000 * (1 + 0.01*np.arange(periods)) + rng.normal(0, 30, periods)                          # dealer demand
    asp = 600 * (1 + 0.002*np.arange(periods)) + rng.normal(0, 5, periods)                               # selling price
    gross_margin_pct = 0.22 + 0.02*np.sin(np.linspace(0, 2*np.pi, periods))                              # mix
    fx_rate = 1.50 + 0.05*np.sin(np.linspace(0, 3*np.pi, periods)) + rng.normal(0, 0.01, periods)        # AUD per USD
    headcount = np.full(periods, 18) + rng.integers(-1, 2, periods)                                       # warehouse/admin
    labour_cost_per_emp_m = np.full(periods, 6000) + rng.normal(0, 150, periods)                          # per month
    overheads_pct_rev = 0.07 + 0.005*np.cos(np.linspace(0, 2*np.pi, periods))                             # % revenue
    inventory_days = 120 + 10*np.sin(np.linspace(0, 2*np.pi, periods)) + rng.normal(0, 2, periods)        # days
    dso_days = 35 + 5*np.cos(np.linspace(0, 3*np.pi, periods)) + rng.normal(0, 1, periods)                # receivables days
    dpo_days = 45 + 5*np.sin(np.linspace(0, 1.5*np.pi, periods)) + rng.normal(0, 1, periods)              # payables days
    interest_rate_annual = np.full(periods, 0.09)                                                          # 9% annual
    capex_pct_revenue = np.full(periods, 0.015)                                                            # 1.5% capex
    tax_rate = np.full(periods, 0.30)                                                                      # corporate tax

    df = cal.copy()
    df["units"] = units
    df["asp"] = asp
    df["gross_margin_pct"] = gross_margin_pct
    df["fx_rate"] = fx_rate
    df["headcount"] = headcount
    df["labour_cost_per_emp_m"] = labour_cost_per_emp_m
    df["overheads_pct_rev"] = overheads_pct_rev
    df["inventory_days"] = inventory_days
    df["dso_days"] = dso_days
    df["dpo_days"] = dpo_days
    df["interest_rate_annual"] = interest_rate_annual
    df["capex_pct_revenue"] = capex_pct_revenue
    df["tax_rate"] = tax_rate
    return df

def compute_financials_asa(dr: pd.DataFrame, fx_ref=1.50) -> pd.DataFrame:
    """Compute ASA-specific financials from driver dataframe."""
    df = dr.copy().sort_values("month").reset_index(drop=True)

    # Revenue & COGS (FX affects landed cost)
    df["revenue"] = df["units"] * df["asp"]
    # COGS share adjusted by FX: higher AUD/USD -> higher AUD cost -> higher COGS
    cogs_share_base = (1 - df["gross_margin_pct"]).clip(lower=0.02, upper=0.98)
    fx_adj = (df["fx_rate"] / fx_ref).clip(lower=0.5, upper=2.0)
    df["cogs"] = df["revenue"] * cogs_share_base * fx_adj
    df["gross_profit"] = df["revenue"] - df["cogs"]

    # Operating expenses
    df["labour"] = df["headcount"] * df["labour_cost_per_emp_m"]
    df["overheads"] = df["overheads_pct_rev"] * df["revenue"]

    # EBITDA (simplified)
    df["ebitda"] = df["gross_profit"] - df["labour"] - df["overheads"]

    # Taxes (approximate on positive EBITDA)
    df["taxes"] = np.where(df["ebitda"] > 0, df["ebitda"] * df["tax_rate"], 0.0)

    # Capex
    df["capex"] = df["capex_pct_revenue"] * df["revenue"]

    # Working capital balances from days
    df["inventory"] = df["cogs"] * (df["inventory_days"] / 365.0)
    df["receivables"] = df["revenue"] * (df["dso_days"] / 365.0)
    # Purchases approximated by COGS here (could refine with pipeline timing)
    df["payables"] = df["cogs"] * (df["dpo_days"] / 365.0)

    df["nwc"] = df["inventory"] + df["receivables"] - df["payables"]
    df["delta_nwc"] = df["nwc"].diff().fillna(df["nwc"])

    # Interest: finance NWC (if positive). Interest is annual rate * average monthly balance.
    avg_nwc = (df["nwc"] + df["nwc"].shift(1).fillna(0)) / 2.0
    df["interest"] = np.maximum(avg_nwc, 0) * (df["interest_rate_annual"] / 12.0)

    # Free Cash Flow
    df["fcf"] = df["ebitda"] - df["taxes"] - df["capex"] - df["delta_nwc"] - df["interest"]
    df["year"] = df["month"].dt.year
    return df

def aggregate_fcf(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(["scenario","year"], as_index=False)["fcf"].sum()
              .rename(columns={"fcf":"fcf_yearly"}))

def total_fcf(df: pd.DataFrame) -> float:
    return float(df["fcf"].sum())

# -----------------------------
# Scenarios & Elasticities
# -----------------------------
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

def run_scenario_asa(base_drivers: pd.DataFrame, name: str, overrides: dict) -> pd.DataFrame:
    scen_dr = apply_overrides(base_drivers, overrides)
    scen_fin = compute_financials_asa(scen_dr)
    scen_fin["scenario"] = name
    return scen_fin

def compute_elasticity_asa(base_dr: pd.DataFrame, base_fin: pd.DataFrame, fx_ref=1.50) -> pd.DataFrame:
    """Local sensitivity of Total FCF to small bumps on each driver."""
    bumps = {
        "units": ("+1%", lambda s: s * 1.01),
        "asp": ("+1%", lambda s: s * 1.01),
        "gross_margin_pct": ("+1pp", lambda s: s + 0.01),
        "fx_rate": ("+1%", lambda s: s * 1.01),
        "headcount": ("+1", lambda s: s + 1),
        "labour_cost_per_emp_m": ("+1%", lambda s: s * 1.01),
        "overheads_pct_rev": ("+1pp", lambda s: s + 0.01),
        "inventory_days": ("+1 day", lambda s: s + 1.0),
        "dso_days": ("+1 day", lambda s: s + 1.0),
        "dpo_days": ("+1 day", lambda s: s + 1.0),
        "interest_rate_annual": ("+1pp", lambda s: s + 0.01),
        # Optional:
        "capex_pct_revenue": ("+1pp", lambda s: s + 0.01),
    }
    base_total = total_fcf(base_fin)
    rows = []
    for drv, (label, bump_fn) in bumps.items():
        bumped_fin = run_scenario_asa(base_dr, "bump", {drv: bump_fn})
        delta = total_fcf(bumped_fin) - base_total
        rows.append({"driver": drv, "bump": label, "delta_total_fcf": delta})
    df = pd.DataFrame(rows).sort_values("delta_total_fcf", ascending=True)
    return df

# -----------------------------
# UI: Data & Scenario Controls
# -----------------------------
st.sidebar.header("ASA Data & Scenario Controls")
use_mock = st.sidebar.checkbox("Use built-in mock data", value=True)

if use_mock:
    base_drivers = generate_mock_drivers_asa()
else:
    st.sidebar.info("Upload ASA drivers CSV with columns: month, units, asp, gross_margin_pct, fx_rate, headcount, labour_cost_per_emp_m, overheads_pct_rev, inventory_days, dso_days, dpo_days, interest_rate_annual, capex_pct_revenue, tax_rate")
    uploaded = st.sidebar.file_uploader("Upload ASA drivers CSV", type=["csv"])
    if uploaded is not None:
        base_drivers = pd.read_csv(uploaded, parse_dates=["month"])
    else:
        st.stop()

# Scenario sliders (ASA-specific)
st.sidebar.subheader("Scenario Adjustments")
units_mult = st.sidebar.slider("Units multiplier", 0.5, 1.5, 1.00, 0.01)
asp_mult = st.sidebar.slider("ASP multiplier", 0.5, 1.5, 1.00, 0.01)
gm_pp = st.sidebar.slider("Gross Margin Δ (pp)", -0.10, 0.10, 0.00, 0.005)
fx_pp = st.sidebar.slider("FX rate Δ (%)", -0.10, 0.10, 0.00, 0.005)  # +/- percentage change
headcount_delta = st.sidebar.slider("Headcount Δ (people)", -10, 10, 0, 1)
labour_mult = st.sidebar.slider("Labour cost per emp Δ (%)", -0.20, 0.20, 0.00, 0.01)
oh_pp = st.sidebar.slider("Overheads % Rev Δ (pp)", -0.10, 0.10, 0.00, 0.005)
inv_days_delta = st.sidebar.slider("Inventory Days Δ (days)", -60, 60, 0, 1)
dso_days_delta = st.sidebar.slider("DSO Δ (days)", -30, 30, 0, 1)
dpo_days_delta = st.sidebar.slider("DPO Δ (days)", -30, 30, 0, 1)
rate_pp = st.sidebar.slider("Interest rate Δ (pp)", -0.10, 0.10, 0.00, 0.005)
capex_pp = st.sidebar.slider("Capex % Rev Δ (pp)", -0.05, 0.05, 0.00, 0.005)

overrides = {
    "units": (lambda s: s * units_mult),
    "asp": (lambda s: s * asp_mult),
    "gross_margin_pct": (lambda s: s + gm_pp),
    "fx_rate": (lambda s: s * (1 + fx_pp)),
    "headcount": (lambda s: s + headcount_delta),
    "labour_cost_per_emp_m": (lambda s: s * (1 + labour_mult)),
    "overheads_pct_rev": (lambda s: s + oh_pp),
    "inventory_days": (lambda s: s + inv_days_delta),
    "dso_days": (lambda s: s + dso_days_delta),
    "dpo_days": (lambda s: s + dpo_days_delta),
    "interest_rate_annual": (lambda s: s + rate_pp),
    "capex_pct_revenue": (lambda s: s + capex_pp),
}

# Compute scenarios
base_fin = run_scenario_asa(base_drivers, "Base", {})
scen_fin = run_scenario_asa(base_drivers, "Scenario", overrides)
all_fin = pd.concat([base_fin, scen_fin], ignore_index=True)

# Header & KPIs
st.title("Project Ganesha – ASA FCF Modeller")
st.caption("For ASA (import/distribution): Drivers → Revenue/COGS → EBITDA → Capex/ΔNWC/Interest → FCF.")

col1, col2, col3 = st.columns(3)
col1.metric("Total FCF (Base)", f"{total_fcf(base_fin):,.0f}")
col2.metric("Total FCF (Scenario)", f"{total_fcf(scen_fin):,.0f}", delta=f"{(total_fcf(scen_fin)-total_fcf(base_fin)):,.0f}")
col3.metric("Scenario vs Base (%)", f"{(total_fcf(scen_fin)/max(1,total_fcf(base_fin))-1)*100:,.1f}%")

# Monthly FCF chart
st.subheader("Monthly Free Cash Flow")
pivot = all_fin.pivot_table(index="month", columns="scenario", values="fcf", aggfunc="sum").reset_index()
fig = plt.figure(figsize=(10,4))
plt.plot(pivot["month"], pivot["Base"], label="Base")
plt.plot(pivot["month"], pivot["Scenario"], label="Scenario")
plt.title("Monthly Free Cash Flow")
plt.xlabel("Month")
plt.ylabel("FCF")
plt.legend()
st.pyplot(fig)

# Yearly table
st.subheader("Yearly FCF by Scenario")
st.dataframe(aggregate_fcf(all_fin))

# -----------------------------
# 1) Tornado – Constraint Ranking
# -----------------------------
st.markdown("---")
st.header("Constraints: Which drivers impact ASA's FCF the most? (Tornado)")
elasticity_df = compute_elasticity_asa(base_drivers, base_fin)
plot_df = elasticity_df.copy().sort_values("delta_total_fcf")  # most constraining at top (most negative)

fig_t = plt.figure(figsize=(9, max(4, 0.4*len(plot_df))))
plt.barh(plot_df["driver"], plot_df["delta_total_fcf"])
plt.axvline(0, linestyle="--")
plt.xlabel("Δ Total FCF from small bump (see labels)")
plt.title("FCF Constraint Ranking (local sensitivity around Base)")
st.pyplot(fig_t)
with st.expander("View elasticity table"):
    st.dataframe(plot_df.reset_index(drop=True))

# -----------------------------
# 2) Sankey – Value Flow
# -----------------------------
st.markdown("---")
st.header("Value Flow (Sankey)")
mode = st.radio("Flow mode", ["Base average", "Scenario − Base (Δ)"], horizontal=True)

def build_sankey_asa(fin_base: pd.DataFrame, fin_scen: pd.DataFrame, mode: str = "Base average"):
    nodes = ["Revenue","COGS","Gross Profit","Labour","Overheads","EBITDA","Taxes","Capex","ΔNWC","Interest","FCF"]
    node_index = {n:i for i,n in enumerate(nodes)}

    def totals(df: pd.DataFrame):
        return {
            "revenue": float(df["revenue"].mean()),
            "cogs": float(df["cogs"].mean()),
            "gp": float(df["gross_profit"].mean()),
            "labour": float(df["labour"].mean()),
            "overheads": float(df["overheads"].mean()),
            "ebitda": float(df["ebitda"].mean()),
            "taxes": float(df["taxes"].mean()),
            "capex": float(df["capex"].mean()),
            "dnwc": float(df["delta_nwc"].mean()),
            "interest": float(df["interest"].mean()),
            "fcf": float(df["fcf"].mean()),
        }

    tb = totals(base_fin); ts = totals(scen_fin)

    if mode == "Base average":
        links = [
            ("Revenue","COGS", max(tb["cogs"],0)),
            ("Revenue","Gross Profit", max(tb["gp"],0)),
            ("Gross Profit","Labour", max(tb["labour"],0)),
            ("Gross Profit","Overheads", max(tb["overheads"],0)),
            ("Gross Profit","EBITDA", max(tb["ebitda"],0)),
            ("EBITDA","Taxes", max(tb["taxes"],0)),
            ("EBITDA","Capex", max(tb["capex"],0)),
            ("EBITDA","ΔNWC", abs(tb["dnwc"])),
            ("EBITDA","Interest", max(tb["interest"],0)),
            ("Taxes","FCF", max(tb["taxes"],0)),
            ("Capex","FCF", max(tb["capex"],0)),
            ("ΔNWC","FCF", abs(tb["dnwc"])),
            ("Interest","FCF", max(tb["interest"],0)),
            ("EBITDA","FCF", max(tb["ebitda"] - tb["taxes"] - tb["capex"] - abs(tb["dnwc"]) - tb["interest"], 0)),
        ]
    else:
        # deltas (Scenario - Base)
        dif = {k: ts[k]-tb[k] for k in tb.keys()}
        links = [
            ("Revenue","COGS", abs(dif["cogs"])),
            ("Revenue","Gross Profit", abs(dif["gp"])),
            ("Gross Profit","Labour", abs(dif["labour"])),
            ("Gross Profit","Overheads", abs(dif["overheads"])),
            ("Gross Profit","EBITDA", abs(dif["ebitda"])),
            ("EBITDA","Taxes", abs(dif["taxes"])),
            ("EBITDA","Capex", abs(dif["capex"])),
            ("EBITDA","ΔNWC", abs(dif["dnwc"])),
            ("EBITDA","Interest", abs(dif["interest"])),
            ("Taxes","FCF", abs(dif["taxes"])),
            ("Capex","FCF", abs(dif["capex"])),
            ("ΔNWC","FCF", abs(dif["dnwc"])),
            ("Interest","FCF", abs(dif["interest"])),
            ("EBITDA","FCF", abs(dif["ebitda"] - dif["taxes"] - dif["capex"] - dif["dnwc"] - dif["interest"])),
        ]

    src = [node_index[s] for s,t,v in links]
    tgt = [node_index[t] for s,t,v in links]
    val = [float(v) if float(v) > 0 else 0.0 for s,t,v in links]

    fig = go.Figure(data=[go.Sankey(node=dict(label=nodes, pad=20, thickness=20),
                                    link=dict(source=src, target=tgt, value=val))])
    fig.update_layout(height=520, margin=dict(l=10,r=10,t=10,b=10))
    return fig

fig_s = build_sankey_asa(base_fin, scen_fin, mode)
st.plotly_chart(fig_s, use_container_width=True)

# -----------------------------
# 3) Network – Dependencies (Top-N constraints + neighbors)
# -----------------------------
st.markdown("---")
st.header("Driver Dependency Graph (Top-N constraints and neighbors)")

top_n = st.slider("Show top-N constrained drivers", min_value=3, max_value=20, value=10, step=1)

# Build graph definition
nodes = []

def add_node(nid, label, ntype):
    nodes.append({"id": nid, "label": label, "type": ntype})

# Driver nodes (ASA core set)
add_node("units", "Units (Dealer Demand)", "driver")
add_node("asp", "Average Selling Price", "driver")
add_node("gross_margin_pct", "Gross Margin % (mix)", "driver")
add_node("fx_rate", "FX Rate (AUD/USD)", "driver")
add_node("headcount", "Headcount", "driver")
add_node("labour_cost_per_emp_m", "Labour Cost per Emp (m)", "driver")
add_node("overheads_pct_rev", "Overheads % Revenue", "driver")
add_node("inventory_days", "Inventory Days", "driver")
add_node("dso_days", "DSO (Receivables Days)", "driver")
add_node("dpo_days", "DPO (Payables Days)", "driver")
add_node("interest_rate_annual", "Interest Rate (annual)", "driver")
add_node("capex_pct_revenue", "Capex % Revenue", "driver")  # optional lever

# Intermediates & output
for nid, label, typ in [
    ("revenue","Revenue","intermediate"),
    ("cogs","COGS","intermediate"),
    ("gross_profit","Gross Profit","intermediate"),
    ("labour","Labour","intermediate"),
    ("overheads","Overheads","intermediate"),
    ("ebitda","EBITDA","intermediate"),
    ("taxes","Taxes","intermediate"),
    ("capex","Capex","intermediate"),
    ("inventory","Inventory","intermediate"),
    ("receivables","Receivables","intermediate"),
    ("payables","Payables","intermediate"),
    ("delta_nwc","ΔNWC","intermediate"),
    ("fcf","FCF","output"),
]:
    add_node(nid, label, typ)

edges = [
    ("units","revenue","+"),
    ("asp","revenue","+"),
    ("gross_margin_pct","cogs","-"),
    ("fx_rate","cogs","+"),
    ("revenue","cogs","+"),
    ("revenue","gross_profit","+"),
    ("cogs","gross_profit","-"),
    ("gross_profit","labour","+"),
    ("headcount","labour","+"),
    ("labour_cost_per_emp_m","labour","+"),
    ("gross_profit","overheads","+"),
    ("overheads_pct_rev","overheads","+"),
    ("gross_profit","ebitda","+"),
    ("labour","ebitda","-"),
    ("overheads","ebitda","-"),
    ("ebitda","taxes","+"),
    ("revenue","capex","+"),
    ("capex_pct_revenue","capex","+"),
    ("cogs","inventory","+"),
    ("revenue","receivables","+"),
    ("cogs","payables","+"),
    ("inventory","delta_nwc","+"),
    ("receivables","delta_nwc","+"),
    ("payables","delta_nwc","-"),
    ("ebitda","fcf","+"),
    ("taxes","fcf","-"),
    ("capex","fcf","-"),
    ("delta_nwc","fcf","-"),
    ("interest_rate_annual","fcf","-"),  # via interest; shown as direct for simplicity
]

G = nx.DiGraph()
for n in nodes:
    G.add_node(n["id"], label=n["label"], type=n["type"])
for s,t,sgn in edges:
    G.add_edge(s,t, sign=sgn)

# Influence sizing: use to-FCF elasticities for driver nodes
elasticity = plot_df.set_index("driver")["delta_total_fcf"].to_dict()
for n in G.nodes:
    if G.nodes[n]["type"] == "driver":
        G.nodes[n]["influence"] = abs(elasticity.get(n, 0.0))
    elif n == "fcf":
        G.nodes[n]["influence"] = 1.0
    else:
        G.nodes[n]["influence"] = 0.5 + 0.2*(G.in_degree(n)+G.out_degree(n))

# Filter to top-N constrained drivers + their neighbors
top_drivers = sorted([k for k in elasticity.keys()], key=lambda k: elasticity[k])[:top_n]
sub_nodes = set(top_drivers)
for d in top_drivers:
    sub_nodes.update(nx.ancestors(G, d))
    sub_nodes.update(nx.descendants(G, d))
H = G.subgraph(sub_nodes).copy()

# Layout and plot
pos = nx.spring_layout(H, k=0.6, seed=42)
type_color = {"driver":"#1f77b4","intermediate":"#7f7f7f","output":"#2ca02c"}
node_x, node_y, node_text, node_size, node_color = [], [], [], [], []
max_infl = max([H.nodes[n].get("influence",1.0) for n in H.nodes] + [1.0])

for n in H.nodes:
    x, y = pos[n]
    node_x.append(x); node_y.append(y)
    node_text.append(f"{H.nodes[n]['label']} (type: {H.nodes[n]['type']})")
    size = 10 + 30*(H.nodes[n].get("influence",0)/max_infl)
    node_size.append(size)
    node_color.append(type_color[H.nodes[n]["type"]])

edge_x, edge_y = [], []
for s,t in H.edges:
    x0, y0 = pos[s]; x1, y1 = pos[t]
    edge_x += [x0, x1, None]; edge_y += [y0, y1, None]

edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines",
                        line=dict(width=1, color="#bbbbbb"), hoverinfo="none")
node_trace = go.Scatter(x=node_x, y=node_y, mode="markers+text",
                        text=[H.nodes[n]["label"] for n in H.nodes],
                        textposition="bottom center",
                        marker=dict(size=node_size, color=node_color, line=dict(width=1, color="#333")),
                        hovertext=node_text, hoverinfo="text")

fig_net = go.Figure(data=[edge_trace, node_trace])
fig_net.update_layout(showlegend=False, margin=dict(l=10,r=10,t=10,b=10), height=620)
st.plotly_chart(fig_net, use_container_width=True)

# Downloads
st.markdown("---")
st.download_button("Download monthly results (CSV)", data=all_fin.to_csv(index=False), file_name="ganesha_asa_results_monthly.csv", mime="text/csv")
st.download_button("Download yearly results (CSV)", data=aggregate_fcf(all_fin).to_csv(index=False), file_name="ganesha_asa_results_yearly.csv", mime="text/csv")
st.download_button("Download elasticity (CSV)", data=plot_df.to_csv(index=False), file_name="ganesha_asa_elasticity.csv", mime="text/csv")

st.markdown("""
**Notes**
- **Tornado** ranks ASA-specific constraints by local sensitivity (Δ Total FCF from a small bump per driver).
- **Sankey** explains value flow from Revenue/COGS to FCF; switch to Δ mode to see Scenario vs Base changes.
- **Network** reveals structure; filtered to the top-N constrained drivers and related nodes.
- Replace mock data by uploading ASA drivers CSV with the listed columns.
""")
