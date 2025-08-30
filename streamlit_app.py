# app.py — Ganesha POC: Value Driver Tree
# Run locally:  streamlit run app.py

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd
import streamlit as st

############################
# 1) Page Config & Styles  #
############################
st.set_page_config(page_title="Ganesha – Value Driver Tree", layout="wide")

CARD_CSS = """
<style>
.node-card {
  border: 1px solid #e5e7eb;
  border-left: 8px solid #ef4444; /* red accent like example */
  border-radius: 12px; padding: 8px 12px; background: #fff;
  box-shadow: 0 1px 2px rgba(0,0,0,0.06);
}
.node-title {font-weight:600; font-size: 0.95rem;}
.node-value {font-size: 1.6rem; font-weight: 700;}
.node-units {font-size: 0.8rem; color:#6b7280; margin-left:4px}
.node-meta {font-size: 0.8rem; color:#374151;}
.small {font-size:0.8rem; color:#6b7280}
</style>
"""
st.markdown(CARD_CSS, unsafe_allow_html=True)

############################
# 2) Inputs (single month) #
############################
with st.sidebar:
    st.header("Assumptions (Monthly)")
    colA, colB = st.columns(2)
    with colA:
        units = st.number_input("Units", min_value=0, value=500, step=10)
        asp = st.number_input("ASP (A$)", min_value=0.0, value=1200.0, step=50.0)
        gm_pct = st.number_input("Gross Margin %", min_value=0.0, max_value=1.0, value=0.28, step=0.01)
        fx_purchases = st.number_input("FX on Purchases (multiplier)", min_value=0.5, max_value=2.0, value=1.05, step=0.01, help="Multiplier applied to COGS to reflect FX on purchases.")
        inv_days = st.number_input("Inventory Days", min_value=0, value=120, step=5)
        dso = st.number_input("DSO (Days)", min_value=0, value=35, step=1)
        dpo = st.number_input("DPO (Days)", min_value=0, value=45, step=1)
    with colB:
        labour = st.number_input("Labour (A$)", min_value=0.0, value=140000.0, step=5000.0)
        overheads = st.number_input("Overheads (A$)", min_value=0.0, value=80000.0, step=5000.0)
        tax_rate = st.number_input("Tax Rate %", min_value=0.0, max_value=1.0, value=0.30, step=0.01)
        capex = st.number_input("Capex (A$)", min_value=0.0, value=20000.0, step=5000.0)
        interest_rate = st.number_input("Interest Rate %", min_value=0.0, max_value=1.0, value=0.12, step=0.01)
        prev_inv = st.number_input("Prev Inventory (A$)", min_value=0.0, value=900000.0, step=10000.0)
        prev_ar = st.number_input("Prev Receivables (A$)", min_value=0.0, value=650000.0, step=10000.0)
        prev_ap = st.number_input("Prev Payables (A$)", min_value=0.0, value=500000.0, step=10000.0)

st.caption("Values are for the current month; ΔNWC compares to the previous month balances.")

############################
# 3) Calculations          #
############################
# Revenue & COGS
revenue = units * asp
cogs_pre_fx = revenue * (1 - gm_pct)
cogs = cogs_pre_fx * fx_purchases  # FX-adjusted purchases

# EBITDA
gross_profit = revenue * gm_pct
ebitda = gross_profit - labour - overheads

# Working Capital balances
inventory = cogs * (inv_days / 365)
receivables = revenue * (dso / 365)
payables = cogs * (dpo / 365)

# ΔNWC (positive Δ means cash outflow)
nwc_prev = prev_inv + prev_ar - prev_ap
nwc_now = inventory + receivables - payables
change_nwc = nwc_now - nwc_prev

# Debt funding need = NWC (simple POC assumption). Interest on average debt.
avg_debt = (nwc_prev + nwc_now)/2
interest = avg_debt * interest_rate / 12.0  # monthly interest from annual rate

# Taxes (simple POC: corporate tax on positive EBITDA only)
taxes = max(ebitda, 0) * tax_rate

# FCF
fcf = ebitda - taxes - capex - change_nwc - interest

############################
# 4) Helper renderers      #
############################

def card(title: str, value: float, units: str = "A$", precision: int = 0):
    formatted = f"{value:,.{precision}f}" if precision else f"{value:,.0f}"
    if units == "%":
        formatted = f"{value*100:.1f}%"
    st.markdown(
        f"""
        <div class='node-card'>
          <div class='node-title'>{title}</div>
          <div class='node-value'>{formatted}<span class='node-units'>{units if units!='%' else ''}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

############################
# 5) Layout – Inline Cards + Tree (Right→Left) #
############################

# --- Scenario handling (Base vs Scenario) ---
if 'base_metrics' not in st.session_state:
    st.session_state.base_metrics = None

col_top1, col_top2 = st.columns([1,1])
with col_top1:
    set_base = st.button("Set current inputs as BASE case")
with col_top2:
    st.write("")

# Calculate a dict of all key metrics for convenience
metrics_now = {
    'Units': units,
    'ASP': asp,
    'Revenue': revenue,
    'GM%': gm_pct,
    'COGS': cogs,
    'Labour': labour,
    'Overheads': overheads,
    'EBITDA': ebitda,
    'Inventory Days': inv_days,
    'DSO': dso,
    'DPO': dpo,
    'Inventory': inventory,
    'Receivables': receivables,
    'Payables': payables,
    'ΔNWC': change_nwc,
    'Avg Debt': avg_debt,
    'Interest': interest,
    'Taxes': taxes,
    'Capex': capex,
    'FCF': fcf,
}

if set_base or st.session_state.base_metrics is None:
    st.session_state.base_metrics = metrics_now.copy()
    st.toast("Base case captured from current inputs.")

base = st.session_state.base_metrics

# --- Helpers for variance & formatting ---
PREFER_LOWER = {  # lower-is-better nodes
    'COGS', 'Inventory', 'Receivables', 'Payables', 'ΔNWC', 'Interest', 'Taxes', 'Capex', 'Inventory Days', 'DSO', 'DPO', 'Labour', 'Overheads'
}
UNITS = {
    'Units': '', 'ASP':'A$', 'Revenue':'A$', 'GM%':'%', 'COGS':'A$', 'Labour':'A$', 'Overheads':'A$', 'EBITDA':'A$',
    'Inventory Days':'days','DSO':'days','DPO':'days','Inventory':'A$','Receivables':'A$','Payables':'A$','ΔNWC':'A$',
    'Avg Debt':'A$','Interest':'A$','Taxes':'A$','Capex':'A$','FCF':'A$'
}

import random
random.seed(7)

# Simple mock sparkline (12 months) using unicode blocks
SPARK_BLOCKS = ['▁','▂','▃','▄','▅','▆','▇']

def sparkline_for(val: float):
    # create a tiny synthetic 12m series around current value
    series = [val*(0.9+0.2*random.random()) for _ in range(12)]
    mn, mx = min(series), max(series)
    rng = (mx - mn) or 1.0
    idxs = [int((x-mn)/rng* (len(SPARK_BLOCKS)-1)) for x in series]
    return ''.join(SPARK_BLOCKS[i] for i in idxs)

# Decide node color based on variance vs base
def color_for(name: str, now: float, base_val: float):
    if base_val is None:
        return '#9ca3af'  # neutral grey
    diff = now - base_val
    better = diff < 0 if name in PREFER_LOWER else diff > 0
    if abs(diff) < (abs(base_val)+1e-6)*0.005:  # within 0.5%
        return '#9ca3af'  # neutral
    return '#10b981' if better else '#ef4444'  # green or red

# Format value at top, then bold name (units) on same line, then variances, then sparkline
def label_for(name: str, val: float):
    unit = UNITS.get(name, '')
    if unit == '%':
        v_str = f"{val*100:.1f}%"
    elif unit in ('A$', ''):
        v_str = f"{val:,.0f}{' A$' if unit=='A$' else ''}"
    else:
        v_str = f"{val:,.0f} {unit}"

    b = base.get(name) if base else None
    abs_var = (val - b) if b is not None else 0.0
    pct_var = (abs_var / (abs(b)+1e-9)) if b not in (None, 0) else 0.0
    var_str = f"Δ {abs_var:,.0f} ({pct_var*100:.1f}%)" if base else ""

    # Largest value first line, then bold name with units in brackets
    title = f"<b>{name}</b> ({unit})" if unit else f"<b>{name}</b>"
    spark = sparkline_for(val)
    return f"{v_str}
{title}
{var_str}
{spark}"

############################
# Inline Tree with streamlit-agraph (Right→Left)
############################
from streamlit_agraph import agraph, Node, Edge, Config

# Build nodes with fixed positions (spacious, RL: outcomes left)
nodes = []
edges = []

# X coordinates: lower = more to the left (FCF at x=0), move drivers to the right
# Y coordinates: stack groups vertically with spacing
X = {
    'FCF': 0,
    'EBITDA': 200, 'Taxes': 200, 'Capex': 200, 'ΔNWC': 200, 'Interest': 200,
    'Revenue': 400, 'GM%': 400, 'Labour': 400, 'Overheads': 400,
    'Inventory': 400, 'Receivables': 400, 'Payables': 400, 'Avg Debt': 400,
    'Units': 600, 'ASP': 600, 'COGS': 600, 'Inventory Days': 600, 'DSO': 600, 'DPO': 600,
}
Y = {
    'FCF': 0,
    'EBITDA': -180, 'Taxes': -20, 'Capex': 140, 'ΔNWC': 300, 'Interest': 460,
    'Revenue': -260, 'GM%': -100, 'Labour': 40, 'Overheads': 180,
    'Inventory': 220, 'Receivables': 340, 'Payables': 460, 'Avg Debt': 580,
    'Units': -300, 'ASP': -180, 'COGS': 40, 'Inventory Days': 220, 'DSO': 340, 'DPO': 460,
}

KEYS = ['Units','ASP','Revenue','GM%','COGS','Labour','Overheads','EBITDA','Inventory Days','DSO','DPO','Inventory','Receivables','Payables','ΔNWC','Avg Debt','Interest','Taxes','Capex','FCF']

for k in KEYS:
    val = metrics_now['GM%'] if k=='GM%' else metrics_now.get(k, 0)
    base_val = base['GM%'] if k=='GM%' else base.get(k) if base else None
    nodes.append(Node(
        id=k,
        label=label_for(k, val),
        title=k,
        shape='box',
        font={'multi':'html','size':16,'color':'#111827'}, # black text
        color={'border': color_for(k, val, base_val), 'background': '#ffffff'},
        x=X.get(k, 800), y=Y.get(k, 0),
        fixed=True,
        shadow=True,
        borderWidth=2,
        margin=12
    ))

# Edges with formula annotations
def add_edge(a,b,label):
    edges.append(Edge(source=a, target=b, arrows='to', label=label, smooth=False))

add_edge('EBITDA','FCF','EBITDA → FCF')
add_edge('Taxes','FCF','- Taxes')
add_edge('Capex','FCF','- Capex')
add_edge('ΔNWC','FCF','- ΔNWC')
add_edge('Interest','FCF','- Interest')

add_edge('Revenue','EBITDA','Revenue × GM% − Labour − Overheads')
add_edge('GM%','EBITDA','')
add_edge('Labour','EBITDA','')
add_edge('Overheads','EBITDA','')

add_edge('Units','Revenue','Units × ASP')
add_edge('ASP','Revenue','')

add_edge('Revenue','COGS','COGS = Revenue × (1−GM%) × FX')
add_edge('GM%','COGS','')

add_edge('COGS','Inventory','Inventory = COGS × InvDays/365')
add_edge('Inventory Days','Inventory','')

add_edge('Revenue','Receivables','Receivables = Revenue × DSO/365')
add_edge('DSO','Receivables','')

add_edge('COGS','Payables','Payables = COGS × DPO/365')
add_edge('DPO','Payables','')

add_edge('Inventory','ΔNWC','ΔNWC = Δ(Inv+AR−AP)')
add_edge('Receivables','ΔNWC','')
add_edge('Payables','ΔNWC','')

add_edge('ΔNWC','Avg Debt','Avg Debt ≈ (NWC_prev+NWC_now)/2')
add_edge('Avg Debt','Interest','Interest = AvgDebt × r / 12')

config = Config(
    width=1500, height=820, directed=True, physics=False,
    hierarchical=True, hierarchical_direction='RL',
    hierarchical_level_separation=200, hierarchical_node_spacing=140, hierarchical_tree_spacing=220
)

selected = agraph(nodes=nodes, edges=edges, config=config)

# Right-hand details drawer mimic
with st.expander("Selected node details", expanded=True):
    node_id = selected if isinstance(selected, str) else None
    if node_id and node_id in metrics_now:
        st.markdown(f"### {node_id}")
        v = metrics_now[node_id]
        b = base.get(node_id) if base else None
        unit = UNITS.get(node_id,'')
        if unit == '%':
            st.write(f"Value: **{v*100:.1f}%**  |  Base: {'' if b is None else f'{b*100:.1f}%'}")
        else:
            st.write(f"Value: **{v:,.0f} {unit}**  |  Base: {'' if b is None else f'{b:,.0f} {unit}' }")
        if b is not None:
            abs_var = v - b
            pct_var = (abs_var/(abs(b)+1e-9))*100 if b else 0.0
            st.write(f"Variance vs Base: **{abs_var:,.0f} {unit} ({pct_var:.1f}%)**")
        st.write("Sparkline (mock last 12m): ")
        st.code(sparkline_for(v))
    else:
        st.write("Click any node to focus and see its details here.")

############################
# 6) Dataframe Summary     #
############################
summary = pd.DataFrame({
    'Metric': ['Units','ASP','Revenue','GM %','COGS (FX Adj)','Labour','Overheads','EBITDA','Inventory','Receivables','Payables','ΔNWC','Avg Debt','Interest','Taxes','Capex','FCF'],
    'Value': [units, asp, revenue, gm_pct, cogs, labour, overheads, ebitda, inventory, receivables, payables, change_nwc, avg_debt, interest, taxes, capex, fcf]
})
st.dataframe(summary, hide_index=True, use_container_width=True)

st.success("Inline tree ready: right→left flow, spacious cards, base vs scenario variances, colored accents, formulas on edges. Use the button above to capture a BASE case.")
