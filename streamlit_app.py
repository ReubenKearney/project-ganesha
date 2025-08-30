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
# 5) Layout                #
############################
left, right = st.columns([2,3])

with left:
    st.subheader("Key Drivers")
    card("Revenue", revenue)
    card("COGS (FX Adj)", cogs)
    card("Gross Margin %", gm_pct, units="%")
    card("EBITDA", ebitda)
    card("Inventory", inventory)
    card("Receivables", receivables)
    card("Payables", payables)
    card("Δ NWC", change_nwc)
    card("Interest", interest)
    card("Free Cash Flow", fcf)

with right:
    st.subheader("Value Driver Tree (Structure)")
    st.write(
        "This POC shows structure and values. Next iteration will add per-node sparklines and variances."
    )

    from streamlit_agraph import agraph, Node, Edge, Config

    # Build interactive nodes/edges (vis.js via streamlit_agraph)
    nodes = []
    edges = []

    def N(id, title, val, unit="A$", precision=0):
        if unit == "%":
            v = f"{val*100:.1f}%"
        else:
            v = f"{val:,.{precision}f}" if precision else f"{val:,.0f}"
        label = f"{title}\n{v}{'' if unit=='%' else (' ' + unit if unit else '')}"
        nodes.append(Node(id=id, label=label, shape="box", 
                          font={"multi":"html","size":14},
                          borderWidth=1, shadow=True))

    def E(a,b):
        edges.append(Edge(source=a, target=b, smooth=True, arrows="to"))

    # Nodes
    N('Units', 'Units', units, unit="")
    N('ASP', 'ASP', asp)
    N('Revenue', 'Revenue', revenue)
    N('GM', 'Gross Margin %', gm_pct, unit='%')
    N('COGS', 'COGS (FX Adj)', cogs)
    N('Labour', 'Labour', labour)
    N('Overheads', 'Overheads', overheads)
    N('EBITDA', 'EBITDA', ebitda)
    N('InvDays', 'Inventory Days', inv_days, unit="days")
    N('DSO', 'DSO', dso, unit="days")
    N('DPO', 'DPO', dpo, unit="days")
    N('Inventory', 'Inventory', inventory)
    N('Receivables', 'Receivables', receivables)
    N('Payables', 'Payables', payables)
    N('DeltaNWC', 'ΔNWC', change_nwc)
    N('AvgDebt', 'Avg Debt', avg_debt)
    N('Interest', 'Interest', interest)
    N('Taxes', 'Taxes', taxes)
    N('Capex', 'Capex', capex)
    N('FCF', 'Free Cash Flow', fcf)

    # Edges
    E('Units','Revenue'); E('ASP','Revenue')
    E('Revenue','COGS'); E('GM','COGS')
    E('Revenue','EBITDA'); E('GM','EBITDA'); E('Labour','EBITDA'); E('Overheads','EBITDA')
    E('COGS','Inventory'); E('InvDays','Inventory')
    E('Revenue','Receivables'); E('DSO','Receivables')
    E('COGS','Payables'); E('DPO','Payables')
    E('Inventory','DeltaNWC'); E('Receivables','DeltaNWC'); E('Payables','DeltaNWC')
    E('DeltaNWC','AvgDebt'); E('AvgDebt','Interest')
    E('EBITDA','FCF'); E('Taxes','FCF'); E('Capex','FCF'); E('DeltaNWC','FCF'); E('Interest','FCF')

    config = Config(width=1200, height=700, directed=True, physics=True,
                    hierarchical=True,
                    hierarchical_sort_method='directed',
                    hierarchical_enabled=True,
                    hierarchical_level_separation=140,
                    hierarchical_node_spacing=110,
                    hierarchical_tree_spacing=160)

    agraph(nodes=nodes, edges=edges, config=config)

############################
# 6) Dataframe Summary     #
############################
summary = pd.DataFrame({
    'Metric': ['Units','ASP','Revenue','GM %','COGS (FX Adj)','Labour','Overheads','EBITDA','Inventory','Receivables','Payables','ΔNWC','Avg Debt','Interest','Taxes','Capex','FCF'],
    'Value': [units, asp, revenue, gm_pct, cogs, labour, overheads, ebitda, inventory, receivables, payables, change_nwc, avg_debt, interest, taxes, capex, fcf]
})
st.dataframe(summary, hide_index=True, use_container_width=True)

st.success("POC ready. Adjust assumptions in the sidebar to see the tree and FCF update in real time.")
