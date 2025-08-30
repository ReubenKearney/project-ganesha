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

    import graphviz as gv
    g = gv.Digraph(format='svg')
    g.attr('node', shape='box', style='rounded,filled', color='#e5e7eb', fillcolor='#ffffff', fontname='Inter')

    def label(title, val, unit="A$", precision=0):
        if unit == "%":
            v = f"{val*100:.1f}%"
        else:
            v = f"{val:,.{precision}f}" if precision else f"{val:,.0f}"
        return f"<<b>{title}</b><br/><font point-size='18'><b>{v}</b></font>>"

    # Nodes
    g.node('Units', label('Units', units, unit=""))
    g.node('ASP', label('ASP', asp))
    g.node('Revenue', label('Revenue', revenue))

    g.node('GM', label('Gross Margin %', gm_pct, unit='%'))
    g.node('COGS', label('COGS (FX Adj)', cogs))

    g.node('Labour', label('Labour', labour))
    g.node('Overheads', label('Overheads', overheads))
    g.node('EBITDA', label('EBITDA', ebitda))

    g.node('InvDays', label('Inventory Days', inv_days, unit="days"))
    g.node('DSO', label('DSO', dso, unit="days"))
    g.node('DPO', label('DPO', dpo, unit="days"))
    g.node('Inventory', label('Inventory', inventory))
    g.node('Receivables', label('Receivables', receivables))
    g.node('Payables', label('Payables', payables))

    g.node('DeltaNWC', label('ΔNWC', change_nwc))
    g.node('AvgDebt', label('Avg Debt', avg_debt))
    g.node('Interest', label('Interest', interest))

    g.node('Taxes', label('Taxes', taxes))
    g.node('Capex', label('Capex', capex))
    g.node('FCF', label('Free Cash Flow', fcf))

    # Edges
    g.edges([('Units','Revenue'), ('ASP','Revenue')])
    g.edge('Revenue','COGS')
    g.edge('GM','COGS')

    g.edge('Revenue','EBITDA')
    g.edge('GM','EBITDA')
    g.edge('Labour','EBITDA')
    g.edge('Overheads','EBITDA')

    g.edge('COGS','Inventory')
    g.edge('InvDays','Inventory')

    g.edge('Revenue','Receivables')
    g.edge('DSO','Receivables')

    g.edge('COGS','Payables')
    g.edge('DPO','Payables')

    g.edge('Inventory','DeltaNWC')
    g.edge('Receivables','DeltaNWC')
    g.edge('Payables','DeltaNWC')

    g.edge('DeltaNWC','AvgDebt')
    g.edge('AvgDebt','Interest')

    g.edge('EBITDA','FCF')
    g.edge('Taxes','FCF')
    g.edge('Capex','FCF')
    g.edge('DeltaNWC','FCF')
    g.edge('Interest','FCF')

    st.graphviz_chart(g, use_container_width=True)

############################
# 6) Dataframe Summary     #
############################
summary = pd.DataFrame({
    'Metric': ['Units','ASP','Revenue','GM %','COGS (FX Adj)','Labour','Overheads','EBITDA','Inventory','Receivables','Payables','ΔNWC','Avg Debt','Interest','Taxes','Capex','FCF'],
    'Value': [units, asp, revenue, gm_pct, cogs, labour, overheads, ebitda, inventory, receivables, payables, change_nwc, avg_debt, interest, taxes, capex, fcf]
})
st.dataframe(summary, hide_index=True, use_container_width=True)

st.success("POC ready. Adjust assumptions in the sidebar to see the tree and FCF update in real time.")
