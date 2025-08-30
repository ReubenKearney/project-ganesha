# Project Ganesha – ASA FCF Modeller

Drivers → Revenue/COGS → EBITDA → Capex/ΔNWC/Interest → **FCF**. Built for ASA (firearms & accessories import/distribution).

## Features

* **Tornado ranking** of FCF constraints (local sensitivity to each driver)
* **Sankey flow** for Base and Scenario−Base Δ
* **Dependency network** filtered to top‑N constraints + neighbors
* Scenario sliders for ASA-specific drivers (Units, ASP, GM%, FX, Inventory/DSO/DPO, Interest rate, etc.)
* CSV downloads (monthly, yearly, elasticity)

## Run locally

```bash
pip install -r requirements.txt
streamlit run ganesha_app_asa.py
```

## Deploy on Streamlit Community Cloud

1. Push `ganesha_app_asa.py`, `requirements.txt`, and this `README.md` to a GitHub repo.
2. In Streamlit Cloud → **New app** → pick the repo and set **Main file** to `ganesha_app_asa.py`.

## Data schema (optional upload)

If you untick *Use built-in mock data*, upload a CSV with columns:

```
month, units, asp, gross_margin_pct, fx_rate, headcount, labour_cost_per_emp_m,
overheads_pct_rev, inventory_days, dso_days, dpo_days, interest_rate_annual,
capex_pct_revenue, tax_rate
```

## Notes

* Elasticities use intuitive bumps (e.g., +1% Units/ASP/FX, +1 day Inventory/DSO/DPO, +1pp on rates/percentages).
* Interest expense is tied to NWC (positive balances financed at the interest rate).
* You can refine COGS timing vs. purchases, taxes, or add SKU-level mix later.
