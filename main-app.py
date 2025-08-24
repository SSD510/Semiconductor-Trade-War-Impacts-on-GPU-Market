import os
import textwrap
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# =============================
# App Config
# =============================
st.set_page_config(
    page_title="US–China Semiconductor Trade War: Consumer GPU Market Dashboard",
    page_icon="💾",
    layout="wide",
)

# -----------------------------
# Helper: seed/demo data
# -----------------------------
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

GPU_PRICES_FP = os.path.join(DATA_DIR, "gpu_prices.csv")
SHIP_FP = os.path.join(DATA_DIR, "shipments.csv")
POLICY_FP = os.path.join(DATA_DIR, "policy_events.csv")
ETF_FP = os.path.join(DATA_DIR, "equity_prices.csv")

np.random.seed(7)


def _ensure_demo_gpu_prices(path: str):
    if os.path.exists(path):
        return
    dates = pd.date_range("2020-01-01", datetime.today().date(), freq="MS")
    models = ["NVIDIA RTX 3060", "NVIDIA RTX 4060", "AMD RX 6600", "AMD RX 7600"]
    regions = ["US", "CN", "EU"]
    rows = []
    base = {"NVIDIA RTX 3060": 350, "NVIDIA RTX 4060": 399, "AMD RX 6600": 299, "AMD RX 7600": 269}
    for d in dates:
        for m in models:
            for r in regions:
                t = (d.year - 2020) * 12 + (d.month - 1)
                price = base[m]
                price += 30 * np.sin(t / 6)
                if d.year == 2022 and d.month >= 6:
                    price -= 40
                if d >= pd.Timestamp("2023-10-01"):
                    if r == "CN":
                        price += 60
                    elif r == "US":
                        price += 15
                if d >= pd.Timestamp("2024-07-01") and r == "CN":
                    price += 25
                price += np.random.normal(0, 12)
                rows.append([d.date(), m, r, max(120, round(price, 2)), "demo"])
    df = pd.DataFrame(rows, columns=["date", "model", "region", "median_price", "source"])
    df.to_csv(path, index=False)


def _ensure_demo_shipments(path: str):
    if os.path.exists(path):
        return
    dates = pd.date_range("2020-01-01", datetime.today().date(), freq="QS")
    vendors = ["NVIDIA", "AMD"]
    rows = []
    for d in dates:
        for v in vendors:
            base = 8 if v == "NVIDIA" else 4
            season = 1 + 0.15 * np.sin(d.quarter * np.pi / 2)
            trade_headwind = 0.9 if d >= pd.Timestamp("2023-10-01") else 1.0
            cn_offset = 0.85 if d >= pd.Timestamp("2024-07-01") else 1.0
            shipments = base * season * trade_headwind * cn_offset
            shipments += np.random.normal(0, 0.4)
            rows.append([d.date(), v, max(1.0, round(shipments, 2)), "demo"])
    pd.DataFrame(rows, columns=["date", "vendor", "shipments_m", "source"]).to_csv(path, index=False)


def _ensure_demo_policy(path: str):
    if os.path.exists(path):
        return
    events = [
        ("2020-09-15", "US", "Restrictions on Huawei (chip supply tightened)", "export_control", "down", "Tighter access to advanced fabs and EDA for Chinese firms."),
        ("2022-08-09", "US", "CHIPS and Science Act signed", "subsidy", "mixed", "US fabs incentives; long-run supply up, short-term neutral."),
        ("2022-10-07", "US", "Sweeping export controls on advanced chips/EDA to China", "export_control", "up", "Scarcity risk for advanced GPUs in CN."),
        ("2023-10-17", "US", "Updated AI chip export rules (A800/H800 etc.)", "export_control", "up", "Further restricts AI accelerators; consumer spillover risk."),
        ("2024-07-01", "US", "Rumors/plans of tighter controls; partners adjust SKUs", "expectation", "up", "Market prices in CN rise on anticipation."),
    ]
    pd.DataFrame(events, columns=["date", "country", "title", "category", "impact_direction", "notes"]).to_csv(path, index=False)


def _ensure_demo_equities(path: str):
    if os.path.exists(path):
        return
    dates = pd.date_range("2020-01-01", datetime.today().date(), freq="B")
    tickers = ["NVDA", "AMD"]
    rows = []
    for tkr in tickers:
        price = 50 if tkr == "AMD" else 60
        for d in dates:
            drift = 1.0006 if tkr == "NVDA" else 1.0004
            price = price * drift * (1 + np.random.normal(0, 0.01))
            if d == pd.Timestamp("2023-10-18"):
                price *= 0.98
            if d == pd.Timestamp("2024-07-02"):
                price *= 0.99
            rows.append([d.date(), tkr, round(price, 2), "demo"])
    pd.DataFrame(rows, columns=["date", "ticker", "close", "source"]).to_csv(path, index=False)


_ensure_demo_gpu_prices(GPU_PRICES_FP)
_ensure_demo_shipments(SHIP_FP)
_ensure_demo_policy(POLICY_FP)
_ensure_demo_equities(ETF_FP)

@st.cache_data
def load_data():
    prices = pd.read_csv(GPU_PRICES_FP, parse_dates=["date"])
    ships = pd.read_csv(SHIP_FP, parse_dates=["date"])
    policies = pd.read_csv(POLICY_FP, parse_dates=["date"])
    equities = pd.read_csv(ETF_FP, parse_dates=["date"])
    return prices, ships, policies, equities

prices, ships, policies, equities = load_data()

min_date = prices["date"].min().date()
max_date = prices["date"].max().date()

st.sidebar.header("Filters")
start, end = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)
region = st.sidebar.selectbox("Region", options=["All"] + sorted(prices["region"].unique().tolist()))
models = st.sidebar.multiselect("GPU models", options=sorted(prices["model"].unique().tolist()))

st.title("US–China Semiconductor Trade War → Consumer GPU Market")
st.caption("Interactive dashboard exploring price, supply, and market sentiment pathways.")

mask = (prices["date"].between(pd.to_datetime(start), pd.to_datetime(end)))
if region != "All":
    mask &= prices["region"].eq(region)
if models:
    mask &= prices["model"].isin(models)

f_prices = prices.loc[mask].copy()

col1, col2, col3, col4 = st.columns(4)

def _median(x):
    return np.nan if x.empty else x["median_price"].median()

anchor = pd.Timestamp("2023-10-17")
post = f_prices[f_prices["date"] >= anchor]
pre = f_prices[f_prices["date"] < anchor]
pre_med = _median(pre)
post_med = _median(post)

with col1:
    st.metric("Median price (selected)", f"${_median(f_prices):.0f}")
with col2:
    delta = None if np.isnan(pre_med) or np.isnan(post_med) else (post_med - pre_med) / pre_med * 100
    st.metric("Since Oct 17, 2023", f"{0 if delta is None else delta:.1f}%")
with col3:
    last_month = f_prices[f_prices["date"] >= (pd.to_datetime(end) - pd.DateOffset(months=1))]
    st.metric("Last 30d median", f"${_median(last_month):.0f}")
with col4:
    st.metric("Data points", f"{len(f_prices):,}")

if f_prices.empty:
    st.warning("No data matches filters. Try expanding the date range or clearing filters.")
else:
    st.subheader("GPU Median Prices Over Time")
    chart_df = f_prices.pivot_table(index="date", columns="model", values="median_price", aggfunc="mean")
    st.line_chart(chart_df)

st.subheader("Discrete GPU Shipments (proxy)")
s_mask = ships["date"].between(pd.to_datetime(start), pd.to_datetime(end))
ship_df = ships.loc[s_mask].pivot_table(index="date", columns="vendor", values="shipments_m")
st.line_chart(ship_df)

st.subheader("Equity Sentiment Proxy (NVDA/AMD)")
e_mask = equities["date"].between(pd.to_datetime(start), pd.to_datetime(end))
eq_df = equities.loc[e_mask].pivot_table(index="date", columns="ticker", values="close")
st.line_chart(eq_df)

st.subheader("Regional Price Distribution (selected period)")
r_mask = prices["date"].between(pd.to_datetime(start), pd.to_datetime(end))
reg_df = prices.loc[r_mask]
st.bar_chart(reg_df.groupby("region")["median_price"].median())

with st.expander("Policy & Timeline Events"):
    st.dataframe(policies.sort_values("date", ascending=False).assign(date=lambda d: d["date"].dt.date))

with st.expander("Data Sources & How to Replace Demo Data", expanded=False):
    st.markdown(
        textwrap.dedent(
            """
            **Drop your own CSVs into `data/` with these schemas:**

            1. `gpu_prices.csv`: date, model, region, median_price, source
            2. `shipments.csv`: date, vendor, shipments_m, source
            3. `policy_events.csv`: date, country, title, category, impact_direction, notes
            4. `equity_prices.csv`: date, ticker, close, source
            """
        )
    )

st.caption("Demo data included. Replace with real sources for accurate insights. This tool is for educational visualization, not investment advice.")
