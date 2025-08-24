import os
import textwrap
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
    # Monthly from 2020-01 to today
    dates = pd.date_range("2020-01-01", datetime.today().date(), freq="MS")
    models = [
        "NVIDIA RTX 3060",
        "NVIDIA RTX 4060",
        "AMD RX 6600",
        "AMD RX 7600",
    ]
    regions = ["US", "CN", "EU"]
    rows = []
    base = {
        "NVIDIA RTX 3060": 350,
        "NVIDIA RTX 4060": 399,
        "AMD RX 6600": 299,
        "AMD RX 7600": 269,
    }
    for d in dates:
        for m in models:
            for r in regions:
                # create a synthetic price path that responds to a few known timeline bumps
                t = (d.year - 2020) * 12 + (d.month - 1)
                price = base[m]
                price += 30 * np.sin(t / 6)  # cyclical market swings
                # 2022 crypto collapse -> lower used prices
                if d.year == 2022 and d.month >= 6:
                    price -= 40
                # 2023 Oct US export controls tighten -> CN up, US mild up
                if d >= pd.Timestamp("2023-10-01"):
                    if r == "CN":
                        price += 60
                    elif r == "US":
                        price += 15
                # 2024 rumored restrictions on AI chips -> spillover
                if d >= pd.Timestamp("2024-07-01") and r == "CN":
                    price += 25
                # noise
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
    # simple synthetic price paths for NVDA, AMD to correlate
    dates = pd.date_range("2020-01-01", datetime.today().date(), freq="B")
    tickers = ["NVDA", "AMD"]
    rows = []
    for tkr in tickers:
        price = 50 if tkr == "AMD" else 60
        for d in dates:
            drift = 1.0006 if tkr == "NVDA" else 1.0004
            price = price * drift * (1 + np.random.normal(0, 0.01))
            # add mild shock around policy events
            if d == pd.Timestamp("2023-10-18"):
                price *= 0.98
            if d == pd.Timestamp("2024-07-02"):
                price *= 0.99
            rows.append([d.date(), tkr, round(price, 2), "demo"])
    pd.DataFrame(rows, columns=["date", "ticker", "close", "source"]).to_csv(path, index=False)


# Create demo data if missing
_ensure_demo_gpu_prices(GPU_PRICES_FP)
_ensure_demo_shipments(SHIP_FP)
_ensure_demo_policy(POLICY_FP)
_ensure_demo_equities(ETF_FP)

# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data():
    prices = pd.read_csv(GPU_PRICES_FP, parse_dates=["date"])  # date, model, region, median_price, source
    ships = pd.read_csv(SHIP_FP, parse_dates=["date"])  # date, vendor, shipments_m, source
    policies = pd.read_csv(POLICY_FP, parse_dates=["date"])  # date, country, title, category, impact_direction, notes
    equities = pd.read_csv(ETF_FP, parse_dates=["date"])  # date, ticker, close, source
    return prices, ships, policies, equities

prices, ships, policies, equities = load_data()

# -----------------------------
# Sidebar Controls
# -----------------------------
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
show_median = st.sidebar.checkbox("Show overall median line", value=True)

# -----------------------------
# Layout
# -----------------------------
st.title("US–China Semiconductor Trade War → Consumer GPU Market")
st.caption("Interactive dashboard exploring price, supply, and market sentiment pathways.")

with st.expander("About this dashboard", expanded=False):
    st.markdown(
        """
        **What this shows**
        - Median *street prices* for popular consumer GPUs by region.
        - Vendor *shipment proxies* over time.
        - *Policy timeline* with event overlays.
        - Simple *equity proxies* (NVDA/AMD) for market sentiment.

        **How to use**
        - Adjust the date range and filters in the sidebar.
        - Hover on charts to view event annotations.

        **Data note**
        - This app ships with demo/synthetic data so it runs out-of-the-box. Replace files in `data/` with real sources (eBay sold listings, retailers, customs data, analyst reports). Schemas are documented in each CSV header.
        """
    )

# Filter data
mask = (prices["date"].between(pd.to_datetime(start), pd.to_datetime(end)))
if region != "All":
    mask &= prices["region"].eq(region)
if models:
    mask &= prices["model"].isin(models)

f_prices = prices.loc[mask].copy()

# -----------------------------
# KPI Cards
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

# Price delta since policy milestone (2023-10-17)
anchor = pd.Timestamp("2023-10-17")
post = f_prices[f_prices["date"] >= anchor]
pre = f_prices[f_prices["date"] < anchor]

def _median(x):
    return np.nan if x.empty else x["median_price"].median()

pre_med = _median(pre)
post_med = _median(post)

with col1:
    st.metric("Median price (selected)", f"${_median(f_prices):.0f}")
with col2:
    delta = None if np.isnan(pre_med) or np.isnan(post_med) else (post_med - pre_med) / pre_med * 100
    st.metric("Since Oct 17, 2023", f"{0 if delta is None else delta:.1f}%", delta=f"{0 if delta is None else delta:.1f}%")
with col3:
    last_month = f_prices[f_prices["date"] >= (pd.to_datetime(end) - pd.DateOffset(months=1))]
    st.metric("Last 30d median", f"${_median(last_month):.0f}")
with col4:
    st.metric("Data points", f"{len(f_prices):,}")

# -----------------------------
# Price Time Series with Event Overlays
# -----------------------------
if f_prices.empty:
    st.warning("No data matches filters. Try expanding the date range or clearing filters.")
else:
    fig = px.line(
        f_prices,
        x="date",
        y="median_price",
        color="model",
        line_group="model",
        markers=False,
        title="GPU Median Prices Over Time",
    )

    # compute overall median series (optional)
    if show_median:
        med = (
            f_prices.groupby("date")["median_price"].median().reset_index().rename(columns={"median_price": "median_all"})
        )
        fig.add_trace(
            go.Scatter(x=med["date"], y=med["median_all"], mode="lines", name="Median (all)", line=dict(dash="dash"))
        )

    # add policy markers within range
    in_range = policies[(policies["date"].between(pd.to_datetime(start), pd.to_datetime(end)))]
    for _, row in in_range.iterrows():
        color = {
            "export_control": "#ef4444",
            "subsidy": "#22c55e",
            "expectation": "#f59e0b",
        }.get(row["category"], "#6b7280")
        fig.add_vline(x=row["date"], line_width=1, line_dash="dot", line_color=color)
        fig.add_annotation(x=row["date"], y=max(f_prices["median_price"]) * 1.02, text=row["title"], showarrow=False, yanchor="bottom", font=dict(size=10), textangle=90)

    fig.update_layout(height=480, margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Shipments and Equities
# -----------------------------
cc1, cc2 = st.columns(2)
with cc1:
    s_mask = ships["date"].between(pd.to_datetime(start), pd.to_datetime(end))
    s_fig = px.line(ships.loc[s_mask], x="date", y="shipments_m", color="vendor", title="Discrete GPU Shipments (proxy)")
    s_fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(s_fig, use_container_width=True)

with cc2:
    e_mask = equities["date"].between(pd.to_datetime(start), pd.to_datetime(end))
    e_fig = px.line(equities.loc[e_mask], x="date", y="close", color="ticker", title="Equity Sentiment Proxy (NVDA/AMD)")
    e_fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(e_fig, use_container_width=True)

# -----------------------------
# Regional Price Spread
# -----------------------------
rfig = px.box(
    prices[prices["date"].between(pd.to_datetime(start), pd.to_datetime(end))],
    x="region",
    y="median_price",
    color="region",
    points="suspectedoutliers",
    title="Regional Price Distribution (selected period)",
)
rfig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(rfig, use_container_width=True)

# -----------------------------
# Event Table
# -----------------------------
with st.expander("Policy & Timeline Events"):
    st.dataframe(
        policies.sort_values("date", ascending=False).assign(
            date=lambda d: d["date"].dt.date
        )
    )

# -----------------------------
# Notes & Replace with Real Data
# -----------------------------
with st.expander("Data Sources & How to Replace Demo Data", expanded=False):
    st.markdown(
        textwrap.dedent(
            """
            **Drop your own CSVs into `data/` with these schemas:**

            1. `gpu_prices.csv`
               - Columns: `date` (YYYY-MM-DD), `model` (str), `region` (str: US/CN/EU/...), `median_price` (float), `source` (str)
               - Example sources: retailer price trackers, eBay *sold* listings (use median by month), price APIs.

            2. `shipments.csv`
               - Columns: `date` (YYYY-MM-DD), `vendor` (str), `shipments_m` (float, millions of units), `source` (str)
               - Example sources: analyst reports (JPR), company filings; if NDA-bound, use public aggregates.

            3. `policy_events.csv`
               - Columns: `date`, `country`, `title`, `category` (export_control/subsidy/expectation/other), `impact_direction` (up/down/mixed), `notes`
               - Example sources: BIS export rules, MOFCOM notices, customs rules, tariff changes.

            4. `equity_prices.csv`
               - Columns: `date`, `ticker`, `close`, `source`
               - Example: daily closes for NVDA, AMD (Yahoo Finance, official filings).

            **Tips**
            - Keep dates monthly for prices; daily is fine but can be noisy—consider median-of-month.
            - Align policy dates to *effective* dates, not just announcement dates, and add `notes`.
            - To compare regions, ensure same SKU (e.g., RTX 4060 8GB) and normalize for VAT/imports.
            """
        )
    )

# -----------------------------
# Footer
# -----------------------------
st.caption(
    "Demo data included. Replace with real sources for accurate insights. This tool is for educational visualization, not investment advice."
)
