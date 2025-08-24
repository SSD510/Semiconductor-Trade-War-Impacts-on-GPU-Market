# US–China Semiconductor Trade War: Consumer GPU Market Dashboard

This project is an **interactive dashboard built with Streamlit** that visualizes how the US–China semiconductor trade war impacts the **consumer GPU market**. It uses demo (synthetic) data out-of-the-box, and you can replace it with your own CSV datasets.

---

## Features

* 📈 **GPU Price Trends**: Median prices by model and region over time.
* 🏭 **Shipments**: Vendor-level shipment proxies.
* 📰 **Policy Timeline**: Key trade policy and export control events.
* 💹 **Equity Proxies**: NVDA and AMD stock as market sentiment indicators.
* 🌍 **Regional Price Spreads**: Compare GPU prices across regions.

---

## Requirements

* Python 3.9+
* [Streamlit](https://streamlit.io/) (no extra visualization libraries required)

Install Streamlit:

```bash
pip install streamlit pandas numpy
```

---

## Run the App

Clone this repository and run:

```bash
streamlit run app.py
```

The app will open in your browser.

---

## Data Sources

The app looks for CSV files in the `data/` folder. If not found, it will generate demo datasets.

### Expected Schemas

1. **`gpu_prices.csv`**

   * `date` (YYYY-MM-DD)
   * `model` (e.g., RTX 4060)
   * `region` (US, CN, EU, ...)
   * `median_price` (float)
   * `source` (string)

2. **`shipments.csv`**

   * `date`
   * `vendor` (e.g., NVIDIA, AMD)
   * `shipments_m` (float, millions of units)
   * `source`

3. **`policy_events.csv`**

   * `date`
   * `country`
   * `title`
   * `category` (export\_control / subsidy / expectation / other)
   * `impact_direction` (up / down / mixed)
   * `notes`

4. **`equity_prices.csv`**

   * `date`
   * `ticker` (NVDA, AMD, ...)
   * `close` (float)
   * `source`

---

## Using Real Data

* Replace the demo CSVs in `data/` with real datasets (e.g., scraped GPU prices, analyst shipment reports, official export control dates).
* Ensure the format matches the schemas above.
* Restart the app.

---

## Notes

* This dashboard is for **educational and visualization purposes** only.
* Demo data is **synthetic** and does not reflect actual GPU prices, shipments, or financials.
* Not intended as financial or investment advice.

---

## License

MIT License
