# Revenue Forecast Dashboard

Interactive time series dashboard for forecasting daily store revenue (sales) using a SARIMA model and Streamlit.

The project demonstrates:

- Automatic dataset download from a public GitHub mirror
- Data aggregation from item-level sales to daily total revenue
- Time series modeling with SARIMA
- Saved model artifacts (local only, not committed)
- Interactive Streamlit dashboard with adjustable forecast horizon

---

## 1. Problem Overview

Retail and e-commerce businesses often need to forecast future revenue to plan inventory, staffing, and marketing.

This project:

- Uses a public Store Item Demand Forecasting dataset (daily sales by store and item)
- Aggregates sales to a single daily revenue series
- Trains a SARIMA model on historical data
- Provides an interactive dashboard to visualize history and forecast future daily revenue

---

## 2. Tech Stack

- Python 3.13
- pandas, numpy, statsmodels (SARIMAX)
- Streamlit
- requests, pathlib

---

## 3. Data Source

The raw dataset is automatically downloaded from a public GitHub mirror of the Store Item Demand Forecasting Challenge (originally Kaggle).

Columns:

- date (daily timestamps)
- store (store ID)
- item (item ID)
- sales (units sold)

The project aggregates sales across stores and items to create:

- date
- total_sales

Dataset and model artifacts are not committed to the repository. They are downloaded/generated at runtime.

---

## 4. Project Structure

revenue-forecast-dashboard/
├── dashboards/
│   └── app.py                    # Streamlit dashboard
├── src/
│   └── revenue_forecast/
│       ├── __init__.py
│       ├── data.py               # Auto-download + load raw store-item sales
│       ├── transform.py          # Aggregate to daily total_sales time series
│       ├── model.py              # SARIMA training + save/load
│       └── train.py              # Training entrypoint with hold-out evaluation
├── data/                         # (ignored) downloaded raw data
├── models/                       # (ignored) trained model artifacts
├── .gitignore                    # excludes data, models, artifacts
├── requirements.txt
└── README.md

---

## 5. Setup

git clone https://github.com/connorkozlowski1/revenue-forecast-dashboard.git
cd revenue-forecast-dashboard

python -m venv .venv
# Windows:
.venv\Scripts\activate

pip install -r requirements.txt

---

## 6. Data Ingestion and Transformation

### Load and inspect the raw data

python -m src.revenue_forecast.data

This will:

- Download the CSV from the configured GitHub URL (if not already present)
- Save it under data/raw/
- Print the head, dtypes, and shape

### Build the daily revenue series

python -m src.revenue_forecast.transform

This aggregates item-level store sales to:

- date
- total_sales

and prints a preview.

---

## 7. Training the Forecasting Model

Train the SARIMA model on the daily total_sales series and evaluate the last 180 days:

python -m src.revenue_forecast.train

This script:

- Loads raw data
- Aggregates to daily total_sales
- Splits train/test (last 180 days)
- Fits SARIMA: order=(1,1,1), seasonal_order=(1,1,1,7)
- Computes MAE and MAPE
- Saves the model to models/daily_sarima.pkl (ignored by git)

---

## 8. Running the Dashboard

Start the Streamlit dashboard:

streamlit run dashboards/app.py

Open the URL printed in the terminal (usually http://localhost:8501).

The dashboard shows:

- Historical total_sales
- Slider for forecast horizon (30–365 days)
- Forecast plot
- Table of predicted values

---

## 9. Notes and Possible Extensions

Potential improvements:

- Add Prophet or auto-ARIMA models
- Allow model comparisons
- Add store/item filtering options
- Add confidence intervals
- Deploy dashboard to Streamlit Cloud or Render

---

## 10. License

MIT License.
