import sys
import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
import pandas as pd
import streamlit as st
from pathlib import Path

from src.revenue_forecast.data import load_sales_data
from src.revenue_forecast.transform import make_daily_series
from src.revenue_forecast.model import load_model


MODEL_PATH = Path("models") / "daily_sarima.pkl"


st.title("📈 Revenue Forecast Dashboard")
st.write("Interactive forecasting tool using SARIMA on daily sales data.")


@st.cache_data
def load_data():
    raw = load_sales_data()
    daily = make_daily_series(raw)
    return daily


@st.cache_resource
def load_forecasting_model():
    model = load_model(MODEL_PATH)
    return model


# Load data + model
daily = load_data()
model = load_forecasting_model()

st.subheader("Historical Daily Sales")
st.line_chart(daily.set_index("date")["total_sales"])

# Forecast horizon selector
h = st.slider("Forecast Horizon (Days)", min_value=30, max_value=365, value=90, step=30)

# Forecast
forecast = model.forecast(steps=h)

forecast_df = pd.DataFrame({
    "date": pd.date_range(start=daily["date"].max() + pd.Timedelta(days=1), periods=h),
    "forecast": forecast.values
})

st.subheader("Forecast")
st.line_chart(forecast_df.set_index("date")["forecast"])

st.write("Preview of forecasted values:")
st.dataframe(forecast_df.head())
