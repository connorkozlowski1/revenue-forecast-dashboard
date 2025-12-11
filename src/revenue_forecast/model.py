from pathlib import Path

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults

from .data import load_sales_data
from .transform import make_daily_series

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "daily_sarima.pkl"


def train_daily_sarima(daily: pd.DataFrame, forecast_horizon: int = 180):
    """
    Train a SARIMA model on daily total_sales and evaluate on a hold-out period.

    Returns:
        fitted_model, metrics (dict), test_df (with actuals + forecast)
    """
    daily = daily.copy()
    daily = daily.sort_values("date")
    daily = daily.set_index("date")
    y = daily["total_sales"].asfreq("D")

    # Define train/test split: last `forecast_horizon` days as test
    train_y = y.iloc[:-forecast_horizon]
    test_y = y.iloc[-forecast_horizon:]

    # Simple baseline SARIMA config
    model = SARIMAX(
        train_y,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    results = model.fit(disp=False)

    # Forecast over the test horizon
    forecast = results.forecast(steps=forecast_horizon)
    forecast.index = test_y.index

    test_df = pd.DataFrame(
        {
            "actual": test_y,
            "forecast": forecast,
        }
    )

    mae = (test_df["forecast"] - test_df["actual"]).abs().mean()
    mape = (test_df["forecast"] - test_df["actual"]).abs().div(test_df["actual"]).mean() * 100

    metrics = {
        "mae": float(mae),
        "mape": float(mape),
        "horizon_days": int(forecast_horizon),
    }

    return results, metrics, test_df


def save_model(results: SARIMAXResults, path: Path = MODEL_PATH):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    results.save(path)


def load_model(path: Path = MODEL_PATH) -> SARIMAXResults:
    return SARIMAXResults.load(path)
