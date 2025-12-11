from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults

from .data import load_sales_data
from .transform import make_daily_series

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "daily_sarima.pkl"


def train_daily_sarima(daily: pd.DataFrame, forecast_horizon: int = 180):
    """
    Train a SARIMA model on log-transformed daily total_sales and evaluate on a hold-out period.

    Returns:
        fitted_model, metrics (dict), test_df (with actuals + forecast)
    """
    daily = daily.copy()
    daily = daily.sort_values("date")
    daily = daily.set_index("date")
    y = daily["total_sales"].asfreq("D")

    # Log-transform to stabilize variance
    y_log = np.log1p(y)

    # Train/test split on the transformed series
    train_y_log = y_log.iloc[:-forecast_horizon]
    test_y_log = y_log.iloc[-forecast_horizon:]

    model = SARIMAX(
        train_y_log,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    results = model.fit(disp=False)

    # Forecast on log scale
    forecast_log = results.forecast(steps=forecast_horizon)
    forecast_log.index = test_y_log.index

    # Inverse transform back to original scale
    forecast = np.expm1(forecast_log)
    test_actual = np.expm1(test_y_log)

    test_df = pd.DataFrame(
        {
            "actual": test_actual,
            "forecast": forecast,
        }
    )

    mae = (test_df["forecast"] - test_df["actual"]).abs().mean()
    mape = (
        (test_df["forecast"] - test_df["actual"])
        .abs()
        .div(test_df["actual"])
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .mean()
        * 100
    )

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
