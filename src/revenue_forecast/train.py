from pathlib import Path

from .data import load_sales_data
from .transform import make_daily_series
from .model import train_daily_sarima, save_model, MODEL_PATH


def main():
    print("Loading raw sales data...")
    raw = load_sales_data()
    daily = make_daily_series(raw)

    print(f"Daily series shape: {daily.shape}")
    print(f"Date range: {daily['date'].min()} to {daily['date'].max()}")

    print("\nTraining SARIMA model on daily total_sales...")
    model, metrics, test_df = train_daily_sarima(daily, forecast_horizon=180)

    print("\n=== HOLD-OUT METRICS (last 180 days) ===")
    print(f"MAE : {metrics['mae']:.2f}")
    print(f"MAPE: {metrics['mape']:.2f}%")

    save_model(model, MODEL_PATH)
    print(f"\nModel saved to: {Path(MODEL_PATH).resolve()}")

    print("\nTest sample (tail):")
    print(test_df.tail())


if __name__ == "__main__":
    main()
