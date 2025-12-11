import pandas as pd
from pathlib import Path
from .data import load_sales_data


def make_daily_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate item-level store sales into a single daily revenue time series.
    
    Output:
        date | total_sales
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    daily = (
        df.groupby("date")["sales"]
        .sum()
        .reset_index()
        .sort_values("date")
    )

    daily.rename(columns={"sales": "total_sales"}, inplace=True)
    return daily


def make_store_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate into per-store daily series:
    
        date | store | total_sales
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    store_daily = (
        df.groupby(["date", "store"])["sales"]
        .sum()
        .reset_index()
        .sort_values(["store", "date"])
    )

    store_daily.rename(columns={"sales": "total_sales"}, inplace=True)
    return store_daily


if __name__ == "__main__":
    raw = load_sales_data()
    daily = make_daily_series(raw)

    print("Daily series preview:")
    print(daily.head())
    print(daily.tail())
    print("Shape:", daily.shape)
