from pathlib import Path
import pandas as pd
import requests

DATA_DIR = Path("data")
RAW_PATH = DATA_DIR / "raw" / "store_item_sales.csv"

# TODO: replace this with the actual raw GitHub CSV URL you choose
SOURCE_URL = "https://raw.githubusercontent.com/jgonzalezab/Store-Item-Demand-Forecasting/master/Data/train.csv"


def _ensure_raw_data(path: Path = RAW_PATH) -> Path:
    path = Path(path)

    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)

    resp = requests.get(SOURCE_URL, timeout=30)
    resp.raise_for_status()

    with path.open("wb") as f:
        f.write(resp.content)

    return path


def load_sales_data(path: str | Path = RAW_PATH) -> pd.DataFrame:
    """
    Load the raw store-item daily sales dataset.

    Expected columns (after you inspect the actual file):
    - date
    - store
    - item
    - sales
    """
    path = Path(path)
    path = _ensure_raw_data(path)

    df = pd.read_csv(path, parse_dates=["date"])
    return df


if __name__ == "__main__":
    df = load_sales_data()
    print(df.head())
    print(df.dtypes)
    print(df.shape)
