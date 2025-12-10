import numpy as np
import h5py
from pathlib import Path
import polars as pl


DATA_DIR = Path("/data/share/data")

# TODO: add caching (reentrant)
def load_data(category: str, field: str, time_and_ticker=False):
    path = DATA_DIR / category / f"{field}.h5"
    with h5py.File(path, "r") as f:
        time = f["time"][:]
        ticker = f["ticker"][:]
        values = f["values"][:]
        
    if time.shape[0] > 3643:
        time = time[:3643]
        values = values[:3643, :]
    assert time.shape == (3643,)
    assert ticker.shape == (5480,)
    
    if time_and_ticker:
        return time, ticker, values
    else:
        return values


DATA_RAW_DIR = Path("/data/share/data_raw")

INTRADAY_PERIODS = [
    "10:00:00",  # Period 1: 09:30-10:00
    "10:30:00",  # Period 2: 10:00-10:30
    "11:00:00",  # Period 3: 10:30-11:00
    "11:30:00",  # Period 4: 11:00-11:30
    "13:30:00",  # Period 5: 13:00-13:30 (after lunch break)
    "14:00:00",  # Period 6: 13:30-14:00
    "14:30:00",  # Period 7: 14:00-14:30
    "15:00:00",  # Period 8: 14:30-15:00 (market close)
]

def load_intraday_data(date: str) -> "pl.DataFrame":
    """Load intraday 30-min bar data for a single day.

    Parameters
    ----------
    date : str
        Date in format 'YYYY-MM-DD'

    Returns
    -------
    pl.DataFrame with columns: open, close, high, low, volume, order_book_id, datetime
    """

    path = DATA_RAW_DIR / "intra_30min" / f"{date}.parquet"
    return pl.read_parquet(path)


def compute_intraday_returns(df: "pl.DataFrame") -> "pl.DataFrame":
    """Compute returns for each intraday period.

    Returns
    -------
    DataFrame with columns: order_book_id, period, return
    where period is 1-8 for intraday periods
    """

    # Sort by ticker and time
    df = df.sort(["order_book_id", "datetime"])

    # Compute return = (close - open) / open for each bar
    df = df.with_columns([
        ((pl.col("close") - pl.col("open")) / pl.col("open")).alias("return"),
        pl.col("datetime").dt.time().cast(pl.Utf8).alias("time_str"),
    ])

    # Map time to period number
    time_to_period = {t: i + 1 for i, t in enumerate(INTRADAY_PERIODS)}

    df = df.with_columns([
        pl.col("time_str").replace(time_to_period, default=0).alias("period")
    ])

    return df.select(["order_book_id", "period", "return", "datetime"])