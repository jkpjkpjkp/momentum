import numpy as np
import h5py
from pathlib import Path
import polars as pl


_TOP_1000_MASK: np.ndarray | None = None
_TOP_1000_TICKERS: set[bytes] | None = None
def get_top_1000_mask(reference_date: str = "2020-01-04") -> np.ndarray:
    """Get boolean mask for top 1000 stocks by market cap at reference date.

    Returns mask of shape (5480,) where True = in top 1000.
    """
    global _TOP_1000_MASK
    if _TOP_1000_MASK is not None:
        return _TOP_1000_MASK

    time, ticker, close = load_data("daily", "close", time_and_ticker=True, top1000=False)
    shares = load_data("shares", "total_a", top1000=False)
    market_cap = close * shares

    # Find reference date index
    from datetime import datetime
    dates_ns = {
        datetime.fromtimestamp(t / 1e9).strftime("%Y-%m-%d"): i
        for i, t in enumerate(time)
    }

    if reference_date not in dates_ns:
        # Find nearest date
        available = sorted(dates_ns.keys())
        for d in available:
            if d >= reference_date:
                reference_date = d
                break
        else:
            reference_date = available[-1]

    t_idx = dates_ns[reference_date]
    mcap_at_ref = market_cap[t_idx, :]

    # Get indices of top 1000 by market cap (ignoring NaN)
    valid_mcap = np.where(np.isnan(mcap_at_ref), -np.inf, mcap_at_ref)
    top_indices = np.argsort(valid_mcap)[-1000:]

    mask = np.zeros(5480, dtype=bool)
    mask[top_indices] = True

    _TOP_1000_MASK = mask
    print(f"Filtered to top 1000 stocks by market cap at {reference_date}")
    return _TOP_1000_MASK


def get_top_1000_tickers() -> set[bytes]:
    """Get set of ticker bytes for top 1000 stocks."""
    global _TOP_1000_TICKERS
    if _TOP_1000_TICKERS is not None:
        return _TOP_1000_TICKERS

    _, ticker, _ = load_data("daily", "close", time_and_ticker=True)
    mask = get_top_1000_mask()
    _TOP_1000_TICKERS = set(ticker[mask])
    return _TOP_1000_TICKERS


DATA_DIR = Path("/data/share/data")
def load_data(category: str, field: str, time_and_ticker=False, top1000=True):
    """Load data, filtered to top 1000 stocks by market cap."""
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

    if top1000:
        mask = get_top_1000_mask()
        ticker = ticker[mask]
        values = values[:, mask]

    if time_and_ticker:
        return time, ticker, values
    else:
        return values


DATA_RAW_DIR = Path("/data/share/data_raw")
def load_intraday_data(date: str) -> "pl.DataFrame":
    path = DATA_RAW_DIR / "intra_30min" / f"{date}.parquet"
    return pl.read_parquet(path)


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
def compute_intraday_returns(df: "pl.DataFrame") -> "pl.DataFrame":
    """Compute returns for each intraday period.

    Returns
    -------
    DataFrame with columns: order_book_id, period, return
    where period is 1-8 for intraday periods
    """

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