"""
Systematic Momentum - Li, Yuan & Zhou (2025)
"""
import numpy as np
from typing import Callable
import polars as pl
from datetime import datetime
from tqdm import tqdm

from the_short_of_it import anomalies as short_anomalies
from factor_momentum import anomalies as fm_anomalies
from .utils import (
    load_data,
    load_intraday_data,
    get_available_dates,
)


def to_bytes_list(tickers: list) -> list[bytes]:
    """Convert list of tickers to bytes (parquet gives str, h5 gives bytes)."""
    return [tk.encode() if isinstance(tk, str) else tk for tk in tickers]


def build_regression_matrices(
    tickers: list[bytes],
    returns: np.ndarray,
    ticker_to_idx: dict[bytes, int],
    anomalies: np.ndarray,
    char_t_idx: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Build X (characteristics) and y (returns) matrices for regression."""
    y_list = []
    X_list = []

    for i, tk in enumerate(tickers):
        if tk not in ticker_to_idx:
            continue
        s_idx = ticker_to_idx[tk]
        char_values = anomalies[char_t_idx, s_idx, :]

        if np.any(np.isnan(char_values)) or np.isnan(returns[i]):
            continue

        y_list.append(returns[i])
        X_list.append(char_values)

    if len(y_list) == 0:
        return None, None

    return np.array(y_list), np.array(X_list)


def fit_ols_with_intercept(X: np.ndarray, y: np.ndarray, reg_lambda: float = 1e-8) -> np.ndarray:
    """Fit OLS with intercept and regularization, return coefficients (excluding intercept)."""
    X_aug = np.column_stack([np.ones(len(y)), X])
    XtX = X_aug.T @ X_aug
    Xty = X_aug.T @ y
    reg = reg_lambda * np.eye(XtX.shape[0])
    coeffs = np.linalg.solve(XtX + reg, Xty)
    return coeffs[1:]


def compute_sys_scores(
    tickers: list[bytes],
    ticker_to_idx: dict[bytes, int],
    anomalies: np.ndarray,
    char_t_idx: int,
    theta: np.ndarray,
) -> dict[bytes, float]:
    """Compute SYS scores for all stocks: SYS = characteristics @ theta."""
    sys_scores = {}
    for tk in tickers:
        if tk not in ticker_to_idx:
            continue
        s_idx = ticker_to_idx[tk]
        char_values = anomalies[char_t_idx, s_idx, :]
        if not np.any(np.isnan(char_values)):
            sys_scores[tk] = np.dot(char_values, theta)
    return sys_scores


def compute_portfolio_returns(
    tickers: list[bytes],
    returns: np.ndarray,
    scores: dict[bytes, float],
    n_portfolios: int,
) -> list[float] | None:
    """Sort stocks by scores into portfolios and compute average returns."""
    valid_stocks = []
    for i, tk in enumerate(tickers):
        if tk in scores and not np.isnan(returns[i]):
            valid_stocks.append((tk, scores[tk], returns[i]))

    if len(valid_stocks) < n_portfolios * 5:
        return None

    valid_stocks.sort(key=lambda x: x[1])

    n_per_port = len(valid_stocks) // n_portfolios
    port_ret = []
    for p in range(n_portfolios):
        start_idx = p * n_per_port
        end_idx = start_idx + n_per_port if p < n_portfolios - 1 else len(valid_stocks)
        stocks_in_port = valid_stocks[start_idx:end_idx]
        avg_ret = np.mean([s[2] for s in stocks_in_port])
        port_ret.append(avg_ret)

    return port_ret


def cross_sectional_standardize(values: np.ndarray) -> np.ndarray:
    """Standardize values cross-sectionally (each row has mean=0, std=1)."""
    mean = np.nanmean(values, axis=1, keepdims=True)
    std = np.nanstd(values, axis=1, keepdims=True)
    result = (values - mean) / std

    n_valid = np.sum(~np.isnan(values), axis=1)
    invalid_rows = (n_valid < 10) | (std.squeeze() <= 1e-10)
    result[invalid_rows, :] = np.nan

    return result


def load_anomaly_set(anomaly_list: list[Callable]) -> np.ndarray:
    results = []
    for j, anomaly_func in enumerate(anomaly_list):
        print(f"  Loading anomaly {j+1}/{len(anomaly_list)}: {anomaly_func.__name__}")
        x = anomaly_func()
        x = cross_sectional_standardize(x)
        results.append(x)
    return np.stack(results, axis=2)


def load_all_anomalies(min_valid_pct: float = 0.3) -> tuple[np.ndarray, list[str]]:
    """Load and filter anomalies, removing those with insufficient valid data."""
    all_funcs = []
    all_names = []

    for func in short_anomalies:
        all_funcs.append(func)
        all_names.append(f"short_{func.__name__}")
    for func in fm_anomalies:
        all_funcs.append(func)
        all_names.append(f"fm_{func.__name__}")

    anomalies = load_anomaly_set(all_funcs)

    n_factors = anomalies.shape[2]
    valid_mask = []
    filtered_names = []

    for j in range(n_factors):
        valid_pct = np.sum(~np.isnan(anomalies[:, :, j])) / anomalies[:, :, j].size
        if valid_pct >= min_valid_pct:
            valid_mask.append(j)
            filtered_names.append(all_names[j])
        else:
            print(f"   Excluding {all_names[j]}: only {valid_pct*100:.1f}% valid data")

    if len(valid_mask) < n_factors:
        anomalies = anomalies[:, :, valid_mask]
        print(f"   Kept {len(valid_mask)}/{n_factors} anomalies after filtering")

    return anomalies, filtered_names


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
N_INTRADAY_PERIODS = len(INTRADAY_PERIODS)
def compute_intraday_returns(df) -> "pl.DataFrame":
    df = df.sort("order_book_id", "datetime")

    time_to_period = {t: i + 1 for i, t in enumerate(INTRADAY_PERIODS)}
    df = df.with_columns(
        (pl.col("close") / pl.col("open") - 1).alias("ret"),
        pl.col("datetime").dt.time().cast(pl.Utf8).replace(time_to_period).alias("period")
    )

    return df.select(["order_book_id", "period", "ret", "datetime"])


def compute_intraday_systematic_momentum_by_period(
    start_date: str = "2020-01-01",
    end_date: str = "2024-12-31",
    n_portfolios: int = 10,
) -> dict:
    time, ticker, _ = load_data("daily", "close", time_and_ticker=True)
    dates_ns = {
        datetime.fromtimestamp(t / 1e9).strftime("%Y-%m-%d"): i
        for i, t in enumerate(time)
    }
    ticker_to_idx = {t: i for i, t in enumerate(ticker)}

    anomalies, anomaly_names = load_all_anomalies()
    n_factors = len(anomaly_names)
    all_dates = get_available_dates()
    dates = [d for d in all_dates if start_date <= d <= end_date]

    # Storage for period-specific thetas: thetas_by_period[period][date] = theta
    thetas_by_period = {p: {} for p in range(1, N_INTRADAY_PERIODS + 1)}
    portfolio_returns_by_period = {p: [] for p in range(1, N_INTRADAY_PERIODS + 1)}

    print("\n4. Running period-specific cross-sectional regressions...")

    for date_idx, date in tqdm(enumerate(dates)):
        # Get previous trading day index for characteristics
        assert date in dates_ns, f"Date {date} not found in daily data"
        t_idx = dates_ns[date]
        if t_idx == 0:
            continue
        char_t_idx = t_idx - 1  # Lagged characteristics

        df = load_intraday_data(date)
        df = compute_intraday_returns(df)

        # Run regression for each period separately
        for period in range(1, N_INTRADAY_PERIODS + 1):
            period_data = df.filter(pl.col("period") == period)

            tickers_period = to_bytes_list(period_data["order_book_id"].to_list())
            returns_period = period_data["ret"].to_numpy()

            y, X = build_regression_matrices(
                tickers_period, returns_period, ticker_to_idx, anomalies, char_t_idx
            )

            if y is None or len(y) < n_factors + 10:
                continue

            thetas_by_period[period][date] = fit_ols_with_intercept(X, y)

    print("\n5. Computing SYS and creating portfolios...")
    # Trading
    # use SYS i-1 to predict return i
    for date_idx, date in enumerate(dates[1:], 1):
        prev_date = dates[date_idx - 1]

        if date not in dates_ns:
            continue
        t_idx = dates_ns[date]
        if t_idx == 0:
            continue
        char_t_idx = t_idx - 1

        df = load_intraday_data(date)
        ret_df = compute_intraday_returns(df)

        prev_sys = None  # SYS from previous period

        for period in range(1, N_INTRADAY_PERIODS + 1):
            # Get theta for this period (use yesterday's estimate)
            if prev_date not in thetas_by_period[period]:
                if date not in thetas_by_period[period]:
                    continue
                theta = thetas_by_period[period][date]
            else:
                theta = thetas_by_period[period][prev_date]

            period_data = ret_df.filter(pl.col("period") == period)
            if period_data.height == 0:
                continue

            tickers_period = to_bytes_list(period_data["order_book_id"].to_list())
            returns_period = period_data["ret"].to_numpy()

            current_sys = compute_sys_scores(
                tickers_period, ticker_to_idx, anomalies, char_t_idx, theta
            )

            # Trading: sort by previous period's SYS
            if prev_sys is not None and len(prev_sys) > n_portfolios * 5:
                port_ret = compute_portfolio_returns(
                    tickers_period, returns_period, prev_sys, n_portfolios
                )
                if port_ret is not None:
                    portfolio_returns_by_period[period].append({
                        "date": date,
                        "portfolio_returns": port_ret,
                        "long_short": port_ret[-1] - port_ret[0],
                    })

            prev_sys = current_sys

    # Print results
    print("SYSTEMATIC MOMENTUM RESULTS BY INTRADAY PERIOD")
    print(f"{'Period':<20} {'N Days':>8} {'Ann Ret':>10} {'Ann Vol':>10} {'Sharpe':>8}")

    all_ls_returns = []

    for period in range(1, N_INTRADAY_PERIODS + 1):
        port_rets = portfolio_returns_by_period[period]

        ls_returns = [pr["long_short"] for pr in port_rets]
        all_ls_returns.extend(ls_returns)

        # Annualize (8 periods per day, 252 days per year)
        ann_factor = 252 * N_INTRADAY_PERIODS
        ls_mean = np.mean(ls_returns) * ann_factor
        ls_std = np.std(ls_returns) * np.sqrt(ann_factor)
        ls_sharpe = ls_mean / ls_std if ls_std > 0 else np.nan

        print(
            f"{INTRADAY_PERIODS[period-1]:<20} {len(port_rets):>8} "
            f"{ls_mean*100:>9.2f}% {ls_std*100:>9.2f}% {ls_sharpe:>8.2f}"
        )

    ann_factor = 252 * N_INTRADAY_PERIODS
    agg_mean = np.mean(all_ls_returns) * ann_factor
    agg_std = np.std(all_ls_returns) * np.sqrt(ann_factor)
    agg_sharpe = agg_mean / agg_std if agg_std > 0 else np.nan
    print(
        f"{'AGGREGATE':<20} {len(all_ls_returns):>8} "
        f"{agg_mean*100:>9.2f}% {agg_std*100:>9.2f}% {agg_sharpe:>8.2f}"
    )

    return {
        "thetas_by_period": thetas_by_period,
        "portfolio_returns_by_period": portfolio_returns_by_period,
        "anomaly_names": anomaly_names,
    }


if __name__ == "__main__":
    results = compute_intraday_systematic_momentum_by_period(
        start_date="2018-01-01",
        end_date="2022-12-31",
    )
