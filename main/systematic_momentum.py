"""
Systematic Momentum - Li, Yuan & Zhou (2025)

Methodology:
1. Cross-sectional regression at each period i:
   RET_s,d,i = alpha_d,i + sum_j(C_s,d-1,j * theta_d,i,j) + epsilon_s,d,i
   where C_s,d-1,j are standardized anomaly characteristics from day d-1
2. SYS_s,d,i = sum_j(C_s,d-1,j * theta_hat_d,i,j) is the systematic component
3. Trading: At each period i, sort stocks by SYS from period i-1,
   go long top decile, short bottom decile
"""
import numpy as np
from typing import Callable
import polars as pl
from datetime import datetime

from the_short_of_it import anomalies as short_anomalies
from factor_momentum import anomalies as fm_anomalies
from .utils import (
    load_data,
    load_intraday_data,
    compute_intraday_returns,
    INTRADAY_PERIODS,
    DATA_RAW_DIR,
)

N_INTRADAY_PERIODS = len(INTRADAY_PERIODS)


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


def load_all_anomalies(min_valid_pct: float = 0.10) -> tuple[np.ndarray, list[str]]:
    """Load and filter anomalies, removing those with insufficient valid data."""
    all_funcs = []
    all_names = []

    print("Loading the_short_of_it anomalies...")
    for func in short_anomalies:
        all_funcs.append(func)
        all_names.append(f"short_{func.__name__}")

    print("Loading factor_momentum anomalies...")
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


def get_available_dates() -> list[str]:
    """Get list of available dates with intraday data."""
    intraday_dir = DATA_RAW_DIR / "intra_30min"
    dates = sorted([f.stem for f in intraday_dir.glob("*.parquet")])
    return dates


def compute_intraday_systematic_momentum_by_period(
    start_date: str = "2020-01-01",
    end_date: str = "2024-12-31",
    n_portfolios: int = 10,
) -> dict:
    """
    Systematic momentum with period-specific regressions.

    Following Li, Yuan & Zhou (2025): run separate cross-sectional regression
    for each intraday period i, as factor returns (theta) vary by time-of-day.

    RET_s,d,i = alpha_d,i + sum_j(C_s,d-1,j * theta_d,i,j) + epsilon_s,d,i
    """
    print("=" * 70)
    print("INTRADAY SYSTEMATIC MOMENTUM - Period-Specific Regressions")
    print("Following Li, Yuan & Zhou (2025)")
    print("=" * 70)

    # Load daily characteristics
    print("\n1. Loading daily data...")
    time, ticker, _ = load_data("daily", "return", time_and_ticker=True)

    # Create date lookup from nanosecond timestamps
    dates_ns = {
        datetime.fromtimestamp(t / 1e9).strftime("%Y-%m-%d"): i
        for i, t in enumerate(time)
    }
    assert len(dates_ns) == len(time)
    ticker_to_idx = {t: i for i, t in enumerate(ticker)}

    print(f"   Daily data: {len(time)} days, {len(ticker)} stocks")

    print("\n2. Loading anomaly characteristics...")
    anomalies, anomaly_names = load_all_anomalies()
    n_factors = len(anomaly_names)
    print(f"   {n_factors} anomalies loaded")

    print("\n3. Loading intraday dates...")
    all_dates = get_available_dates()
    dates = [d for d in all_dates if start_date <= d <= end_date]
    print(f"   {len(dates)} trading days in range")

    # Storage for period-specific thetas: thetas_by_period[period][date] = theta
    thetas_by_period = {p: {} for p in range(1, N_INTRADAY_PERIODS + 1)}
    portfolio_returns_by_period = {p: [] for p in range(1, N_INTRADAY_PERIODS + 1)}

    print("\n4. Running period-specific cross-sectional regressions...")

    for date_idx, date in enumerate(dates):
        if date_idx % 50 == 0:
            print(f"   Processing {date} ({date_idx+1}/{len(dates)})")

        # Get previous trading day index for characteristics
        if date not in dates_ns:
            continue
        t_idx = dates_ns[date]
        if t_idx == 0:
            continue
        char_t_idx = t_idx - 1  # Lagged characteristics

        df = load_intraday_data(date)
        ret_df = compute_intraday_returns(df)

        # Run regression for each period separately
        for period in range(1, N_INTRADAY_PERIODS + 1):
            period_data = ret_df.filter(pl.col("period") == period)
            if period_data.height == 0:
                continue

            tickers_period = period_data["order_book_id"].to_list()
            returns_period = period_data["return"].to_numpy()

            # Build X (characteristics) and y (returns) matrices
            y_list = []
            X_list = []

            for i, tk in enumerate(tickers_period):
                tk_bytes = tk if isinstance(tk, bytes) else tk.encode()
                s_idx = ticker_to_idx[tk_bytes]
                char_values = anomalies[char_t_idx, s_idx, :]

                if np.any(np.isnan(char_values)) or np.isnan(returns_period[i]):
                    continue

                y_list.append(returns_period[i])
                X_list.append(char_values)

            if len(y_list) < n_factors + 10:
                continue

            y = np.array(y_list)
            X = np.array(X_list)

            # OLS: y = alpha + X @ theta + epsilon
            X_with_intercept = np.column_stack([np.ones(len(y)), X])

            XtX = X_with_intercept.T @ X_with_intercept
            Xty = X_with_intercept.T @ y
            reg = 1e-8 * np.eye(XtX.shape[0])
            coeffs = np.linalg.solve(XtX + reg, Xty)
            thetas_by_period[period][date] = coeffs[1:]

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

            tickers_period = period_data["order_book_id"].to_list()
            returns_period = period_data["return"].to_numpy()

            # Compute SYS for all stocks
            current_sys = {}
            for i, tk in enumerate(tickers_period):
                tk_str = tk.decode() if isinstance(tk, bytes) else tk
                tk_bytes = tk if isinstance(tk, bytes) else tk.encode()

                if tk_bytes not in ticker_to_idx:
                    continue

                s_idx = ticker_to_idx[tk_bytes]
                char_values = anomalies[char_t_idx, s_idx, :]

                if not np.any(np.isnan(char_values)):
                    current_sys[tk_str] = np.dot(char_values, theta)

            # Trading: sort by previous period's SYS
            if prev_sys is not None and len(prev_sys) > n_portfolios * 5:
                valid_stocks = []
                for i, tk in enumerate(tickers_period):
                    tk_str = tk.decode() if isinstance(tk, bytes) else tk
                    if tk_str in prev_sys and not np.isnan(returns_period[i]):
                        valid_stocks.append((tk_str, prev_sys[tk_str], returns_period[i]))

                if len(valid_stocks) >= n_portfolios * 5:
                    valid_stocks.sort(key=lambda x: x[1])

                    n_per_port = len(valid_stocks) // n_portfolios
                    port_ret = []
                    for p in range(n_portfolios):
                        start_idx = p * n_per_port
                        end_idx = (
                            start_idx + n_per_port
                            if p < n_portfolios - 1
                            else len(valid_stocks)
                        )
                        stocks_in_port = valid_stocks[start_idx:end_idx]
                        avg_ret = np.mean([s[2] for s in stocks_in_port])
                        port_ret.append(avg_ret)

                    portfolio_returns_by_period[period].append({
                        "date": date,
                        "portfolio_returns": port_ret,
                        "long_short": port_ret[-1] - port_ret[0],
                    })

            prev_sys = current_sys

    # Print results
    print("\n" + "=" * 70)
    print("SYSTEMATIC MOMENTUM RESULTS BY INTRADAY PERIOD")
    print("=" * 70)
    print(f"{'Period':<20} {'N Days':>8} {'Ann Ret':>10} {'Ann Vol':>10} {'Sharpe':>8}")
    print("-" * 70)

    all_ls_returns = []

    for period in range(1, N_INTRADAY_PERIODS + 1):
        port_rets = portfolio_returns_by_period[period]
        if not port_rets:
            print(f"{INTRADAY_PERIODS[period-1]:<20} {'N/A':>8}")
            continue

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

    # Aggregate statistics
    if all_ls_returns:
        print("-" * 70)
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
        start_date="2020-01-01",
        end_date="2024-12-31",
    )
