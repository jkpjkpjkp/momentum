"""Systematic Momentum - Li, Yuan & Zhou (2025)

Methodology (adapted for Chinese market with 8 intraday periods):
1. Intraday periods: 8 half-hour intervals (10:00-15:00 with lunch break)
   Plus overnight period (previous close to 10:00 open)
2. Cross-sectional regression at each period i:
   RET_s,d,i = alpha_d,i + sum_j(C_s,d-1,j * theta_d,i,j) + epsilon_s,d,i
   where C_s,d-1,j are standardized anomaly characteristics from day d-1
3. SYS_s,d,i = sum_j(C_s,d-1,j * theta_hat_d,i,j) is the systematic component
4. Trading: At each period i, sort stocks by SYS from period i-1,
   go long top decile, short bottom decile

Uses two anomaly sets:
- the_short_of_it: 11 anomalies (financial distress, momentum, profitability, etc.)
- factor_momentum: 15 anomalies (Ehsani & Linnainmaa factors)
"""

import numpy as np
from typing import Callable
import polars as pl

# Import anomaly sets
from the_short_of_it import anomalies as short_anomalies
from factor_momentum import anomalies as fm_anomalies
from .utils import (
    load_data,
    load_intraday_data,
    compute_intraday_returns,
    INTRADAY_PERIODS,
    N_INTRADAY_PERIODS,
    DATA_RAW_DIR,
)


def cross_sectional_standardize(values: np.ndarray) -> np.ndarray:
    """Standardize values cross-sectionally (each row has mean=0, std=1)."""
    result = np.full_like(values, np.nan, dtype=np.float64)

    for t in range(values.shape[0]):
        row = values[t, :]
        valid = ~np.isnan(row)
        n_valid = valid.sum()

        if n_valid < 10:  # Need minimum observations
            continue

        mu = np.nanmean(row)
        sigma = np.nanstd(row)

        if sigma > 1e-10:
            result[t, valid] = (row[valid] - mu) / sigma

    return result


def load_anomaly_set(
    anomaly_list: list[Callable],
) -> np.ndarray:
    n_anomalies = len(anomaly_list)

    results = []
    for j, anomaly_func in enumerate(anomaly_list):
        print(f"  Loading anomaly {j+1}/{n_anomalies}: {anomaly_func.__name__}")

        x = anomaly_func()
        x = cross_sectional_standardize(x)
        results.append(x)

    return np.stack(results, axis=2)


def load_all_anomalies() -> tuple[np.ndarray, list[str]]:
    """
    Load all anomalies from selected sets.

    Parameters
    ----------
    reference_time : np.ndarray
        Daily time grid
    reference_ticker : np.ndarray
        Stock ticker array

    Returns
    -------
    anomalies : np.ndarray
        (N_time, N_stocks, N_anomalies) standardized anomaly values
    names : list[str]
        Names of anomalies
    """
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

    return anomalies, all_names


def run_cross_sectional_regression(
    returns: np.ndarray,
    characteristics: np.ndarray,
    use_inverse_variance_weights: bool = True,
):
    """
    Run cross-sectional regression for each time period.

    RET_s,d = alpha_d + sum_j(C_s,d-1,j * theta_d,j) + epsilon_s,d

    Following the paper, we use lagged characteristics (d-1) to predict returns (d).

    Parameters
    ----------
    returns : np.ndarray
        (N_time, N_stocks) daily returns
    characteristics : np.ndarray
        (N_time, N_stocks, N_factors) standardized anomaly characteristics
    use_inverse_variance_weights : bool
        If True, use inverse variance weighting for OLS (improves efficiency)

    Returns
    -------
    alphas : np.ndarray
        (N_time,) daily intercepts
    thetas : np.ndarray
        (N_time, N_factors) estimated factor returns (slopes)
    residuals : np.ndarray
        (N_time, N_stocks) regression residuals
    """
    n_time, n_stocks = returns.shape
    n_factors = characteristics.shape[2]

    alphas = np.full(n_time, np.nan)
    thetas = np.full((n_time, n_factors), np.nan)
    residuals = np.full((n_time, n_stocks), np.nan)

    for t in range(1, n_time):  # Start from 1 to use lagged characteristics
        y = returns[t, :]  # Returns on day t
        X = characteristics[t - 1, :, :]  # Characteristics from day t-1

        # Find valid observations (no NaN in returns or any characteristic)
        valid_y = ~np.isnan(y)
        valid_X = ~np.any(np.isnan(X), axis=1)
        valid = valid_y & valid_X

        n_valid = valid.sum()
        if n_valid < n_factors + 10:  # Need enough observations
            continue

        y_valid = y[valid]
        X_valid = X[valid, :]

        # Add intercept
        X_with_intercept = np.column_stack([np.ones(n_valid), X_valid])

        # OLS regression
        try:
            if use_inverse_variance_weights:
                # Inverse variance weighting (mentioned in paper for efficiency)
                # Use simple OLS first, then compute weights from residual variance
                # For simplicity, use standard OLS here
                pass

            # Solve: (X'X)^-1 X'y
            XtX = X_with_intercept.T @ X_with_intercept
            Xty = X_with_intercept.T @ y_valid

            # Add small regularization for numerical stability
            reg = 1e-8 * np.eye(XtX.shape[0])
            coeffs = np.linalg.solve(XtX + reg, Xty)

            alphas[t] = coeffs[0]
            thetas[t, :] = coeffs[1:]

            # Compute residuals for all stocks
            fitted = X @ thetas[t, :] + alphas[t]
            residuals[t, :] = y - fitted

        except np.linalg.LinAlgError:
            continue

    return alphas, thetas, residuals


def compute_systematic_component(
    characteristics: np.ndarray,
    thetas: np.ndarray,
) -> np.ndarray:
    """
    Compute systematic return component SYS_s,d = sum_j(C_s,d-1,j * theta_d,j)

    Parameters
    ----------
    characteristics : np.ndarray
        (N_time, N_stocks, N_factors) standardized anomaly characteristics
    thetas : np.ndarray
        (N_time, N_factors) estimated factor returns

    Returns
    -------
    sys : np.ndarray
        (N_time, N_stocks) systematic component
    """
    n_time, n_stocks, n_factors = characteristics.shape
    sys = np.full((n_time, n_stocks), np.nan)

    for t in range(1, n_time):
        # Use lagged characteristics with current theta
        C = characteristics[t - 1, :, :]  # (N_stocks, N_factors)
        theta = thetas[t, :]  # (N_factors,)

        if np.any(np.isnan(theta)):
            continue

        # SYS = C @ theta
        sys[t, :] = C @ theta

    return sys


def create_decile_portfolios(
    signal: np.ndarray,
    returns: np.ndarray,
    n_portfolios: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create decile portfolios based on a signal and compute their returns.

    Parameters
    ----------
    signal : np.ndarray
        (N_time, N_stocks) sorting signal (e.g., lagged SYS)
    returns : np.ndarray
        (N_time, N_stocks) forward returns
    n_portfolios : int
        Number of portfolios (default 10 for deciles)

    Returns
    -------
    portfolio_returns : np.ndarray
        (N_time, n_portfolios) value-weighted portfolio returns
    long_short : np.ndarray
        (N_time,) long-short portfolio returns (top - bottom decile)
    """
    n_time, n_stocks = signal.shape

    portfolio_returns = np.full((n_time, n_portfolios), np.nan)

    for t in range(1, n_time):
        s = signal[t - 1, :]  # Lagged signal
        r = returns[t, :]  # Forward returns

        # Find valid observations
        valid = ~np.isnan(s) & ~np.isnan(r)
        n_valid = valid.sum()

        if n_valid < n_portfolios * 5:  # Need enough stocks
            continue

        s_valid = s[valid]
        r_valid = r[valid]

        # Compute percentile breakpoints
        percentiles = np.linspace(0, 100, n_portfolios + 1)
        breakpoints = np.percentile(s_valid, percentiles)

        # Assign stocks to portfolios
        assignments = np.digitize(s_valid, breakpoints[1:-1])  # 0 to n_portfolios-1

        # Compute value-weighted returns for each portfolio
        # (Using equal weights for simplicity; can add market cap weighting)
        for p in range(n_portfolios):
            mask = assignments == p
            if mask.sum() > 0:
                portfolio_returns[t, p] = np.mean(r_valid[mask])

    # Long-short: top decile minus bottom decile
    long_short = portfolio_returns[:, -1] - portfolio_returns[:, 0]

    return portfolio_returns, long_short


def compute_systematic_momentum(
    n_portfolios: int = 10,
) -> dict:
    """
    Main function to compute systematic momentum following Li, Yuan & Zhou (2025).

    Parameters
    ----------
    n_portfolios : int
        Number of portfolios for sorting

    Returns
    -------
    dict with keys:
        - time: timestamp array
        - ticker: ticker array
        - returns: raw returns
        - sys: systematic component
        - res: residual component
        - thetas: factor returns
        - anomaly_names: list of anomaly names
        - portfolio_returns: decile portfolio returns
        - long_short: long-short portfolio returns
    """
    print("=" * 60)
    print("SYSTEMATIC MOMENTUM")
    print("Implementation of Li, Yuan & Zhou (2025)")
    print("=" * 60)

    # Load reference data (daily returns)
    print("\n1. Loading daily returns...")
    time, ticker, returns = load_data("daily", "return", time_and_ticker=True)
    print(f"   Time periods: {len(time):,}")
    print(f"   Stocks: {len(ticker):,}")
    print(f"   Returns shape: {returns.shape}")

    # Load and standardize anomalies
    print("\n2. Loading and standardizing anomalies...")
    anomalies, anomaly_names = load_all_anomalies()
    print(f"   Anomalies shape: {anomalies.shape}")
    print(f"   Anomaly names: {anomaly_names}")

    # Run cross-sectional regression
    print("\n3. Running cross-sectional regressions...")
    alphas, thetas, residuals = run_cross_sectional_regression(returns, anomalies)
    print(f"   Valid theta estimates: {np.sum(~np.isnan(thetas[:, 0])):,} days")

    # Compute systematic component
    print("\n4. Computing systematic component (SYS)...")
    sys = compute_systematic_component(anomalies, thetas)
    valid_sys = np.sum(~np.isnan(sys))
    print(f"   Valid SYS observations: {valid_sys:,}")

    # Residual component
    res = returns - sys

    # Create decile portfolios sorted by lagged SYS
    print("\n5. Creating decile portfolios...")
    portfolio_returns, long_short = create_decile_portfolios(sys, returns, n_portfolios)

    # Compute summary statistics
    valid_ls = ~np.isnan(long_short)
    if valid_ls.sum() > 0:
        ls_mean = np.nanmean(long_short) * 252  # Annualized
        ls_std = np.nanstd(long_short) * np.sqrt(252)
        ls_sharpe = ls_mean / ls_std if ls_std > 0 else np.nan

        print("\n" + "=" * 60)
        print("SYSTEMATIC MOMENTUM STRATEGY RESULTS")
        print("=" * 60)
        print(f"   Period: {valid_ls.sum():,} trading days")
        print(f"   Annualized Return: {ls_mean * 100:.2f}%")
        print(f"   Annualized Volatility: {ls_std * 100:.2f}%")
        print(f"   Sharpe Ratio: {ls_sharpe:.3f}")

    return {
        "time": time,
        "ticker": ticker,
        "returns": returns,
        "sys": sys,
        "res": res,
        "alphas": alphas,
        "thetas": thetas,
        "anomaly_names": anomaly_names,
        "portfolio_returns": portfolio_returns,
        "long_short": long_short,
    }


def analyze_factor_returns(thetas: np.ndarray, names: list[str]) -> None:
    """Analyze and print factor return statistics."""
    print("\n" + "=" * 60)
    print("FACTOR RETURN STATISTICS (theta)")
    print("=" * 60)
    print(f"{'Factor':<30} {'Mean':>10} {'Std':>10} {'t-stat':>10}")
    print("-" * 60)

    for j, name in enumerate(names):
        theta_j = thetas[:, j]
        valid = ~np.isnan(theta_j)
        if valid.sum() < 30:
            continue

        mean = np.nanmean(theta_j) * 252  # Annualized
        std = np.nanstd(theta_j) * np.sqrt(252)
        t_stat = mean / (std / np.sqrt(valid.sum())) if std > 0 else np.nan

        print(f"{name:<30} {mean*100:>9.2f}% {std*100:>9.2f}% {t_stat:>10.2f}")


if __name__ == "__main__":
    # Run systematic momentum analysis
    results = compute_systematic_momentum()

    # Analyze factor returns
    analyze_factor_returns(results["thetas"], results["anomaly_names"])

    # Print portfolio statistics
    print("\n" + "=" * 60)
    print("DECILE PORTFOLIO RETURNS (Annualized)")
    print("=" * 60)

    port_ret = results["portfolio_returns"]
    for p in range(port_ret.shape[1]):
        valid = ~np.isnan(port_ret[:, p])
        if valid.sum() > 0:
            ann_ret = np.nanmean(port_ret[:, p]) * 252 * 100
            print(f"   Decile {p+1}: {ann_ret:>8.2f}%")


# =============================================================================
# INTRADAY SYSTEMATIC MOMENTUM
# =============================================================================


def get_available_dates() -> list[str]:
    """Get list of available dates with intraday data."""
    intraday_dir = DATA_RAW_DIR / "intra_30min"
    dates = sorted([f.stem for f in intraday_dir.glob("*.parquet")])
    return dates


def load_intraday_returns_for_period(
    dates: list[str],
    period: int,
) -> tuple[list[str], dict[str, np.ndarray]]:
    """
    Load intraday returns for a specific period across all dates.

    Parameters
    ----------
    dates : list[str]
        List of dates in 'YYYY-MM-DD' format
    period : int
        Intraday period (1-8)

    Returns
    -------
    valid_dates : list[str]
        Dates with valid data
    returns_by_ticker : dict
        {ticker: array of returns for each date}
    """
    returns_data = []

    for date in dates:
        df = load_intraday_data(date)
        if df is None:
            continue

        ret_df = compute_intraday_returns(df)
        period_ret = ret_df.filter(pl.col("period") == period)

        returns_data.append({
            "date": date,
            "tickers": period_ret["order_book_id"].to_list(),
            "returns": period_ret["return"].to_numpy(),
        })

    return returns_data


def run_intraday_cross_sectional_regression(
    returns_data: list[dict],
    characteristics: np.ndarray,
    char_time: np.ndarray,
    char_ticker: np.ndarray,
) -> tuple[dict, dict]:
    """
    Run cross-sectional regression for intraday returns.

    Parameters
    ----------
    returns_data : list[dict]
        List of {date, tickers, returns} for each day
    characteristics : np.ndarray
        (N_time, N_stocks, N_factors) standardized anomaly characteristics
    char_time : np.ndarray
        Time array for characteristics (nanosecond timestamps)
    char_ticker : np.ndarray
        Ticker array for characteristics

    Returns
    -------
    thetas : dict
        {date: theta array}
    sys_values : dict
        {date: {ticker: sys_value}}
    """
    import pandas as pd

    # Create lookup for characteristics
    ticker_to_idx = {t: i for i, t in enumerate(char_ticker)}

    # Convert nanosecond timestamps to dates for matching
    char_dates = pd.to_datetime(char_time).strftime("%Y-%m-%d").tolist()
    date_to_idx = {d: i for i, d in enumerate(char_dates)}

    n_factors = characteristics.shape[2]
    thetas = {}
    sys_values = {}

    for day_data in returns_data:
        date = day_data["date"]
        tickers = day_data["tickers"]
        returns = day_data["returns"]

        # Find previous trading day for characteristics
        # (simplified: use the previous day in our char_dates if available)
        if date not in date_to_idx:
            continue
        t_idx = date_to_idx[date]
        if t_idx == 0:
            continue

        # Use characteristics from previous day
        char_t_idx = t_idx - 1

        # Build X and y matrices
        valid_indices = []
        y_list = []
        X_list = []

        for i, ticker in enumerate(tickers):
            if isinstance(ticker, bytes):
                ticker = ticker.decode()
            ticker_bytes = ticker.encode() if isinstance(ticker, str) else ticker

            if ticker_bytes not in ticker_to_idx:
                continue

            s_idx = ticker_to_idx[ticker_bytes]
            char_values = characteristics[char_t_idx, s_idx, :]

            if np.any(np.isnan(char_values)) or np.isnan(returns[i]):
                continue

            valid_indices.append(i)
            y_list.append(returns[i])
            X_list.append(char_values)

        if len(y_list) < n_factors + 10:
            continue

        y = np.array(y_list)
        X = np.array(X_list)

        # Add intercept
        X_with_intercept = np.column_stack([np.ones(len(y)), X])

        # OLS regression
        try:
            XtX = X_with_intercept.T @ X_with_intercept
            Xty = X_with_intercept.T @ y
            reg = 1e-8 * np.eye(XtX.shape[0])
            coeffs = np.linalg.solve(XtX + reg, Xty)

            thetas[date] = coeffs[1:]  # Exclude intercept

            # Compute SYS for all stocks
            sys_values[date] = {}
            for i, ticker in enumerate(tickers):
                if isinstance(ticker, bytes):
                    ticker = ticker.decode()
                ticker_bytes = ticker.encode() if isinstance(ticker, str) else ticker

                if ticker_bytes not in ticker_to_idx:
                    continue

                s_idx = ticker_to_idx[ticker_bytes]
                char_values = characteristics[char_t_idx, s_idx, :]

                if not np.any(np.isnan(char_values)):
                    sys_val = np.dot(char_values, coeffs[1:])
                    sys_values[date][ticker] = sys_val

        except np.linalg.LinAlgError:
            continue

    return thetas, sys_values


def compute_intraday_systematic_momentum(
    start_date: str = "2020-01-01",
    end_date: str = "2020-12-31",
    period: int = 1,
    n_portfolios: int = 10,
) -> dict:
    """
    Compute systematic momentum for a specific intraday period.

    Parameters
    ----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    period : int
        Intraday period (1-8)
    n_portfolios : int
        Number of portfolios for sorting

    Returns
    -------
    dict with results
    """
    print("=" * 60)
    print(f"INTRADAY SYSTEMATIC MOMENTUM - Period {period}")
    print(f"({INTRADAY_PERIODS[period-1]} bar)")
    print("=" * 60)

    # Get available dates
    all_dates = get_available_dates()
    dates = [d for d in all_dates if start_date <= d <= end_date]
    print(f"\n1. Date range: {dates[0]} to {dates[-1]} ({len(dates)} days)")

    # Load daily characteristics
    print("\n2. Loading daily characteristics...")
    time, ticker, _ = load_data("daily", "return", time_and_ticker=True)
    anomalies, anomaly_names = load_all_anomalies()
    print(f"   Anomalies: {len(anomaly_names)}")

    # Load intraday returns for this period
    print(f"\n3. Loading intraday returns for period {period}...")
    returns_data = load_intraday_returns_for_period(dates, period)
    print(f"   Days with data: {len(returns_data)}")

    # Run cross-sectional regression
    print("\n4. Running cross-sectional regressions...")
    thetas, sys_values = run_intraday_cross_sectional_regression(
        returns_data, anomalies, time, ticker
    )
    print(f"   Valid theta estimates: {len(thetas)} days")

    # Create portfolios sorted by lagged SYS
    print("\n5. Creating decile portfolios...")
    portfolio_returns = []

    prev_sys = None
    for day_data in returns_data:
        date = day_data["date"]
        tickers = day_data["tickers"]
        returns = day_data["returns"]

        if prev_sys is None:
            if date in sys_values:
                prev_sys = sys_values[date]
            continue

        # Sort stocks by previous period's SYS
        valid_stocks = []
        for i, ticker in enumerate(tickers):
            if ticker in prev_sys and not np.isnan(returns[i]):
                valid_stocks.append((ticker, prev_sys[ticker], returns[i]))

        if len(valid_stocks) < n_portfolios * 5:
            if date in sys_values:
                prev_sys = sys_values[date]
            continue

        # Sort by SYS
        valid_stocks.sort(key=lambda x: x[1])

        # Assign to portfolios
        n_per_port = len(valid_stocks) // n_portfolios
        port_ret = []
        for p in range(n_portfolios):
            start = p * n_per_port
            end = start + n_per_port if p < n_portfolios - 1 else len(valid_stocks)
            stocks_in_port = valid_stocks[start:end]
            avg_ret = np.mean([s[2] for s in stocks_in_port])
            port_ret.append(avg_ret)

        portfolio_returns.append({
            "date": date,
            "portfolio_returns": port_ret,
            "long_short": port_ret[-1] - port_ret[0],
        })

        if date in sys_values:
            prev_sys = sys_values[date]

    # Compute statistics
    if portfolio_returns:
        ls_returns = [pr["long_short"] for pr in portfolio_returns]
        ls_mean = np.mean(ls_returns) * 252 * N_INTRADAY_PERIODS  # Annualized
        ls_std = np.std(ls_returns) * np.sqrt(252 * N_INTRADAY_PERIODS)
        ls_sharpe = ls_mean / ls_std if ls_std > 0 else np.nan

        print("\n" + "=" * 60)
        print(f"SYSTEMATIC MOMENTUM RESULTS - Period {period}")
        print("=" * 60)
        print(f"   Trading days: {len(portfolio_returns)}")
        print(f"   Annualized Return: {ls_mean * 100:.2f}%")
        print(f"   Annualized Volatility: {ls_std * 100:.2f}%")
        print(f"   Sharpe Ratio: {ls_sharpe:.3f}")

    return {
        "period": period,
        "dates": dates,
        "thetas": thetas,
        "sys_values": sys_values,
        "anomaly_names": anomaly_names,
        "portfolio_returns": portfolio_returns,
    }


def compute_all_intraday_systematic_momentum(
    start_date: str = "2020-01-01",
    end_date: str = "2020-12-31",
) -> dict:
    """
    Compute systematic momentum for all 8 intraday periods.

    Returns combined results and the aggregate portfolio.
    """
    print("=" * 60)
    print("SYSTEMATIC MOMENTUM - ALL INTRADAY PERIODS")
    print("=" * 60)

    all_results = {}

    for period in range(1, N_INTRADAY_PERIODS + 1):
        print(f"\n{'='*60}")
        result = compute_intraday_systematic_momentum(
            start_date=start_date,
            end_date=end_date,
            period=period,
        )
        all_results[period] = result

    # Compute aggregate statistics
    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS ACROSS ALL PERIODS")
    print("=" * 60)

    for period, result in all_results.items():
        if result["portfolio_returns"]:
            ls_returns = [pr["long_short"] for pr in result["portfolio_returns"]]
            ls_mean = np.mean(ls_returns) * 252 * N_INTRADAY_PERIODS
            ls_sharpe = ls_mean / (np.std(ls_returns) * np.sqrt(252 * N_INTRADAY_PERIODS))
            print(f"   Period {period} ({INTRADAY_PERIODS[period-1]}): "
                  f"Return={ls_mean*100:.1f}%, Sharpe={ls_sharpe:.2f}")

    return all_results
