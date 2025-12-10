"""Anomalies 3 and 4: Net stock issues and composite equity issues."""

import numpy as np
import h5py
from pathlib import Path

from ..main.utils import load_data


"""
The stock issuing market has been long viewed as
producing an anomaly arising from sentiment-driven292
R.F. Stambaugh et al. / Journal of Financial Economics 104 (2012) 288–302
mispricing: Smart managers issue shares when sentiment-
driven traders push prices to overvalued levels. Ritter
(1991) and Loughran and Ritter (1995) show that, in
post-issue years, equity issuers under-perform matching
nonissuers with similar characteristics (anomaly 3). We
measure net stock issues as the growth rate of the split-
adjusted shares outstanding in the previous ﬁscal year."""
def net_stock_issues() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute net stock issues (Anomaly 3).

    Net stock issues = growth rate of split-adjusted shares outstanding
                     = (shares_t / shares_{t-1}) - 1

    We use total_a shares adjusted by ex_factor for split adjustment.

    Returns:
        (time, ticker, net_stock_issues) arrays
    """
    # Load shares outstanding (_a stands for adjusted)
    time, ticker, total_a = load_data("shares", "total_a")

    # Load adjustment factor for split adjustment (on daily grid, need to align)
    time_d, _, ex_factor = load_data("daily", "ex_factor")

    # Find matching timestamp
    time_d_dict = {t: i for i, t in enumerate(time_d)}
    ex_aligned = np.full_like(total_a, np.nan)
    for i, t in enumerate(time):
        if t in time_d_dict:
            ex_aligned[i, :] = ex_factor[time_d_dict[t], :]
        else:
            # Find closest earlier date
            earlier = time_d[time_d <= t]
            if len(earlier) > 0:
                ex_aligned[i, :] = ex_factor[np.where(time_d == earlier[-1])[0][0], :]

    adjusted_shares = total_a * ex_aligned

    # Compute year-over-year growth (approximately 252 trading days)
    # Use ~1 year lag for annual growth rate
    lag = 252
    n_time = len(time)

    net_issues = np.full_like(adjusted_shares, np.nan)
    if n_time > lag:
        shares_curr = adjusted_shares[lag:, :]
        shares_prev = adjusted_shares[:-lag, :]
        # Growth rate = (current / previous) - 1
        with np.errstate(divide="ignore", invalid="ignore"):
            growth = (shares_curr / shares_prev) - 1
        net_issues[lag:, :] = growth

    return time, ticker, net_issues


"""
Daniel and Titman (2006) study an alternative measure,
composite equity issuance, deﬁned as the amount of
equity a ﬁrm issues (or retires) in exchange for cash or
services. Under this measure, seasoned issues and share-
based acquisitions increase the issuance measure, while
repurchases, dividends, and other actions that take cash
out of the ﬁrm reduce this issuance measure. They also
ﬁnd that issuers under-perform nonissuers (anomaly 4)."""
def composite_equity_issues() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute composite equity issuance (Anomaly 4) following Daniel & Titman (2006).

    Composite equity issuance measures the amount of equity issued/retired.
    Following the paper's methodology:
        CEI = log(ME_t / ME_{t-1}) - r_t

    Where:
        ME = market equity (price * shares)
        r_t = cumulative log return over the period

    This captures issuance activity: if ME grows faster than returns explain,
    the firm must have issued equity.

    Returns:
        (time, ticker, composite_equity_issuance) arrays
    """
    # Load daily data
    time, ticker, close = load_data("daily", "close")
    _, _, ret = load_data("daily", "return")

    # Load shares outstanding (on different time grid)
    time_s, _, total_a = load_data("shares", "total_a")

    # Align shares to daily grid
    time_s_dict = {t: i for i, t in enumerate(time_s)}
    n_time = len(time)
    n_ticker = len(ticker)
    shares_aligned = np.full((n_time, n_ticker), np.nan)

    last_idx = 0
    for i, t in enumerate(time):
        if t in time_s_dict:
            last_idx = time_s_dict[t]
        shares_aligned[i, :] = total_a[last_idx, :] if last_idx < len(time_s) else np.nan

    # Market equity = close * shares
    market_equity = close * shares_aligned

    # Compute 1-year (252 trading days) composite equity issuance
    lag = 252

    # Log market equity growth
    with np.errstate(divide="ignore", invalid="ignore"):
        log_me = np.log(market_equity)
        log_me_growth = log_me[lag:, :] - log_me[:-lag, :]

        # Cumulative log return over the same period
        log_ret = np.log(1 + ret)
        cum_log_ret = np.full((n_time - lag, n_ticker), np.nan)
        for i in range(lag, n_time):
            cum_log_ret[i - lag, :] = np.nansum(log_ret[i - lag + 1 : i + 1, :], axis=0)

        # CEI = log(ME_t/ME_{t-1}) - cumulative log return
        cei = log_me_growth - cum_log_ret

    # Pad with NaN for initial period
    composite_issues = np.full((n_time, n_ticker), np.nan)
    composite_issues[lag:, :] = cei

    return time, ticker, composite_issues