"""
15 Anomalies from Ehsani & Linnainmaa (2022) "Factor Momentum and the Momentum Factor"

Each function returns a (N_time, N_stocks) array of factor exposures (characteristics).
Higher values indicate stronger exposure to the long side of the factor.
"""

import numpy as np
from pathlib import Path
from main.utils import load_data

def align_to_daily(
    daily_time: np.ndarray,
    quarterly_time: np.ndarray,
    quarterly_values: np.ndarray,
) -> np.ndarray:
    """Forward-fill quarterly/shares data to daily frequency.

    Parameters
    ----------
    daily_time : np.ndarray
        Daily timestamps (N_daily,)
    quarterly_time : np.ndarray
        Quarterly timestamps (N_quarterly,)
    quarterly_values : np.ndarray
        Quarterly values (N_quarterly, N_stocks)

    Returns
    -------
    np.ndarray
        (N_daily, N_stocks) forward-filled values
    """
    n_daily = len(daily_time)
    n_stocks = quarterly_values.shape[1]
    result = np.full((n_daily, n_stocks), np.nan)

    # Use searchsorted for efficient forward-fill
    # For each daily timestamp, find the index of the last quarterly timestamp <= it
    indices = np.searchsorted(quarterly_time, daily_time, side="right") - 1

    for i in range(n_daily):
        idx = indices[i]
        if idx >= 0:
            result[i, :] = quarterly_values[idx, :]

    return result


# -----------------------------------------------------------------------------
# 1. Size (SMB) - Small Minus Big
# -----------------------------------------------------------------------------
def size() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Market capitalization. Lower = small cap (long side of SMB).

    Returns negative market cap so higher values = smaller firms.
    """
    time, ticker, close = load_data("daily", "close")
    shares_time, _, shares = load_data("shares", "total_a")
    shares_aligned = align_to_daily(time, shares_time, shares)
    market_cap = close * shares_aligned
    return time, ticker, -market_cap  # Negative so small = high exposure


# -----------------------------------------------------------------------------
# 2. Book-to-Market (Value)
# -----------------------------------------------------------------------------
def book_to_market() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Book-to-market ratio. Higher = value stock (long side of HML)."""
    time, ticker, close = load_data("daily", "close")
    shares_time, _, shares = load_data("shares", "total_a")
    be_time, _, book_equity = load_data("balance_sheet", "total_equity_mrq_0")

    shares_aligned = align_to_daily(time, shares_time, shares)
    be_aligned = align_to_daily(time, be_time, book_equity)

    market_cap = close * shares_aligned
    bm = be_aligned / market_cap
    return time, ticker, bm


# -----------------------------------------------------------------------------
# 3. Momentum (UMD) - Up Minus Down
# -----------------------------------------------------------------------------
def momentum() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Past 12-month return, skipping most recent month. Higher = winner."""
    time, ticker, ret = load_data("daily", "return")

    n_time, n_stocks = ret.shape
    mom = np.full((n_time, n_stocks), np.nan)

    # Convert to cumulative returns for easier window calculations
    # Skip most recent ~21 days, use prior ~252-21=231 days
    skip_days = 21
    lookback = 252 - skip_days

    for t in range(skip_days + lookback, n_time):
        # Returns from t-252 to t-21 (skip recent month)
        period_ret = ret[t - skip_days - lookback : t - skip_days, :]
        # Cumulative return = product of (1 + r) - 1
        cum_ret = np.nanprod(1 + period_ret, axis=0) - 1
        mom[t, :] = cum_ret

    return time, ticker, mom


# -----------------------------------------------------------------------------
# 4. Short-Term Reversals
# -----------------------------------------------------------------------------
def short_term_reversal() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prior month return. Lower = reversal long side (losers become winners)."""
    time, ticker, ret = load_data("daily", "return")

    n_time, n_stocks = ret.shape
    str_factor = np.full((n_time, n_stocks), np.nan)
    lookback = 21  # ~1 month

    for t in range(lookback, n_time):
        period_ret = ret[t - lookback : t, :]
        cum_ret = np.nanprod(1 + period_ret, axis=0) - 1
        str_factor[t, :] = -cum_ret  # Negative: losers have high exposure

    return time, ticker, str_factor


# -----------------------------------------------------------------------------
# 5. Long-Term Reversals
# -----------------------------------------------------------------------------
def long_term_reversal() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns from month -60 to -13. Lower = reversal long side."""
    time, ticker, ret = load_data("daily", "return")

    n_time, n_stocks = ret.shape
    ltr = np.full((n_time, n_stocks), np.nan)

    # t-60 months to t-13 months (~1260 to ~273 days back)
    start_lag = 1260  # ~60 months
    end_lag = 273  # ~13 months

    for t in range(start_lag, n_time):
        period_ret = ret[t - start_lag : t - end_lag, :]
        cum_ret = np.nanprod(1 + period_ret, axis=0) - 1
        ltr[t, :] = -cum_ret  # Negative: past losers have high exposure

    return time, ticker, ltr


# -----------------------------------------------------------------------------
# 6. Accruals
# -----------------------------------------------------------------------------
def accruals() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Operating accruals. Lower = conservative (long side)."""
    time, ticker, current_assets = load_data("balance_sheet", "current_assets_mrq_0")
    _, _, current_assets_lag = load_data("balance_sheet", "current_assets_mrq_4")
    _, _, cash = load_data("balance_sheet", "cash_equivalent_mrq_0")
    _, _, cash_lag = load_data("balance_sheet", "cash_equivalent_mrq_4")
    _, _, current_liab = load_data("balance_sheet", "current_liabilities_mrq_0")
    _, _, current_liab_lag = load_data("balance_sheet", "current_liabilities_mrq_4")
    _, _, total_assets = load_data("balance_sheet", "total_assets_mrq_0")
    _, _, total_assets_lag = load_data("balance_sheet", "total_assets_mrq_4")

    # Change in non-cash current assets
    delta_ca = current_assets - current_assets_lag
    delta_cash = cash - cash_lag
    delta_cl = current_liab - current_liab_lag

    # Accruals = (delta_CA - delta_Cash) - delta_CL, scaled by avg assets
    avg_assets = (total_assets + total_assets_lag) / 2
    acc = ((delta_ca - delta_cash) - delta_cl) / avg_assets

    return time, ticker, -acc  # Negative: low accruals = high exposure


# -----------------------------------------------------------------------------
# 7. Profitability (ROA)
# -----------------------------------------------------------------------------
def profitability() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return on assets. Higher = more profitable (long side)."""
    time, ticker, net_profit = load_data("income_statement", "net_profit_mrq_0")
    _, _, total_assets = load_data("balance_sheet", "total_assets_mrq_0")
    roa = net_profit / total_assets
    return time, ticker, roa


# -----------------------------------------------------------------------------
# 8. Investment
# -----------------------------------------------------------------------------
def investment() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Asset growth. Lower = conservative investment (long side)."""
    time, ticker, total_assets = load_data("balance_sheet", "total_assets_mrq_0")
    _, _, total_assets_lag = load_data("balance_sheet", "total_assets_mrq_4")
    asset_growth = (total_assets - total_assets_lag) / total_assets_lag
    return time, ticker, -asset_growth  # Negative: low growth = high exposure


# -----------------------------------------------------------------------------
# 9. Earnings-to-Price
# -----------------------------------------------------------------------------
def earnings_to_price() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Earnings yield. Higher = value (long side)."""
    time, ticker, net_profit = load_data("income_statement", "net_profit_mrq_0")
    _, _, close = load_data("daily", "close")
    _, _, shares = load_data("shares", "total_a")
    market_cap = close * shares
    ep = net_profit / market_cap
    return time, ticker, ep


# -----------------------------------------------------------------------------
# 10. Cash-Flow to Price
# -----------------------------------------------------------------------------
def cashflow_to_price() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Operating cash flow yield. Higher = value (long side)."""
    time, ticker, ocf = load_data("cash_flow_statement", "cash_flow_from_operating_activities_mrq_0")
    _, _, close = load_data("daily", "close")
    _, _, shares = load_data("shares", "total_a")
    market_cap = close * shares
    cfp = ocf / market_cap
    return time, ticker, cfp


# -----------------------------------------------------------------------------
# 11. Betting Against Beta (BAB)
# -----------------------------------------------------------------------------
def betting_against_beta() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Market beta. Lower beta = long side of BAB."""
    time, ticker, ret = load_data("daily", "return")
    _, _, mkt_weights = load_data("index_weights", "000300.XSHG")

    n_time, n_stocks = ret.shape
    beta = np.full((n_time, n_stocks), np.nan)

    lookback = 252  # 12 months

    for t in range(lookback, n_time):
        # Market return as weighted average of constituent returns
        weights = mkt_weights[t, :]
        valid_weights = np.where(np.isnan(weights) | (weights == 0), 0, weights)
        if valid_weights.sum() == 0:
            continue
        valid_weights = valid_weights / valid_weights.sum()

        stock_ret = ret[t - lookback : t, :]
        mkt_ret = np.nansum(stock_ret * valid_weights, axis=1)

        # Compute beta for each stock
        mkt_var = np.var(mkt_ret)
        if mkt_var == 0:
            continue

        for j in range(n_stocks):
            sr = stock_ret[:, j]
            valid = ~np.isnan(sr)
            if valid.sum() < 60:
                continue
            cov = np.cov(sr[valid], mkt_ret[valid])[0, 1]
            beta[t, j] = cov / mkt_var

    return time, ticker, -beta  # Negative: low beta = high exposure


# -----------------------------------------------------------------------------
# 12. Residual Variance (Idiosyncratic Volatility)
# -----------------------------------------------------------------------------
def residual_variance() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Idiosyncratic volatility from CAPM. Higher = long side (lottery demand)."""
    time, ticker, ret = load_data("daily", "return")
    _, _, mkt_weights = load_data("index_weights", "000300.XSHG")

    n_time, n_stocks = ret.shape
    resid_var = np.full((n_time, n_stocks), np.nan)

    lookback = 252

    for t in range(lookback, n_time):
        weights = mkt_weights[t, :]
        valid_weights = np.where(np.isnan(weights) | (weights == 0), 0, weights)
        if valid_weights.sum() == 0:
            continue
        valid_weights = valid_weights / valid_weights.sum()

        stock_ret = ret[t - lookback : t, :]
        mkt_ret = np.nansum(stock_ret * valid_weights, axis=1)

        mkt_var = np.var(mkt_ret)
        if mkt_var == 0:
            continue

        for j in range(n_stocks):
            sr = stock_ret[:, j]
            valid = ~np.isnan(sr)
            if valid.sum() < 60:
                continue

            # CAPM regression
            cov = np.cov(sr[valid], mkt_ret[valid])[0, 1]
            beta_j = cov / mkt_var
            alpha_j = np.mean(sr[valid]) - beta_j * np.mean(mkt_ret[valid])

            # Residuals
            resid = sr[valid] - alpha_j - beta_j * mkt_ret[valid]
            resid_var[t, j] = np.var(resid)

    return time, ticker, resid_var


# -----------------------------------------------------------------------------
# 13. Net Share Issues
# -----------------------------------------------------------------------------
def net_share_issues() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """12-month change in shares outstanding. Lower = buybacks (long side)."""
    time, ticker, shares = load_data("shares", "total_a")

    n_time, n_stocks = shares.shape
    nsi = np.full((n_time, n_stocks), np.nan)

    lag = 252  # ~12 months

    for t in range(lag, n_time):
        shares_now = shares[t, :]
        shares_past = shares[t - lag, :]
        nsi[t, :] = (shares_now - shares_past) / shares_past

    return time, ticker, -nsi  # Negative: share reduction = high exposure


# -----------------------------------------------------------------------------
# 14. Liquidity (Turnover)
# -----------------------------------------------------------------------------
def liquidity() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average turnover. Higher = more liquid (long side)."""
    time, ticker, volume = load_data("daily", "volume")
    _, _, shares = load_data("shares", "circulation_a")

    n_time, n_stocks = volume.shape
    turnover = np.full((n_time, n_stocks), np.nan)

    lookback = 21  # 1 month average

    for t in range(lookback, n_time):
        vol = volume[t - lookback : t, :]
        shr = shares[t, :]  # Use current float shares
        avg_vol = np.nanmean(vol, axis=0)
        turnover[t, :] = avg_vol / shr

    return time, ticker, turnover


# -----------------------------------------------------------------------------
# 15. Quality Minus Junk (QMJ)
# -----------------------------------------------------------------------------
def quality_minus_junk() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Composite quality score. Higher = quality (long side).

    Combines: profitability, safety (low leverage, low volatility), growth
    """
    time, ticker, net_profit = load_data("income_statement", "net_profit_mrq_0")
    _, _, total_assets = load_data("balance_sheet", "total_assets_mrq_0")
    _, _, total_equity = load_data("balance_sheet", "total_equity_mrq_0")
    _, _, total_liab = load_data("balance_sheet", "total_liabilities_mrq_0")
    _, _, ret = load_data("daily", "return")

    n_time, n_stocks = ret.shape

    # 1. Profitability: ROA
    roa = net_profit / total_assets

    # 2. Safety: equity ratio (low leverage)
    equity_ratio = total_equity / total_assets

    # 3. Safety: low volatility (rolling 12-month)
    volatility = np.full((n_time, n_stocks), np.nan)
    lookback = 252
    for t in range(lookback, n_time):
        volatility[t, :] = np.nanstd(ret[t - lookback : t, :], axis=0)

    # Combine into composite z-score
    def zscore(arr):
        """Cross-sectional z-score."""
        result = np.full_like(arr, np.nan)
        for t in range(arr.shape[0]):
            row = arr[t, :]
            valid = ~np.isnan(row)
            if valid.sum() > 10:
                mu = np.nanmean(row)
                sigma = np.nanstd(row)
                if sigma > 0:
                    result[t, :] = (row - mu) / sigma
        return result

    z_roa = zscore(roa)
    z_equity = zscore(equity_ratio)
    z_vol = zscore(-volatility)  # Negative: low vol = high quality

    # Equal-weighted composite
    qmj = (z_roa + z_equity + z_vol) / 3

    return time, ticker, qmj