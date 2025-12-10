"""
15 Anomalies from Ehsani & Linnainmaa (2022) "Factor Momentum and the Momentum Factor"

Each function returns a (N_time, N_stocks) array of factor exposures (characteristics).
Higher values indicate stronger exposure to the long side of the factor.
"""

import numpy as np
import h5py
from pathlib import Path
from numba import njit, prange
from main.utils import load_data, DATA_DIR


def _load_index_weights(ticker):
    """Load index weights aligned to standard 5480 ticker list."""
    _, std_ticker, _ = load_data("daily", "return", time_and_ticker=True)
    path = DATA_DIR / "index_weights" / f"{ticker}.h5"
    with h5py.File(path, "r") as f:
        iw_ticker = f["ticker"][:]
        values = f["values"][:3643, :]
    ticker_to_idx = {t: i for i, t in enumerate(iw_ticker)}
    aligned = np.full((3643, 5480), np.nan)
    for i, t in enumerate(std_ticker):
        if t in ticker_to_idx:
            aligned[:, i] = values[:, ticker_to_idx[t]]
    return aligned

# -----------------------------------------------------------------------------
# 1. Size (SMB) - Small Minus Big
# -----------------------------------------------------------------------------
def size():
    """(negative) Market capitalization. Lower = small cap (long side of SMB)."""
    close = load_data("daily", "close")
    shares = load_data("shares", "total_a")
    market_cap = close * shares
    return -market_cap  # small = high exposure


# -----------------------------------------------------------------------------
# 2. Book-to-Market (Value)
# -----------------------------------------------------------------------------
def book_to_market():
    """Book-to-market ratio. Higher = value stock (long side of HML)."""
    close = load_data("daily", "close")
    shares = load_data("shares", "total_a")
    be = load_data("balance_sheet", "total_equity_mrq_0")

    market_cap = close * shares
    bm = be / market_cap
    return bm


# -----------------------------------------------------------------------------
# 3. Momentum (UMD) - Up Minus Down
# -----------------------------------------------------------------------------
def momentum():
    """Past 12-month return, skipping most recent month. Higher = winner."""
    time, ticker, ret = load_data("daily", "return", time_and_ticker=True)

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

    return mom


# -----------------------------------------------------------------------------
# 4. Short-Term Reversals
# -----------------------------------------------------------------------------
def short_term_reversal():
    """Prior month return. Lower = reversal long side (losers become winners)."""
    time, ticker, ret = load_data("daily", "return", time_and_ticker=True)

    n_time, n_stocks = ret.shape
    str_factor = np.full((n_time, n_stocks), np.nan)
    lookback = 21  # ~1 month

    for t in range(lookback, n_time):
        period_ret = ret[t - lookback : t, :]
        cum_ret = np.nanprod(1 + period_ret, axis=0) - 1
        str_factor[t, :] = -cum_ret  # Negative: losers have high exposure

    return str_factor


# -----------------------------------------------------------------------------
# 5. Long-Term Reversals
# -----------------------------------------------------------------------------
def long_term_reversal():
    """Returns from month -60 to -13. Lower = reversal long side."""
    time, ticker, ret = load_data("daily", "return", time_and_ticker=True)

    n_time, n_stocks = ret.shape
    ltr = np.full((n_time, n_stocks), np.nan)

    # t-60 months to t-13 months (~1260 to ~273 days back)
    start_lag = 1260  # ~60 months
    end_lag = 273  # ~13 months

    for t in range(start_lag, n_time):
        period_ret = ret[t - start_lag : t - end_lag, :]
        cum_ret = np.nanprod(1 + period_ret, axis=0) - 1
        ltr[t, :] = -cum_ret  # Negative: past losers have high exposure

    return ltr


# -----------------------------------------------------------------------------
# 6. Accruals
# -----------------------------------------------------------------------------
def accruals():
    """Operating accruals. Lower = conservative (long side)."""
    time, ticker, current_assets = load_data("balance_sheet", "current_assets_mrq_0", time_and_ticker=True)
    current_assets_lag = load_data("balance_sheet", "current_assets_mrq_4")
    cash = load_data("balance_sheet", "cash_equivalent_mrq_0")
    cash_lag = load_data("balance_sheet", "cash_equivalent_mrq_4")
    current_liab = load_data("balance_sheet", "current_liabilities_mrq_0")
    current_liab_lag = load_data("balance_sheet", "current_liabilities_mrq_4")
    total_assets = load_data("balance_sheet", "total_assets_mrq_0")
    total_assets_lag = load_data("balance_sheet", "total_assets_mrq_4")

    # Change in non-cash current assets
    delta_ca = current_assets - current_assets_lag
    delta_cash = cash - cash_lag
    delta_cl = current_liab - current_liab_lag

    # Accruals = (delta_CA - delta_Cash) - delta_CL, scaled by avg assets
    avg_assets = (total_assets + total_assets_lag) / 2
    acc = ((delta_ca - delta_cash) - delta_cl) / avg_assets

    return -acc  # Negative: low accruals = high exposure


# -----------------------------------------------------------------------------
# 7. Profitability (ROA)
# -----------------------------------------------------------------------------
def profitability():
    """Return on assets. Higher = more profitable (long side)."""
    net_profit = load_data("income_statement", "net_profit_mrq_0")
    total_assets = load_data("balance_sheet", "total_assets_mrq_0")
    roa = net_profit / total_assets
    return roa


# -----------------------------------------------------------------------------
# 8. Investment
# -----------------------------------------------------------------------------
def investment():
    """Asset growth. Lower = conservative investment (long side)."""
    total_assets = load_data("balance_sheet", "total_assets_mrq_0")
    total_assets_lag = load_data("balance_sheet", "total_assets_mrq_4")
    asset_growth = (total_assets - total_assets_lag) / total_assets_lag
    return -asset_growth  # Negative: low growth = high exposure


# -----------------------------------------------------------------------------
# 9. Earnings-to-Price
# -----------------------------------------------------------------------------
def earnings_to_price():
    """Earnings yield. Higher = value (long side)."""
    net_profit = load_data("income_statement", "net_profit_mrq_0")
    close = load_data("daily", "close")
    shares = load_data("shares", "total_a")
    market_cap = close * shares
    ep = net_profit / market_cap
    return ep


# -----------------------------------------------------------------------------
# 10. Cash-Flow to Price
# -----------------------------------------------------------------------------
def cashflow_to_price():
    """Operating cash flow yield. Higher = value (long side)."""
    ocf = load_data("cash_flow_statement", "cash_flow_from_operating_activities_mrq_0")
    close = load_data("daily", "close")
    shares = load_data("shares", "total_a")
    market_cap = close * shares
    cfp = ocf / market_cap
    return cfp


# -----------------------------------------------------------------------------
# 11. Betting Against Beta (BAB)
# -----------------------------------------------------------------------------
def betting_against_beta():
    """(negative) beta. Lower beta = long side of BAB."""
    time, ticker, ret = load_data("daily", "return", time_and_ticker=True)
    mkt_weights = _load_index_weights("000300.XSHG")

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

    return -beta


# -----------------------------------------------------------------------------
# 12. Residual Variance (Idiosyncratic Volatility)
# -----------------------------------------------------------------------------
@njit(parallel=True, cache=True)
def _residual_variance_kernel(ret, mkt_weights, lookback, min_valid):
    n_time, n_stocks = ret.shape
    resid_var = np.full((n_time, n_stocks), np.nan)

    # Precompute normalized weights and market returns for all time points
    mkt_ret_all = np.empty((n_time, lookback))
    mkt_var_all = np.empty(n_time)
    valid_time = np.zeros(n_time, dtype=np.bool_)

    for t in range(lookback, n_time):
        weights = mkt_weights[t, :]
        weight_sum = 0.0
        for i in range(n_stocks):
            w = weights[i]
            if np.isnan(w) or w == 0:
                weights[i] = 0.0
            else:
                weight_sum += w

        if weight_sum == 0:
            continue

        # Normalize weights
        for i in range(n_stocks):
            weights[i] /= weight_sum

        # Compute market return for this lookback window
        stock_ret = ret[t - lookback : t, :]
        mkt_ret = np.zeros(lookback)
        for day in range(lookback):
            for j in range(n_stocks):
                sr = stock_ret[day, j]
                if not np.isnan(sr):
                    mkt_ret[day] += sr * weights[j]

        mkt_ret_all[t, :] = mkt_ret

        # Market variance
        mkt_mean = 0.0
        for day in range(lookback):
            mkt_mean += mkt_ret[day]
        mkt_mean /= lookback

        mkt_var = 0.0
        for day in range(lookback):
            diff = mkt_ret[day] - mkt_mean
            mkt_var += diff * diff
        mkt_var /= lookback

        mkt_var_all[t] = mkt_var
        if mkt_var > 0:
            valid_time[t] = True

    # Parallel loop over time
    for t in prange(lookback, n_time):
        if not valid_time[t]:
            continue

        mkt_ret = mkt_ret_all[t, :]
        mkt_var = mkt_var_all[t]
        stock_ret = ret[t - lookback : t, :]

        # Precompute market mean
        mkt_mean = 0.0
        for day in range(lookback):
            mkt_mean += mkt_ret[day]
        mkt_mean /= lookback

        # Process each stock
        for j in range(n_stocks):
            sr = stock_ret[:, j]

            # Count valid and compute means
            valid_count = 0
            sr_sum = 0.0
            mkt_sum_valid = 0.0

            for day in range(lookback):
                if not np.isnan(sr[day]):
                    valid_count += 1
                    sr_sum += sr[day]
                    mkt_sum_valid += mkt_ret[day]

            if valid_count < min_valid:
                continue

            sr_mean = sr_sum / valid_count
            mkt_mean_valid = mkt_sum_valid / valid_count

            # Compute covariance
            cov = 0.0
            for day in range(lookback):
                if not np.isnan(sr[day]):
                    cov += (sr[day] - sr_mean) * (mkt_ret[day] - mkt_mean_valid)
            cov /= valid_count

            # Beta and alpha
            beta_j = cov / mkt_var
            alpha_j = sr_mean - beta_j * mkt_mean_valid

            # Residual variance
            resid_var_sum = 0.0
            for day in range(lookback):
                if not np.isnan(sr[day]):
                    resid = sr[day] - alpha_j - beta_j * mkt_ret[day]
                    resid_var_sum += resid * resid
            resid_var[t, j] = resid_var_sum / valid_count

    return resid_var


def residual_variance():
    """Idiosyncratic volatility from CAPM. Higher = long side (lottery demand)."""
    ret = load_data("daily", "return")
    mkt_weights = _load_index_weights("000300.XSHG")

    lookback = 252
    min_valid = 60

    # Make a copy of mkt_weights since we modify it in the kernel
    mkt_weights = mkt_weights.copy()

    return _residual_variance_kernel(ret, mkt_weights, lookback, min_valid)


# -----------------------------------------------------------------------------
# 13. Net Share Issues
# -----------------------------------------------------------------------------
def net_share_issues():
    """(negative) 12-month change in shares outstanding. Lower = buybacks (long side)."""
    shares = load_data("shares", "total_a")

    n_time, n_stocks = shares.shape
    nsi = np.full((n_time, n_stocks), np.nan)

    lag = 252  # ~12 months

    for t in range(lag, n_time):
        shares_now = shares[t, :]
        shares_past = shares[t - lag, :]
        nsi[t, :] = (shares_now - shares_past) / shares_past

    return -nsi  # Negative: share reduction = high exposure


# -----------------------------------------------------------------------------
# 14. Liquidity (Turnover)
# -----------------------------------------------------------------------------
def liquidity() -> np.ndarray:
    """Average turnover."""
    volume = load_data("daily", "volume")
    shares = load_data("shares", "circulation_a")

    n_time, n_stocks = volume.shape
    turnover = np.full((n_time, n_stocks), np.nan)

    lookback = 21  # 1 month average

    for t in range(lookback, n_time):
        vol = volume[t - lookback : t, :]
        shr = shares[t, :]  # Use current float shares
        avg_vol = np.nanmean(vol, axis=0)
        turnover[t, :] = avg_vol / shr

    return turnover


# -----------------------------------------------------------------------------
# 15. Quality Minus Junk (QMJ)
# -----------------------------------------------------------------------------
def quality_minus_junk():
    """Composite quality score. Higher = quality (long side).

    Combines: profitability, safety (low leverage, low volatility), growth
    """
    net_profit = load_data("income_statement", "net_profit_mrq_0")
    total_assets = load_data("balance_sheet", "total_assets_mrq_0")
    total_equity = load_data("balance_sheet", "total_equity_mrq_0")
    total_liab = load_data("balance_sheet", "total_liabilities_mrq_0")
    ret = load_data("daily", "return")

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

    return qmj