"""Anomalies 1
and 2: Financial distress. Financial distress is often invoked to
explain otherwise anomalous patterns in the cross section of
stock returns. However, Campbell, Hilscher, and Szilagyi
(2008) ﬁnd that ﬁrms with high failure probability have
lower, not higher, subsequent returns (anomaly 1). Campbell,
Hilscher, and Szilagyi suggest that their ﬁnding is a challenge
to standard models of rational asset pricing. The failure
probability is estimated by a dynamic logit model with both
accounting and equity market variables as explanatory vari-
ables. Using the Ohlson (1980) O-score as the distress
measure yields similar results (anomaly 2). The Ohlson O-
score is calculated as the probability of bankruptcy in a static
model using accounting variables, such as net income divided
by assets, working capital divided by market assets, current
liability divided by current assets, etc. The failure probability
is different from the O-score in that it is estimated by a
dynamic, rather than a static model, and that the model uses
several equity market variables, such as stock prices, book-to-
market, stock volatility, relative size to the Standard and
Poor’s (S&P) 500, and cumulative excess return relative to
S&P 500."""
import numpy as np
import h5py
from pathlib import Path

from main.utils import load_data


# 1
# lower subsequent returns -> high failure probability
# Campbell, Hilscher, and Szilagyi (2008) Failure Probability Model
def financial_distress_chs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute CHS (2008) failure probability.

    CHS = -9.16 - 20.26*NIMTAAVG + 1.42*TLMTA - 7.13*EXRETAVG + 1.41*SIGMA
          - 0.045*RSIZE - 2.13*CASHMTA + 0.075*MB - 0.058*PRICE

    P(failure) = exp(CHS) / (1 + exp(CHS))

    Returns:
        (time, ticker, failure_probability) arrays
    """
    time_d, ticker, close = load_data("daily", "close")
    _, _, ret = load_data("daily", "return")

    # shares outstanding
    time_s, _, total_a = load_data("shares", "total_a")

    time_bs, _, total_assets = load_data("balance_sheet", "total_assets_mrq_0")
    _, _, total_liabilities = load_data("balance_sheet", "total_liabilities_mrq_0")
    _, _, total_equity = load_data("balance_sheet", "total_equity_mrq_0")
    _, _, cash_equivalent = load_data("balance_sheet", "cash_equivalent_mrq_0")

    _, _, net_profit = load_data("income_statement", "net_profit_mrq_0")

    # Align to balance sheet time (quarterly data aligned to daily)
    # Use the balance sheet timestamps as output
    time_out = time_bs
    n_time = len(time_out)
    n_ticker = len(ticker)

    # Create index mapping from balance sheet time to daily time
    time_d_set = set(time_d)
    daily_idx = []
    for t in time_out:
        if t in time_d_set:
            daily_idx.append(np.where(time_d == t)[0][0])
        else:
            # Find closest earlier date
            earlier = time_d[time_d <= t]
            if len(earlier) > 0:
                daily_idx.append(np.where(time_d == earlier[-1])[0][0])
            else:
                daily_idx.append(0)
    daily_idx = np.array(daily_idx)

    # Get aligned close prices and shares
    close_aligned = close[daily_idx, :]
    total_a_aligned = total_a  # Already on same time grid as balance sheet

    # Market equity = close * shares outstanding
    market_equity = close_aligned * total_a_aligned

    # Market value of total assets = book assets - book equity + market equity
    mva = total_assets - total_equity + market_equity

    # Avoid division by zero
    mva = np.where(mva > 0, mva, np.nan)
    total_equity_safe = np.where(total_equity > 0, total_equity, np.nan)

    # NIMTA: Net income / market value of total assets
    # Note: For NIMTAAVG we'd need moving average, using point-in-time for simplicity
    nimta = net_profit / mva

    # TLMTA: Total liabilities / market value of total assets
    tlmta = total_liabilities / mva

    # CASHMTA: Cash / market value of total assets
    cashmta = cash_equivalent / mva

    # MB: Market-to-book ratio
    mb = market_equity / total_equity_safe

    # PRICE: log(price per share), capped at log(15) as in original paper
    price = np.log(np.clip(close_aligned, 1e-8, None))
    price = np.minimum(price, np.log(15))

    # SIGMA: Standard deviation of daily returns (past 3 months ~63 trading days)
    # Compute rolling std for each stock
    sigma = np.full((n_time, n_ticker), np.nan)
    window = 63
    for i, d_idx in enumerate(daily_idx):
        start_idx = max(0, d_idx - window + 1)
        if d_idx >= window - 1:
            sigma[i, :] = np.nanstd(ret[start_idx : d_idx + 1, :], axis=0)

    # EXRETAVG: Excess return vs market (simplified: use mean return as proxy)
    # Without market index, we use cross-sectional mean as market proxy
    exret = np.full((n_time, n_ticker), np.nan)
    window_ex = 63
    for i, d_idx in enumerate(daily_idx):
        start_idx = max(0, d_idx - window_ex + 1)
        if d_idx >= window_ex - 1:
            stock_ret = ret[start_idx : d_idx + 1, :]
            market_ret = np.nanmean(stock_ret, axis=1, keepdims=True)
            excess = np.log(1 + stock_ret) - np.log(1 + market_ret)
            exret[i, :] = np.nanmean(excess, axis=0)

    # RSIZE: log(firm market cap) - log(market total cap)
    # Use cross-sectional percentile as proxy
    total_market_cap = np.nansum(market_equity, axis=1, keepdims=True)
    rsize = np.log(np.clip(market_equity, 1e-8, None)) - np.log(
        np.clip(total_market_cap, 1e-8, None)
    )

    # CHS score
    chs = (
        -9.16
        - 20.26 * nimta
        + 1.42 * tlmta
        - 7.13 * exret
        + 1.41 * sigma
        - 0.045 * rsize
        - 2.13 * cashmta
        + 0.075 * mb
        - 0.058 * price
    )

    # Failure probability (numerically stable sigmoid)
    failure_prob = np.where(chs >= 0, 1 / (1 + np.exp(-chs)), np.exp(chs) / (1 + np.exp(chs)))

    return time_out, ticker, failure_prob


def financial_distress_1():
    """Alias for CHS failure probability model."""
    return financial_distress_chs()


# 2
# lower subsequent returns -> high O-Score (bankruptcy probability)
# Ohlson (1980) O-Score Model
def financial_distress_oscore() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Ohlson (1980) O-Score bankruptcy probability.

    O-Score = -1.32 - 0.407*log(TA/GNP) + 6.03*TLTA - 1.43*WCTA + 0.076*CLCA
              - 1.72*OENEG - 2.37*NITA - 1.83*FUTL + 0.285*INTWO - 0.521*CHIN

    Where:
        TA = Total assets
        GNP = Gross National Product price index (proxy with constant/market cap)
        TLTA = Total liabilities / Total assets
        WCTA = Working capital / Total assets
        CLCA = Current liabilities / Current assets
        OENEG = 1 if total liabilities > total assets, 0 otherwise
        NITA = Net income / Total assets
        FUTL = Funds from operations / Total liabilities
        INTWO = 1 if net income negative for last 2 years, 0 otherwise
        CHIN = (NI_t - NI_{t-1}) / (|NI_t| + |NI_{t-1}|)

    P(bankruptcy) = exp(O) / (1 + exp(O))

    Returns:
        (time, ticker, oscore_probability) arrays
    """
    # Load balance sheet data
    time, ticker, total_assets = load_data("balance_sheet", "total_assets_mrq_0")
    _, _, total_liabilities = load_data("balance_sheet", "total_liabilities_mrq_0")
    _, _, current_assets = load_data("balance_sheet", "current_assets_mrq_0")
    _, _, current_liabilities = load_data("balance_sheet", "current_liabilities_mrq_0")

    # Load income statement data
    _, _, net_profit = load_data("income_statement", "net_profit_mrq_0")

    n_time, n_ticker = total_assets.shape

    # Avoid division by zero
    ta_safe = np.where(total_assets > 0, total_assets, np.nan)
    tl_safe = np.where(total_liabilities > 0, total_liabilities, np.nan)
    ca_safe = np.where(current_assets > 0, current_assets, np.nan)

    # SIZE: log(Total Assets / GNP deflator)
    # Use cross-sectional median as deflator proxy
    ta_median = np.nanmedian(total_assets, axis=1, keepdims=True)
    ta_median = np.where(ta_median > 0, ta_median, 1.0)
    size = np.log(np.clip(total_assets / ta_median, 1e-8, None))

    # TLTA: Total liabilities / Total assets
    tlta = total_liabilities / ta_safe

    # WCTA: Working capital / Total assets
    working_capital = current_assets - current_liabilities
    wcta = working_capital / ta_safe

    # CLCA: Current liabilities / Current assets
    clca = current_liabilities / ca_safe

    # OENEG: 1 if total liabilities > total assets
    oeneg = (total_liabilities > total_assets).astype(float)

    # NITA: Net income / Total assets
    nita = net_profit / ta_safe

    # FUTL: Funds from operations / Total liabilities
    # Approximate funds from operations as net income (simplified)
    futl = net_profit / tl_safe

    # INTWO: 1 if net income negative for last 2 periods
    # Check current and previous quarter
    intwo = np.zeros((n_time, n_ticker))
    if n_time > 1:
        ni_curr_neg = net_profit[1:, :] < 0
        ni_prev_neg = net_profit[:-1, :] < 0
        intwo[1:, :] = (ni_curr_neg & ni_prev_neg).astype(float)

    # CHIN: Change in net income scaled by sum of absolute values
    chin = np.full((n_time, n_ticker), np.nan)
    if n_time > 1:
        ni_curr = net_profit[1:, :]
        ni_prev = net_profit[:-1, :]
        denom = np.abs(ni_curr) + np.abs(ni_prev)
        denom = np.where(denom > 0, denom, np.nan)
        chin[1:, :] = (ni_curr - ni_prev) / denom

    # O-Score calculation
    oscore = (
        -1.32
        - 0.407 * size
        + 6.03 * tlta
        - 1.43 * wcta
        + 0.076 * clca
        - 1.72 * oeneg
        - 2.37 * nita
        - 1.83 * futl
        + 0.285 * intwo
        - 0.521 * chin
    )

    # Bankruptcy probability (numerically stable sigmoid)
    oscore_prob = np.where(
        oscore >= 0,
        1 / (1 + np.exp(-oscore)),
        np.exp(oscore) / (1 + np.exp(oscore)),
    )

    return time, ticker, oscore_prob


def financial_distress_2():
    """Alias for Ohlson O-Score bankruptcy probability model."""
    return financial_distress_oscore()