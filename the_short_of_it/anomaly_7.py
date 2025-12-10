"""Anomaly 7: Momentum. The momentum effect, discovered by Jegadeesh and
Titman (1993), is one of the most robust anomalies in asset pricing. It
refers to the phenomenon that high past recent returns forecast high future
returns. In a contemporaneous study, Antoniou, Doukas, and Subrahmanyam
(2010) find that the momentum effect is stronger when sentiment is high,
and they suggest this result is consistent with the slow spread of bad news
during high-sentiment periods. The portfolios we use are ranked on cumulative
returns from month-7 to month-2, and the holding period for each portfolio
is six months. That is, a momentum return for a given month is the equally
weighted average return on six portfolios in that month."""

import numpy as np

from main.utils import load_data


def momentum():
    """
    Compute momentum signal following Jegadeesh and Titman (1993).

    Momentum = Cumulative return from month -7 to month -2
             = Product of (1 + r_t) from t-147 to t-42, minus 1

    Using ~21 trading days per month:
        - Month -7 start: ~147 days ago
        - Month -2 end: ~42 days ago

    High past returns predict high future returns.
    """
    ret = load_data("daily", "return")

    n_time, n_ticker = ret.shape

    # Month -7 to month -2: approximately day -147 to day -42
    # (skipping the most recent month to avoid short-term reversal)
    start_lag = 147  # ~7 months ago
    end_lag = 42     # ~2 months ago

    momentum_signal = np.full((n_time, n_ticker), np.nan)

    # Compute cumulative return from t-start_lag to t-end_lag
    for i in range(start_lag, n_time):
        # Returns from i-start_lag to i-end_lag (inclusive)
        period_ret = ret[i - start_lag : i - end_lag + 1, :]

        # Cumulative return = product of (1 + r) - 1
        cum_ret = np.nanprod(1 + period_ret, axis=0) - 1
        momentum_signal[i, :] = cum_ret

    return momentum_signal


if __name__ == "__main__":
    time, ticker, mom = momentum()
    print(f"Time periods: {len(time)}")
    print(f"Stocks: {len(ticker)}")
    print(f"Momentum shape: {mom.shape}")
    print(f"Momentum range: [{np.nanmin(mom):.4f}, {np.nanmax(mom):.4f}]")
    print(f"Momentum mean: {np.nanmean(mom):.4f}")
    print(f"Valid observations: {np.sum(~np.isnan(mom)):,}")
