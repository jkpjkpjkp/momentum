"""Anomaly 9: Asset growth. Cooper, Gulen, and Schill (2008) find companies
that grow their total assets more earn lower subsequent returns. They suggest
that this phenomenon is due to investors' initial overreaction to changes in
future business prospects implied by asset expansions. Asset growth is measured
as the growth rate of total assets in the previous fiscal year."""

import numpy as np

from main.utils import load_data


def asset_growth():
    """
    Compute asset growth following Cooper, Gulen, and Schill (2008).

    Asset Growth = (Total Assets_t - Total Assets_{t-1}) / Total Assets_{t-1}
                 = Total Assets_t / Total Assets_{t-1} - 1

    Higher asset growth predicts lower future returns.
    """
    total_assets = load_data("balance_sheet", "total_assets_mrq_0")
    total_assets_lag = load_data("balance_sheet", "total_assets_mrq_4")

    # Avoid division by zero
    total_assets_lag_safe = np.where(total_assets_lag > 0, total_assets_lag, np.nan)

    # Asset growth rate
    ag = (total_assets / total_assets_lag_safe) - 1

    return ag


if __name__ == "__main__":
    time, ticker, ag = asset_growth()
    print(f"Time periods: {len(time)}")
    print(f"Stocks: {len(ticker)}")
    print(f"Asset growth shape: {ag.shape}")
    print(f"Asset growth range: [{np.nanmin(ag):.4f}, {np.nanmax(ag):.4f}]")
    print(f"Asset growth mean: {np.nanmean(ag):.4f}")
    print(f"Valid observations: {np.sum(~np.isnan(ag)):,}")
