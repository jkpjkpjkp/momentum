"""Anomaly 9: Asset growth. Cooper, Gulen, and Schill (2008) find companies
that grow their total assets more earn lower subsequent returns. They suggest
that this phenomenon is due to investors' initial overreaction to changes in
future business prospects implied by asset expansions. Asset growth is measured
as the growth rate of total assets in the previous fiscal year."""
import numpy as np
from main.utils import load_data


def asset_growth():
    """Asset growth rate"""
    total_assets = load_data("balance_sheet", "total_assets_mrq_0")
    total_assets_lag = load_data("balance_sheet", "total_assets_mrq_4")

    total_assets_lag_safe = np.where(total_assets_lag > 0, total_assets_lag, np.nan)
    return (total_assets / total_assets_lag_safe) - 1


if __name__ == "__main__":
    ag = asset_growth()
    print(f"Asset growth shape: {ag.shape}")
    print(f"Asset growth range: [{np.nanmin(ag):.4f}, {np.nanmax(ag):.4f}]")
    print(f"Asset growth mean: {np.nanmean(ag):.4f}")
    print(f"Valid observations: {np.sum(~np.isnan(ag)):,}")
