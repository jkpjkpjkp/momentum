"""Anomaly 5: Total accruals. Sloan (1996) shows that
firms with high accruals earn abnormal lower returns on
average than firms with low accruals and suggests that
investors overestimate the persistence of the accrual
component of earnings when forming earnings expectations.
Here, total accruals are calculated as changes in
noncash working capital minus depreciation expense
scaled by average total assets for the previous two
fiscal years."""

import numpy as np
import h5py
from pathlib import Path

from main.utils import load_data


def total_accruals():
    """
    Compute total accruals following Sloan (1996).

    Total Accruals = (ΔCA - ΔCash) - ΔCL - Depreciation
                     ----------------------------------------
                           Average Total Assets

    Where:
    - ΔCA = Change in current assets
    - ΔCash = Change in cash & equivalents
    - ΔCL = Change in current liabilities
    - Depreciation is estimated from change in accumulated depreciation
      (or fixed assets if not available)

    Since depreciation is not directly available, we use the balance sheet
    approach: Total Accruals = Δ(Noncash Working Capital) / Avg Total Assets

    Noncash Working Capital = (Current Assets - Cash) - Current Liabilities
    """
    # Load balance sheet data
    current_assets = load_data("balance_sheet", "current_assets_mrq_0")
    cash = load_data("balance_sheet", "cash_equivalent_mrq_0")
    current_liabilities = load_data("balance_sheet", "current_liabilities_mrq_0")
    total_assets = load_data("balance_sheet", "total_assets_mrq_0")

    # Load lagged data (mrq_4 = 4 quarters ago = 1 year ago)
    current_assets_lag = load_data("balance_sheet", "current_assets_mrq_4")
    cash_lag = load_data("balance_sheet", "cash_equivalent_mrq_4")
    current_liabilities_lag = load_data("balance_sheet", "current_liabilities_mrq_4")
    total_assets_lag = load_data("balance_sheet", "total_assets_mrq_4")

    # Noncash working capital = (Current Assets - Cash) - Current Liabilities
    nwc = (current_assets - cash) - current_liabilities
    nwc_lag = (current_assets_lag - cash_lag) - current_liabilities_lag

    # Change in noncash working capital
    delta_nwc = nwc - nwc_lag

    # Average total assets over past two years (current and 1 year ago)
    avg_total_assets = (total_assets + total_assets_lag) / 2.0

    # Avoid division by zero
    avg_total_assets = np.where(avg_total_assets > 0, avg_total_assets, np.nan)

    # Total accruals scaled by average total assets
    accruals = delta_nwc / avg_total_assets

    return accruals


if __name__ == "__main__":
    time, ticker, accruals = total_accruals()
    print(f"Time points: {len(time)}")
    print(f"Tickers: {len(ticker)}")
    print(f"Accruals shape: {accruals.shape}")
    print(f"Sample accruals (first 5 stocks, last date):")
    print(f"  {accruals[-1, :5]}")
    print(f"Accruals range: [{np.nanmin(accruals):.4f}, {np.nanmax(accruals):.4f}]")
    print(f"Accruals mean: {np.nanmean(accruals):.4f}")
