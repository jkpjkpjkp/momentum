"""Anomaly 6: Net operating assets. Hirshleifer, Hou, Teoh
and Zhang (2004) ﬁnd that net operating assets, deﬁned
as the difference on the balance sheet between all
operating assets and all operating liabilities scaled by
total assets, is a strong negative predictor of long-run
stock returns. They suggest that investors with limited
attention tend to focus on accounting proﬁtability,
neglecting information about cash proﬁtability, in which
case net operating assets, equivalently measured as the
cumulative difference between operating income and free
cash ﬂow, captures such a bias."""

import numpy as np
import h5py
from pathlib import Path

from main.utils import load_data


def net_operating_assets():
    """
    Compute Net Operating Assets (NOA) scaled by total assets.

    NOA = (Operating Assets - Operating Liabilities) / Total Assets

    Where:
    - Operating Assets = Total Assets - Cash - Financial Assets (trading)
    - Operating Liabilities = Total Liabilities - Total Debt

    High NOA is a negative predictor of future stock returns.

    Returns:
        (time, ticker, noa) arrays
    """
    total_assets = load_data("balance_sheet", "total_assets_mrq_0")
    total_liabilities = load_data("balance_sheet", "total_liabilities_mrq_0")
    cash = load_data("balance_sheet", "cash_equivalent_mrq_0")
    financial_assets = load_data("balance_sheet", "financial_asset_held_for_trading_mrq_0")
    short_term_loans = load_data("balance_sheet", "short_term_loans_mrq_0")
    long_term_loans = load_data("balance_sheet", "long_term_loans_mrq_0")

    # Replace NaN with 0 for items that may not exist for all firms
    cash = np.nan_to_num(cash, nan=0.0)
    financial_assets = np.nan_to_num(financial_assets, nan=0.0)
    short_term_loans = np.nan_to_num(short_term_loans, nan=0.0)
    long_term_loans = np.nan_to_num(long_term_loans, nan=0.0)

    operating_assets = total_assets - cash - financial_assets

    total_debt = short_term_loans + long_term_loans
    operating_liabilities = total_liabilities - total_debt

    noa_raw = operating_assets - operating_liabilities

    total_assets_safe = np.where(total_assets > 0, total_assets, np.nan)
    return noa_raw / total_assets_safe


if __name__ == "__main__":
    noa = net_operating_assets()
    print(f"NOA shape: {noa.shape}")
    print(f"NOA range: [{np.nanmin(noa):.4f}, {np.nanmax(noa):.4f}]")
    print(f"NOA mean: {np.nanmean(noa):.4f}")
    print(f"NOA median: {np.nanmedian(noa):.4f}")
    print(f"Valid observations: {np.sum(~np.isnan(noa)):,}")