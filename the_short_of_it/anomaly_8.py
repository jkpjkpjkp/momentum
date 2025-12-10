"""Anomaly 8: Gross profitability premium. Novy-Marx (2010) discovers that
sorting on gross profit-to-assets creates abnormal benchmark-adjusted returns,
with more profitable firms having higher returns than less profitable ones.
Novy-Marx argues that gross profits scaled by assets is the cleanest accounting
measure of true economic profitability. The farther down the income statement
one goes, the more polluted profitability measures become, and the less related
they are to true economic profitability."""

import numpy as np

from main.utils import load_data


def gross_profitability():
    """
    Compute gross profitability following Novy-Marx (2010).

    Gross Profitability = Gross Profit / Total Assets
                        = (Revenue - COGS) / Total Assets

    Since gross profit may not be directly available, we approximate using:
        Gross Profit = Operating Revenue - Operating Costs

    Higher gross profitability predicts higher future returns.

    """
    total_assets = load_data("balance_sheet", "total_assets_mrq_0")
    operating_revenue = load_data("income_statement", "operating_revenue_mrq_0")
    operating_costs = load_data("income_statement", "operating_costs_mrq_0")

    # Gross profit = Operating Revenue - Operating Costs
    gross_profit = operating_revenue - operating_costs

    # Scale by total assets
    total_assets_safe = np.where(total_assets > 0, total_assets, np.nan)

    gp = gross_profit / total_assets_safe

    return gp


if __name__ == "__main__":
    time, ticker, gp = gross_profitability()
    print(f"Time periods: {len(time)}")
    print(f"Stocks: {len(ticker)}")
    print(f"Gross profitability shape: {gp.shape}")
    print(f"Gross profitability range: [{np.nanmin(gp):.4f}, {np.nanmax(gp):.4f}]")
    print(f"Gross profitability mean: {np.nanmean(gp):.4f}")
    print(f"Valid observations: {np.sum(~np.isnan(gp)):,}")
