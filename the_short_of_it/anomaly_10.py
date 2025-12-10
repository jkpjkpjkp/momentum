"""Anomaly 10: Return on assets. Fama and French (2006) find that more
profitable firms have higher expected returns than less profitable firms.
Chen, Novy-Marx, and Zhang (2010) show that firms with higher past return
on assets earn abnormally higher subsequent returns. Return on assets is
measured as the ratio of quarterly earnings to last quarter's assets.
Wang and Yu (2010) find that the anomaly exists primarily among firms with
high arbitrage costs and high information uncertainty, suggesting that
mispricing is a culprit."""
import numpy as np
from main.utils import load_data


def return_on_assets():
    """Return on assets 
    
    (Fama and French (2006) / Chen et al. (2010).)

    ROA = Quarterly Earnings / Last Quarter's Total Assets
        = Net Profit (MRQ) / Total Assets (MRQ-1)

    Higher ROA predicts higher future returns.
    """
    net_profit = load_data("income_statement", "net_profit_mrq_0")
    total_assets_lag = load_data("balance_sheet", "total_assets_mrq_1")

    total_assets_lag_safe = np.where(total_assets_lag > 0, total_assets_lag, np.nan)
    return net_profit / total_assets_lag_safe


if __name__ == "__main__":
    roa = return_on_assets()
    print(f"ROA shape: {roa.shape}")
    print(f"ROA range: [{np.nanmin(roa):.4f}, {np.nanmax(roa):.4f}]")
    print(f"ROA mean: {np.nanmean(roa):.4f}")
    print(f"Valid observations: {np.sum(~np.isnan(roa)):,}")
