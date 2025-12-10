"""Anomaly 11: Investment-to-assets. Titman, Wei, and Xie (2004) and Xing
(2008) show that higher past investment predicts abnormally lower future
returns. Titman, Wei, and Xie (2004) attribute this anomaly to investors'
initial underreactions to the overinvestment caused by managers' empire-
building behavior. Here, investment-to-assets is measured as the annual
change in gross property, plant, and equipment plus the annual change in
inventories scaled by the lagged book value of assets."""
import numpy as np
from main.utils import load_data


def investment_to_assets():
    """Investment-to-assets
    (Titman, Wei, and Xie (2004).)

    I/A = (ΔPPE + ΔInventory) / Total Assets_{t-1}

    Where:
    - ΔPPE = Change in gross property, plant, and equipment
    - ΔInventory = Change in inventories

    Higher investment-to-assets predicts lower future returns.
    """
    fixed_assets = load_data("balance_sheet", "total_fixed_assets_mrq_0")
    fixed_assets_lag = load_data("balance_sheet", "total_fixed_assets_mrq_4")
    inventory = load_data("balance_sheet", "inventory_mrq_0")
    inventory_lag = load_data("balance_sheet", "inventory_mrq_4")
    total_assets_lag = load_data("balance_sheet", "total_assets_mrq_4")

    # Replace NaN with 0 for inventory (some firms may not have inventory)
    inventory = np.nan_to_num(inventory, nan=0.0)
    inventory_lag = np.nan_to_num(inventory_lag, nan=0.0)

    # Change in fixed assets (proxy for PPE)
    delta_ppe = fixed_assets - fixed_assets_lag

    # Change in inventory
    delta_inv = inventory - inventory_lag

    # Total investment
    investment = delta_ppe + delta_inv

    total_assets_lag_safe = np.where(total_assets_lag > 0, total_assets_lag, np.nan)
    return investment / total_assets_lag_safe


if __name__ == "__main__":
    ia = investment_to_assets()
    print(f"Investment-to-assets shape: {ia.shape}")
    print(f"Investment-to-assets range: [{np.nanmin(ia):.4f}, {np.nanmax(ia):.4f}]")
    print(f"Investment-to-assets mean: {np.nanmean(ia):.4f}")
    print(f"Valid observations: {np.sum(~np.isnan(ia)):,}")
