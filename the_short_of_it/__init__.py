from .anomaly_1_2 import (
    financial_distress_chs,
    financial_distress_oscore,
    financial_distress_1,
    financial_distress_2,
)
from .anomaly_3_4 import net_stock_issues, composite_equity_issues
from .anomaly_5 import total_accruals
from .anomaly_6 import net_operating_assets
from .anomaly_7 import momentum
from .anomaly_8 import gross_profitability
from .anomaly_9 import asset_growth
from .anomaly_10 import return_on_assets
from .anomaly_11 import investment_to_assets

anomalies = [
    financial_distress_chs,      # 1: CHS failure probability
    financial_distress_oscore,   # 2: Ohlson O-Score
    net_stock_issues,            # 3: Net stock issues
    composite_equity_issues,     # 4: Composite equity issues
    total_accruals,              # 5: Total accruals
    net_operating_assets,        # 6: Net operating assets
    momentum,                    # 7: Momentum
    gross_profitability,         # 8: Gross profitability premium
    asset_growth,                # 9: Asset growth
    return_on_assets,            # 10: Return on assets
    investment_to_assets,        # 11: Investment-to-assets
]