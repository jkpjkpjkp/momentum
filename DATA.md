## structure of /data/share/data

All data stored in HDF5 (.h5) files with consistent structure:
- `ticker`: (N_stocks,) array of stock IDs (bytes, e.g. b'000004.XSHE')
- `time`: (N_time,) array of timestamps (int64, nanoseconds)
- `values`: (N_time, N_stocks) 2D array of values (float64)

### Directories

- `daily/` - Daily market data (4858 days, 5480 stocks)
  - close.h5, open.h5, high.h5, low.h5, volume.h5
  - return.h5 (daily returns)
  - prev_close.h5, limit_up.h5, limit_down.h5
  - ex_factor.h5 (adjustment factor)

- `balance_sheet/` - Quarterly balance sheet (3643 days aligned)
  - total_assets_mrq_0.h5, total_liabilities_mrq_0.h5, total_equity_mrq_0.h5
  - cash_equivalent_mrq_0.h5, current_assets_mrq_0.h5, current_liabilities_mrq_0.h5
  - Suffix: `_mrq_N` = most recent quarter (N=0 is latest), `_lyr_N` = last year

- `income_statement/` - Quarterly income (same time grid as balance_sheet)
  - net_profit_mrq_0.h5, net_profit_lyr_0.h5

- `shares/` - Share counts (same time grid as balance_sheet)
  - total_a.h5 (total A-shares), circulation_a.h5 (float), total.h5

- `index_weights/` - Index constituent weights
  - 000300.XSHG.h5 (CSI 300), 000905.XSHG.h5 (CSI 500), etc.

- `valid_mask/`, `universe/`, `industry/` - Filtering and classification data