I'll search for the exact mathematical formulations of these financial distress models from their original sources.
Let me search for publicly available sources containing these mathematical formulas.
Based on the provided text and corroborated by original source material, here are the exact mathematical formulations:

---

### **Anomaly 1: Campbell, Hilscher, and Szilagyi (2008) Failure Probability Model**

A **dynamic logit model** with both accounting and equity market variables:

**Logit Specification:**
$$
\begin{align*}
CHS_t = &-9.16 - 20.26 \cdot NIMTAAVG_t + 1.42 \cdot TLMTA_t \\
        &- 7.13 \cdot EXRETAVG_t + 1.41 \cdot SIGMA_t - 0.045 \cdot RSIZE_t \\
        &- 2.13 \cdot CASHMTA_t + 0.075 \cdot MB_t - 0.058 \cdot PRICE_t
\end{align*}
$$

**Failure Probability:**
$$
P(\text{bankruptcy}) = \frac{\exp(CHS_t)}{1 + \exp(CHS_t)}
$$

**Variable Definitions:**
- $NIMTAAVG_t$ = Moving average of net income / market value of total assets
- $TLMTA_t$ = Total liabilities / market value of total assets  
- $EXRETAVG_t$ = Moving average of log(gross firm stock return / S&P 500 return)
- $SIGMA_t$ = Standard deviation of daily stock returns (past 3 months)
- $RSIZE_t$ = Log(firm market cap) / log(S&P 500 market cap)
- $CASHMTA_t$ = Cash & short-term investments / market value of total assets
- $MB_t$ = Market-to-book ratio of equity
- $PRICE_t$ = Log(price per share)

**Moving Average Construction** ($\phi = 2^{-1/3}$):
$$
NIMTAAVG_{t-1,t-12} = \frac{1-\phi^3}{1-\phi^{12}}\sum_{k=0}^{9}\phi^k \cdot NIMTA_{t-3k-1,t-3k-3}
$$

---

### **Anomaly 2: Ohlson (1980) O-Score**

A **static logit model** using only accounting variables:

**O-Score Formula:**
$$
\begin{align*}
O\text{-score} = &-1.32 - 0.407 \cdot \log\left(\frac{\text{Total assets}}{\text{GNP price-level index}}\right) \\
                 &+ 6.03 \cdot \frac{\text{Total liabilities}}{\text{Total assets}} \\
                 &- 1.43 \cdot \frac{\text{Working capital}}{\text{Total assets}} \\
                 &+ 0.0757 \cdot \frac{\text{Current liabilities}}{\text{Current assets}} \\
                 &- 1.72 \cdot I(\text{Total liabilities} > \text{Total assets}) \\
                 &- 2.37 \cdot \frac{\text{Net income}}{\text{Total assets}} \\
                 &- 1.83 \cdot \frac{\text{Funds from operations}}{\text{Total liabilities}} \\
                 &+ 0.285 \cdot I(\text{Net loss for last 2 years}) \\
                 &- 0.521 \cdot \frac{\Delta \text{Net income}}{|\Delta \text{Net income}|}
\end{align*}
$$

**Bankruptcy Probability:**
$$
P(\text{bankruptcy}) = \frac{\exp(O\text{-score})}{1 + \exp(O\text{-score})}
$$

**Note**: The final term is a sign indicator for the change in net income (±1).

---

### **Key Distinction**

| Feature | CHS (2008) Model | Ohlson O-score |
|---------|------------------|----------------|
| **Type** | Dynamic logit | Static logit |
| **Variables** | Accounting + market (stock price, volatility, relative size, excess returns) | Accounting only |
| **Time structure** | Includes moving averages (dynamic) | Single-period values (static)



















Here's a practical implementation guide to calculate both models from your raw data fields:

---

### **1. Campbell-Hilscher-Szilagyi (2008) Failure Probability**

```python
# Formula:
CHS = -9.16 - 20.26*NIMTAAVG + 1.42*TLMTA - 7.13*EXRETAVG + 1.41*SIGMA - 0.045*RSIZE - 2.13*CASHMTA + 0.075*MB - 0.058*PRICE

P_failure = exp(CHS) / (1 + exp(CHS))
```

#### **Variable Mapping & Calculation:**

| Variable | Calculation from Your Fields | Notes |
|----------|------------------------------|-------|
| **NIMTAAVG** | `net_profit_mrq_ / (total_assets_mrq_ - total_equity_mrq_ + (close.h * total_a.h))` | Moving average with φ=2^(-1/3) decay over 12 months |
| **TLMTA** | `total_liabilities_mrq_ / (total_assets_mrq_ - total_equity_mrq_ + (close.h * total_a.h))` | Market-valued total assets |
| **CASHMTA** | `cash_equivalent_mrq_ / (total_assets_mrq_ - total_equity_mrq_ + (close.h * total_a.h))` | Same denominator |
| **EXRETAVG** | `MA[log(1 + return.h) - log(1 + SP500_return)]` | **Requires S&P 500 daily returns** (3-month MA) |
| **SIGMA** | `std_dev(daily_return)` | Calculate from `close.h` over past 3 months |
| **MB** | `(close.h * total_a.h) / total_equity_mrq_` | Market-to-book ratio |
| **RSIZE** | `log(close.h * total_a.h) / log(SP500_market_cap)` | **Requires S&P 500 market cap** |
| **PRICE** | `log(close.h)` | Current price per share |

#### **Moving Average Construction:**
For NIMTAAVG, use quarterly or monthly data with geometric decay:
```python
phi = 2**(-1/3) ≈ 0.7937
weights = [phi**0, phi**3, phi**6, phi**9]  # For 4 quarters
NIMTAAVG = weighted_average(NIMTA_quarters, weights)
```

---

### **2. Ohlson (1980) O-Score**

```python
# Formula:
O_score = -1.32 
          - 0.407*log(TA/GNP_deflator) 
          + 6.03*TL/TA 
          - 1.43*WC/TA 
          + 0.0757*CL/CA 
          - 1.72*I(TL > TA) 
          - 2.37*NI/TA 
          - 1.83*FFO/TL 
          + 0.285*I(NI_lt_0_2yrs) 
          - 0.521*(ΔNI/|ΔNI|)

P_bankruptcy = exp(O_score) / (1 + exp(O_score))
```

#### **Variable Mapping & Calculation:**

| Variable | Calculation from Your Fields | Notes |
|----------|------------------------------|-------|
| **TA** | `total_assets_mrq_` | Total assets (book value) |
| **GNP_deflator** | Use CPI index or set = 1 for relative analysis | **Macro data needed** |
| **TL** | `total_liabilities_mrq_` | Total liabilities |
| **WC** | `current_assets_mrq_ - current_liabilities_mrq_` | Working capital |
| **CA** | `current_assets_mrq_` | Current assets |
| **CL** | `current_liabilities_mrq_` | Current liabilities |
| **NI** | `net_profit_mrq_` | Net income |
| **FFO** | `net_profit_mrq_ + depreciation_estimate` | **Depreciation not directly available**; approximate using change in fixed assets or set = NI |
| **I(TL > TA)** | `1 if total_liabilities_mrq_ > total_assets_mrq_ else 0` | Balance sheet insolvency flag |
| **I(NI_lt_0_2yrs)** | `1 if net_profit_lyr_ < 0 and net_profit_mrq_ < 0 else 0` | Requires 2 years of data |
| **ΔNI/|ΔNI|** | `sign(net_profit_mrq_ - net_profit_lyr_)` | -1, 0, or 1 |

---

### **Implementation Notes & Data Limitations**

#### **Critical Issues:**

1. **Missing Market Data**: EXRETAVG and RSIZE require S&P 500 returns/market cap. **Solutions**:
   - Use local market index (CSI 300, if Chinese data)
   - Omit these variables (reduces model accuracy but still functional)
   - Set EXRETAVG=0 and RSIZE=1 as neutral values

2. **Share Count Identification**: 
   - `total_a.h` likely = total A-shares outstanding (in shares)
   - Verify units: If `total_a.h` is in millions, multiply by 1,000,000
   - Alternative: Use `circulation_a.h` (free float) if `total_a.h` unavailable

3. **GNP Deflator**: 
   - For US stocks: Use quarterly GNP deflator from FRED
   - For Chinese stocks: Use quarterly CPI from NBS or set = 1

4. **Depreciation (FFO)**: 
   - Ohlson's original FFO = Net Income + Depreciation + Deferred Taxes
   - **Proxy**: FFO ≈ `net_profit_mrq_` (simplification) or estimate from fixed asset changes

5. **Frequency Alignment**:
   - `_mrq_` fields = Most Recent Quarter (use for point-in-time calculation)
   - `_lyr_` fields = Last Year (for ΔNI and 2-year loss flag)
   - `*.h` fields = Daily market data (use latest values)

#### **Python Skeleton Code:**

```python
# Market value of assets (common for CHS)
def market_value_assets():
    market_equity = close_h * total_a_h  # total_a.h is shares outstanding
    book_assets = total_assets_mrq
    book_equity = total_equity_mrq
    return book_assets - book_equity + market_equity

mva = market_value_assets()

# CHS Components
NIMTA = net_profit_mrq / mva
TLMTA = total_liabilities_mrq / mva
CASHMTA = cash_equivalent_mrq / mva
MB = (close_h * total_a_h) / total_equity_mrq
PRICE = np.log(close_h)

# EXRETAVG and SIGMA require 3-month daily history
# SIGMA = close_h.pct_change().rolling(60).std()  # ~3 months

# O-Score Components
WC = current_assets_mrq - current_liabilities_mrq
CA = current_assets_mrq
CL = current_liabilities_mrq
NI_TA = net_profit_mrq / total_assets_mrq
TL_TA = total_liabilities_mrq / total_assets_mrq
WC_TA = WC / total_assets_mrq
CL_CA = current_liabilities_mrq / current_assets_mrq
FFO_TL = net_profit_mrq / total_liabilities_mrq  # Simplified
I_TL_GT_TA = 1 if total_liabilities_mrq > total_assets_mrq else 0
I_NI_2YR = 1 if net_profit_lyr < 0 and net_profit_mrq < 0 else 0
dNI_sign = np.sign(net_profit_mrq - net_profit_lyr)
```

---

### **Data Quality Checks**

- **Negative Book Equity**: If `total_equity_mrq_` ≤ 0, set MB = 0
- **Zero Assets**: Skip calculation if `total_assets_mrq_` = 0
- **Missing Values**: Replace with cross-sectional median or lagged values
- **Outliers**: Winsorize ratios at 1st/99th percentiles

Would you like me to provide a complete, runnable implementation for a specific market (US/China) with data fetching for the missing macro variables?