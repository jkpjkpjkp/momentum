# Implementation Brief: Key Asset Pricing Factors from Raw Data

Based on the HDF5 data structure and standard factor definitions from the literature, here is a concise implementation guide for the 15 specified factors:

---

### **1. Size (SMB)**
**Data:** `daily/close.h5`, `shares/total_a.h5` (or `circulation_a.h5`)  
**Calculation:**  
- Market Cap = `close × shares`  
- Rank stocks by Market Cap each month  
- **Long:** Smallest 30% | **Short:** Largest 30%  
- Value-weight or equal-weight within each group

---

### **2. Book-to-Market (Value)**
**Data:** `balance_sheet/total_equity_mrq_0.h5`, `daily/close.h5`, `shares/total_a.h5`  
**Calculation:**  
- BM = `total_equity_mrq_0 / (close × shares)`  
- Forward-fill quarterly book equity to daily frequency  
- **Long:** Highest 30% BM | **Short:** Lowest 30% BM

---

### **3. Momentum (UMD)**
**Data:** `daily/return.h5`  
**Calculation:**  
- Cumulative return = `(t-12 to t-2)` (skip most recent month)  
- **Long:** Winners (top 30%) | **Short:** Losers (bottom 30%)  
- Exclude stocks with <12 months return history

---

### **4. Short-Term Reversals**
**Data:** `daily/return.h5`  
**Calculation:**  
- Prior month return = `t-1`  
- **Long:** Losers (bottom 30%) | **Short:** Winners (top 30%)  

---

### **5. Long-Term Reversals**
**Data:** `daily/return.h5`  
**Calculation:**  
- Cumulative return = `(t-60 to t-13)` (past 3-5 years)  
- **Long:** Past losers (bottom 30%) | **Short:** Past winners (top 30%)  

---

### **6. Accruals**
**Data:** `balance_sheet/total_assets_mrq_0.h5`, `current_assets_mrq_0.h5`, `cash_equivalent_mrq_0.h5`, `current_liabilities_mrq_0.h5`, `income_statement/net_profit_mrq_0.h5`  
**Calculation:**  
- Operating Accruals = `[(ΔCurrentAssets - ΔCash) - (ΔCurrentLiabilities - ΔDebt)] / AverageTotalAssets`  
- If operating cash flow unavailable, use balance sheet approach  
- **Long:** Lowest accruals (conservative) | **Short:** Highest accruals (aggressive)

---

### **7. Profitability**
**Data:** `income_statement/net_profit_mrq_0.h5`, `balance_sheet/total_assets_mrq_0.h5`  
**Calculation:**  
- ROA = `net_profit_mrq_0 / total_assets_mrq_0`  
- **Long:** Highest ROA (profitable) | **Short:** Lowest ROA  

---

### **8. Investment**
**Data:** `balance_sheet/total_assets_mrq_0.h5`  
**Calculation:**  
- Asset Growth = `(TotalAssets_t - TotalAssets_{t-4}) / TotalAssets_{t-4}` (quarterly YoY)  
- **Long:** Low investment (conservative) | **Short:** High investment (aggressive)  

---

### **9. Earnings-to-Price**
**Data:** `income_statement/net_profit_mrq_0.h5`, `daily/close.h5`, `shares/total_a.h5`  
**Calculation:**  
- E/P = `net_profit_mrq_0 / (close × shares)`  
- **Long:** Highest E/P (value) | **Short:** Lowest E/P  

---

### **10. Betting Against Beta (BAB)**
**Data:** `daily/return.h5`, `index_weights/000300.XSHG.h5` (market proxy)  
**Calculation:**  
- Rolling 12-month beta regression: `stock_return = α + β × market_return`  
- Rank by estimated β  
- **Long:** Lowest β (bottom 30%) | **Short:** Highest β (top 30%)  
- Leverage long/short legs to β=1 for market neutrality

---

### **11. Residual Variance**
**Data:** `daily/return.h5`, `index_weights/000300.XSHG.h5`  
**Calculation:**  
- Rolling 12-month CAPM regression  
- Compute residual variance = Var(stock_return - β × market_return)  
- **Long:** Highest residual variance | **Short:** Lowest residual variance  

---

### **12. Net Share Issues**
**Data:** `shares/total_a.h5`  
**Calculation:**  
- NSI = `(Shares_t - Shares_{t-12}) / Shares_{t-12}` (12-month change)  
- **Long:** Largest share reductions | **Short:** Largest share issuances  

---

### **13. Liquidity**
**Data:** `daily/volume.h5`, `shares/circulation_a.h5`, `daily/return.h5`  
**Calculation (Turnover):**  
- Turnover = `volume / shares` (monthly average)  
- **Long:** Highest turnover (liquid) | **Short:** Lowest turnover  

**Alternative (Pastor-Stambaugh):**  
- Regress daily returns on signed volume changes (complex; use turnover proxy)

---

### **14. Quality Minus Junk (QMJ)**
**Data:** Multiple: profitability, safety, growth metrics  
**Calculation (simplified):**  
- Combine z-scores of:  
  - Profitability: ROA  
  - Safety: Low leverage (total_equity/total_assets), low volatility  
  - Growth: Low asset growth  
- **Long:** High quality (top 30%) | **Short:** Low quality/junk (bottom 30%)  

---

### **15. Industry-Adjusted Momentum**
**Data:** `daily/return.h5`, `industry/` classification  
**Calculation:**  
- Compute industry returns using industry/ data  
- Stock return - Industry return = Industry-adjusted return  
- Use adjusted return for momentum ranking (t-12 to t-2)  

---

### **Key Implementation Notes**

**Data Alignment:**  
- Quarterly accounting data (`_mrq_0`) must be forward-filled to daily/monthly frequency  
- Use `valid_mask/` to filter for tradable stocks  
- Use `universe/` to define investment universe (e.g., exclude microcaps)  

**Portfolio Construction:**  
- Sort stocks into deciles or terciles based on NYSE breakpoints  
- For each factor, **long top decile, short bottom decile** (or 30%/30% splits)  
- Rebalance monthly (or quarterly for accounting-based factors)  
- Value-weight or equal-weight within portfolios  

**Point-in-Time:**  
- Use `_mrq_0` for most recent quarter data available at month-end  
- Avoid look-ahead bias: use only data available at portfolio formation date  

**Industry Neutrality:**  
- Use `industry/` classifications for industry-adjusted factors  
- Can also neutralize other factors by residualizing characteristics  

**Factor Returns:**  
- Factor return = Return of long portfolio - Return of short portfolio  
- For long-short factors (e.g., HML, SMB), ensure weights sum to zero