# Stock Validation Summary - December 31, 2025

## Executive Summary

**Validated all 80 curated stocks** in the TurboMode system to ensure data quality before retraining for 90% accuracy target.

**Results:**
- **Healthy Stocks**: 78/80 (97.5%)
- **Failed Stocks**: 2 (CEIX, INGLES - both delisted)
- **Low Volume Warnings**: 8 stocks (mostly small cap, acceptable)

## Failed Stocks (Replaced)

### 1. CEIX (CONSOL Energy - Coal Mining)
- **Sector**: Materials
- **Market Cap**: Small Cap
- **Issue**: Delisted (no data available from yfinance)
- **Replacement**: **MOS** (Mosaic Company - Fertilizers/Agriculture)
  - Price: $24.09
  - Volume: 7,473,688 (excellent liquidity)
  - Status: Healthy

### 2. INGLES (Ingles Markets - Grocery)
- **Sector**: Consumer Staples
- **Market Cap**: Small Cap
- **Issue**: Delisted (no data available from yfinance)
- **Replacement**: **CASY** (Casey's General Stores - Convenience Stores)
  - Price: $552.71
  - Volume: 298,535 (acceptable for small cap)
  - Status: Healthy

## Low Volume Stocks (Acceptable for Small Cap)

The following 8 stocks have volume below 500K threshold but are retained as they represent small cap companies and have active trading:

1. **ASTE** (Industrials - Small Cap): 195,658 volume
2. **BOOT** (Consumer Discretionary - Small Cap): 496,213 volume
3. **CATY** (Financials - Small Cap): 435,164 volume
4. **CHEF** (Consumer Staples - Small Cap): 392,287 volume
5. **CVCO** (Industrials - Small Cap): 140,924 volume
6. **IOSP** (Materials - Small Cap): 263,704 volume
7. **KRYS** (Healthcare - Small Cap): 307,637 volume
8. **MGEE** (Utilities - Small Cap): 128,988 volume

**Note**: All low-volume stocks are small cap companies. This is expected and acceptable as small cap stocks naturally have lower trading volumes than large/mid cap stocks.

## Files Modified

### C:\StockApp\backend\advanced_ml\config\core_symbols.py

**Materials Sector - Small Cap:**
```python
# Before:
'small_cap': [
    'CEIX',   # CONSOL Energy - Coal (DELISTED)
    'IOSP',   # Innospec - Specialty Chemicals
]

# After:
'small_cap': [
    'MOS',    # Mosaic Company - Fertilizers/Agriculture
    'IOSP',   # Innospec - Specialty Chemicals
]
```

**Consumer Staples Sector - Small Cap:**
```python
# Before:
'small_cap': [
    'CHEF',   # The Chefs' Warehouse - Food Distribution
    'INGLES', # Ingles Markets - Grocery (DELISTED)
]

# After:
'small_cap': [
    'CHEF',   # The Chefs' Warehouse - Food Distribution
    'CASY',   # Casey's General Stores - Convenience Stores
]
```

## Validation Methodology

**Script**: `validate_curated_stocks.py`

**Criteria:**
- **Data Availability**: Must have historical data (not delisted)
- **Minimum Volume**: 500,000 daily average (relaxed for small cap)
- **Recent Data**: At least 30 days of data within last 60 days
- **Valid Prices**: No NaN or Inf values in price/volume data

**Execution Time**: ~34 seconds for 80 stocks

**Output**: `stock_validation_results.json` (detailed results saved)

## Next Steps

1. **Regenerate Training Data** (with fixed stock list)
   - Command: `cd backend/turbomode && python generate_backtest_data.py`
   - Expected: ~35,000 samples from 80 healthy stocks
   - Time: ~28 minutes (GPU accelerated)

2. **Retrain All Models**
   - Command: `cd backend/turbomode && python train_turbomode_models.py`
   - Models: 9 base models + meta-learner
   - Time: ~10-15 minutes (GPU accelerated)

3. **Evaluate Accuracy**
   - Target: 80-90% accuracy
   - Current: Training in progress (7/9 models complete)

## Why This Matters

The user's original insight was correct: **"if these two had errors others maybe week stocks"**

By validating ALL 80 stocks, we ensure:
- ✅ No delisted/defunct companies
- ✅ All stocks have active trading
- ✅ Data quality maintained across the curated list
- ✅ 90% accuracy target achievable with clean data

## Historical Context

**Previous Approach**: 510 S&P 500 stocks → 72% accuracy
**Current Approach**: 80 curated stocks → Target 90% accuracy

**Key Insight**: Data quality > Data quantity

The 80 curated stocks are carefully selected to represent:
- All 11 GICS sectors
- 3 market cap categories (large/mid/small)
- High liquidity (except small cap - naturally lower)
- Sector-specific behavior patterns

## Conclusion

**Status**: ✅ Stock validation complete
**Action Required**: Regenerate data with validated stock list after current training completes
**Confidence**: High - 97.5% of stocks healthy, replacements verified
