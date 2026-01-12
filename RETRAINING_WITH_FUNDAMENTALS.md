# Retraining Models with Fundamental Features

## Overview

The system now extracts **191 features** (176 technical + 12 fundamental + 3 metadata), up from 179 features. You need to retrain all 8 models to use the new fundamental data.

## Feature Breakdown

### Technical Features (176)
- Momentum indicators (RSI, Stochastic, ROC, Williams %R)
- Trend indicators (Moving averages, MACD, Bollinger Bands)
- Volume indicators (OBV, VWAP, Volume patterns)
- Volatility indicators (ATR, Standard deviation)
- Price patterns (Support/resistance, candlestick patterns)
- Statistical features (Sharpe ratio, linear regression)
- Market structure (Highs/lows, breakouts)
- Multi-timeframe indicators

### Fundamental Features (12) - NEW!
1. **beta** - Market correlation/volatility
2. **short_percent_of_float** - Short squeeze potential
3. **short_ratio** - Days to cover shorts
4. **analyst_target_price** - Analyst consensus
5. **profit_margin** - Profitability %
6. **debt_to_equity** - Financial leverage
7. **price_to_book** - Value metric
8. **price_to_sales** - Sales multiple
9. **return_on_equity** - ROE quality
10. **current_ratio** - Liquidity health
11. **revenue_growth** - Growth rate
12. **forward_pe** - Forward valuation

### Metadata Features (3)
- sector_code (GICS sector 1-60)
- market_cap_tier (1=large, 2=mid, 3=small)
- symbol_hash (unique identifier)

## Retraining Process

### Step 1: Refresh Fundamental Cache

Run this BEFORE training to populate the cache for all 80 stocks:

```bash
python refresh_fundamental_cache.py
```

**Time:** ~1.8 minutes
**Output:** `backend/data/fundamentals_cache.json`

This ensures all subsequent feature extractions use cached data (no slowdown).

### Step 2: Run Backtest Training

The existing training script should work, but you need to verify it uses the updated feature engineer.

**Check these files:**
- `backend/advanced_ml/backtest_training.py` - Main training script
- Uses `GPUFeatureEngineer` which now includes fundamentals

**Command:**
```bash
cd backend/advanced_ml
python backtest_training.py
```

**What happens:**
1. Downloads 7 years of price data for all 80 stocks
2. Extracts features (now 191 instead of 179)
3. Generates 10%/10% binary labels (strong BUY vs strong SELL)
4. Trains 8 GPU models:
   - XGBoost (standard)
   - XGBoost ET (Extra Trees)
   - XGBoost Hist
   - XGBoost DART
   - XGBoost GBLinear
   - XGBoost Approx
   - LightGBM
   - CatBoost
5. Trains meta-learner (stacks 8 model predictions)
6. Saves models to `backend/data/turbomode_models/`

**Expected time:** 15-30 minutes (GPU), 1-2 hours (CPU)

### Step 3: Verify Model Output

After training, check model feature counts:

```bash
python -c "
import sys
sys.path.insert(0, 'backend')
from advanced_ml.models.xgboost_model import XGBoostModel
import os

model = XGBoostModel(model_path='backend/data/turbomode_models/xgboost')
model.load()
print(f'Model expects {model.model.n_features_in_} features')
"
```

Should print: `Model expects 191 features`

### Step 4: Test Prediction Pipeline

Test that predictions work with new features:

```bash
python -c "
import sys
sys.path.insert(0, 'backend')
from turbomode.overnight_scanner import OvernightScanner

scanner = OvernightScanner()
result = scanner.scan_symbol('AAPL')
if result:
    print(f'✓ Prediction works! {result[\"signal_type\"]} at {result[\"confidence\"]:.1%}')
else:
    print('Symbol filtered or no signal')
"
```

## Cache Maintenance

### Daily Refresh (Automated)

Add to your scheduler (Task Scheduler on Windows, cron on Linux):

```bash
# Run every day at 6 AM before market open
python refresh_fundamental_cache.py
```

### Manual Refresh

Anytime you want fresh fundamental data:

```bash
python refresh_fundamental_cache.py
```

### Cache Location

File: `backend/data/fundamentals_cache.json`

Example entry:
```json
{
  "metadata": {
    "AAPL": {
      "timestamp": "2026-01-02T14:30:00",
      "source": "yfinance"
    }
  },
  "data": {
    "AAPL": {
      "beta": 1.107,
      "short_percent_of_float": 0.0083,
      "profit_margin": 0.26915,
      "debt_to_equity": 152.411,
      ...
    }
  }
}
```

## Performance Notes

### With Cache (Normal Operation)
- Feature extraction: Same speed as before (~0.76s per stock)
- No slowdown thanks to 24-hour cache
- Daily cache refresh: 1.8 minutes

### Without Cache (First Run)
- Feature extraction: +0.56s per stock (74% slower)
- Only happens once per day on first run
- Then cache kicks in for rest of day

## Troubleshooting

### "Model expects 179 features but got 191"

**Cause:** Using old models with new feature engineer

**Fix:** Retrain models (Step 2 above)

### "Failed to fetch fundamentals for {symbol}"

**Cause:** Network issue or invalid symbol

**Fix:** Safe defaults are used automatically. Check symbol is in 80-stock list.

### Cache file not found

**Cause:** Haven't run refresh script yet

**Fix:** Run `python refresh_fundamental_cache.py`

## Feature Importance Analysis

After retraining, you can analyze which fundamental features are most predictive:

```bash
cd backend/advanced_ml
python -c "
from models.xgboost_model import XGBoostModel
import pandas as pd

model = XGBoostModel(model_path='backend/data/turbomode_models/xgboost')
model.load()

# Get feature importances
importances = model.model.feature_importances_
feature_names = model.model.feature_names_in_

# Create dataframe
df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

# Show top 20
print('\\nTop 20 Most Important Features:')
print(df.head(20).to_string(index=False))

# Show fundamental features
fund_features = ['beta', 'short_percent_of_float', 'profit_margin',
                'debt_to_equity', 'analyst_target_price', 'price_to_book',
                'price_to_sales', 'return_on_equity', 'current_ratio',
                'revenue_growth', 'forward_pe', 'short_ratio']

print('\\nFundamental Feature Importance:')
fund_df = df[df['feature'].isin(fund_features)]
print(fund_df.to_string(index=False))
"
```

This will show you if fundamentals are actually helping predictions!

## Expected Improvements

With fundamental features, you should see:

1. **Better filtering of bad trades**
   - High debt stocks avoided
   - Unprofitable companies flagged
   - Buyout targets detected (via low growth + stable price)

2. **Improved confidence calibration**
   - Fundamentally strong stocks get higher confidence
   - Risky stocks get lower confidence

3. **Reduced false positives**
   - Technical patterns + weak fundamentals = filtered
   - Technical patterns + strong fundamentals = signal

4. **Better sector rotation detection**
   - Profit margins track sector health
   - Growth rates identify emerging trends

## Summary

1. ✅ Fundamental cache implemented (24-hour expiration)
2. ✅ GPU feature engineer updated (191 features)
3. ✅ No performance penalty with cache
4. ⏳ **TODO: Retrain 8 models with new features**
5. ⏳ **TODO: Test predictions with new models**
6. ⏳ **TODO: Analyze feature importance**

Run `python refresh_fundamental_cache.py` to get started!
