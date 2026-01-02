# Dual Meta-Learner System - Testing Guide

## Overview
This system uses TWO meta-learners for better accuracy:
- **General Meta-Learner**: Trained on all 80 stocks (71.29% accuracy)
- **Specialized Meta-Learner**: Trained on top 10 stocks only (expected 75-80% accuracy)

## Quick Start (When You Get Home)

### Step 1: Run Stock Ranking Analysis (2-3 min)
```bash
cd C:\StockApp\backend\turbomode
python adaptive_stock_ranker.py
```

**What this does:**
- Analyzes backtest performance for all 80 stocks
- Calculates 30/60/90-day rolling win rates
- Ranks stocks by composite score
- Saves top 10 to `backend/data/stock_rankings.json`

**Expected output:**
```
TOP 10 MOST PREDICTABLE STOCKS
Rank Symbol  Score    WR_30d  WR_60d  WR_90d  Sig/Yr  Regime
1    AAPL    0.823    82.0%   80.5%   79.2%   28.5    stable
2    MSFT    0.791    79.5%   78.0%   77.5%   24.2    stable
...
```

---

### Step 2: Train Specialized Meta-Learner (5-10 min)
```bash
python train_specialized_meta_learner.py
```

**What this does:**
- Loads top 10 stocks from rankings.json
- Filters training data to ONLY those 10 stocks
- Uses existing 8 base models (already trained)
- Trains NEW specialized meta-learner
- Compares accuracy: General vs Specialized
- Saves to `backend/data/turbomode_models/meta_learner_top10/`

**Expected output:**
```
ACCURACY COMPARISON ON TOP 10 STOCKS
General Meta-Learner (all 80 stocks):     71.29%
Specialized Meta-Learner (top 10 stocks): 76.84%

🎯 Improvement: +5.55 percentage points
✅ Specialized meta-learner is BETTER for top 10 stocks!
```

---

### Step 3: Verify Files Created

Check these files exist:
```
C:\StockApp\backend\data\stock_rankings.json
C:\StockApp\backend\data\ranking_history.json
C:\StockApp\backend\data\turbomode_models\meta_learner_top10\model.json
C:\StockApp\backend\data\turbomode_models\meta_learner_top10\metadata.json
```

---

## Testing Scenarios

### Test 1: Compare Predictions on Same Stock

```python
# Test both meta-learners on AAPL (should be in top 10)
cd C:\StockApp
python

from backend.advanced_ml.models.meta_learner import MetaLearner
# ... load base models ...

# General meta-learner
general = MetaLearner(model_path="backend/data/turbomode_models/meta_learner")
general_pred = general.predict(base_predictions)
print(f"General: {general_pred['confidence']:.2%}")

# Specialized meta-learner
specialized = MetaLearner(model_path="backend/data/turbomode_models/meta_learner_top10")
specialized_pred = specialized.predict(base_predictions)
print(f"Specialized: {specialized_pred['confidence']:.2%}")
```

**Expected:**
- General: ~68-72% confidence
- Specialized: ~75-85% confidence (higher on top 10 stocks!)

---

### Test 2: Check Top 10 List

```python
import json

with open('backend/data/stock_rankings.json', 'r') as f:
    rankings = json.load(f)

top_10 = [stock['symbol'] for stock in rankings['top_10']]
print(f"Top 10: {', '.join(top_10)}")

# Print full details
for i, stock in enumerate(rankings['top_10'], 1):
    print(f"{i}. {stock['symbol']}: {stock['composite_score']:.3f} "
          f"(30d: {stock['win_rate_30d']*100:.1f}%)")
```

---

### Test 3: Verify Specialized Model Performance

```bash
# Re-run the specialized training script to see results
cd C:\StockApp\backend\turbomode
python train_specialized_meta_learner.py
```

Look for:
- ✅ Improvement > +3 percentage points = **SUCCESS**
- ⚠️ Improvement 0-3 points = **Marginal (still usable)**
- ❌ Improvement < 0 = **Problem (investigate)**

---

## Next Steps (After Testing)

### If Specialized Meta-Learner Performs Better:

**Update overnight scanner to use BOTH meta-learners**

Current scanner behavior (to be updated later):
```python
# Will scan ONLY top 10 stocks
# Will use specialized meta-learner for predictions
# Expected: 2-3 high-quality signals per night
```

---

## Troubleshooting

### Error: "Rankings file not found"
**Solution:** Run Step 1 first (adaptive_stock_ranker.py)

### Error: "No training data found for top 10 stocks"
**Problem:** Top 10 stocks don't have backtest data
**Solution:** Check that backtest data exists in advanced_ml_system.db

### Low Improvement (<1 percentage point)
**Possible causes:**
1. Top 10 stocks aren't significantly easier to predict
2. Not enough training samples per stock
3. General meta-learner already performing optimally

**Solution:** Still usable - even 1-2% improvement is valuable

### Specialized Performs Worse
**Possible causes:**
1. Overfitting on small dataset
2. Top 10 stocks are too diverse
3. Need more training data

**Solution:** Stick with general meta-learner for now

---

## Expected Timeline

| Task | Time | Status |
|------|------|--------|
| Run stock ranking | 2-3 min | ⏳ Pending |
| Train specialized meta-learner | 5-10 min | ⏳ Pending |
| Test predictions | 5 min | ⏳ Pending |
| Verify improvement | Instant | ⏳ Pending |
| **TOTAL** | **12-18 min** | |

---

## Success Criteria

✅ **Minimum Success:**
- Specialized meta-learner trains without errors
- Accuracy on top 10 stocks ≥ general meta-learner
- Improvement ≥ +1 percentage point

🎯 **Target Success:**
- Improvement ≥ +3 percentage points
- Accuracy on top 10 stocks ≥ 75%
- Clear confidence score differences

🚀 **Exceptional Success:**
- Improvement ≥ +5 percentage points
- Accuracy on top 10 stocks ≥ 78%
- Consistent improvement across all top 10 stocks

---

## Files Created

1. `train_specialized_meta_learner.py` - Training script
2. `DUAL_META_LEARNER_TESTING_GUIDE.md` - This guide
3. Session notes updated with implementation plan

## What To Do Tonight

1. Open this guide: `DUAL_META_LEARNER_TESTING_GUIDE.md`
2. Follow "Quick Start" steps 1-3
3. Check results
4. Report back findings

**Good luck! 🚀**
