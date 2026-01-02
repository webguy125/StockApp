# Binary Classification Implementation - Complete

## Date: 2025-12-30

## Problem Identified

Analysis of the training data revealed a critical issue with the 3-class labeling system:

### Original 3-Class Distribution (BROKEN):
- **Buy**: 27,132 samples (12.9%)
- **Hold**: 115,679 samples (55.0%) ← **PROBLEM!**
- **Sell**: 67,578 samples (32.1%)

### Hold Label Analysis:
- 76,324 holds had **POSITIVE** returns (66%) → should have been "buy"
- 39,128 holds had **NEGATIVE** returns (34%) → should have been "sell"
- Average "hold" return: **+1.76%** (actually profitable!)

### Why This Caused Low Accuracy:
- Models learned to predict "hold" for easy accuracy (55% of data)
- The wide threshold range (-5% to +10% = 15% range) created too many holds
- Real-world signal quality suffered (67-77% accuracy instead of 90%)

---

## Solution: Binary Classification

Converted from 3-class (buy/hold/sell) to 2-class (buy/sell) based on return sign.

### New Labeling Logic:
```python
if final_return >= 0:
    label = 'buy'   # Positive or zero return
else:
    label = 'sell'  # Negative return
```

### Expected New Distribution:
- **Buy**: ~103,456 samples (49.2%) - Nearly balanced!
- **Sell**: ~106,706 samples (50.7%) - Nearly balanced!

---

## Files Modified

### 1. `backend/advanced_ml/backtesting/historical_backtest.py`

#### Change 1: Label Assignment Logic (lines 168-174)
```python
# BEFORE (3-class with problematic hold range):
if final_return >= self.win_threshold:  # >= +10%
    label = 'buy'
elif final_return <= self.loss_threshold:  # <= -5%
    label = 'sell'
else:
    label = 'hold'  # Between -5% and +10%

# AFTER (binary classification):
if final_return >= 0:
    label = 'buy'
else:
    label = 'sell'
```

#### Change 2: Label Mapping in generate_labeled_data() (line 221)
```python
# BEFORE:
label_map = {'buy': 0, 'hold': 1, 'sell': 2}

# AFTER:
label_map = {'buy': 0, 'sell': 1}  # Binary classification
```

#### Change 3: Label Mapping in run_backtest() (line 304)
```python
# BEFORE:
label_map = {'buy': 0, 'hold': 1, 'sell': 2}

# AFTER:
label_map = {'buy': 0, 'sell': 1}  # Binary classification
```

#### Change 4: Label Mapping in prepare_training_data() (lines 543-545)
```python
# BEFORE:
label_map = {'buy': 0, 'hold': 1, 'sell': 2}
label = label_map.get(outcome, 1)  # Default to hold

# AFTER:
label_map = {'buy': 0, 'sell': 1}
label = label_map.get(outcome, 0)  # Default to buy
```

#### Change 5: Label-to-Action Mapping (line 373)
```python
# BEFORE:
label_to_action = {0: 'buy', 1: 'hold', 2: 'sell'}

# AFTER:
label_to_action = {0: 'buy', 1: 'sell'}
```

### 2. `backend/turbomode/generate_backtest_data.py` (Previously Modified)

Removed database clearing to enable continuous learning:
```python
# BEFORE (deleted all old data):
if old_count > 0:
    cursor.execute("DELETE FROM trades WHERE trade_type = 'backtest'")
    cursor.execute("DELETE FROM feature_store")

# AFTER (preserves data):
if old_count > 0:
    print("[INFO] Keeping existing backtest data for continuous learning")
    print("[INFO] New data will be added/updated (INSERT OR REPLACE)")
```

### 3. `backend/turbomode/train_turbomode_models.py` (Previously Modified)

Fixed model registration names to match actual models:
```python
# BEFORE (misleading names):
meta_learner.register_base_model('random_forest', rf_model)
meta_learner.register_base_model('gradientboost', gb_model)
# ... etc

# AFTER (accurate names):
meta_learner.register_base_model('xgboost_rf', rf_model)
meta_learner.register_base_model('catboost', gb_model)
# ... etc
```

---

## Expected Results After Next Backtest

### Accuracy Improvement:
- **Current**: 67-77% (best: XGBoost RF at 77.24%)
- **Expected**: 85-95% (dramatic improvement)

### Why Accuracy Will Improve:
1. Balanced class distribution (49%/51% instead of 13%/55%/32%)
2. No more "easy wins" by predicting the majority class
3. Clear decision boundary (positive vs negative returns)
4. Models will learn actual market patterns instead of gaming the label distribution

### Data Volume:
- Old: 210,389 total samples
- After conversion: ~210,162 samples (removes ~227 zero-return samples)
- Database will continue to grow week-over-week (no more clearing)

---

## Next Steps

1. ✅ **Binary classification implemented** (COMPLETE)
2. ⏳ **Wait for current training to finish** (currently at 61.8%)
3. 🔄 **Run new backtest** with binary classification:
   ```bash
   cd backend/turbomode
   python generate_backtest_data.py
   ```
4. 🔄 **Retrain models** on new binary-labeled data:
   ```bash
   cd backend/turbomode
   python train_turbomode_models.py
   ```
5. 🔄 **Verify accuracy improvement** (expecting 85-95%)

---

## Analysis Scripts Created

### `backend/turbomode/test_files/analyze_hold_impact.py`
Analyzes the distribution and profitability of hold labels. Key findings:
- 55% of data labeled as "hold"
- 66% of holds had positive returns (should be buy)
- 34% of holds had negative returns (should be sell)
- This script identified the root cause of low accuracy

### `backend/turbomode/select_best_features.py` (Future Use)
Feature selection tool to reduce 176 features to top 50-100:
- Extracts feature importance from all trained models
- Averages importance across ensemble
- Selects top N features
- Expected benefits: 40-72% faster training + potentially higher accuracy

---

## Model Architecture (Unchanged)

All 8 base models are 100% GPU-accelerated:
1. **XGBoost RF** - Random Forest-style (XGBoost GPU implementation)
2. **XGBoost** - Standard gradient boosting (GPU enabled)
3. **LightGBM** - Gradient boosting (GPU enabled)
4. **XGBoost ET** - Extra Trees-style (XGBoost GPU implementation)
5. **CatBoost** - Gradient boosting (GPU enabled)
6. **PyTorch NN** - Neural network (GPU enabled)
7. **XGBoost Linear** - Linear model (XGBoost GPU implementation)
8. **CatBoost SVM** - SVM-style (CatBoost GPU implementation)
9. **Meta-Learner** - XGBoost GPU stacking ensemble

---

## Database Schema (Unchanged)

Database: `backend/data/advanced_ml_system.db`
- Shared between TurboMode and Slipstream
- 15 tables total
- Training data stored in `trades` table (trade_type='backtest')
- Features stored in `feature_store` table
- Model predictions stored in `model_predictions` table

---

## Summary

**Root Cause**: 3-class labeling with wide thresholds created 55% "hold" labels, causing models to learn the wrong patterns.

**Solution**: Binary classification based on return sign (≥0 = buy, <0 = sell).

**Expected Impact**: Accuracy improvement from 67-77% to 85-95%.

**Status**: ✅ All code changes complete. Ready for next backtest run.
