# 14-Day Swing Trade System Redesign
**Date**: 2026-01-21
**Session**: Major Architecture Redesign
**Status**: Implementation Complete, Ready for Retraining

---

## Executive Summary

This session implemented a complete redesign of the TurboMode ML system from a 1-day intraday TP/DD system to a true **14-day swing trading** system suitable for stock and options directional trading.

### Key Changes:
1. ✅ New 14-day swing label function (`compute_labels_14d_swing`)
2. ✅ Updated training pipeline to use 14-day horizon
3. ✅ Updated all documentation to reflect swing trade semantics
4. ⏳ Ready to retrain all 66 models (11 sectors × 6 models each)

---

## 1. Label Redesign

### **Old System (DEPRECATED):**
- **Horizon**: 1 day
- **Method**: Intraday TP/DD (take-profit/drawdown)
- **Thresholds**: ±5% intraday moves
- **Problem**: Too aggressive, not suitable for swing trading

### **New System (14-Day Swing):**
- **Horizon**: 14 trading days (~2-3 weeks)
- **Method**: Close-to-close return
- **Thresholds**: ±3% over 14 days
- **Formula**: `r_14d = (close[t+14] - close[t]) / close[t]`

### **Label Semantics (Unchanged):**
```python
0 = SELL: 14-day return <= -3% (bearish swing)
1 = HOLD: 14-day return between -3% and +3% (no edge)
2 = BUY: 14-day return >= +3% (bullish swing)
```

**IMPORTANT**: Class index mapping remains 0=SELL, 1=HOLD, 2=BUY everywhere.

---

## 2. Files Modified

### **sector_batch_trainer.py**
**Location**: `backend/turbomode/core_engine/sector_batch_trainer.py`

**Changes**:
1. Added `compute_labels_14d_swing()` function (lines 47-149)
2. Marked `compute_labels_1d_5pct()` as DEPRECATED
3. Updated `load_sector_data_once()` to use 14-day horizon
4. Updated module docstring to reflect swing trade architecture
5. Updated `run_sector_training()` docstring

**Key Function:**
```python
def compute_labels_14d_swing(trades: List[Dict], ohlcv_data: Dict) -> Dict:
    """
    Compute 14-day swing trade labels based on close-to-close returns.

    - Horizon: 14 trading days
    - Thresholds: ±3% return
    - Method: Close-to-close (not intraday TP/DD)
    """
```

### **fastmode_inference.py**
**Location**: `backend/turbomode/core_engine/fastmode_inference.py`

**Changes**:
1. Updated module docstring to reflect 14-day predictions
2. Added swing trade semantics documentation
3. Clarified that predictions represent 14-day outcomes

**No code changes** - inference logic remains identical, only documentation updated.

### **overnight_scanner.py**
**Location**: `backend/turbomode/core_engine/overnight_scanner.py`

**Pending Changes** (not yet applied):
1. Adjust entry thresholds from 60% to 55%
2. Update comments to reflect 14-day swing signals
3. Make thresholds configurable

---

## 3. Training Architecture

### **Model Structure (Per Sector):**
- **5 Base Models**:
  1. LightGBM-GPU
  2. CatBoost-GPU
  3. XGBoost-Hist-GPU
  4. XGBoost-Linear (CPU)
  5. RandomForest (CPU)

- **1 MetaLearner**:
  - LogisticRegression (multi-class)
  - Trained on stacked out-of-fold predictions
  - 15 input features (5 models × 3 classes)

### **Total Models:**
- 11 sectors × 6 models = **66 models**

### **Expected Training Time:**
- **Per Sector**: ~10-15 minutes
- **All 11 Sectors**: ~2-3 hours
- **Method**: GPU-accelerated ensemble training

---

## 4. Expected Label Distribution

### **Hypothesis:**
With 14-day ±3% thresholds, we expect:
- **BUY**: 15-25% (stocks moving up >3% in 14 days)
- **SELL**: 15-25% (stocks moving down >3% in 14 days)
- **HOLD**: 50-70% (stocks moving <±3%)

### **Why This is Better:**
1. **Actionable signals**: BUY/SELL should no longer be 0%
2. **Realistic thresholds**: 3% over 14 days is achievable
3. **Options-friendly**: 14-day horizon matches typical option expiry cycles
4. **Risk management**: HOLD correctly identifies low-confidence setups

---

## 5. Next Steps

### **Immediate (Ready to Execute):**

1. **Retrain All Models**:
   ```bash
   cd C:\StockApp\backend\turbomode\core_engine
   python train_all_sectors_optimized_orchestrator.py
   ```
   - Expected time: 2-3 hours
   - Will retrain all 66 models with 14-day labels
   - Saves to: `backend/turbomode/models/trained/<sector>/`

2. **Validate Label Distribution**:
   ```bash
   python analyze_label_distribution.py
   ```
   - Check BUY/SELL/HOLD percentages per sector
   - Verify labels are balanced (~15-25% BUY/SELL each)

3. **Update Scanner Thresholds**:
   - Change entry threshold from 60% to 55%
   - Update `overnight_scanner.py` comments

4. **Run Validation Scan**:
   ```bash
   python overnight_scanner.py
   ```
   - Verify BUY/SELL signals appear (not 0%)
   - Check signal distribution (should see 5-15% actionable signals)

### **Testing & Validation:**

1. **Sample Predictions**:
   ```bash
   python sample_base_model_outputs.py
   ```
   - Print 10-20 symbols with raw predictions
   - Verify BUY/SELL probabilities are non-zero
   - Check argmax distribution

2. **Backtesting**:
   - Run historical backtest on 14-day signals
   - Measure win rate, average return, max drawdown
   - Compare against 1-day system (deprecated)

---

## 6. Technical Details

### **Class Mapping (PRESERVED):**
```python
# Training labels
0 = SELL  # 14-day return <= -3%
1 = HOLD  # 14-day return between -3% and +3%
2 = BUY   # 14-day return >= +3%

# Inference mapping (unchanged)
class_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
probs[0] = prob_sell
probs[1] = prob_hold
probs[2] = prob_buy
```

### **OHLCV Data Loading:**
```python
# OLD: 1-day horizon
ohlcv_data = load_ohlcv_for_trades(db_path, trades, horizon_days=1)

# NEW: 14-day horizon
ohlcv_data = load_ohlcv_for_trades(db_path, trades, horizon_days=14)
```

### **Label Computation:**
```python
# Compute 14-day return
entry_price = trade["entry_price"]
exit_price = closes[exit_idx - 1]  # Close at day 14
r_14d = (exit_price - entry_price) / entry_price

# Classify
if r_14d >= 0.03:    outcome = 2  # BUY
elif r_14d <= -0.03: outcome = 0  # SELL
else:                outcome = 1  # HOLD
```

---

## 7. Risk & Considerations

### **Potential Issues:**
1. **Data Sparsity**: Need 14 days of future data for labeling
   - Solution: Training data should have 14+ days after entry
   - Old trades near present may get HOLD labels (insufficient data)

2. **Model Accuracy**: 14-day predictions harder than 1-day
   - Solution: Ensemble architecture reduces overfitting
   - MetaLearner combines multiple perspectives

3. **Signal Frequency**: Fewer signals than 1-day system
   - Expected: 5-15% of stocks show BUY/SELL signals
   - This is CORRECT - swing trading is selective

### **Mitigation:**
- Use 5-fold cross-validation for OOF predictions
- Log label distribution per sector for debugging
- Monitor prediction confidence (threshold at 55%)
- Keep old 1-day system as backup (marked DEPRECATED)

---

## 8. Success Metrics

### **Post-Retraining Validation:**

✅ **Label Distribution**:
- BUY: 15-25% across sectors
- SELL: 15-25% across sectors
- HOLD: 50-70% across sectors

✅ **Scanner Signals**:
- BUY signals: 10-30 stocks (out of 230)
- SELL signals: 10-30 stocks (out of 230)
- Distribution should vary with market conditions

✅ **Model Performance**:
- Accuracy: >40% (baseline 33% for 3-class)
- Macro F1: >0.35
- No class collapse (all 3 classes predicted)

---

## 9. Rollback Plan

If 14-day system fails validation:

1. **Restore 1-day labels**:
   ```python
   # In sector_batch_trainer.py line 337
   labels_dict = compute_labels_1d_5pct(trade_list, ohlcv_data)  # Restore old
   ```

2. **Revert horizon**:
   ```python
   # In sector_batch_trainer.py line 331
   ohlcv_data = load_ohlcv_for_trades(db_path, trades, horizon_days=1)  # Restore
   ```

3. **Retrain with old labels**:
   ```bash
   python train_all_sectors_optimized_orchestrator.py
   ```

---

## 10. Summary

### **Completed**:
- ✅ Implemented 14-day swing label function
- ✅ Updated training pipeline
- ✅ Updated all documentation
- ✅ Preserved class mapping (0=SELL, 1=HOLD, 2=BUY)

### **Ready to Execute**:
- ⏳ Retrain all 66 models (~2-3 hours)
- ⏳ Validate label distribution
- ⏳ Update scanner thresholds
- ⏳ Run validation scan

### **Expected Outcome**:
- BUY/SELL signals will appear (not 0%)
- 14-day predictions suitable for swing trading
- Options-friendly timeframe
- Realistic ±3% targets over 2-3 weeks

---

**End of Session Notes**
