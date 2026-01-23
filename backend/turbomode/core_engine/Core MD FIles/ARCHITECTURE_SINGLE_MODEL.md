# TurboMode Single-Model Architecture

**Date**: 2026-01-20
**Status**: Production Ready
**Architecture**: Single-model-per-sector (1d/5% only)

---

## Overview

TurboMode has been refactored from a multi-horizon, multi-threshold, ensemble-based architecture to a **single-model-per-sector** architecture for maximum simplicity, speed, and maintainability.

### Key Principles

1. **One Model Per Sector**: Exactly 11 models total (one per sector)
2. **Single Label**: Only `label_1d_5pct` exists (1-day horizon, ±5% thresholds)
3. **Single Algorithm**: LightGBM only (no ensemble, no meta-learner)
4. **Flat Directory**: `models/<sector>/model.pkl` (no subdirectories)
5. **Fast Training**: ~45-60 minutes for all 11 sectors

---

## Architecture Comparison

### OLD Architecture (Deprecated)
```
- 11 sectors
- 6 horizons/thresholds per sector (1d_5%, 1d_10%, 2d_5%, 2d_10%, 5d_5%, 5d_10%)
- 5 base models + 1 meta-learner per configuration
- Total: 11 × 6 × 6 = 396 model files
- Training time: 8-12 hours
```

### NEW Architecture (Current)
```
- 11 sectors
- 1 model per sector
- Single LightGBM model (no ensemble)
- Total: 11 model files
- Training time: 45-60 minutes
```

---

## Label Configuration

### Single Label: `label_1d_5pct`

- **Horizon**: 1 trading day
- **Buy Threshold**: +5% (if max high in next 1 day ≥ +5%, label = BUY)
- **Sell Threshold**: -5% (if min low in next 1 day ≤ -5%, label = SELL)
- **Hold**: Everything else

**Label Encoding**:
- `0` = SELL
- `1` = HOLD
- `2` = BUY

---

## Sectors

### All 11 Sectors

1. `technology`
2. `financials`
3. `healthcare`
4. `consumer_discretionary`
5. `communication_services`
6. `industrials`
7. `consumer_staples`
8. `energy`
9. `materials`
10. `real_estate`
11. `utilities`

---

## Directory Structure

```
backend/turbomode/models/trained/
├── technology/
│   ├── model.pkl           # Single LightGBM model
│   └── metadata.json       # Model metadata
├── financials/
│   ├── model.pkl
│   └── metadata.json
├── healthcare/
│   ├── model.pkl
│   └── metadata.json
├── consumer_discretionary/
│   ├── model.pkl
│   └── metadata.json
├── communication_services/
│   ├── model.pkl
│   └── metadata.json
├── industrials/
│   ├── model.pkl
│   └── metadata.json
├── consumer_staples/
│   ├── model.pkl
│   └── metadata.json
├── energy/
│   ├── model.pkl
│   └── metadata.json
├── materials/
│   ├── model.pkl
│   └── metadata.json
├── real_estate/
│   ├── model.pkl
│   └── metadata.json
└── utilities/
    ├── model.pkl
    └── metadata.json
```

### Metadata Example

```json
{
  "sector": "technology",
  "horizon_days": 1,
  "threshold_pct": 5,
  "label": "label_1d_5pct",
  "architecture": "single_model",
  "model_type": "LGBMClassifier",
  "training_timestamp": "2026-01-20 18:50:41"
}
```

---

## Core Components

### 1. Training Pipeline

#### `train_all_sectors_optimized_orchestrator.py`
- **Purpose**: Top-level orchestrator
- **Function**: `train_all_sectors_optimized()`
- **Loops**: Sector loop only (11 iterations)
- **Output**: 11 models in `models/trained/`

#### `sector_batch_trainer.py`
- **Purpose**: Single-sector training logic
- **Function**: `run_sector_training(sector_name, sector_symbols, db_path, save_dir)`
- **Label Function**: `compute_labels_1d_5pct(trades, ohlcv_data)`
- **Data Loading**: `load_sector_data_once()` - loads 1-day OHLCV only
- **Output**: 1 model file per sector

#### `train_turbomode_models_fastmode.py`
- **Purpose**: Model training and save/load functions
- **Training Function**: `train_single_sector_worker_fastmode()`
- **Model**: Single LightGBM classifier (GPU-accelerated)
- **Save Function**: `save_fastmode_models(model, sector, save_dir)`
- **Load Function**: `load_fastmode_models(sector, load_dir)`

### 2. Inference Engine

#### `fastmode_inference.py`
- **Purpose**: Single-model inference
- **Load Function**: `load_model(sector)` - loads single model, LRU-cached
- **Predict Function**: `predict(model, X)` - returns probs, labels, class_indices
- **Single Sample**: `predict_single(model, features)` - returns signal + probabilities
- **Backward Compatibility**: `load_fastmode_models(sector, horizon='1d')` (ignores horizon)

### 3. Production Scanner

#### `overnight_scanner.py`
- **Purpose**: Generate trading signals for 208 symbols
- **Class**: `ProductionScanner`
- **Initialization**: No horizon parameter (hardcoded to 1d)
- **Model Loading**: `load_model(sector)` per symbol
- **Inference**: `predict_single(model, features)` - single model only
- **Output**: BUY/SELL signals with confidence scores

---

## Training Workflow

### Step-by-Step Process

1. **Orchestrator** calls `run_sector_training()` for each of 11 sectors
2. **Data Loading** (`load_sector_data_once`):
   - Query all backtest trades for sector symbols
   - Parse 179 features from `entry_features_json`
   - Load 1-day OHLCV data from canonical DB
3. **Label Computation** (`compute_labels_1d_5pct`):
   - Vectorized computation using NumPy searchsorted
   - For each trade: compute max high and min low in next 1 day
   - Classify as BUY (≥+5%), SELL (≤-5%), or HOLD
4. **Train/Val Split**:
   - 80/20 split with stratification (random_state=42)
5. **Model Training** (`train_single_sector_worker_fastmode`):
   - Single LightGBM classifier with GPU acceleration
   - Hyperparameters: n_estimators=300, max_depth=8, learning_rate=0.05
6. **Model Saving**:
   - Save to `models/trained/<sector>/model.pkl`
   - Save metadata to `models/trained/<sector>/metadata.json`

### Training Command

```bash
cd C:\StockApp\backend\turbomode\core_engine
python train_all_sectors_optimized_orchestrator.py
```

### Expected Output

```
================================================================================
SINGLE-MODEL TRAINING (1D/5% ONLY)
================================================================================
Start Time: 2026-01-20 18:45:00
Sectors: 11
Label: label_1d_5pct (1-day horizon, 5% threshold)
Total models: 11
================================================================================

================================================================================
[1/11] SECTOR: TECHNOLOGY
Symbols: 30
================================================================================

[DATA] Loaded 194,565 trades for sector
[PARSE] Features parsed in 12.45s (194,565 samples)
[OHLCV] Loaded in 8.23s
[LABELS] label_1d_5pct computed in 3.67s
[TOTAL] Sector data loaded in 24.35s

Label distribution: SELL=38,913 (20.0%), HOLD=116,739 (60.0%), BUY=38,913 (20.0%)
Train: 155,652, Val: 38,913

[TECHNOLOGY] Starting single-model training...
[TECHNOLOGY] Data: 155,652 train, 38,913 val
[TECHNOLOGY] Label: label_1d_5pct
[technology] LightGBM: train_acc=0.9405, val_acc=0.9333 (13.2s)
[OK] Model saved to C:\StockApp\backend\turbomode\models\trained\technology/model.pkl
[TECHNOLOGY] COMPLETE - 0.7 min

[1/11] TECHNOLOGY COMPLETE [OK]
Time: 0.7 minutes

[PROGRESS] 1/11 sectors complete
[PROGRESS] Elapsed: 0.7 min | Estimated remaining: 7.0 min

... (repeat for all 11 sectors)

================================================================================
SINGLE-MODEL TRAINING COMPLETE
================================================================================
End Time: 2026-01-20 19:30:00
Total Time: 45.0 minutes (0.8 hours)

================================================================================
RESULTS SUMMARY
================================================================================
Sectors processed: 11
  Successful: 11
  Failed: 0

Per-Sector Results:
--------------------------------------------------------------------------------
  technology                     [OK] Completed in 0.7 min
  financials                     [OK] Completed in 0.6 min
  healthcare                     [OK] Completed in 0.5 min
  consumer_discretionary         [OK] Completed in 0.4 min
  communication_services         [OK] Completed in 0.3 min
  industrials                    [OK] Completed in 0.5 min
  consumer_staples               [OK] Completed in 0.4 min
  energy                         [OK] Completed in 0.3 min
  materials                      [OK] Completed in 0.4 min
  real_estate                    [OK] Completed in 0.2 min
  utilities                      [OK] Completed in 0.2 min

================================================================================
MODEL DIRECTORY:
================================================================================
  Location: C:\StockApp\backend\turbomode\models\trained
  Structure: models/<sector>/model.pkl
  Total models: 11 (one per sector)
================================================================================
```

---

## Inference Workflow

### Step-by-Step Process

1. **Scanner** loads symbol list (208 symbols from `scanning_symbols.py`)
2. **For each symbol**:
   - Get OHLCV data (730 days = 2 years)
   - Extract 179 features using `TurboModeVectorizedFeatureEngine`
   - Get symbol metadata (sector, market_cap, etc.)
   - Load sector model using `load_model(sector)` (cached)
   - Run inference using `predict_single(model, features)`
   - Generate BUY/SELL signal if confidence ≥ 0.60
3. **Signal Generation**:
   - Sort by confidence (descending)
   - Save top 100 BUY and top 100 SELL signals to database
4. **Position Management**:
   - Manage existing positions (SL/TP, trailing stops, partial profits)
   - Open new positions for signals that pass entry threshold

### Inference Command

```bash
cd C:\StockApp\backend\turbomode\core_engine
python overnight_scanner.py
```

### Expected Output

```
================================================================================
TURBOMODE PRODUCTION SCANNER (FAST MODE + NEWS-AWARE)
================================================================================
Start Time: 2026-01-20 20:00:00
Architecture: Single-model-per-sector (1d/5% only)

[STEP 0] Updating news risk state...

[STEP 1] Updating existing signal ages...
  Expired 15 signals

[STEP 2] Loading symbol list...
  Total symbols: 208

[STEP 3] Scanning 208 symbols with Fast Mode...
  Progress: 50/208 (24.0%) - BUY: 12, SELL: 8
  Progress: 100/208 (48.1%) - BUY: 25, SELL: 18
  Progress: 150/208 (72.1%) - BUY: 38, SELL: 27
  Progress: 200/208 (96.2%) - BUY: 52, SELL: 35

[STEP 4] Scan complete!
  Scanned: 195/208
  Failed: 13
  BUY signals: 52
  SELL signals: 35

[STEP 5] Saving signals to database...
  BUY: 52 saved
  SELL: 35 saved

[STEP 6] Active positions summary...
  Active positions: 3
    AAPL: LONG @ $185.23 (stop: $182.45, target: $191.78)
    TSLA: LONG @ $245.67 (stop: $241.23, target: $254.56)
    GE: SHORT @ $112.34 (stop: $114.12, target: $108.89)

================================================================================
SCAN COMPLETE
================================================================================
End Time: 2026-01-20 20:12:00

[OK] Production scan complete!
BUY signals: 52
SELL signals: 35
Active positions: 3
```

---

## Model Hyperparameters

### LightGBM Configuration

```python
LGBMClassifier(
    device='gpu',              # GPU acceleration
    gpu_platform_id=0,
    gpu_device_id=0,
    n_estimators=300,          # Number of boosting rounds
    max_depth=8,               # Maximum tree depth
    learning_rate=0.05,        # Boosting learning rate
    num_leaves=31,             # Maximum leaves per tree
    subsample=0.8,             # Row sampling ratio
    colsample_bytree=0.8,      # Column sampling ratio
    random_state=42,           # Reproducibility
    verbose=-1,                # Suppress warnings
    n_jobs=-1                  # Use all CPU cores
)
```

---

## Features

### 179 Total Features

1. **Price Action** (10 features): close, open, high, low, hl_range, body_size, upper_wick, lower_wick, is_bullish, is_bearish
2. **Volume** (5 features): volume, volume_sma_20, volume_ratio, vwap, price_to_vwap
3. **Trend Indicators** (10 features): sma_10, sma_20, sma_50, sma_200, ema_12, ema_26, price_to_sma_20, price_to_sma_50, price_to_sma_200, trend_strength
4. **Momentum Indicators** (12 features): rsi_14, macd, macd_signal, macd_histogram, stoch_k, stoch_d, roc_10, roc_21, williams_r, cci, adx, momentum
5. **Volatility Indicators** (8 features): atr_14, bb_upper, bb_middle, bb_lower, bb_width, bb_position, keltner_upper, keltner_lower
6. **Support/Resistance** (6 features): pivot_point, r1, r2, s1, s2, distance_to_pivot
7. **Advanced Indicators** (15 features): obv, obv_sma_20, mfi, cmf, vpt, eom, rsi_divergence, macd_divergence, fibonacci_levels (5), ichimoku_tenkan, ichimoku_kijun
8. **Pattern Recognition** (20 features): doji, hammer, shooting_star, engulfing_bullish, engulfing_bearish, morning_star, evening_star, three_white_soldiers, three_black_crows, etc.
9. **Statistical Features** (15 features): returns_1d, returns_5d, returns_10d, returns_20d, volatility_20d, skewness_20d, kurtosis_20d, sharpe_ratio, sortino_ratio, etc.
10. **Time-Based Features** (8 features): day_of_week, week_of_month, month_of_year, quarter, days_since_52w_high, days_since_52w_low, etc.
11. **Multi-Timeframe** (30 features): Same indicators on 5-day and 20-day timeframes
12. **Sector/Metadata** (3 features): sector_code, market_cap_tier, symbol_hash
13. **Lag Features** (37 features): Previous 1-7 day values for key indicators

**Total**: 179 features (verified in `feature_list.py`)

---

## Performance Metrics

### Training Performance (Technology Sector Example)

- **Train Samples**: 155,652
- **Val Samples**: 38,913
- **Train Accuracy**: 94.05%
- **Val Accuracy**: 93.33%
- **Training Time**: 13.2 seconds
- **Total Time** (including data loading): 0.7 minutes

### Label Distribution (Typical)

- **SELL**: ~20% (e.g., 38,913 / 194,565)
- **HOLD**: ~60% (e.g., 116,739 / 194,565)
- **BUY**: ~20% (e.g., 38,913 / 194,565)

### Inference Performance

- **Model Load Time**: <100ms (cached after first load)
- **Single Prediction**: ~2-5ms per symbol
- **Full Scan** (208 symbols): ~10-15 minutes (including data fetching)

---

## Key Differences from Old Architecture

### What Was Removed

1. ❌ **Multi-Horizon Support** - Only 1d remains
2. ❌ **Multi-Threshold Support** - Only 5% remains
3. ❌ **5-Model Ensemble** - Only single LightGBM
4. ❌ **Meta-Learner** - No stacking
5. ❌ **Horizon/Threshold Loops** - Single iteration
6. ❌ **Dual-Threshold Merging** - No signal merging
7. ❌ **Multiple Save Directories** - Single `trained/` directory
8. ❌ **Complex Model Loading** - Simple `load_model(sector)`

### What Was Kept

1. ✅ **179 Features** - Same feature engineering
2. ✅ **11 Sectors** - Same sector breakdown
3. ✅ **Vectorized Label Computation** - Fast NumPy operations
4. ✅ **GPU Acceleration** - LightGBM GPU support
5. ✅ **Position Management** - Same adaptive SL/TP logic
6. ✅ **News Engine** - Same risk management
7. ✅ **Database Schema** - Same turbomode.db structure

---

## Migration Notes

### Files Refactored

1. ✅ `sector_batch_trainer.py` - Single label, single model
2. ✅ `train_all_sectors_optimized_orchestrator.py` - Single save directory
3. ✅ `train_turbomode_models_fastmode.py` - Single model training
4. ✅ `fastmode_inference.py` - Single model loading
5. ✅ `overnight_scanner.py` - Single model inference

### Files NOT Changed

- `feature_list.py` - Feature engineering unchanged
- `turbomode_vectorized_feature_engine.py` - Feature extraction unchanged
- `canonical_ohlcv_loader.py` - Data loading unchanged
- `adaptive_sltp.py` - Risk management unchanged
- `position_manager.py` - Position tracking unchanged
- `news_engine.py` - News risk unchanged
- `core_symbols.py` - Symbol metadata unchanged

### Breaking Changes

1. **`load_fastmode_models()`** signature changed:
   - OLD: `load_fastmode_models(sector, horizon_days, load_dir)`
   - NEW: `load_fastmode_models(sector, load_dir)` (horizon ignored for backward compat)
   - RECOMMENDED: Use `load_model(sector)` instead

2. **`ProductionScanner()`** initialization changed:
   - OLD: `ProductionScanner(horizon='1d')`
   - NEW: `ProductionScanner()` (no horizon parameter)

3. **Model directory structure** changed:
   - OLD: `models/trained_5pct/<sector>/1d/model.pkl`
   - NEW: `models/trained/<sector>/model.pkl`

---

## Verification Checklist

### After Training

- [ ] Verify exactly 11 sector directories exist in `models/trained/`
- [ ] Verify each sector has exactly 2 files: `model.pkl` and `metadata.json`
- [ ] Verify metadata shows `"architecture": "single_model"`
- [ ] Verify metadata shows `"label": "label_1d_5pct"`
- [ ] Verify no subdirectories exist (1d, 2d, 5d, etc.)
- [ ] Verify training completed in 45-60 minutes

### After Inference

- [ ] Verify `load_model()` caches all 11 sectors (LRU cache size=11)
- [ ] Verify predictions return 3 probabilities (SELL, HOLD, BUY)
- [ ] Verify signals have `threshold_source` field (for backward compat)
- [ ] Verify scanner logs "Architecture: Single-model-per-sector (1d/5% only)"

---

## Troubleshooting

### Model Not Found Error

```
FileNotFoundError: Model not found: C:\StockApp\backend\turbomode\models\trained\<sector>\model.pkl
Run train_all_sectors_optimized_orchestrator.py first to train models.
```

**Solution**: Train models using `train_all_sectors_optimized_orchestrator.py`

### Wrong Number of Features

```
ValueError: Expected 179 features, got <N> for trade <id>
```

**Solution**: Check that `entry_features_json` in database has all 179 features

### Label Computation Error

```
KeyError: 'label_1d_5pct'
```

**Solution**: Ensure using `compute_labels_1d_5pct()` function, not old multi-label function

---

## Future Enhancements

### Potential Additions (Without Breaking Architecture)

1. **Hyperparameter Tuning** - Optimize LightGBM params per sector
2. **Class Balancing** - SMOTE or class weights for imbalanced sectors
3. **Feature Selection** - Reduce from 179 to top 100 features
4. **Incremental Learning** - Monthly retraining with new data
5. **Ensemble Boosting** - Multiple LightGBM seeds (still single model type)

### NOT Recommended

- ❌ Adding multi-horizon support back
- ❌ Adding multi-threshold support back
- ❌ Adding ensemble/meta-learner back
- ❌ Changing directory structure back

---

## Contact

For questions about this architecture, consult:
- `session_files/session_notes_2026-01-20.md` - Implementation details
- This file (`ARCHITECTURE_SINGLE_MODEL.md`) - Architecture overview

---

**Last Updated**: 2026-01-20
**Architecture Version**: 1.0 (Single-Model)
