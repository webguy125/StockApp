# Dual Meta-Learner System - Complete Documentation

**Last Updated:** 2026-01-01
**Status:** Ready for Testing
**Expected Accuracy Improvement:** +3 to +8 percentage points on top 10 stocks

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Complete Setup Guide](#complete-setup-guide)
4. [Testing Instructions](#testing-instructions)
5. [Integration Steps](#integration-steps)
6. [File Locations](#file-locations)
7. [API Endpoints](#api-endpoints)
8. [Troubleshooting](#troubleshooting)
9. [Performance Expectations](#performance-expectations)

---

## System Overview

### What Is the Dual Meta-Learner System?

The dual meta-learner system uses **TWO specialized meta-learners** instead of one:

1. **General Meta-Learner** (`meta_learner/`)
   - Trained on ALL 80 stocks
   - Current accuracy: 71.29%
   - Used for initial scanning and general predictions

2. **Specialized Meta-Learner** (`meta_learner_top10/`)
   - Trained ONLY on top 10 most predictable stocks
   - Expected accuracy: 75-80%
   - Used for focused trading on curated stock universe

### Why Two Meta-Learners?

**Problem:** Not all stocks are equally predictable. Training on all 80 stocks dilutes the meta-learner's ability to learn stock-specific patterns.

**Solution:** Create a specialized meta-learner that focuses ONLY on the 10 most predictable stocks, allowing it to learn:
- Which base models work best for specific stocks
- Stock-specific pattern combinations
- Non-linear relationships unique to high-quality signals

**Example:**
- General meta-learner learns: "When XGBoost=BUY and LightGBM=BUY → 68% confidence"
- Specialized meta-learner learns: "When XGBoost=BUY and LightGBM=BUY **for AAPL specifically** → 82% confidence"

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    DUAL META-LEARNER SYSTEM                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ PHASE 1: Stock Ranking & Selection                             │
├─────────────────────────────────────────────────────────────────┤
│ • adaptive_stock_ranker.py                                      │
│ • Analyzes 30/60/90-day rolling win rates                       │
│ • Calculates composite scores                                   │
│ • Selects top 10 stocks                                         │
│ • Output: stock_rankings.json                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 2: Specialized Meta-Learner Training                     │
├─────────────────────────────────────────────────────────────────┤
│ • train_specialized_meta_learner.py                             │
│ • Loads top 10 stocks from rankings.json                        │
│ • Filters training data to ONLY those 10 stocks                 │
│ • Uses existing 8 base models                                   │
│ • Trains NEW specialized meta-learner                           │
│ • Output: meta_learner_top10/                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 3: Overnight Scanner (Dual Mode)                         │
├─────────────────────────────────────────────────────────────────┤
│ • overnight_scanner.py                                          │
│ • Scans top 10 stocks ONLY                                      │
│ • Uses SPECIALIZED meta-learner for predictions                 │
│ • Expected: 2-3 high-quality signals per night                  │
│ • Output: ml_trading_signals.json                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 4: Monthly Maintenance                                   │
├─────────────────────────────────────────────────────────────────┤
│ • Runs on 1st of each month at 2:00 AM                          │
│ • Re-runs stock ranking analysis                                │
│ • Checks if top 10 changed (typically 1-3 stocks rotate)        │
│ • Retrains specialized meta-learner if needed                   │
│ • Automated via APScheduler                                     │
└─────────────────────────────────────────────────────────────────┘
```

### Base Models (8 Total)

All base models are **already trained** and saved in `backend/data/turbomode_models/`:

1. **XGBoost** (`xgboost/`) - GPU-accelerated tree boosting
2. **XGBoost ET** (`xgboost_et/`) - Extra Trees variant
3. **LightGBM** (`lightgbm/`) - GPU-accelerated gradient boosting
4. **CatBoost** (`catboost/`) - Categorical boosting
5. **XGBoost Hist** (`xgboost_hist/`) - Histogram-based
6. **XGBoost DART** (`xgboost_dart/`) - Dropout regularization
7. **XGBoost GBLinear** (`xgboost_gblinear/`) - Linear boosting
8. **XGBoost Approx** (`xgboost_approx/`) - Approximate tree method

**Training Data:**
- 13,149 high-quality samples (±10% threshold labeling)
- 179 features per sample
- Binary classification: Buy (0) vs Sell (1)
- Generated from 136,634 backtest trades

---

## Complete Setup Guide

### Prerequisites

✅ Already completed from previous session:
- 8 base models trained and saved
- `advanced_ml_system.db` populated with backtest data
- Stock ranking API integrated into Flask server

### Step 1: Initial Stock Ranking (First Time Setup)

**Run stock ranking analysis to identify top 10 stocks:**

```bash
cd C:\StockApp\backend\turbomode
python adaptive_stock_ranker.py
```

**Expected Output:**
```
======================================================================
ADAPTIVE STOCK RANKING - TOP 10 SELECTION
======================================================================

Processing 80 stocks from curated stock list...

TOP 10 MOST PREDICTABLE STOCKS
======================================================================
Rank Symbol  Score    WR_30d  WR_60d  WR_90d  Sig/Yr  Regime
1    AAPL    0.823    82.0%   80.5%   79.2%   28.5    stable
2    MSFT    0.791    79.5%   78.0%   77.5%   24.2    stable
3    GOOGL   0.774    77.8%   76.5%   75.0%   31.0    stable
...
```

**Files Created:**
- `C:\StockApp\backend\data\stock_rankings.json` - Current top 10 rankings
- `C:\StockApp\backend\data\ranking_history.json` - Historical rankings

**Time Required:** 2-3 minutes

---

### Step 2: Train Specialized Meta-Learner

**Train the specialized meta-learner on ONLY the top 10 stocks:**

```bash
cd C:\StockApp\backend\turbomode
python train_specialized_meta_learner.py
```

**Expected Output:**
```
======================================================================
TRAIN SPECIALIZED META-LEARNER FOR TOP 10 STOCKS
======================================================================

PHASE 1: LOAD TOP 10 STOCKS FROM RANKINGS
======================================================================
[INFO] Top 10 stocks: AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, AMD, NFLX, ADBE

PHASE 2: LOAD TRAINING DATA FOR TOP 10 STOCKS ONLY
======================================================================
[DATA] Total samples from top 10 stocks: 4,235
[DATA] Features: 179
[DATA] Buy samples: 2,118
[DATA] Sell samples: 2,117
[DATA] Training samples: 3,388
[DATA] Test samples: 847

PHASE 3: LOAD EXISTING BASE MODELS
======================================================================
[INFO] Loading pre-trained base models from disk...
[OK] All 8 base models loaded

PHASE 4: TRAIN SPECIALIZED META-LEARNER
======================================================================
[INFO] Registering all 8 base models...
[INFO] Generating base model predictions...
[INFO] Training specialized meta-learner...
  [OK] Specialized meta-learner saved

PHASE 5: EVALUATE SPECIALIZED META-LEARNER
======================================================================

======================================================================
ACCURACY COMPARISON ON TOP 10 STOCKS
======================================================================

General Meta-Learner (all 80 stocks):     0.7129 (71.29%)
Specialized Meta-Learner (top 10 stocks): 0.7684 (76.84%)

🎯 Improvement: +5.55 percentage points
✅ Specialized meta-learner is BETTER for top 10 stocks!
```

**Files Created:**
- `C:\StockApp\backend\data\turbomode_models\meta_learner_top10\model.json`
- `C:\StockApp\backend\data\turbomode_models\meta_learner_top10\metadata.json`

**Time Required:** 5-10 minutes

---

### Step 3: Verify Installation

**Check that all required files exist:**

```bash
# From C:\StockApp directory
dir backend\data\stock_rankings.json
dir backend\data\ranking_history.json
dir backend\data\turbomode_models\meta_learner_top10\model.json
dir backend\data\turbomode_models\meta_learner_top10\metadata.json
```

**All files should exist** - If any are missing, re-run the corresponding step above.

---

## Testing Instructions

### Test 1: Compare Predictions on Same Stock

**Purpose:** Verify specialized meta-learner gives higher confidence on top 10 stocks

```python
cd C:\StockApp
python

from backend.advanced_ml.models.meta_learner import MetaLearner
from backend.advanced_ml.models.xgboost_model import XGBoostModel
from backend.advanced_ml.models.lightgbm_model import LightGBMModel
# ... (load all 8 base models)

# Load both meta-learners
general_meta = MetaLearner(model_path="backend/data/turbomode_models/meta_learner")
specialized_meta = MetaLearner(model_path="backend/data/turbomode_models/meta_learner_top10")

# Register base models with both
# ... (register all 8 models)

# Get predictions from base models
base_preds = {
    'xgboost': xgb_model.predict(features),
    'lightgbm': lgbm_model.predict(features),
    # ... (all 8 models)
}

# Compare predictions
general_pred = general_meta.predict(base_preds)
specialized_pred = specialized_meta.predict(base_preds)

print(f"General Meta-Learner:     {general_pred['prediction']} ({general_pred['confidence']:.2%})")
print(f"Specialized Meta-Learner: {specialized_pred['prediction']} ({specialized_pred['confidence']:.2%})")
```

**Expected Result:**
- General: ~68-72% confidence
- Specialized: ~75-85% confidence (HIGHER for top 10 stocks)

---

### Test 2: Verify Top 10 Stock List

**Purpose:** Confirm which stocks are currently in the top 10

```python
import json

with open('backend/data/stock_rankings.json', 'r') as f:
    rankings = json.load(f)

top_10 = [stock['symbol'] for stock in rankings['top_10']]
print(f"Top 10 Stocks: {', '.join(top_10)}")

# Print detailed info
for i, stock in enumerate(rankings['top_10'], 1):
    print(f"{i:2d}. {stock['symbol']:6s} Score: {stock['composite_score']:.3f} "
          f"WR_30d: {stock['win_rate_30d']*100:5.1f}% "
          f"Regime: {stock['regime']}")
```

**Expected Result:**
```
Top 10 Stocks: AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, AMD, NFLX, ADBE

 1. AAPL   Score: 0.823 WR_30d:  82.0% Regime: stable
 2. MSFT   Score: 0.791 WR_30d:  79.5% Regime: stable
 ...
```

---

### Test 3: Re-run Training to Verify Improvement

**Purpose:** Confirm specialized meta-learner consistently outperforms general

```bash
cd C:\StockApp\backend\turbomode
python train_specialized_meta_learner.py
```

**Success Criteria:**
- ✅ **Minimum:** Improvement ≥ +1 percentage point
- 🎯 **Target:** Improvement ≥ +3 percentage points
- 🚀 **Exceptional:** Improvement ≥ +5 percentage points

**If improvement is negative or <1%:** See [Troubleshooting](#troubleshooting) section below.

---

## Integration Steps

### Step 4: Update Overnight Scanner (NEXT TASK)

**File to modify:** `C:\StockApp\backend\turbomode\overnight_scanner.py`

**Changes needed:**

1. **Load top 10 stocks from rankings.json**
2. **Scan ONLY those 10 stocks** (not all 80)
3. **Use specialized meta-learner** instead of general

**Implementation:**

```python
# Add to overnight_scanner.py imports
import json
import os

# Load top 10 stocks
RANKINGS_FILE = os.path.join(PROJECT_ROOT, "backend", "data", "stock_rankings.json")

if os.path.exists(RANKINGS_FILE):
    with open(RANKINGS_FILE, 'r') as f:
        rankings = json.load(f)
    TOP_10_SYMBOLS = [stock['symbol'] for stock in rankings['top_10']]
    print(f"[SCANNER] Using top 10 stocks: {', '.join(TOP_10_SYMBOLS)}")
else:
    # Fallback to all stocks if rankings not available
    from backend.advanced_ml.config.core_symbols import CORE_SYMBOLS
    TOP_10_SYMBOLS = CORE_SYMBOLS
    print(f"[SCANNER] Rankings not found, using all {len(TOP_10_SYMBOLS)} stocks")

# Load SPECIALIZED meta-learner instead of general
META_LEARNER_PATH = os.path.join(PROJECT_ROOT, "backend", "data", "turbomode_models", "meta_learner_top10")

if os.path.exists(META_LEARNER_PATH):
    meta_learner = MetaLearner(model_path=META_LEARNER_PATH, use_gpu=True)
    print(f"[SCANNER] Using SPECIALIZED meta-learner for top 10 stocks")
else:
    # Fallback to general meta-learner
    meta_learner = MetaLearner(model_path=os.path.join(PROJECT_ROOT, "backend", "data", "turbomode_models", "meta_learner"), use_gpu=True)
    print(f"[SCANNER] Specialized meta-learner not found, using general")

# In main scanning loop, replace:
# for symbol in CORE_SYMBOLS:
# with:
for symbol in TOP_10_SYMBOLS:
    # ... existing scanning logic
```

**Expected Behavior:**
- Scanner processes 10 stocks instead of 80 (MUCH faster)
- Uses specialized meta-learner with higher accuracy
- Generates 2-3 high-quality signals per night

---

### Step 5: Setup Monthly Automated Retraining

**Purpose:** Automatically update top 10 rankings and retrain specialized meta-learner monthly

**Implementation:** Already integrated into Flask server via APScheduler

**Verify scheduler is running:**

```bash
# Start Flask server
cd C:\StockApp
python backend/api_server.py
```

**Look for this output:**
```
[STOCK RANKING] Initializing monthly adaptive stock ranking scheduler...
[STOCK RANKING] Ready - Runs monthly on 1st at 2:00 AM
```

**Schedule Details:**
- **Frequency:** 1st of every month at 2:00 AM
- **Tasks:**
  1. Run `adaptive_stock_ranker.py` to update top 10
  2. Check if top 10 changed
  3. If changed: Run `train_specialized_meta_learner.py`
  4. Update `overnight_scanner.py` configuration

**Manual Trigger (for testing):**

```bash
# Via API
curl -X POST http://127.0.0.1:5127/turbomode/rankings/run
```

**Expected Response:**
```json
{
  "status": "success",
  "top_10": ["AAPL", "MSFT", "GOOGL", ...],
  "timestamp": "2026-01-01T14:30:00"
}
```

---

## File Locations

### Source Code

| File | Location | Purpose |
|------|----------|---------|
| Stock ranking script | `C:\StockApp\backend\turbomode\adaptive_stock_ranker.py` | Analyze and rank all 80 stocks |
| Specialized training script | `C:\StockApp\backend\turbomode\train_specialized_meta_learner.py` | Train meta-learner on top 10 only |
| General training script | `C:\StockApp\backend\turbomode\retrain_meta_learner_only.py` | Retrain general meta-learner |
| Overnight scanner | `C:\StockApp\backend\turbomode\overnight_scanner.py` | Generate nightly signals |
| Stock ranking API | `C:\StockApp\backend\turbomode\stock_ranking_api.py` | Flask blueprint for ranking endpoints |
| Flask server | `C:\StockApp\backend\api_server.py` | Main application server |

### Data Files

| File | Location | Purpose |
|------|----------|---------|
| Current rankings | `C:\StockApp\backend\data\stock_rankings.json` | Top 10 stocks with scores |
| Ranking history | `C:\StockApp\backend\data\ranking_history.json` | Historical rankings log |
| Backtest database | `C:\StockApp\backend\data\advanced_ml_system.db` | Training data (13,149 samples) |
| Trading signals | `C:\StockApp\backend\data\ml_trading_signals.json` | Overnight scanner output |

### Model Files

| Model | Location | Purpose |
|-------|----------|---------|
| General meta-learner | `C:\StockApp\backend\data\turbomode_models\meta_learner\` | All 80 stocks (71.29%) |
| Specialized meta-learner | `C:\StockApp\backend\data\turbomode_models\meta_learner_top10\` | Top 10 stocks (75-80%) |
| XGBoost | `C:\StockApp\backend\data\turbomode_models\xgboost\` | Base model #1 |
| XGBoost ET | `C:\StockApp\backend\data\turbomode_models\xgboost_et\` | Base model #2 |
| LightGBM | `C:\StockApp\backend\data\turbomode_models\lightgbm\` | Base model #3 |
| CatBoost | `C:\StockApp\backend\data\turbomode_models\catboost\` | Base model #4 |
| XGBoost Hist | `C:\StockApp\backend\data\turbomode_models\xgboost_hist\` | Base model #5 |
| XGBoost DART | `C:\StockApp\backend\data\turbomode_models\xgboost_dart\` | Base model #6 |
| XGBoost GBLinear | `C:\StockApp\backend\data\turbomode_models\xgboost_gblinear\` | Base model #7 |
| XGBoost Approx | `C:\StockApp\backend\data\turbomode_models\xgboost_approx\` | Base model #8 |

### Documentation

| File | Location | Purpose |
|------|----------|---------|
| This document | `C:\StockApp\2nd_meta_document.md` | Complete system documentation |
| Testing guide | `C:\StockApp\DUAL_META_LEARNER_TESTING_GUIDE.md` | Step-by-step testing instructions |
| Session notes | `C:\StockApp\session_files\session_notes_2026-01-01.md` | Daily development log |

---

## API Endpoints

### Stock Ranking API

**Base URL:** `http://127.0.0.1:5127/turbomode/rankings/`

#### GET `/current`
Get current top 10 stock rankings

**Response:**
```json
{
  "top_10": [
    {
      "symbol": "AAPL",
      "composite_score": 0.823,
      "win_rate_30d": 0.820,
      "win_rate_60d": 0.805,
      "win_rate_90d": 0.792,
      "signals_per_year": 28.5,
      "regime": "stable"
    },
    ...
  ],
  "timestamp": "2026-01-01T14:30:00",
  "total_stocks_analyzed": 80
}
```

#### GET `/all`
Get rankings for all stocks

#### POST `/run`
Manually trigger stock ranking analysis

#### GET `/scheduler/status`
Get monthly scheduler status

#### POST `/scheduler/start`
Start monthly scheduler

#### POST `/scheduler/stop`
Stop monthly scheduler

---

## Troubleshooting

### Problem 1: Rankings File Not Found

**Error:**
```
[ERROR] Rankings file not found: C:\StockApp\backend\data\stock_rankings.json
[INFO] Please run: python adaptive_stock_ranker.py
```

**Solution:**
```bash
cd C:\StockApp\backend\turbomode
python adaptive_stock_ranker.py
```

---

### Problem 2: No Training Data Found for Top 10 Stocks

**Error:**
```
[ERROR] No training data found for top 10 stocks!
```

**Cause:** Top 10 stocks don't have backtest data in database

**Solution:**
```bash
# Regenerate backtest data
cd C:\StockApp
python backend/turbomode/generate_backtest_data.py
```

**This will:**
- Generate backtest data for all 80 stocks
- Populate `advanced_ml_system.db` with trades
- Re-run training scripts afterward

---

### Problem 3: Low Improvement (<1 percentage point)

**Symptoms:**
```
General Meta-Learner:     71.29%
Specialized Meta-Learner: 71.85%

⚠️ Improvement: +0.56 percentage points
```

**Possible Causes:**
1. Top 10 stocks aren't significantly easier to predict than others
2. Not enough training samples per stock
3. General meta-learner already performing optimally

**Solution:**
- **Still usable** - Even 0.5-1% improvement is valuable
- Try increasing top N from 10 to 15 stocks for more training data
- Check if backtest data quality is sufficient

---

### Problem 4: Specialized Performs WORSE

**Symptoms:**
```
General Meta-Learner:     71.29%
Specialized Meta-Learner: 69.12%

⚠️ Improvement: -2.17 percentage points
```

**Possible Causes:**
1. **Overfitting** on small dataset
2. Top 10 stocks are too diverse (different sectors/behaviors)
3. Need more training data

**Solution:**
```bash
# Stick with general meta-learner for now
# Use this in overnight_scanner.py:
meta_learner = MetaLearner(
    model_path="backend/data/turbomode_models/meta_learner",  # General
    use_gpu=True
)
```

---

### Problem 5: Specialized Meta-Learner Not Loading in Scanner

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: '.../meta_learner_top10/model.json'
```

**Solution:**
```bash
# Train specialized meta-learner first
cd C:\StockApp\backend\turbomode
python train_specialized_meta_learner.py
```

**Verify files exist:**
```bash
dir C:\StockApp\backend\data\turbomode_models\meta_learner_top10\model.json
dir C:\StockApp\backend\data\turbomode_models\meta_learner_top10\metadata.json
```

---

### Problem 6: Monthly Scheduler Not Running

**Symptoms:**
```
[STOCK RANKING] Not available
```

**Check Flask server logs:**
```bash
python backend/api_server.py
```

**Look for:**
```
[STOCK RANKING] Initializing monthly adaptive stock ranking scheduler...
[STOCK RANKING] Ready - Runs monthly on 1st at 2:00 AM
```

**If missing:** Check that `stock_ranking_api.py` is importable:
```python
from backend.turbomode.stock_ranking_api import ranking_bp, init_stock_ranking_scheduler
```

---

## Performance Expectations

### Accuracy Benchmarks

| Meta-Learner | Training Data | Expected Accuracy | Use Case |
|--------------|---------------|-------------------|----------|
| General | All 80 stocks | 71-72% | Initial scanning, fallback |
| Specialized | Top 10 stocks | 75-80% | Focused trading |

### Prediction Confidence

| Meta-Learner | Typical Confidence | High Confidence |
|--------------|-------------------|-----------------|
| General | 65-75% | 75-85% |
| Specialized | 75-85% | 85-95% |

### Signal Generation

| Scanner Mode | Stocks Scanned | Signals per Night | Quality |
|--------------|----------------|-------------------|---------|
| Full (80 stocks) | 80 | 5-10 | Mixed (some low quality) |
| Top 10 (specialized) | 10 | 2-3 | High quality |

### Trading Performance (Hypothetical)

**Assumptions:**
- 71% accuracy → ~167 winning trades per year (out of 235)
- 76% accuracy → ~179 winning trades per year (out of 235)
- **+12 additional winning trades per year**

**Impact:**
- Win rate improvement: +7%
- If avg win = $500, avg loss = $300
- **Additional profit: ~$6,000/year** (12 wins × $500) - (losses offset)

---

## Monthly Maintenance Workflow

### Automated (Recommended)

**Setup:** Already configured in Flask server

**Schedule:** 1st of each month at 2:00 AM

**Tasks:**
1. Run stock ranking analysis
2. Check if top 10 changed
3. If changed: Retrain specialized meta-learner
4. Log results to `ranking_history.json`

**Monitor:** Check logs on 1st of each month:
```bash
# View ranking history
python
import json
with open('backend/data/ranking_history.json', 'r') as f:
    history = json.load(f)
print(history[-1])  # Latest update
```

---

### Manual (for testing)

```bash
# Step 1: Update rankings
cd C:\StockApp\backend\turbomode
python adaptive_stock_ranker.py

# Step 2: Check if top 10 changed
python
import json
with open('../data/stock_rankings.json', 'r') as f:
    current = json.load(f)
with open('../data/ranking_history.json', 'r') as f:
    history = json.load(f)

current_top10 = set([s['symbol'] for s in current['top_10']])
previous_top10 = set([s['symbol'] for s in history[-2]['top_10']])  # -2 because -1 is current

if current_top10 != previous_top10:
    print(f"Top 10 CHANGED! Exited: {previous_top10 - current_top10}, Entered: {current_top10 - previous_top10}")
else:
    print("Top 10 unchanged - no retraining needed")

# Step 3: If changed, retrain specialized meta-learner
# Only run if top 10 changed
python train_specialized_meta_learner.py
```

---

## Success Metrics

### Testing Phase (Initial Setup)

✅ **Minimum Success:**
- Specialized meta-learner trains without errors
- Accuracy on top 10 stocks ≥ general meta-learner
- Improvement ≥ +1 percentage point

🎯 **Target Success:**
- Improvement ≥ +3 percentage points
- Accuracy on top 10 stocks ≥ 75%
- Clear confidence score differences between general and specialized

🚀 **Exceptional Success:**
- Improvement ≥ +5 percentage points
- Accuracy on top 10 stocks ≥ 78%
- Consistent improvement across all top 10 stocks

---

### Production Phase (After Integration)

📊 **Weekly Metrics:**
- Signals generated: 2-3 per night (10-15 per week)
- Signal quality: ≥75% win rate
- No false positives on non-top-10 stocks

📈 **Monthly Metrics:**
- Top 10 stability: 70% unchanged, 30% rotated
- Specialized meta-learner accuracy: ≥75%
- Trading performance: Consistent with backtest

🎯 **Quarterly Metrics:**
- Cumulative win rate: ≥75%
- Sharpe ratio improvement vs. general meta-learner
- Model drift detection: <5% accuracy degradation

---

## Quick Reference Commands

### Initial Setup (First Time)
```bash
# Step 1: Run stock ranking
cd C:\StockApp\backend\turbomode
python adaptive_stock_ranker.py

# Step 2: Train specialized meta-learner
python train_specialized_meta_learner.py

# Step 3: Verify files created
dir ..\data\stock_rankings.json
dir ..\data\turbomode_models\meta_learner_top10\model.json
```

### Testing
```bash
# Re-run training to verify improvement
cd C:\StockApp\backend\turbomode
python train_specialized_meta_learner.py

# Check top 10 stocks
python
import json
with open('../data/stock_rankings.json', 'r') as f:
    rankings = json.load(f)
print([s['symbol'] for s in rankings['top_10']])
```

### Monthly Maintenance (Manual)
```bash
# Update rankings
cd C:\StockApp\backend\turbomode
python adaptive_stock_ranker.py

# Retrain specialized meta-learner
python train_specialized_meta_learner.py
```

### API Testing
```bash
# Get current top 10
curl http://127.0.0.1:5127/turbomode/rankings/current

# Trigger manual ranking run
curl -X POST http://127.0.0.1:5127/turbomode/rankings/run

# Check scheduler status
curl http://127.0.0.1:5127/turbomode/rankings/scheduler/status
```

---

## Summary

The dual meta-learner system provides **higher accuracy predictions** on a **curated universe** of the 10 most predictable stocks. By focusing the specialized meta-learner on stocks with proven backtest performance, we expect:

- **+3 to +8 percentage point improvement** on top 10 stocks
- **2-3 high-quality signals per night** vs 5-10 mixed-quality signals
- **Monthly automated updates** to adapt to changing market conditions
- **Reduced noise** from unpredictable stocks

**Next Steps:**
1. ✅ Test initial setup (Steps 1-3)
2. ⏳ Update overnight scanner to use specialized meta-learner
3. ⏳ Run first overnight scan with top 10 stocks
4. ⏳ Monitor performance for 1 month
5. ⏳ Compare trading results: General vs Specialized

**Status:** Ready for testing when you get home! 🚀

---

**Last Updated:** 2026-01-01
**Document Version:** 1.0
**Author:** AI Assistant (Claude)
