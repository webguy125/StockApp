# TurboMode Complete Pipeline & Automation Guide

**System**: TurboMode Autonomous ML Trading System
**Database**: `backend/data/turbomode.db`
**Models**: `backend/data/turbomode_models/`
**Date**: January 5, 2026

---

## 🔄 Complete Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    TURBOMODE PIPELINE                           │
└─────────────────────────────────────────────────────────────────┘

1. DATA GENERATION (One-time/Manual)
   ↓
2. MODEL TRAINING (One-time/Periodic)
   ↓
3. NIGHTLY PREDICTIONS (Automated - 11 PM)
   ↓
4. OUTCOME TRACKING (Missing - Need to Build)
   ↓
5. RETRAINING WITH NEW DATA (Missing - Need to Build)
```

---

## 📋 Pipeline Steps (Detailed)

### STEP 1: Generate Training Data
**Script**: `backend/turbomode/generate_backtest_data.py`
**Frequency**: One-time (or when you want to refresh historical data)
**Duration**: ~2-4 hours (for 82 stocks × 7 years)
**Purpose**: Create labeled training samples from historical price data

**What it does**:
1. Fetches 7 years of historical OHLCV data from yfinance
2. For each day:
   - Extracts 179 features
   - Simulates 14-day forward trade
   - Labels outcome: BUY (≥10% gain) or SELL (<10% gain)
3. Stores samples in `turbomode.db` → `trades` table

**Command**:
```bash
python backend/turbomode/generate_backtest_data.py
```

**Expected Output**:
- ~10,000-15,000 labeled samples in `trades` table
- Features stored in JSON format
- Date range: 2019-present

**Current Status**: ❌ NOT RUN (trades table is empty)

---

### STEP 2: Train Models
**Script**: `backend/turbomode/train_turbomode_models.py`
**Frequency**: One-time / Monthly / After data refresh
**Duration**: ~30-60 minutes (GPU accelerated)
**Purpose**: Train 8-model ensemble + meta-learner

**What it does**:
1. Loads training samples from `turbomode.db` → `trades` table
2. Splits into train/test (80/20)
3. Trains 8 models in parallel:
   - XGBoost (standard)
   - XGBoost Extra Trees
   - XGBoost DART
   - XGBoost Histogram
   - XGBoost Approx
   - XGBoost GBLinear
   - LightGBM
   - CatBoost
4. Trains meta-learner (stacks predictions)
5. Saves all 9 models to `backend/data/turbomode_models/`

**Command**:
```bash
python backend/turbomode/train_turbomode_models.py
```

**Expected Output**:
- 9 model directories with `.cbm`, `.json`, `.txt` files
- Training metrics (accuracy, precision, recall)
- Model metadata with timestamp

**Current Status**: ✅ MODELS EXIST (trained Jan 2, 2026)
**Note**: Models will need retraining after Step 1 generates fresh data

---

### STEP 3: Nightly Prediction Scan
**Script**: `backend/turbomode/overnight_scanner.py`
**Frequency**: ⏰ **AUTOMATED - Every night at 11:00 PM**
**Duration**: ~60 minutes (for all 82 stocks)
**Purpose**: Generate predictions for all curated stocks

**What it does**:
1. Loads 9 pre-trained models from disk
2. For each of 82 curated stocks:
   - Fetches latest price data from yfinance
   - Generates 179 features on-the-fly
   - Runs through all 8 models
   - Meta-learner combines predictions
   - Outputs: BUY/SELL + confidence %
3. Saves results to:
   - `turbomode.db` → `active_signals` table
   - `backend/data/all_predictions.json`
4. Calculates entry range (±3% of current price)

**Scheduler**: APScheduler (already configured in Flask)
**Location**: `backend/turbomode/turbomode_scheduler.py`

**Current Status**: ✅ RUNNING AUTOMATICALLY
**Last Run**: January 5, 2026 at 11:00 PM

**Output Example**:
```
BOOT:  BUY  84.2% confidence  $150.25  Entry: $145.74-$154.76
SHAK:  BUY  82.0% confidence  $110.50  Entry: $107.19-$113.82
NVDA:  BUY  81.9% confidence  $525.00  Entry: $509.25-$540.75
```

---

### STEP 4: Outcome Tracking
**Status**: ❌ **NOT IMPLEMENTED - NEEDS TO BE BUILT**
**Frequency**: Should run daily
**Purpose**: Track if predictions were correct

**What it SHOULD do**:
1. Look at signals from 14 days ago
2. Check current price vs entry price
3. Calculate actual return
4. Determine if prediction was correct:
   - BUY signal + ≥10% gain = CORRECT
   - BUY signal + <10% gain = INCORRECT
5. Move from `active_signals` → `signal_history` table
6. Store outcome for future retraining

**Pseudocode**:
```python
def track_outcomes():
    # Get signals from 14 days ago
    signals = get_signals_from_date(today - 14_days)

    for signal in signals:
        current_price = get_current_price(signal.symbol)
        return_pct = (current_price - signal.entry_price) / signal.entry_price

        if signal.prediction == 'buy':
            outcome = 'correct' if return_pct >= 0.10 else 'incorrect'

        # Save to signal_history table
        save_outcome(signal, outcome, return_pct)

        # Remove from active_signals
        deactivate_signal(signal.id)
```

**Scheduler Needed**: ✅ YES - Daily at 2 AM (after market close data is available)

---

### STEP 5: Generate New Training Samples from Outcomes
**Status**: ❌ **NOT IMPLEMENTED - NEEDS TO BE BUILT**
**Frequency**: Weekly or after X outcomes tracked
**Purpose**: Convert tracked outcomes into training samples

**What it SHOULD do**:
1. Fetch outcomes from `signal_history` table
2. For each outcome with complete data:
   - Extract original features (stored in JSON)
   - Label: 'buy' (correct) or 'sell' (incorrect)
3. Insert new samples into `trades` table
4. Update training data freshness

**Scheduler Needed**: ✅ YES - Weekly (Sunday 3 AM)

---

### STEP 6: Automated Model Retraining
**Status**: ❌ **NOT IMPLEMENTED - NEEDS TO BE BUILT**
**Frequency**: Monthly (1st of month at 4 AM)
**Purpose**: Retrain models with accumulated new data

**What it SHOULD do**:
1. Check if enough new samples exist (e.g., ≥100)
2. Run `train_turbomode_models.py` automatically
3. Compare new model accuracy vs old models
4. If better: Replace production models
5. If worse: Keep old models, log warning
6. Send email/notification with results

**Scheduler Needed**: ✅ YES - Monthly (1st at 4 AM)

---

## 🕐 Scheduler Configuration

### Current Scheduler (Flask APScheduler)
**File**: `backend/turbomode/turbomode_scheduler.py`
**Integration**: `backend/api_server.py`

**Current Jobs**:
```python
# ALREADY SCHEDULED ✅
scheduler.add_job(
    func=run_overnight_scan,
    trigger=CronTrigger(hour=23, minute=0),  # 11 PM daily
    id='turbomode_overnight_scan',
    name='TurboMode Overnight Scan',
    replace_existing=True
)
```

### Jobs That NEED TO BE ADDED:

#### 1. Outcome Tracker (Daily)
```python
scheduler.add_job(
    func=track_signal_outcomes,
    trigger=CronTrigger(hour=2, minute=0),  # 2 AM daily
    id='turbomode_outcome_tracker',
    name='TurboMode Outcome Tracker',
    replace_existing=True
)
```

#### 2. Generate Training Samples from Outcomes (Weekly)
```python
scheduler.add_job(
    func=generate_training_samples_from_outcomes,
    trigger=CronTrigger(day_of_week='sun', hour=3, minute=0),  # Sunday 3 AM
    id='turbomode_sample_generator',
    name='TurboMode Sample Generator',
    replace_existing=True
)
```

#### 3. Monthly Model Retraining (Monthly)
```python
scheduler.add_job(
    func=automated_model_retraining,
    trigger=CronTrigger(day=1, hour=4, minute=0),  # 1st of month, 4 AM
    id='turbomode_monthly_retrain',
    name='TurboMode Monthly Retraining',
    replace_existing=True
)
```

#### 4. Quarterly Stock Curation (Quarterly)
```python
# ALREADY SCHEDULED ✅
scheduler.add_job(
    func=run_quarterly_curation,
    trigger=CronTrigger(month='1,4,7,10', day=1, hour=2, minute=0),
    id='turbomode_quarterly_curation',
    name='TurboMode Quarterly Stock Curation',
    replace_existing=True
)
```

#### 5. Adaptive Stock Ranking (Monthly)
```python
# ALREADY SCHEDULED ✅
scheduler.add_job(
    func=run_adaptive_ranking,
    trigger=CronTrigger(day=1, hour=2, minute=0),
    id='turbomode_adaptive_ranking',
    name='TurboMode Adaptive Stock Ranking',
    replace_existing=True
)
```

---

## 📅 Complete Automation Schedule

| Time | Frequency | Job | Status |
|------|-----------|-----|--------|
| 11:00 PM | Daily | Overnight Scan (82 stocks) | ✅ ACTIVE |
| 2:00 AM | Daily | **Outcome Tracker** | ❌ NEEDS BUILD |
| 2:00 AM | 1st of month | Adaptive Stock Ranking | ✅ ACTIVE |
| 2:00 AM | Quarterly | Stock Curation | ✅ ACTIVE |
| 3:00 AM | Sunday | **Generate Training Samples** | ❌ NEEDS BUILD |
| 4:00 AM | 1st of month | **Automated Retraining** | ❌ NEEDS BUILD |

---

## 🛠️ What Needs to Be Built

### HIGH PRIORITY (Missing Feedback Loop)

#### 1. Outcome Tracker (`backend/turbomode/outcome_tracker.py`)
```python
"""
Track TurboMode prediction outcomes after 14 days
Moves signals from active_signals → signal_history
"""

def track_signal_outcomes():
    # Implementation needed
    pass
```

#### 2. Training Sample Generator (`backend/turbomode/training_sample_generator.py`)
```python
"""
Convert tracked outcomes into training samples
Adds new samples to trades table
"""

def generate_training_samples_from_outcomes():
    # Implementation needed
    pass
```

#### 3. Automated Retrainer (`backend/turbomode/automated_retrainer.py`)
```python
"""
Monthly model retraining with accumulated data
Validates new models before deployment
"""

def automated_model_retraining():
    # Implementation needed
    pass
```

#### 4. Update Scheduler (`backend/turbomode/turbomode_scheduler.py`)
Add the 3 new jobs to the existing scheduler

---

## 🎯 Immediate Next Steps

### To Get TurboMode Fully Operational:

1. **Generate Initial Training Data** (One-time)
   ```bash
   python backend/turbomode/generate_backtest_data.py
   ```
   Duration: 2-4 hours
   Output: ~10,000+ training samples

2. **Retrain Models** (One-time)
   ```bash
   python backend/turbomode/train_turbomode_models.py
   ```
   Duration: 30-60 minutes
   Output: Fresh models with current data

3. **Build Outcome Tracker** (Development)
   - Create `backend/turbomode/outcome_tracker.py`
   - Implement 14-day outcome checking
   - Update scheduler

4. **Build Training Sample Generator** (Development)
   - Create `backend/turbomode/training_sample_generator.py`
   - Convert outcomes → training samples
   - Update scheduler

5. **Build Automated Retrainer** (Development)
   - Create `backend/turbomode/automated_retrainer.py`
   - Implement validation logic
   - Update scheduler

---

## 📊 Data Flow Diagram

```
HISTORICAL DATA (7 years)
         ↓
[generate_backtest_data.py] → trades table (10K samples)
         ↓
[train_turbomode_models.py] → 9 trained models
         ↓
[overnight_scanner.py - 11 PM] → active_signals + all_predictions.json
         ↓
[outcome_tracker.py - 2 AM] → signal_history (after 14 days)
         ↓
[training_sample_generator.py - Weekly] → NEW samples in trades table
         ↓
[automated_retrainer.py - Monthly] → UPDATED models
         ↓
[Loop back to overnight_scanner]
```

---

## 🔍 Current Status Summary

### ✅ Working Components:
- Database separation (turbomode.db)
- Model training pipeline
- Nightly prediction scan (automated)
- Stock curation (quarterly)
- Adaptive ranking (monthly)

### ❌ Missing Components:
- Outcome tracking (no feedback loop)
- Training sample generation from outcomes
- Automated model retraining
- Performance monitoring/alerting

### 📈 To Achieve Full Autonomy:
Build the 3 missing components + update scheduler = **Self-improving ML system**

---

**Generated**: January 5, 2026
**System**: TurboMode Autonomous ML Trading System
**Next Review**: After implementing outcome tracker
