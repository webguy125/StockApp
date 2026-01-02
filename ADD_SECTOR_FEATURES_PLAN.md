# Plan to Add 3 Missing Features (176 → 179)

## Summary
Add sector and market cap metadata as model features to match the original 90% accuracy setup.

**New Features:**
1. `sector_code` (0-10): GICS sector integer encoding
2. `market_cap_tier` (0-2): 0=large, 1=mid, 2=small
3. `symbol_hash` (0-79): Symbol index in curated list

**Impact:** Expected +5-10% accuracy improvement

---

## Files to Modify

### 1. ✅ `backend/advanced_ml/config/symbol_metadata.py` (CREATED)
Helper module to provide sector/market_cap features for any symbol.

**Status:** Complete - already created

---

### 2. `backend/advanced_ml/backtesting/historical_backtest.py`

**Location:** Feature extraction in `generate_labeled_data()` method

**Change:** Add metadata features after technical features extracted

**Line ~258 (after `features = self.feature_engineer.extract_features(...)`):**

```python
# BEFORE:
features = self.feature_engineer.extract_features(historical_data, symbol=symbol)

# AFTER:
features = self.feature_engineer.extract_features(historical_data, symbol=symbol)

# Add sector + market_cap metadata (3 features)
from advanced_ml.config.symbol_metadata import get_symbol_metadata
metadata = get_symbol_metadata(symbol)
features.update(metadata)  # Adds sector_code, market_cap_tier, symbol_hash
```

**Also need to update batch mode (line ~197):**

```python
# AFTER line 197:
all_features = self.feature_engineer.extract_features_batch(df, start_indices, symbol)

# Add metadata to each feature dict
from advanced_ml.config.symbol_metadata import get_symbol_metadata
metadata = get_symbol_metadata(symbol)
for features in all_features:
    features.update(metadata)
```

---

### 3. `backend/advanced_ml/backtesting/historical_backtest.py`

**Location:** Feature filtering in `prepare_training_data()` method

**Line ~528 (exclude list):**

```python
# BEFORE:
exclude_keys = ['feature_count', 'symbol', 'last_price', 'last_volume', 'timestamp', 'error']

# AFTER (keep the metadata features):
exclude_keys = ['feature_count', 'symbol', 'last_price', 'last_volume', 'timestamp', 'error']
# Note: sector_code, market_cap_tier, symbol_hash are NOT excluded
```

**Status:** No change needed - metadata features will automatically be included

---

### 4. Update Expected Feature Count

**File:** `backend/turbomode/train_turbomode_models.py`

**Line ~97-98:**

```python
# BEFORE:
if X_train.shape[1] != 179:
    print(f"[WARNING] Expected 179 features, got {X_train.shape[1]}")

# AFTER (this is already correct):
if X_train.shape[1] != 179:
    print(f"[WARNING] Expected 179 features, got {X_train.shape[1]}")
```

**Status:** Already correct - will automatically pass when we add the 3 features

---

## Implementation Steps

### Step 1: Modify historical_backtest.py (2 locations)

**Location 1: Line ~258 (loop-based feature extraction)**
```python
features = self.feature_engineer.extract_features(historical_data, symbol=symbol)

# ADD THIS:
from advanced_ml.config.symbol_metadata import get_symbol_metadata
metadata = get_symbol_metadata(symbol)
features.update(metadata)
```

**Location 2: Line ~207 (batch feature extraction)**
```python
all_features = self.feature_engineer.extract_features_batch(df, start_indices, symbol)

# ADD THIS (after line 207):
from advanced_ml.config.symbol_metadata import get_symbol_metadata
metadata = get_symbol_metadata(symbol)
for features in all_features:
    features.update(metadata)
```

### Step 2: Regenerate Training Data
```bash
cd backend/turbomode
python generate_backtest_data.py
```

Expected output: "Features: 179" (not 176)

### Step 3: Retrain Models
```bash
cd backend/turbomode
python train_turbomode_models.py
```

Expected result: No "[WARNING] Expected 179 features, got 176"

---

## Why This Matters

### Original 90% Setup
The documentation and curated stock approach emphasizes **sector stratification** and **market cap tiers** as critical to the "winning formula". These weren't just for UI filtering - they were **model features**.

### Evidence
1. TURBOMODE_SYSTEM_OVERVIEW.md mentions "stratified by sector + market cap" as key to 90% accuracy
2. The system organizes by sector/cap for a reason - models need to learn sector-specific patterns
3. Banking stocks behave differently than tech stocks - models need this context

### Example Pattern Models Will Learn
- **Sector patterns:** Tech (sector_code=45) has higher volatility than Utilities (sector_code=55)
- **Cap patterns:** Small cap (tier=2) has different risk/return than large cap (tier=0)
- **Symbol-specific quirks:** AAPL (hash=0) might have different behavior than WFC (hash=10)

---

## Expected Results

**Before (176 features):**
- Best model: LightGBM 71.63%
- Meta-learner: 41.62% (broken) → ~72% (after bug fix)

**After (179 features):**
- Best model: 75-80% (sector context helps)
- Meta-learner: 75-85% (better ensemble)
- **Target: 85-90%** (with overfitting fixes)

---

## Next Steps After This

1. **Fix overfitting** (LSTM 83%→51%, XGBoost RF 100%→71%)
   - Add dropout to LSTM/PyTorch NN
   - Add L2 regularization to XGBoost models
   - Reduce model complexity

2. **Consider regime/macro features** (optional +25 features)
   - Only if we don't hit 85%+ with sector features alone
   - Adds market regime detection

---

## Quick Test Script

Create `test_metadata.py`:
```python
from advanced_ml.config.symbol_metadata import get_symbol_metadata, get_sector_and_cap

# Test a few symbols
for symbol in ['AAPL', 'JPM', 'SMCI', 'GBCI']:
    metadata = get_symbol_metadata(symbol)
    sector, cap = get_sector_and_cap(symbol)
    print(f"{symbol}: sector={sector}, cap={cap}")
    print(f"  Features: {metadata}")
    print()
```

Expected output:
```
AAPL: sector=technology, cap=large_cap
  Features: {'sector_code': 45.0, 'market_cap_tier': 0.0, 'symbol_hash': 0.0}

JPM: sector=financials, cap=large_cap
  Features: {'sector_code': 40.0, 'market_cap_tier': 0.0, 'symbol_hash': 9.0}

SMCI: sector=technology, cap=small_cap
  Features: {'sector_code': 45.0, 'market_cap_tier': 2.0, 'symbol_hash': 8.0}

GBCI: sector=financials, cap=small_cap
  Features: {'sector_code': 40.0, 'market_cap_tier': 2.0, 'symbol_hash': 15.0}
```
