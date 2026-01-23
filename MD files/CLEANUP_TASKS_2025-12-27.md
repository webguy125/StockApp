# Cleanup Tasks - December 27, 2025

## Summary
TurboMode model separation is complete and working. The following cleanup tasks remain to improve system maintainability and eliminate confusion.

---

## Task 1: Remove Unused Databases (HIGH PRIORITY)

**Problem:** Multiple duplicate/scattered database files causing confusion and wrong database connections

**Databases to Remove:**
```
1. backend/advanced_ml/backtesting/backend/data/advanced_ml_system.db
2. backend/backend/backend/data/advanced_ml_system.db
3. backend/data/rare_event_archive/scripts/backend/data/advanced_ml_system.db
4. backend/turbomode/backend/data/advanced_ml_system.db
```

**Databases to KEEP:**
```
✅ backend/backend/data/advanced_ml_system.db (PRODUCTION - 525MB, 34,086 samples)
   - Used by: Slipstream (ML Automation) for training
   - Used by: TurboMode for training (shared resource)

✅ backend/data/turbomode.db
   - Used by: TurboMode for storing overnight scan signals

✅ backend/data/advanced_ml_system_TEST_ONLY_875_SAMPLES.db
   - Keep for testing purposes (renamed to prevent accidental use)
```

**How to Clean:**
```bash
# Verify production database exists
ls -lh backend/backend/data/advanced_ml_system.db

# Remove duplicates (DO NOT DELETE PRODUCTION DATABASE)
rm backend/advanced_ml/backtesting/backend/data/advanced_ml_system.db
rm backend/backend/backend/data/advanced_ml_system.db
rm backend/data/rare_event_archive/scripts/backend/data/advanced_ml_system.db
rm backend/turbomode/backend/data/advanced_ml_system.db

# Verify only correct databases remain
find backend -name "*.db" -type f
```

**Expected Result:**
- 3 databases total (production, turbomode signals, test)
- No path confusion
- Scripts always find correct database

---

## Task 2: Fix LightGBM Feature Name Warnings (MEDIUM PRIORITY)

**Problem:** Scanner output flooded with warnings:
```
UserWarning: X does not have valid feature names, but LGBMClassifier was fitted with feature names
```

**Root Cause:**
- LightGBM trained with feature names (feature_0, feature_1, etc.)
- Predictions made with plain numpy arrays (no feature names)
- Mismatch triggers warning on every prediction

**Solution Option 1: Add Feature Names During Prediction**
File: `backend/advanced_ml/models/lightgbm_model.py`

Find the `predict()` method and convert numpy array to pandas DataFrame with feature names before prediction:
```python
# Before:
prediction = self.model.predict(X)

# After:
import pandas as pd
feature_names = [f'feature_{i}' for i in range(X.shape[1])]
X_df = pd.DataFrame(X.reshape(1, -1), columns=feature_names)
prediction = self.model.predict(X_df)
```

**Solution Option 2: Disable Feature Names During Training**
File: `backend/advanced_ml/models/lightgbm_model.py`

Add parameter to LGBMClassifier initialization:
```python
self.model = LGBMClassifier(
    ...,
    verbose=-1,
    enable_categorical=False  # Add this
)
```

Then retrain LightGBM model only (5 minutes).

**Recommendation:** Option 1 (easier, no retraining needed)

---

## Task 3: Pin Library Versions (HIGH PRIORITY)

**Problem:** Numpy version mismatches cause pickle compatibility issues

**Current Situation:**
- System Python: numpy 1.26.4
- Venv Python: numpy 2.2.6
- Models trained with venv but can break if numpy updates

**Solution:**
Pin versions in `requirements.txt`:

```bash
# Check current versions
./venv/Scripts/pip.exe freeze | grep -E "numpy|scikit-learn|joblib|lightgbm|xgboost"
```

Update `requirements.txt` with exact versions:
```txt
numpy==2.2.6
scikit-learn==1.6.0
joblib==1.4.2
lightgbm==4.5.0
xgboost==2.1.3
```

**Benefit:** Prevents automatic updates from breaking pickled models

---

## Task 4: Document System Architecture (LOW PRIORITY)

**Create:** `SYSTEM_ARCHITECTURE.md`

**Content:**
```markdown
# System Architecture - StockApp ML Systems

## TurboMode (Overnight Scanner)
- **Purpose:** Generate nightly BUY/SELL signals for S&P 500
- **Models:** backend/data/turbomode_models/ (9 models, 88.93% accuracy)
- **Database:** backend/data/turbomode.db (stores signals)
- **Training Data:** Shared from backend/backend/data/advanced_ml_system.db
- **Schedule:** Automated scan at 11 PM daily
- **Frontend:** frontend/turbomode.html

## Slipstream (ML Automation)
- **Purpose:** Continuous learning and model improvement
- **Models:** backend/data/ml_models/ (9 models)
- **Database:** backend/backend/data/advanced_ml_system.db (34,086 samples)
- **Training:** On-demand or scheduled retraining
- **Frontend:** frontend/ml_trading.html

## Shared Resources
- **Training Database:** backend/backend/data/advanced_ml_system.db
  - 34,086 labeled samples
  - 78 symbols (stocks + crypto)
  - 179 technical features
  - Used by BOTH systems for training
```

---

## Task 5: Create Batch Files for Common Operations (LOW PRIORITY)

**Create convenience scripts:**

**`train_turbomode.bat`:**
```batch
@echo off
echo Training TurboMode Models...
venv\Scripts\python.exe backend\turbomode\train_turbomode_models.py
pause
```

**`run_turbomode_scan.bat`:**
```batch
@echo off
echo Running TurboMode Overnight Scan...
venv\Scripts\python.exe backend\turbomode\overnight_scanner.py
pause
```

**`view_turbomode_signals.bat`:**
```batch
@echo off
echo Opening TurboMode Frontend...
start http://127.0.0.1:5000/turbomode
pause
```

---

## Completion Checklist

- [ ] Remove 4 unused/duplicate databases
- [ ] Verify only 3 databases remain
- [ ] Fix LightGBM warnings (choose Option 1 or 2)
- [ ] Pin library versions in requirements.txt
- [ ] Document system architecture
- [ ] Create convenience batch files
- [ ] Test TurboMode scan runs without warnings
- [ ] Verify tonight's 11 PM scheduled scan works

---

**Priority Order:**
1. Remove unused databases (eliminates confusion)
2. Pin library versions (prevents future pickle issues)
3. Fix LightGBM warnings (clean output)
4. Documentation (helps future work)
5. Batch files (convenience)

**Time Estimate:**
- Task 1: 10 minutes
- Task 2: 15 minutes
- Task 3: 5 minutes
- Task 4: 20 minutes
- Task 5: 10 minutes
- **Total: ~1 hour**

---

**Created:** December 27, 2025 5:35 PM
**Status:** TurboMode model separation COMPLETE, cleanup tasks pending
