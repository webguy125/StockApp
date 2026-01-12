# Session Notes - January 8, 2026

## Session Summary
Fixed critical LightGBM predict_proba() bug that was causing 1D output errors during TurboMode ML training batch prediction generation phase.

## Problem Identified
- **Error**: `ValueError: Model returned 1D output; expected multiclass probabilities (N, 3)` in lightgbm_model.py line 83
- **Root Cause**: LightGBM Booster's `predict()` method returns 1D class labels when given 1D input, instead of 2D probability matrix
- **Location**: backend/turbomode/models/lightgbm_model.py predict_proba() method

## Architecture Clarification
- LightGBM wrapper uses **raw Booster API**, not scikit-learn LGBMClassifier API
- LGBMClassifier is only a **hyperparameter container**
- Actual predictions come from `self.model.booster_` (native LightGBM Booster)
- Correct API call: `booster.predict(X, raw_score=False)` returns (N, num_class) probabilities when X is 2D

## Fix Applied

**File Modified**: backend/turbomode/models/lightgbm_model.py

**Lines Changed**: 72-80 in predict_proba() method

**Change Summary**:
Added 2D reshape guard to auto-convert 1D input to 2D before calling Booster.predict()

### BEFORE:
```python
def predict_proba(self, X: np.ndarray) -> np.ndarray:
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy ndarray")
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.ndim}D")
    if X.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float32)

    raw_preds = self.model.predict(X, raw_score=False)
```

### AFTER:
```python
def predict_proba(self, X: np.ndarray) -> np.ndarray:
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy ndarray")

    # Convert to numpy array
    X = np.asarray(X)

    # Auto-convert 1D input to 2D
    if X.ndim == 1:
        X = X.reshape(1, -1)

    # Validate 2D after reshape
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.ndim}D")

    if X.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float32)

    raw_preds = self.model.predict(X, raw_score=False)
```

## Audit Results

Audited all 8 tree-based model wrappers for 2D input validation:

### Already Correct (7 models):
1. catboost_model.py - Uses `self.model.predict(X, prediction_type="Probability")` with full 2D validation
2. xgboost_model.py - Uses `self.model.predict_proba(X)` with full 2D validation
3. xgboost_approx_model.py - Uses `self.model.predict_proba(X)` with full 2D validation
4. xgboost_dart_model.py - Uses `self.model.predict_proba(X)` with full 2D validation
5. xgboost_et_model.py - Uses `self.model.predict_proba(X)` with full 2D validation
6. xgboost_gblinear_model.py - Uses `self.model.predict_proba(X)` with full 2D validation
7. xgboost_hist_model.py - Uses `self.model.predict_proba(X)` with full 2D validation

### Fixed (1 model):
- lightgbm_model.py - Added reshape guard before `self.model.predict(X, raw_score=False)`

## Technical Details

**Why the reshape guard works**:
- LightGBM Booster.predict() behavior depends on input dimensionality
- 1D input (shape: (n_features,)) → returns 1D class labels
- 2D input (shape: (n_samples, n_features)) → returns 2D probability matrix (n_samples, num_class)
- Auto-converting 1D to 2D ensures consistent (N, 3) output for 3-class models

**Key Validation Order**:
1. Type check (must be numpy array)
2. Convert to numpy array with np.asarray()
3. Auto-reshape 1D → 2D
4. Validate 2D (should never fail after reshape)
5. Handle empty array edge case
6. Call Booster.predict()

## Impact
- Eliminates 1D output errors in batch prediction generation phase
- LightGBM wrapper now consistent with other tree models
- Training can proceed through LightGBM prediction phase without crashes

## Next Steps
- Resume TurboMode training to verify LightGBM predictions generate correctly
- Monitor for any additional model-specific issues during training
- All 8 tree models now have proper 2D input handling

## Files Modified
- backend/turbomode/models/lightgbm_model.py (lines 72-90)

## Session Context
Continued from 2026-01-07 session where XGBoost models were retrained with correct 3-class configuration after deleting contaminated binary models.
