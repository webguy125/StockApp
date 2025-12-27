# Phase 1 Improved Model Configurations

## Summary of Changes

**Added:**
- 25 new features (regime + macro) → 179 + 25 = **204 total features**
- Better regularization to fix overfitting
- 5 years of historical data (vs 2 years)

**Expected Results:**
- Accuracy: 85.18% → 90-92%
- Overfitting gap: 10.12% → <5%
- Better performance in bear/choppy markets

---

## 1. Random Forest - Improved Configuration

**File:** `backend/advanced_ml/models/random_forest_model.py`

**Current (Overfitting):**
```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=None,  # No limit → overfits
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    n_jobs=-1,
    random_state=42
)
```

**Improved (Better Generalization):**
```python
RandomForestClassifier(
    n_estimators=200,  # More trees, better averaging
    max_depth=15,  # Limit depth to prevent overfitting
    min_samples_split=10,  # Require more samples before split
    min_samples_leaf=5,  # Require more samples in leaf nodes
    max_features='sqrt',  # Keep sqrt
    min_impurity_decrease=0.0001,  # Minimum improvement required
    max_samples=0.8,  # Bootstrap 80% of data (more diversity)
    n_jobs=-1,
    random_state=42,
    oob_score=True  # Out-of-bag scoring for validation
)
```

**Expected Impact:** Reduce overfitting gap from 10% to 5%

---

## 2. XGBoost - Add Early Stopping

**File:** `backend/advanced_ml/models/xgboost_model.py`

**Current:**
```python
xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=7,
    objective='multi:softprob',
    n_jobs=-1,
    random_state=42
)
```

**Improved:**
```python
xgb.XGBClassifier(
    n_estimators=500,  # More iterations
    learning_rate=0.05,  # Lower learning rate (more conservative)
    max_depth=6,  # Slightly shallower trees
    min_child_weight=5,  # More conservative splits
    subsample=0.8,  # Use 80% of data per tree
    colsample_bytree=0.8,  # Use 80% of features per tree
    gamma=0.1,  # Minimum loss reduction for split
    reg_alpha=0.1,  # L1 regularization
    reg_lambda=1.0,  # L2 regularization
    objective='multi:softprob',
    n_jobs=-1,
    random_state=42,
    early_stopping_rounds=10  # Stop if no improvement
)
```

**Expected Impact:** Better generalization, +1-2% accuracy

---

## 3. Neural Network - Add Dropout & Regularization

**File:** `backend/advanced_ml/models/neural_network_model.py`

**Current:**
```python
MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    alpha=0.0001,  # Too low
    max_iter=200,
    random_state=42
)
```

**Improved:**
```python
MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    alpha=0.01,  # 100x stronger L2 regularization
    learning_rate_init=0.001,
    batch_size=64,
    max_iter=500,
    early_stopping=True,  # Enable early stopping
    validation_fraction=0.2,  # Use 20% for validation
    n_iter_no_change=15,  # Patience for early stopping
    random_state=42
)
```

**Expected Impact:** Reduce overfitting, +1-2% accuracy

---

## 4. LightGBM - Better Regularization

**File:** `backend/advanced_ml/models/lightgbm_model.py`

**Current:**
```python
lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=7,
    n_jobs=-1,
    random_state=42
)
```

**Improved:**
```python
lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,  # Limit complexity
    min_child_samples=10,  # More conservative splits
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,  # L1 regularization
    reg_lambda=1.0,  # L2 regularization
    min_split_gain=0.01,  # Minimum gain for split
    n_jobs=-1,
    random_state=42
)
```

**Expected Impact:** Better generalization

---

## 5. SVM - Add Feature Scaling

**File:** `backend/advanced_ml/models/svm_model.py`

**Current (14.44% accuracy - broken!):**
```python
SVC(
    C=1.0,
    kernel='rbf',
    gamma='scale',
    max_iter=1000,  # Too low
    probability=True,
    random_state=42
)
```

**Improved:**
```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# SVM REQUIRES scaled features!
Pipeline([
    ('scaler', StandardScaler()),  # Critical for SVM
    ('svm', SVC(
        C=10.0,  # Slightly more complex boundary
        kernel='rbf',
        gamma='scale',
        max_iter=2000,  # More iterations
        probability=True,
        random_state=42,
        class_weight='balanced'  # Handle class imbalance
    ))
])
```

**Expected Impact:** 14% → 75-80% accuracy (HUGE fix!)

---

## 6. Update Feature Integration

**File:** `backend/advanced_ml/backtesting/historical_backtest.py`

Add regime + macro features to the feature extraction:

```python
# In get_features_for_date() method

from advanced_ml.features.regime_macro_features import get_regime_macro_features

# After extracting base features
base_features = feature_engineer.extract_features(price_data, symbol)

# Add regime + macro features
regime_macro = get_regime_macro_features(date, symbol)
base_features.update(regime_macro)

# Now we have 179 + 25 = 204 features total
```

---

## 7. Update Step 11 for 5 Years

**File:** `run_step_11.py` (line 42)

**Current:**
```python
results = pipeline.run_full_pipeline(
    symbols=all_core,
    years=2,  # ← Change this
    test_size=0.2,
    use_existing_data=False
)
```

**Improved:**
```python
results = pipeline.run_full_pipeline(
    symbols=all_core,
    years=5,  # ← 5 years for more data
    test_size=0.2,
    use_existing_data=False
)
```

**Expected Impact:**
- 2 years: ~33,930 samples
- 5 years: ~85,000 samples (2.5x more data!)
- Better learning, less overfitting

---

## Implementation Steps

**Tonight (in order):**

1. ✅ Created `regime_macro_features.py` (DONE)
2. ⏳ Update `random_forest_model.py` with better regularization
3. ⏳ Update `xgboost_model.py` with early stopping
4. ⏳ Update `neural_network_model.py` with stronger regularization
5. ⏳ Update `lightgbm_model.py` with better params
6. ⏳ Fix `svm_model.py` with feature scaling
7. ⏳ Update `historical_backtest.py` to include new features
8. ⏳ Update `run_step_11.py` to use 5 years
9. ⏳ Run overnight validation

**Expected Runtime:** 8-12 hours (more data = longer training)

**Target Results:**
- Meta-Learner: 90-92% accuracy
- Overfitting gap: <5%
- Individual models: 88-93%

---

## Quick Implementation Commands

**To implement all changes quickly:**

```bash
# 1. Copy improved model files (will create these next)
# 2. Run Step 11 with improvements
python run_step_11_phase1.py  # Will create this

# 3. Wait 8-12 hours
# 4. Review results in morning
```

---

**Ready to implement?** Say "yes" and I'll create all the improved model files!
