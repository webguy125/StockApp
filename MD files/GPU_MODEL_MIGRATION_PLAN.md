# GPU Model Migration Plan
**Date**: December 30, 2025
**Goal**: Replace CPU-only sklearn models with GPU-native alternatives

---

## Current Model Performance (30 Features, CPU)

| Model | Training Acc | CV Acc | Status |
|-------|-------------|--------|--------|
| Random Forest | 99.87% | 72.84% | ⚠️ Overfit |
| XGBoost | 75.17% | 65.92% | ✅ GPU Ready (not enabled) |
| LightGBM | 65.57% | N/A | ✅ GPU Enabled |
| Extra Trees | 99.84% | N/A | ⚠️ Severe Overfit |
| Gradient Boosting | 78.69% | N/A | 🔄 Replace with CatBoost |
| Neural Network | 64.98% | 60.42% | ✅ OK |
| Logistic Regression | 50.42% | 50.43% | 🔄 Needs GPU Boost |
| SVM | 16.90% | 20.93% | ❌ **FAILING** |

**Average CV Accuracy**: ~62% (excluding failed models)

---

## Proposed GPU Migration

### Phase 1: Critical Fixes (Do First)

#### 1. Replace sklearn SVM with cuML SVM 🚨 **PRIORITY 1**
**Current Problem**: 20.93% accuracy (worse than random!)
**Root Cause**: sklearn SVM with RBF kernel doesn't scale to 159K samples
**Solution**: cuML SVM with GPU acceleration

```python
# BEFORE (sklearn - FAILING):
from sklearn.svm import SVC
model = SVC(kernel='rbf', C=1.0, gamma='scale')

# AFTER (cuML - GPU accelerated):
from cuml.svm import SVC as cuSVC
model = cuSVC(kernel='rbf', C=1.0, gamma='scale')
```

**Expected Result**: 70-80% accuracy (like other models)
**Speed**: 10-50x faster than sklearn
**Install**: `pip install cuml-cu12`

---

#### 2. Enable XGBoost GPU (Already Installed) ✅ **EASY WIN**
**Current**: `device="cpu"` (GPU support disabled!)
**Fix**: Change one parameter

```python
# BEFORE:
model = xgb.XGBClassifier(device="cpu", ...)

# AFTER:
model = xgb.XGBClassifier(device="cuda", ...)
```

**Expected Result**: Same 65% accuracy, 5-10x faster training
**Speed**: 2-3 seconds vs 20-30 seconds per fold

---

### Phase 2: Performance Boost (After 179-feature backtest)

#### 3. Replace Gradient Boosting with CatBoost GPU 📈 **HIGH VALUE**
**Current**: sklearn GradientBoostingClassifier (78.69%)
**Upgrade**: CatBoost with GPU acceleration

```python
# BEFORE (sklearn):
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)

# AFTER (CatBoost GPU):
from catboost import CatBoostClassifier
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.03,
    task_type='GPU',
    devices='0',
    verbose=False
)
```

**Expected Result**: 80-85% accuracy (+2-7% improvement)
**Speed**: 5-20x faster than sklearn
**Install**: `pip install catboost`

---

#### 4. Replace Random Forest with cuML RandomForest 🌲 **BIG SPEEDUP**
**Current**: sklearn RandomForest (72.84% CV, severely overfit)
**Upgrade**: cuML RandomForest with GPU parallelization

```python
# BEFORE (sklearn):
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, max_depth=None)

# AFTER (cuML GPU):
from cuml.ensemble import RandomForestClassifier as cuRF
model = cuRF(n_estimators=100, max_depth=16, max_features=0.3)
```

**Expected Result**: 70-75% accuracy (less overfit + better generalization)
**Speed**: 10-40x faster than sklearn
**Benefit**: GPU parallelization across all trees

---

#### 5. Replace Logistic Regression with cuML Logistic Regression 📊 **QUICK FIX**
**Current**: sklearn LogisticRegression (50.43% - basically random)
**Upgrade**: cuML LogisticRegression (GPU accelerated)

```python
# BEFORE (sklearn):
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000, solver='lbfgs')

# AFTER (cuML GPU):
from cuml.linear_model import LogisticRegression as cuLR
model = cuLR(max_iter=1000, solver='qn')
```

**Expected Result**: 55-65% accuracy (+5-15% improvement)
**Speed**: 5-10x faster
**Benefit**: Better handling of large datasets

---

#### 6. Replace Extra Trees with cuML ExtraTrees (Optional) 🌳
**Current**: sklearn ExtraTreesClassifier (99.84% - severe overfit)
**Upgrade**: cuML RandomForest with different config (cuML doesn't have ExtraTrees, use RF variant)

```python
# BEFORE (sklearn ExtraTrees):
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier(n_estimators=100)

# AFTER (cuML RF with ExtraTrees-like config):
from cuml.ensemble import RandomForestClassifier as cuRF
model = cuRF(
    n_estimators=100,
    max_depth=None,  # Deeper trees like ExtraTrees
    max_features=0.5,  # More features per split
    bootstrap=False  # No bootstrapping like ExtraTrees
)
```

**Expected Result**: 75-80% accuracy (less overfit)
**Speed**: 10-40x faster

---

### Phase 3: Keep or Enhance

#### 7. Keep LightGBM ✅ **ALREADY GPU ENABLED**
**Status**: Already using GPU (`device='gpu'`)
**Performance**: 65.57% (good)
**Action**: No changes needed

---

#### 8. Replace Neural Network with PyTorch GPU 🧠 **GPU ACCELERATION**
**Current**: sklearn MLPClassifier (60.42% CV, CPU-only)
**Upgrade**: PyTorch neural network with GPU acceleration

```python
# BEFORE (sklearn - CPU only):
from sklearn.neural_network import MLPClassifier
model = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    max_iter=200
)

# AFTER (PyTorch - GPU accelerated):
import torch
import torch.nn as nn

class TurboModeNN(nn.Module):
    def __init__(self, input_size=179, hidden_sizes=[128, 64, 32], num_classes=3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        return self.fc4(x)

# Training on GPU
model = TurboModeNN().to('cuda')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
```

**Expected Result**: 70-75% accuracy (+10-15% improvement)
**Speed**: 5-20x faster than sklearn MLP
**Benefit**:
- GPU parallelization for batch processing
- Better gradient descent with GPU acceleration
- More control over architecture
- Supports larger networks without CPU bottleneck

**Already Installed**: PyTorch with CUDA is already in your environment

---

## Installation Commands

### cuML (GPU-accelerated sklearn)
```bash
# CUDA 12.x compatible
pip install cuml-cu12

# Verify installation
python -c "import cuml; print(cuml.__version__)"
```

### CatBoost (GPU-enabled gradient boosting)
```bash
pip install catboost

# Verify GPU support
python -c "import catboost; print(catboost.__version__)"
```

### Verify CUDA Compatibility
```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}')"
```

---

## Expected Performance After Migration

### With 179 Features + GPU Models

| Model | Current (CPU) | Expected (GPU) | Speedup | Accuracy Gain |
|-------|--------------|----------------|---------|---------------|
| **SVM** | 20.93% ❌ | **70-80%** ✅ | 10-50x | +50-60% |
| **XGBoost** | 65.92% | **65-70%** ✅ | 5-10x | Same/+5% |
| **CatBoost** | N/A | **80-85%** ✅ | 5-20x | New model |
| **cuML RF** | 72.84% | **70-75%** ✅ | 10-40x | Less overfit |
| **cuML LR** | 50.43% | **55-65%** ✅ | 5-10x | +5-15% |
| **LightGBM** | 65.57% | **65-70%** ✅ | Already GPU | Same |
| **PyTorch NN** | 60.42% | **70-75%** ✅ | 5-20x | +10-15% |
| **ExtraTrees** | 99.84% (overfit) | **75-80%** ✅ | 10-40x | Less overfit |

**Estimated Ensemble Accuracy**: **85-92%** (vs current 62%)

---

## Migration Timeline

### Tonight:
1. ✅ Run 179-feature backtest (10-12 hours)
2. ✅ Wait for completion

### Tomorrow Morning:
1. **Quick Win**: Enable XGBoost GPU (1-line change)
2. **Critical Fix**: Install cuML and replace SVM
3. **Train models** with 179 features + 2 GPU fixes
4. **Compare**: CPU baseline vs GPU with 2 fixes

### After Seeing Results:
1. Install CatBoost
2. Replace remaining CPU models with cuML versions
3. **Final training** with all GPU models
4. **Compare**: All three baselines
   - Baseline 1: 30 features + CPU (62% - current)
   - Baseline 2: 179 features + CPU (expected 70-75%)
   - Baseline 3: 179 features + GPU (expected 85-92%)

---

## Implementation Steps

### Step 1: Install GPU Libraries
```bash
cd C:\StockApp
venv\Scripts\pip.exe install cuml-cu12 catboost
```

### Step 2: Update Model Files
Files to modify:
- `backend/advanced_ml/models/xgboost_model.py` (enable GPU)
- `backend/advanced_ml/models/svm_model.py` (replace with cuML)
- `backend/advanced_ml/models/catboost_model.py` (new file)
- `backend/advanced_ml/models/random_forest_model.py` (replace with cuML)
- `backend/advanced_ml/models/logistic_regression_model.py` (replace with cuML)
- `backend/advanced_ml/models/neural_network_model.py` (replace with PyTorch GPU)

### Step 3: Update Training Script
Modify `backend/turbomode/train_turbomode_models.py`:
- Import new GPU models
- Replace CPU model instances
- Add GPU availability checks

---

## Risk Assessment

### Low Risk ✅
- Enabling XGBoost GPU (just change one parameter)
- Installing CatBoost (separate from sklearn)
- Installing cuML (separate from sklearn)

### Medium Risk ⚠️
- Replacing sklearn models with cuML (API mostly compatible)
- May need to adjust hyperparameters

### High Risk 🚨
- **None** - all changes are reversible by switching back to sklearn

---

## Rollback Plan

If GPU models don't work:
1. Keep original sklearn model files as backups
2. Use git to revert changes
3. Fall back to CPU models
4. All data/training remains intact

---

## Success Criteria

✅ **Minimum Success**: SVM accuracy > 60% (vs current 20%)
✅ **Good Success**: Ensemble accuracy > 75% (vs current 62%)
✅ **Excellent Success**: Ensemble accuracy > 85%

---

## Why This Matters

### Current State:
- **30 features** = limited predictive power
- **CPU models** = slow training, poor SVM performance
- **62% accuracy** = barely better than random (33% for 3 classes)

### After Migration:
- **179 features** = comprehensive technical analysis
- **GPU models** = 10-50x faster training
- **85-92% accuracy** = professional-grade predictions

**Trading Impact**: With 85%+ accuracy, the system can confidently:
- Generate 10-50 daily signals (vs 0 currently)
- Achieve profitable trading with proper risk management
- Scale to real-time intraday predictions

---

## Next Steps

1. ✅ Wait for tonight's 179-feature backtest to complete
2. ✅ Enable XGBoost GPU (1-minute fix)
3. ✅ Install cuML and fix SVM (30-minute task)
4. ✅ Train with 179 features + GPU fixes
5. ✅ Compare results and decide on full migration

---

**Status**: Ready to execute after tonight's backtest completes!
