# 179-FEATURE SPEED ISSUE - CRITICAL

## Problem Discovered

**179-feature GPU batch mode is EXTREMELY SLOW:**
- **Actual speed**: 25+ minutes per symbol (stuck on chunk 6/9)
- **Projected time**: 200+ hours for 510 symbols (8+ days!)
- **30-feature vectorized mode**: ~20 seconds per symbol (total: 2.8 hours)

## Root Cause

The `extract_features_batch()` method in `GPUFeatureEngineer` is much slower than expected. It's processing 179 features per window, but something is causing massive slowdown.

## Current Status

1. ✓ Fixed the code to prevent silent 30-feature fallback
2. ✗ 179-feature extraction is too slow for practical use
3. We have 199,053 samples with 30 features from overnight run (b0eb99)

## Options Going Forward

### OPTION 1: Train with 30 features NOW (Quick Win)
**Pros:**
- Data already exists (199,053 samples from overnight)
- Can train models immediately
- Will show SOME improvement over current 49-75%

**Cons:**
- Still won't have full 179-feature accuracy
- Band-aid solution

**Action:**
```bash
# Restore the 30-feature data (it was deleted)
cd backend/turbomode
../../venv/Scripts/python.exe generate_backtest_data.py  # Uses fast 30-feature mode
# Then train
../../venv/Scripts/python.exe train_turbomode_models.py
```

### OPTION 2: Optimize 179-feature extraction (Best Long-term)
**What needs investigation:**
-  `backend/advanced_ml/features/gpu_feature_engineer.py:extract_features_batch()`
- Check if GPU is actually being used efficiently
- Profile the slow chunks to find bottleneck
- May need to optimize feature calculation algorithms

**Time required:** 1-2 hours of debugging + optimization

### OPTION 3: Hybrid Approach
- Use 30-feature mode for quick iteration/testing
- Run optimized 179-feature extraction overnight when ready
- Keeps momentum while working on optimization

## Recommendation

**IMMEDIATE (Tonight):**
1. Restore 30-feature data from overnight run
2. Train models with 30 features
3. See improvement from 199K samples (even if not full 179 features)

**TOMORROW:**
1. Profile and optimize `extract_features_batch()`
2. Target: <2 minutes per symbol (vs current 25+ min)
3. Run optimized version overnight for 179 features

## The Real Issue

The 30-feature vectorized mode was COMMENTED OUT in the code as "fallback", but it's actually 75x faster! The 179-feature "batch" mode needs serious optimization work before it's practical for 510 symbols.

---

**Status:** 179-feature extraction confirmed working, but impractically slow
**Next Step:** Decide whether to optimize speed first or train with 30 features
**Created:** Dec 30, 2025 - 1:00 PM
