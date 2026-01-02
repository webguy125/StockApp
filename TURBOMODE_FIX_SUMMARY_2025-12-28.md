# TurboMode Critical Bug Fix - December 28, 2025

## EXECUTIVE SUMMARY

**CRITICAL BUGS DISCOVERED:**
1. ❌ ALL 200 signals were BUY (zero SELL signals)
2. ❌ All confidence scores clustered in 0.014% range (99.135-99.149%)
3. ❌ Signal replacement logic broken (can't replace when all confidences identical)
4. ❌ Model trained with BACKWARDS labels

**ROOT CAUSE:**
Training data label mapping bug in `historical_backtest.py` caused model to learn inverted predictions.

**STATUS:** ✅ Fix applied, currently testing with 10-symbol dataset

---

## DETAILED FINDINGS

### Issue #1: Zero SELL Signals

**Evidence:**
- Morning scan (3:47 AM): 100 BUY signals, 0 SELL
- Evening scan (5:30 PM): 100 BUY signals, 0 SELL
- Total: 200/200 signals are BUY (100%)

**Statistical Impossibility:**
Out of 500+ S&P stocks, having ZERO bearish signals is mathematically impossible for a properly functioning model.

### Issue #2: Identical Confidence Scores

**Morning Scan:**
- Min: 99.135%
- Max: 99.149%
- Spread: 0.014% (absurdly narrow)

**Evening Scan:**
- Min: 98.627%
- Max: 98.664%
- Spread: 0.037% (absurdly narrow)

**Normal Model Behavior:**
Confidence scores should vary widely (60-99%) based on different stocks' characteristics.

### Issue #3: Root Cause - Label Mapping Bug

**File:** `backend/advanced_ml/backtesting/historical_backtest.py`

**The Bug (Lines 289 & 434):**
```python
# WRONG - Saved labels as outcomes, not actions:
Line 289: 'win' if sample['label'] == 0 else 'loss' if sample['label'] == 2 else 'neutral'
Line 434: label_map = {'win': 0, 'neutral': 1, 'loss': 2}
```

**What Happened:**
1. Backtest generates labels: 0='buy', 1='hold', 2='sell' ✓ CORRECT
2. Save to database converts: 0→'win', 1→'neutral', 2→'loss' ❌ WRONG!
3. Training reads back: 'win'→0, 'neutral'→1, 'loss'→2
4. Result: Model thinks 0='win' (outcome), but code expects 0='buy' (action)

**Training Data Distribution (WRONG):**
- win: 4,917 samples (14.4%) → misinterpreted as 'buy'
- neutral: 18,390 samples (53.9%) → misinterpreted as 'hold'
- loss: 10,779 samples (31.6%) → misinterpreted as 'sell'

**Effect:**
Model learned to predict 'buy' for historically profitable patterns. Since current market looks "bullish" to the model, it predicts BUY for everything!

---

## THE FIX

### Changes Made

**File:** `backend/advanced_ml/backtesting/historical_backtest.py`

**Fix #1 (Lines 277-293):**
```python
# Map label integer to action name (buy/hold/sell)
label_to_action = {0: 'buy', 1: 'hold', 2: 'sell'}
action = label_to_action[sample['label']]

cursor.execute('''...''', (
    ...
    action,  # Store 'buy', 'hold', or 'sell' instead of 'win'/'neutral'/'loss'
    ...
))
```

**Fix #2 (Lines 437-440):**
```python
# Map outcome (action) to label integer
# outcome field now stores 'buy', 'hold', or 'sell' (not 'win'/'neutral'/'loss')
label_map = {'buy': 0, 'hold': 1, 'sell': 2}
label = label_map.get(outcome, 1)  # Default to 1 (hold) if unknown
```

**Result:**
Labels now correctly represent ACTIONS (what to do) not OUTCOMES (what happened).

---

## CURRENT STATUS (AS OF 4:00 PM)

### ✅ Completed
1. Fixed label mapping in `historical_backtest.py`
2. Installed APScheduler for automated scans
3. Created `regenerate_training_data.py` script
4. Started 10-symbol fast test (running now)

### ⏳ In Progress
- **Backtesting 10 symbols** (~40 minutes)
  - ETA: 4:40 PM
  - Will generate ~670 training samples
  - Enough to verify fix works

### 📋 Next Steps (MUST DO AFTER BACKTEST COMPLETES)

#### Step 1: Retrain Models (~15 min)
```bash
cd backend/turbomode
../../venv/Scripts/python.exe train_turbomode_models.py
```

Expected result: Models train on corrected labels

#### Step 2: Clear Bad Signals
```bash
../../venv/Scripts/python.exe -c "from database_schema import TurboModeDB; db = TurboModeDB(); db.clear_all_data(); print('Cleared bad signals')"
```

#### Step 3: Run Test Scan (~5 min)
```bash
python overnight_scanner.py
```

**Verification Checklist:**
- [ ] Some SELL signals generated (not all BUY)
- [ ] Confidence scores vary (not all 99%)
- [ ] Signal distribution looks reasonable:
  - BUY: 30-70% (not 100%)
  - SELL: 10-40% (not 0%)
  - Confidence spread: > 5% range

#### Step 4: If Test Passes - Set Up Full Training

## ⚠️⚠️⚠️ CRITICAL: SET BACKTEST TO FULL ⚠️⚠️⚠️

**YOU MUST DO THIS BEFORE RUNNING PRODUCTION TRAINING!**

**File:** `backend/turbomode/regenerate_training_data.py`

**Line 68 - CHANGE FROM:**
```python
USE_ALL_SYMBOLS = False  # Set to True for full training (run overnight)
```

**TO:**
```python
USE_ALL_SYMBOLS = True  # ✅ PRODUCTION MODE - Full 510 symbols
```

**After Changing Flag, Run Overnight:**
```bash
cd backend/turbomode
../../venv/Scripts/python.exe regenerate_training_data.py
```

**WARNING:** This takes ~36 hours! Start Friday evening, completes Sunday morning.

**DO NOT FORGET THIS STEP OR YOU'LL TRAIN ON ONLY 10 SYMBOLS!**

---

## TIMELINE ESTIMATES

### Fast Test (Current)
- **Backtest:** 10 symbols × 4 min = 40 minutes
- **Train models:** 15 minutes
- **Test scan:** 5 minutes
- **Total:** ~60 minutes

### Full Production (After Test Passes)
- **Backtest:** 510 symbols × 4 min = 36 hours (1.5 days)
- **Train models:** 15 minutes
- **Production scan:** 30 minutes
- **Total:** ~36.75 hours

**Recommended Schedule:**
- Friday 6:00 PM: Start full backtest
- Sunday 6:00 AM: Backtest completes
- Sunday 6:15 AM: Train models
- Sunday 6:30 AM: Run production scan
- Sunday 7:00 AM: Review results

---

## VERIFICATION CRITERIA

### ✅ Fix is Working If:
1. SELL signals appear (10-40% of total)
2. Confidence scores vary widely (60-99% range, spread > 5%)
3. Signal replacement logic works (older/weaker signals get replaced)
4. Label distribution in database shows:
   - 'buy': 10-20%
   - 'hold': 50-70%
   - 'sell': 10-20%

### ❌ Fix Failed If:
1. Still 100% BUY or 100% SELL
2. Confidence scores still clustered (< 1% spread)
3. Model accuracy drops below 70%
4. All signals have same sector/market cap

---

## FILES MODIFIED

### Core Fixes
- `backend/advanced_ml/backtesting/historical_backtest.py` (Lines 277-293, 437-440)

### New Files Created
- `backend/turbomode/regenerate_training_data.py` - Automated backtest + verification
- `C:\StockApp\TURBOMODE_FIX_SUMMARY_2025-12-28.md` - This document
- `C:\StockApp\TODO_TOMORROW_2025-12-28.md` - Original task list
- `C:\StockApp\check_db.py` - Database analysis utility
- `C:\StockApp\analyze_signals.py` - Signal distribution analysis

### Helper Scripts
- `check_db.py` - Analyze database contents
- `analyze_signals.py` - Check BUY/SELL distribution

---

## COMMANDS REFERENCE

### Check Database Contents
```bash
cd C:\StockApp
./venv/Scripts/python.exe -c "import sqlite3; conn = sqlite3.connect('backend/backend/data/advanced_ml_system.db'); cursor = conn.cursor(); cursor.execute('SELECT outcome, COUNT(*) FROM trades WHERE trade_type=\"backtest\" GROUP BY outcome'); [print(f'{row[0]}: {row[1]}') for row in cursor.fetchall()]"
```

### Check Signal Distribution
```bash
./venv/Scripts/python.exe analyze_signals.py
```

### Monitor Background Process
```bash
# Check if backtest is running
tasklist | findstr python

# Check database size (grows as data is generated)
dir backend\backend\data\advanced_ml_system.db
```

### Kill Stuck Process
```bash
taskkill /F /IM python.exe
```

---

## TECHNICAL DETAILS

### Label Encoding

**Correct Encoding (After Fix):**
```
0 = 'buy'   → Stock will go UP   → Model should recommend BUY
1 = 'hold'  → Stock will stay flat → Model should recommend HOLD
2 = 'sell'  → Stock will go DOWN  → Model should recommend SELL
```

**Wrong Encoding (Before Fix):**
```
0 = 'win'     → Past trade was profitable   → Misinterpreted as 'buy'
1 = 'neutral' → Past trade was flat         → Misinterpreted as 'hold'
2 = 'loss'    → Past trade lost money       → Misinterpreted as 'sell'
```

### Why This Broke Everything

1. Model trained to predict: 0 when pattern looks like past "winners"
2. Code interprets 0 as: "BUY this stock"
3. Current market conditions look "bullish" to model (like past winners)
4. Result: Model predicts 0 (BUY) for everything

### Confidence Score Issue

The meta-learner uses softmax probabilities. When model is 99%+ confident:
- buy_prob: 0.9914
- hold_prob: 0.0056
- sell_prob: 0.0030

Confidence = max(probabilities) = 0.9914

**Why All Confidences Identical:**
Model is overconfident AND all stocks look similar (because it's using wrong labels). With correct labels, stocks should have varied predictions.

---

## KNOWN ISSUES & WARNINGS

### Issue: Training Takes 36 Hours
**Cause:** Downloading 2 years of data for 510 symbols + feature calculation
**Solutions:**
1. Use cached data (if available)
2. Reduce to 1 year of history (edit `years=2` to `years=1`)
3. Parallelize symbol processing
4. Run on faster machine

### Issue: Database Grows Large
**Size:** ~200 MB for 34,000 samples
**Solution:** This is normal, ensure sufficient disk space

### Issue: LightGBM Warnings (Still Present)
**Error:** "feature_name is overridden"
**Impact:** Cosmetic only, doesn't affect predictions
**Fix:** See `CLEANUP_TASKS_2025-12-27.md` for solutions

---

## CONTACT & SUPPORT

**Session:** December 28, 2025 - 12:00 PM to 6:00 PM
**Assistant:** Claude (Sonnet 4.5)
**User:** Development Team

**Questions?** Re-read this document and check the TODO list.

---

## FINAL CHECKLIST BEFORE SHUTDOWN

- [ ] Backtest process running (check with `tasklist | findstr python`)
- [ ] `USE_ALL_SYMBOLS` flag documented for post-restart change
- [ ] Summary document saved: `TURBOMODE_FIX_SUMMARY_2025-12-28.md`
- [ ] All code changes committed to git (optional but recommended)

---

**END OF DOCUMENT**

Last Updated: December 28, 2025 - 4:00 PM
Status: FIX APPLIED, TESTING IN PROGRESS
Next Session: After 6 PM restart - Resume with Step 1 (Retrain Models)
