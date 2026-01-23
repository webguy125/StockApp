# START FULL TRAINING AT HOME - OVERNIGHT RUN

**Date:** December 25, 2025 (Christmas Night)
**Time Created:** ~6:00 PM

---

## CURRENT STATUS

### What Happened Today
1. ✅ **Fixed all the problematic additions** that caused overfitting:
   - Disabled event features (were pure noise - all zeros)
   - Disabled regime balancing (was losing 40% of training data)
   - Fixed meta-learner training on wrong dataset

2. ✅ **Quick training succeeded** - Got 86.42% with LightGBM (close to 90% baseline)

3. ❌ **Full 82-symbol training got STUCK**:
   - Started at 5:55 PM
   - Hung on AAPL symbol for 6 hours (yfinance API timeout)
   - Had to kill the process

4. ⚠️ **LOST TRAINING DATA**:
   - When we started "full clean training", we deleted the database
   - Lost the 8,760 samples from quick training
   - Database is now EMPTY - need to re-collect data

---

## WHAT TO DO WHEN YOU GET HOME

### Step 1: Open Command Prompt
```batch
# Navigate to project directory
cd C:\StockApp

# Activate virtual environment
venv\Scripts\activate
```

### Step 2: Run Full Training (Visible Window)
```batch
python run_training_with_checkpoints.py
```

**IMPORTANT:** Run this in the command prompt window so you can:
- Watch the progress in real-time
- See which symbols complete successfully
- See if any symbols get stuck (and which ones)
- Know when it finishes

### Step 3: What to Expect

**Phase 1: Historical Backtest (6-9 hours)**
- Processes 82 symbols across 11 GICS sectors
- Downloads 2 years of price data from Yahoo Finance
- Simulates 14-day hold trades
- Creates ~35,000 training samples
- Saves checkpoint after each symbol (can restart if interrupted)

**Phase 2: Model Training (1-2 hours)**
- Trains all 9 models:
  1. Random Forest
  2. XGBoost
  3. LightGBM
  4. Extra Trees
  5. Gradient Boosting
  6. Neural Network
  7. Logistic Regression
  8. SVM
  9. Meta-Learner (ensemble)

**Total Time:** 8-11 hours (perfect for overnight run)

### Step 4: Morning - Check Results

When you wake up, look for this in the command prompt:

**SUCCESS LOOKS LIKE:**
```
======================================================================
ALL 9 MODELS TRAINED!
======================================================================

FINAL RESULTS - ALL 9 MODELS
======================================================================

1. Meta-Learner        Test: 90.24%  Gap:  5.12%  [EXCELLENT]
2. XGBoost            Test: 88.50%  Gap:  6.20%  [EXCELLENT]
3. Random Forest      Test: 87.30%  Gap:  7.10%  [EXCELLENT]
...

SUCCESS! Test accuracy >= 90% - BASELINE RESTORED!
```

**FAILURE LOOKS LIKE:**
- Script stopped/crashed
- Still stuck on same symbol for hours
- Test accuracy < 80%

---

## IF IT GETS STUCK AGAIN

### Identify the Problem Symbol
Look for output like:
```
[42/82] Processing XYZ...
Processing symbols:   0%|          | 0/1 [00:00<?, ?it/s]
```

If this line doesn't change for more than 15-20 minutes, symbol XYZ is stuck.

### Stop and Resume
1. Press `Ctrl+C` to stop the script
2. The checkpoint system saved progress up to the last successful symbol
3. Run the script again - it will skip completed symbols and continue

### Alternative: Quick Training Instead
If full training keeps failing, run quick training (20 symbols, 2-3 hours):
```batch
python run_quick_training.py
```

This trains only the 3 best models but should get you to 85-90% accuracy.

---

## KEY FILES TO CHECK IN THE MORNING

1. **Training Results:**
   - `backend/data/quick_training_results.json` - Final accuracy scores

2. **Checkpoint Status:**
   - `backend/data/checkpoints/training_checkpoint.json` - Progress tracker

3. **Database:**
   - `backend/data/advanced_ml_system.db` - Should be ~50-100 MB if successful

4. **Models:**
   - `backend/data/ml_models/` - Should contain 9 model folders

5. **Log File:**
   - Check the command prompt window for complete output

---

## CONFIGURATION SUMMARY

**Current Settings (All Correct):**
- ✅ Event features: DISABLED (179 technical features only)
- ✅ Regime balancing: DISABLED (keeps all training data)
- ✅ Meta-learner: Fixed (trains on training set, not test set)
- ✅ Feature count: 179 (was 202 with broken event features)
- ✅ Hold period: 14 days (for baseline comparison)
- ✅ Test/train split: 80/20

**Symbols (82 total across 11 sectors):**
```
Tech: AAPL, MSFT, NVDA, GOOGL, META, AMZN, TSLA, AMD, INTC, CRM, ORCL, ADBE, CSCO, AVGO, TXN, QCOM, NFLX, PYPL, SHOP, SQ, PLTR, SNOW, CRWD, ZS, DDOG
Finance: JPM, BAC, WFC, GS, MS, C, BLK, SCHW, AXP, V, MA
Healthcare: UNH, JNJ, LLY, ABBV, MRK, PFE, TMO, ABT, DHR, CVS
Consumer Discretionary: HD, MCD, NKE, SBUX, LOW, TGT
Consumer Staples: PG, KO, PEP, WMT, COST
Energy: XOM, CVX, COP, SLB, EOG
Industrials: CAT, GE, BA, HON, UPS, LMT, RTX
Materials: LIN, APD, NEM
Utilities: NEE, DUK
Real Estate: AMT, PLD
Communication: DIS, CMCSA, T, VZ
```

---

## TROUBLESHOOTING

### Problem: Script crashes immediately
**Solution:** Check that virtual environment is activated
```batch
venv\Scripts\activate
```

### Problem: ModuleNotFoundError
**Solution:** Reinstall requirements
```batch
pip install -r requirements.txt
```

### Problem: Database locked error
**Solution:** Close any open database connections
```batch
# Kill any stuck Python processes
tasklist | findstr python
taskkill //F //PID <process_id>
```

### Problem: Stuck on AAPL again
**Explanation:** Yahoo Finance API sometimes rate-limits AAPL (most popular stock)

**Solutions:**
1. Wait 15 minutes - it may timeout and move to next symbol
2. Press Ctrl+C and restart - checkpoint will skip AAPL
3. Or manually edit the symbol list in the script to skip AAPL

---

## WHAT'S NEXT (AFTER BASELINE RESTORED)

Once you confirm test accuracy ≥ 90%:

1. **Options-Optimized Training** (5-day hold period for puts/calls):
   ```batch
   python run_options_training.py
   ```

2. **Restart Flask Server** (load new models):
   ```batch
   cd backend
   python api_server.py
   ```

3. **Test ML Signals Page:**
   - Open browser: http://127.0.0.1:5000/ml-signals
   - Enter symbols: AAPL, NVDA, TSLA
   - Check predictions are NOT 33/33/33 (should be clear buy/sell/hold)

4. **Compare Performance:**
   - 14-day models (baseline accuracy)
   - 5-day models (options trading)
   - 7-day models (options trading)

---

## NOTES FROM TODAY'S SESSION

### Why Accuracy Dropped from 90% to 34-54%
1. **Event features were pure noise** - All 23 features were 0.0 (no actual data)
2. **Regime balancing lost 40% of data** - Dropped from 27K to 16K samples
3. **Meta-learner trained on test set** - Classic data leakage mistake

### Why Quick Training Got 86.42% (Close to Baseline)
After disabling those bad additions, we got back to clean technical features only (179 features). The 86.42% from LightGBM proves we're on the right track. Full training with more data and meta-learner should hit 90%.

### Why You Trade Options (Not Stocks)
- **Buy signals** = Buy CALL options (bullish)
- **Sell signals** = Buy PUT options (bearish)
- Need shorter hold periods (5-7 days vs 14 days)
- Higher profit targets (+25-30% vs +10%)
- Higher stop losses (-15-20% vs -5%)

---

## FINAL CHECKLIST BEFORE BED

- [x] All stuck training processes killed
- [x] Database cleared (ready for fresh data collection)
- [x] Models cleared (ready for fresh training)
- [x] Checkpoint cleared (ready for fresh run)
- [x] Event features DISABLED
- [x] Regime balancing DISABLED
- [x] Meta-learner training FIXED
- [ ] Run training when home: `python run_training_with_checkpoints.py`
- [ ] Let run overnight (8-11 hours)
- [ ] Check results in morning

---

## EMERGENCY CONTACT

If something goes wrong and you need help:
- All fixes documented in: `TRAINING_FIXES_SUMMARY.md`
- Session notes in: `SESSION_NOTES_2025-12-24.md`
- This file: `START_TRAINING_AT_HOME.md`

---

**Good luck! See you in the morning when the models are trained!**

🎄 Merry Christmas! 🎄
