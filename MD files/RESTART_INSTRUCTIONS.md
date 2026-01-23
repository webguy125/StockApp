# QUICK START - After 6 PM Restart

## 🚨 IMMEDIATE ACTIONS

### 1. Check if Backtest Completed
```bash
cd C:\StockApp
dir backend\backend\data\advanced_ml_system.db
```
**Expected:** File size > 10 MB (means data was generated)

### 2. Verify Training Data
```bash
./venv/Scripts/python.exe -c "import sqlite3; conn = sqlite3.connect('backend/backend/data/advanced_ml_system.db'); cursor = conn.cursor(); cursor.execute('SELECT outcome, COUNT(*) FROM trades WHERE trade_type=\"backtest\" GROUP BY outcome'); print('Label distribution:'); [print(f'  {row[0]}: {row[1]}') for row in cursor.fetchall()]"
```

**Expected Output:**
```
Label distribution:
  buy: 100-200
  hold: 300-400
  sell: 100-200
```

✅ If you see 'buy', 'hold', 'sell' → Fix worked!
❌ If you see 'win', 'neutral', 'loss' → Fix failed!

---

## 🔧 NEXT STEPS

### Step 1: Train Models (~15 min)
```bash
cd backend\turbomode
..\..\venv\Scripts\python.exe train_turbomode_models.py
```

### Step 2: Clear Bad Signals
```bash
..\..\venv\Scripts\python.exe -c "from database_schema import TurboModeDB; db = TurboModeDB(); db.clear_all_data(); print('Cleared old signals')"
```

### Step 3: Run Test Scan
```bash
python overnight_scanner.py
```

### Step 4: Check Results
```bash
cd ..\..
./venv/Scripts/python.exe analyze_signals.py
```

**Look for:**
- ✅ Some SELL signals (not 0%)
- ✅ Varied confidence scores (spread > 5%)
- ✅ BUY: 30-70%, SELL: 10-40%

---

## ⚠️⚠️⚠️ BEFORE FULL PRODUCTION RUN ⚠️⚠️⚠️

### CRITICAL: Set Full Training Flag

**File:** `backend/turbomode/regenerate_training_data.py`
**Line 68:**

**CHANGE THIS:**
```python
USE_ALL_SYMBOLS = False
```

**TO THIS:**
```python
USE_ALL_SYMBOLS = True  # ✅ FULL PRODUCTION MODE
```

### Then Run Full Training (36 hours):
```bash
cd backend\turbomode
..\..\venv\Scripts\python.exe regenerate_training_data.py
```

**Start:** Friday 6 PM
**Finish:** Sunday 6 AM

---

## 📚 Full Details

See: `TURBOMODE_FIX_SUMMARY_2025-12-28.md`

---

**Last Updated:** December 28, 2025 - 4:00 PM
