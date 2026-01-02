# TODO - December 28, 2025

## URGENT: Fix Frontend Not Showing New Signals

**Problem:**
- Fresh scan completed with 100 BUY signals (SMCI 98.66%, INTC 98.66%, etc.)
- Scanner saved to: `C:\StockApp\backend\data\turbomode.db`
- Flask now reads from correct database (absolute path confirmed)
- Frontend still shows OLD signals (99.1% confidence, KO, PG, etc.)

**Likely Causes:**
1. Scanner saving to wrong table (`signals` vs `active_signals`)
2. Frontend API reading from wrong table
3. Database schema mismatch
4. Frontend caching issue

**Next Steps:**
1. Check which table the scanner saves to:
   ```bash
   grep -n "save_signal\|add_signal" backend/turbomode/overnight_scanner.py
   ```

2. Check database contents:
   ```bash
   sqlite3 backend/data/turbomode.db "SELECT name FROM sqlite_master WHERE type='table';"
   sqlite3 backend/data/turbomode.db "SELECT COUNT(*) FROM signals;"
   sqlite3 backend/data/turbomode.db "SELECT COUNT(*) FROM active_signals;"
   sqlite3 backend/data/turbomode.db "SELECT symbol, confidence FROM signals LIMIT 5;"
   sqlite3 backend/data/turbomode.db "SELECT symbol, confidence FROM active_signals LIMIT 5;"
   ```

3. Check which table the API endpoint reads from:
   ```bash
   grep -n "SELECT.*FROM" backend/routes/turbomode_routes.py
   ```

4. Fix table mismatch (scanner and API must use same table)

**Expected Result:**
- Frontend shows: SMCI, INTC, TMDX, PLAY, CABO at 98.66% confidence
- All dated today (Dec 27, 2025)

---

## Cleanup Tasks (from CLEANUP_TASKS_2025-12-27.md)

### Priority 1: Remove Unused Databases
- 4 duplicate databases scattered around project
- Causing confusion and wrong connections
- **Files to delete:**
  - `backend/advanced_ml/backtesting/backend/data/advanced_ml_system.db`
  - `backend/backend/backend/data/advanced_ml_system.db`
  - `backend/data/rare_event_archive/scripts/backend/data/advanced_ml_system.db`
  - `backend/turbomode/backend/data/advanced_ml_system.db`

### Priority 2: Fix LightGBM Warnings
- Scanner output flooded with feature name warnings
- Two solutions available (Option 1 recommended - no retraining)

### Priority 3: Pin Library Versions
- Prevent numpy version mismatches
- Add exact versions to requirements.txt

---

## System Status

### ✅ Working
- TurboMode models trained (88.93% test accuracy)
- Models separated from Slipstream
- Scanner runs successfully
- Database paths fixed (absolute paths)
- Flask server starts correctly

### ⚠️ Not Working
- Frontend not displaying fresh signals
- Table mismatch between scanner and API

### 🔄 Pending
- Cleanup unused databases
- Fix LightGBM warnings
- Pin library versions
- Verify tonight's 11 PM scheduled scan

---

## Files Modified Today

**Created:**
- `backend/data/turbomode_models/` (all 9 models)
- `backend/turbomode/train_turbomode_models.py`
- `CLEANUP_TASKS_2025-12-27.md`
- `TODO_TOMORROW_2025-12-28.md` (this file)

**Modified:**
- `backend/turbomode/overnight_scanner.py` (absolute paths, direct model loading)
- `backend/turbomode/overnight_scanner.py.backup` (safety backup)
- `backend/turbomode/database_schema.py` (absolute path fix)
- `.gitignore` (exclude *.db, *.pkl, *.joblib)

**Training Results:**
- 27,268 training samples
- 6,818 test samples
- 179 features
- Meta-learner: 88.93% test accuracy (best model)
- Scan results: 100 BUY signals generated

---

## Quick Commands for Tomorrow

**Activate venv:**
```bash
venv\Scripts\activate
```

**Start Flask:**
```bash
start_flask.bat
```

**Run TurboMode scan:**
```bash
cd backend\turbomode
python overnight_scanner.py
```

**Check database:**
```bash
sqlite3 backend/data/turbomode.db
```

---

**Session End:** December 27, 2025 - 5:45 PM
**Status:** Model separation complete, frontend issue to resolve tomorrow
