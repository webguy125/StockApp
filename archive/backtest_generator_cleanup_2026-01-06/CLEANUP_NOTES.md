# Backtest Generator Cleanup - 2026-01-06

## Rogue Files/Tables Created

### 1. PHASE_1_2_IMPLEMENTATION_COMPLETE.md
- **Status:** Archived to this directory
- **Issue:** Created documentation referencing non-existent MASTER_MARKET_DATA_ARCHITECTURE.json
- **Resolution:** Moved to archive

### 2. backtest_runs table in TurboMode.db
- **Status:** May exist in database if backtest_generator.py was run before fixes
- **Issue:** Created custom `backtest_runs` table instead of using existing `training_runs` table
- **Location:** `C:\StockApp\backend\data\turbomode.db`
- **SQL to check:** `SELECT name FROM sqlite_master WHERE type='table' AND name='backtest_runs';`
- **SQL to drop if exists:** `DROP TABLE IF EXISTS backtest_runs;`
- **Resolution:** Backtest generator now uses `training_runs` table with type='backtest' in metadata

## Corrected Implementation

### Changes Made to backtest_generator.py:

1. ✅ Removed reference to non-existent MASTER_MARKET_DATA_ARCHITECTURE.json
2. ✅ Uses `get_candles(symbol, timeframe, start_date, end_date)` with correct parameters
3. ✅ Handles actual DataFrame column names from Master DB (lowercase: timestamp, open, high, low, close, volume)
4. ✅ Creates own SQLite connection using `sqlite3.connect(self.turbomode_db.db_path)`
5. ✅ Wraps all DB operations in try/finally for safe connection management
6. ✅ Uses existing `training_runs` table instead of creating new `backtest_runs` table
7. ✅ Stores backtest metadata in JSON format with type='backtest' flag
8. ✅ Returns Optional[int] and properly handles failures

## Manual Cleanup Required

If the backtest generator was run before these fixes, manually clean up the database:

```bash
cd C:\StockApp\backend\data
sqlite3 turbomode.db
```

```sql
-- Check if rogue table exists
SELECT name FROM sqlite_master WHERE type='table' AND name='backtest_runs';

-- If it exists, drop it
DROP TABLE IF EXISTS backtest_runs;

-- Verify it's gone
.tables

-- Exit
.quit
```

## Architecture Compliance

The corrected backtest_generator.py now:
- Reads from Master Market Data DB (read-only) using get_market_data_api()
- Writes to TurboMode.db using existing schema (training_runs table)
- Does not invent new tables or schemas
- Follows the same patterns as overnight_scanner.py and training_orchestrator.py
