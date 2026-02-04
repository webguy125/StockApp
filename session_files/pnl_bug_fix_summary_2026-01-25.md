# P&L Bug Fix Summary
## 2026-01-25

## Problem Identified

SELL (short) positions had inverted profit/loss calculations throughout the system, causing:
- Winners to appear as losers
- Losers to appear as winners
- Corrupted adaptive ranking scores
- Misleading performance metrics

### Root Cause

`signal_closer.py` line 79 used a single formula for ALL signal types:
```python
pnl_pct = ((current_price - entry_price) / entry_price) * 100.0
```

This formula is ONLY correct for BUY (long) positions. For SELL (short) positions, it produces inverted results.

### Real-World Impact

**INTU SELL Example:**
- Entry: $528.95
- Exit: $563.97 (price went UP 6.62%)
- OLD P/L (WRONG): +6.62% (showed profit when it was a loss)
- NEW P/L (CORRECT): -6.62% (short position loses when price rises)

**GE SELL Example:**
- Entry: $312.34
- Exit: $293.87 (price went DOWN 5.91%)
- OLD P/L (WRONG): -5.91% (showed loss when it was a profit)
- NEW P/L (CORRECT): +5.91% (short position profits when price drops)

## Solution Implemented

### 1. Created Canonical P/L Engine

**File:** `backend/turbomode/core_engine/pnl_utils.py`

Single source of truth for P/L calculations:
```python
def calculate_pnl_pct(entry_price: float, exit_price: float, signal_type: str) -> float:
    if signal_type == 'SELL':
        # Short position: profit when price decreases
        pnl_pct = ((entry_price - exit_price) / entry_price) * 100.0
    else:
        # Long position (BUY/HOLD): profit when price increases
        pnl_pct = ((exit_price - entry_price) / entry_price) * 100.0
    return pnl_pct
```

**Validation:** All tests passing (BUY, SELL, HOLD scenarios verified)

### 2. Fixed Historical Data

**Action:** Recalculated P/L for all SELL positions in signal_history

**Results:**
- 2 SELL positions corrected
- GE: -5.91% -> +5.91%
- INTU: +6.62% -> -6.62%

**Corrected Performance Metrics:**
- Before: 2W / 4L, Total P/L: -13.67%
- After: 2W / 4L, Total P/L: -15.08% (accurate)

### 3. Refactored Source Files

**Modified Files:**

1. **backend/turbomode/core_engine/signal_closer.py**
   - Added import: `from backend.turbomode.core_engine.pnl_utils import calculate_pnl_pct`
   - Replaced line 79:
     - OLD: `pnl_pct = ((current_price - entry_price) / entry_price) * 100.0`
     - NEW: `pnl_pct = calculate_pnl_pct(entry_price, current_price, signal_type)`

2. **backend/turbomode/core_engine/position_manager.py**
   - Added import: `from backend.turbomode.core_engine.pnl_utils import calculate_pnl_pct`
   - Replaced long position calc (line ~231):
     - OLD: `pnl_pct = (exit_price / entry_price - 1) * 100`
     - NEW: `pnl_pct = calculate_pnl_pct(entry_price, exit_price, "BUY")`
   - Replaced short position calc (line ~234):
     - OLD: `pnl_pct = (1 - exit_price / entry_price) * 100`
     - NEW: `pnl_pct = calculate_pnl_pct(entry_price, exit_price, "SELL")`

## Files Created

1. **backend/turbomode/core_engine/pnl_utils.py** - Canonical P/L calculation engine
2. **refactor_pnl_calculations.py** - Database fix script
3. **apply_pnl_refactoring.py** - Source code refactoring script
4. **session_files/pnl_bug_fix_summary_2026-01-25.md** - This document

## Verification

All refactoring complete:
- Canonical P/L function created and tested
- Historical data corrected in signal_history
- Source code refactored to use pnl_utils
- Future SELL positions will calculate correctly

## Impact on Existing Systems

**Affected:**
- signal_closer.py (closes active signals, calculates realized P/L)
- position_manager.py (paper trading position tracking)

**NOT Affected:**
- adaptive_stock_ranker.py (reads P/L from signal_history - will now use correct values)
- Scanner/prediction logic (unchanged)
- Model training (unchanged)
- Database schema (unchanged)

## Next Steps

1. Run scanner to generate new signals - P/L will now calculate correctly
2. Monitor next SELL signal that closes to verify fix is working in production
3. Delete temporary scripts after verification:
   - refactor_pnl_calculations.py
   - apply_pnl_refactoring.py

## Lessons Learned

**Prevention:** All P/L calculations must use `calculate_pnl_pct()` from pnl_utils.py
- Single source of truth eliminates bugs
- Handles BUY/SELL/HOLD correctly
- Deterministic and testable

**Testing:** Unit tests in pnl_utils.py verify correctness for all signal types
