# Signal Lifecycle Implementation Status
## Date: 2026-01-23

## COMPLETED TASKS ✅

### 1. Database Schema Migration ✅
**Status**: COMPLETE
**File**: `backend/turbomode/migrate_signal_schema.py`
**Results**: 107 signals migrated successfully

**Changes Made**:
- Added `current_price` field (updated each scan)
- Added `signal_timestamp` field (fixed at creation)
- Added `entry_min`, `entry_max` fields
- Changed `UNIQUE(symbol, signal_type)` to `UNIQUE(symbol)` to allow flipping

**Verification**:
```bash
python backend/turbomode/migrate_signal_schema.py
# Output: [MIGRATION] Verified: 107 signals in new table
```

### 2. Database Methods Updated ✅
**Status**: COMPLETE
**File**: `backend/turbomode/database_schema.py`

**New Methods**:
- `add_or_update_signal(signal, current_price)` - Implements CREATE/UPDATE/FLIP logic
- `update_current_price(symbol, current_price)` - Updates current price during scans

**Updated Methods**:
- `_init_schema()` - Updated table definition with new fields
- `update_signal_age()` - Now calculates age from `signal_timestamp` instead of `entry_date`
- `add_signal()` - Deprecated, now calls `add_or_update_signal()`

**Signal Lifecycle Rules**:
1. **CREATE**: No existing signal → create new with current_price
2. **UPDATE**: Same signal type (BUY→BUY) → update current_price and confidence only
3. **FLIP**: Different type (BUY→SELL) → reset entry_price, signal_timestamp, age_days to 0

### 3. API Updated ✅
**Status**: COMPLETE (requires Flask restart)
**File**: `backend/turbomode/predictions_api.py`

**Changes Made**:
- Fixed critical bug: Was using `entry_price` for `current_price` (line 111)
- Now returns both fields separately
- Added new fields: `signal_timestamp`, `age_days`, `days_remaining`

**New API Response Format**:
```json
{
  "symbol": "CRM",
  "prediction": "BUY",
  "confidence": 0.9877,
  "entry_price": 246.24,
  "current_price": 229.00,
  "signal_timestamp": "2026-01-13T11:26:47",
  "age_days": 10,
  "days_remaining": 4
}
```

---

## PENDING TASKS ⚠️

### 4. Restart Flask Server ⚠️
**Status**: REQUIRED
**Action**: Kill and restart `backend/api_server.py`

The API changes won't take effect until Flask is restarted.

**Steps**:
1. Find Flask process: `netstat -ano | findstr :5000`
2. Kill process: `taskkill /PID <pid> /F`
3. Restart: `python backend/api_server.py`

### 5. Update Scanner Logic ⚠️
**Status**: NOT STARTED
**Files to Update**:
- `backend/turbomode/core_engine/overnight_scanner.py`
- `backend/turbomode/scanner_files/top10_scanner.py`

**Required Changes**:
```python
# OLD CODE (currently in scanner):
if self.db.add_signal(signal):
    print(f"Added signal for {symbol}")

# NEW CODE (required):
current_price = <fetch from yfinance or use latest OHLCV>
result = self.db.add_or_update_signal(signal, current_price)

if result == 'CREATED':
    print(f"[NEW] {symbol} - {signal_type} signal created")
elif result == 'UPDATED':
    print(f"[UPDATE] {symbol} - confidence updated")
elif result == 'FLIPPED':
    print(f"[FLIP] {symbol} - signal flipped to {signal_type}")
```

**Critical**:
- Scanner MUST pass `current_price` (live market price) to `add_or_update_signal()`
- Scanner MUST call `db.update_signal_age()` before generating new signals
- Scanner should log FLIP events prominently

### 6. Update Frontend ⚠️
**Status**: NOT STARTED
**Files to Update**:
- `frontend/turbomode/top_10_stocks.html`
- `frontend/turbomode/all_predictions.html`

**Required Changes**:

**Current Display** (WRONG):
```javascript
<div>Price: $${stock.entry_price.toFixed(2)}</div>
```

**New Display** (CORRECT):
```javascript
<div>Entry Price: $${stock.entry_price.toFixed(2)}</div>
<div>Current Price: $${stock.current_price.toFixed(2)}</div>
<div>Signal Age: ${stock.age_days} days (${stock.days_remaining} days remaining)</div>
```

**Example Display**:
```
CRM - BUY Signal
Entry Price: $246.24 (Jan 13, 2026)
Current Price: $229.00
Change: -$17.24 (-7.0%)
Signal Age: 10 days old
Days Remaining: 4 days
Confidence: 98.8%
```

### 7. Testing ⚠️
**Status**: NOT STARTED

**Test Cases**:
1. **Restart Flask** → Verify API returns new fields
2. **Run Scanner** → Verify signals flip correctly (BUY→SELL)
3. **Frontend** → Verify both prices display correctly
4. **Age Calculation** → Verify `age_days` increments daily
5. **Expiration** → Verify signals expire after 14 days

---

## ACCEPTANCE CRITERIA

### CRM Example (Expected Behavior):
```
Database:
- entry_price: $246.24
- current_price: $229.00  (if scanner ran today)
- signal_type: BUY or SELL (depends on model output today)
- signal_timestamp: 2026-01-13T11:26:47
- age_days: 10
- days_remaining: 4

API Response:
- Should show BOTH entry_price and current_price
- Should show signal age and days remaining

Frontend Display:
- Show "Entry: $246.24" and "Current: $229.00"
- Show "-$17.24 (-7%)" change
- Show "10 days old, 4 days remaining"
```

---

## IMPLEMENTATION NOTES

### Database Fields (active_signals table):
- `entry_price`: FIXED - Set once at signal creation
- `current_price`: DYNAMIC - Updated each scan
- `signal_timestamp`: FIXED - When signal was created
- `entry_date`: FIXED - Date of entry (YYYY-MM-DD)
- `age_days`: DYNAMIC - Calculated from signal_timestamp
- `updated_at`: DYNAMIC - Last time record was touched

### Signal Flipping Logic:
When scanner runs:
1. Generate new prediction for symbol
2. Check if signal exists in database
3. If signal exists:
   - Same type (BUY→BUY): UPDATE current_price, confidence
   - Different type (BUY→SELL): FLIP - reset entry_price, signal_timestamp
4. If no signal: CREATE new signal

### Git Commits:
- Commit 759f115: Database schema and API fixes
- All changes pushed to origin/main

---

## NEXT STEPS

**IMMEDIATE (User)**:
1. Restart Flask server to apply API changes
2. Verify API returns new fields with curl test

**NEXT (Development)**:
3. Update scanner to use `add_or_update_signal()`
4. Update frontend to display both prices
5. Run end-to-end test with live scanner
6. Verify CRM shows correct behavior

**File**: C:\StockApp\SIGNAL_LIFECYCLE_IMPLEMENTATION_STATUS.md
**Last Updated**: 2026-01-23 12:00
