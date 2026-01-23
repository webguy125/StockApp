# Predictions Webpage Integration - 14-Day Swing System
**Date**: 2026-01-22
**Session**: TurboMode Predictions API Integration
**Status**: Partially Complete - Predictions Loading, Sorting/Filtering Issues Remain

---

## Executive Summary

Successfully integrated the 14-day swing trading predictions from the scanner into the webpage. The predictions API now reads directly from the database where the scanner saves signals, eliminating the need for manual JSON file generation. However, sorting and filtering issues remain to be addressed.

### Completed:
1. ✅ Fixed predictions API import errors (lazy loading)
2. ✅ Connected API to scanner database (turbomode.db)
3. ✅ Updated webpage title to "14-Day Swing Predictions"
4. ✅ Flask server restarted with working configuration
5. ✅ Predictions successfully loading on webpage

### Remaining Issues:
1. ⚠️ Sorting not working correctly
2. ⚠️ Filtering (ALL/BUY/SELL) requires page refresh
3. ⚠️ Need to implement client-side filtering/sorting

---

## Problem Statement

After completing the 14-day swing system training (66 models, 2026-01-21), the predictions needed to be displayed on the TurboMode webpage. The initial integration attempts failed due to:
1. Wrong approach (helper script generating JSON files manually)
2. Disconnected data sources (API reading JSON, scanner writing to database)
3. Import errors preventing API blueprint from loading

---

## Solution Architecture

### Data Flow (Final):
```
Scanner (overnight_scanner.py)
    ↓
Saves BUY/SELL signals to database
    ↓
turbomode.db (SQLite)
    ↓
predictions_api.py reads via TurboModeDB.get_active_signals()
    ↓
Flask API endpoint: /turbomode/predictions/all
    ↓
Webpage: all_predictions.html
```

### Key Files Modified:

#### 1. `backend/turbomode/predictions_api.py`
**Location**: `C:\StockApp\backend\turbomode\predictions_api.py`

**Changes Made**:
1. Removed top-level imports that were causing failures
2. Implemented lazy imports for `/all_live` and `/symbol/<symbol>` endpoints
3. Rewrote `/all` endpoint to read from database instead of JSON file

**Before (Broken)**:
```python
from backend.turbomode.overnight_scanner import OvernightScanner
from backend.turbomode.core_symbols import get_all_core_symbols, get_symbol_metadata

@predictions_bp.route('/all', methods=['GET'])
def get_all_predictions():
    # Read from JSON file
    predictions_file = os.path.join(os.path.dirname(__file__), 'data', 'all_predictions.json')
    with open(predictions_file, 'r') as f:
        data = json.load(f)
    return jsonify(data)
```

**After (Working)**:
```python
# Scanner imports only needed for /all_live endpoint (lazy loaded)
scanner = None

def init_scanner():
    """Initialize scanner with models loaded (lazy imports)"""
    global scanner
    if scanner is None:
        # Lazy import to avoid import errors at module load time
        from backend.turbomode.core_engine.overnight_scanner import OvernightScanner
        scanner = OvernightScanner()
        scanner._load_models()

@predictions_bp.route('/all', methods=['GET'])
def get_all_predictions():
    """
    Get predictions from database (scanner saves signals to turbomode.db)

    This reads directly from the database that the scanner writes to,
    so it automatically shows the latest scanner results!
    """
    from backend.turbomode.database_schema import TurboModeDB

    # Connect to database
    db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'turbomode.db')
    db = TurboModeDB(db_path=db_path)

    # Get all active signals from database
    signals = db.get_active_signals(limit=500)

    # Convert to predictions format
    predictions = []
    buy_count = 0
    sell_count = 0

    for signal in signals:
        pred_type = signal['signal_type'].upper()  # BUY or SELL

        prediction_entry = {
            'symbol': signal['symbol'],
            'sector': signal.get('sector', 'unknown'),
            'prediction': pred_type,
            'confidence': round(signal['confidence'], 4),
            'prob_buy': round(signal['confidence'], 4) if pred_type == 'BUY' else 0.0,
            'prob_sell': round(signal['confidence'], 4) if pred_type == 'SELL' else 0.0,
            'prob_hold': 0.0,
            'current_price': round(signal['entry_price'], 2)
        }

        predictions.append(prediction_entry)

        if pred_type == 'BUY':
            buy_count += 1
        elif pred_type == 'SELL':
            sell_count += 1

    return jsonify({
        'timestamp': datetime.now().isoformat(),
        'total': len(predictions),
        'statistics': {
            'buy_count': buy_count,
            'sell_count': sell_count,
            'hold_count': 0
        },
        'predictions': predictions,
        'source': 'database',
        'success': True
    })
```

**Why This Works**:
- No import errors at module load time (lazy imports only when endpoints called)
- Reads directly from scanner database (automatic updates)
- No manual JSON file generation required

#### 2. `frontend/turbomode/all_predictions.html`
**Location**: `C:\StockApp\frontend\turbomode\all_predictions.html`

**Changes Made**:
Updated title and subtitle to reflect 14-day swing trade semantics.

**Before**:
```html
<h1>🔮 Full Model List - All Predictions</h1>
<div class="subtitle">235 stocks showing BUY/SELL/HOLD predictions</div>
```

**After**:
```html
<h1>🔮 Full Model List - 14-Day Swing Predictions</h1>
<div class="subtitle">235 stocks showing 14-Day BUY/SELL/HOLD swing trade predictions</div>
```

#### 3. `frontend/turbomode/generate_predictions_for_web.py`
**Status**: DELETED

**Why Deleted**:
- Was a helper script approach that required manual execution
- Generated independent predictions not connected to scanner
- User explicitly rejected: "delete the helper script and make the webpage work with the scanner results"

---

## Technical Details

### Database Schema Used:
**Table**: `signals` (in turbomode.db)

**Key Columns**:
- `symbol`: Stock ticker
- `signal_type`: "BUY" or "SELL"
- `confidence`: Model confidence (0.0 - 1.0)
- `entry_price`: Current stock price
- `sector`: Sector classification
- `timestamp`: When signal was generated

**Query Method**:
```python
db = TurboModeDB(db_path='backend/data/turbomode.db')
signals = db.get_active_signals(limit=500)
```

### Import Error Resolution:

**Original Problem**:
```
TurboMode not available: No module named 'backend.turbomode.overnight_scanner'
[PREDICTIONS API] Not available: No module named 'backend.turbomode.overnight_scanner'
```

**Root Cause**:
The predictions_api.py had top-level imports:
```python
from backend.turbomode.overnight_scanner import OvernightScanner
from backend.turbomode.core_symbols import get_all_core_symbols, get_symbol_metadata
```

The `overnight_scanner` module path was incorrect (`backend.turbomode.overnight_scanner` vs correct `backend.turbomode.core_engine.overnight_scanner`).

**Solution**:
Lazy imports - only import when endpoint is actually called:
```python
def init_scanner():
    global scanner
    if scanner is None:
        from backend.turbomode.core_engine.overnight_scanner import OvernightScanner
        scanner = OvernightScanner()

@predictions_bp.route('/all_live', methods=['GET'])
def get_all_predictions_live():
    # Lazy import
    from backend.turbomode.core_symbols import get_all_core_symbols, get_symbol_metadata
    init_scanner()
    # ... rest of endpoint
```

### Flask Server Restart:

**Commands Used**:
```bash
# Find Flask PID
netstat -ano | findstr :5000 | findstr LISTENING

# Kill old Flask server
taskkill //F //PID 17312

# Restart Flask
cd /c/StockApp/backend && python api_server.py
```

**New PID**: 36228 (running on port 5000)

---

## API Endpoints

### `/turbomode/predictions/all` (PRIMARY - FAST)
**Method**: GET
**Description**: Reads predictions from scanner database (turbomode.db)
**Response Time**: <100ms
**Use Case**: Default endpoint for webpage

**Response Format**:
```json
{
  "timestamp": "2026-01-22T21:24:31.123456",
  "total": 167,
  "statistics": {
    "buy_count": 85,
    "sell_count": 82,
    "hold_count": 0
  },
  "predictions": [
    {
      "symbol": "AAPL",
      "sector": "technology",
      "prediction": "BUY",
      "confidence": 0.6234,
      "prob_buy": 0.6234,
      "prob_sell": 0.0,
      "prob_hold": 0.0,
      "current_price": 185.23
    }
  ],
  "source": "database",
  "success": true
}
```

### `/turbomode/predictions/all_live` (SLOW - 2-3 minutes)
**Method**: GET
**Description**: Generates predictions in real-time using loaded models
**Response Time**: 2-3 minutes
**Use Case**: Testing or when live data needed

### `/turbomode/predictions/symbol/<symbol>`
**Method**: GET
**Description**: Get prediction for a single symbol
**Response Time**: ~1 second
**Use Case**: Individual stock queries

---

## Known Issues (To Fix Tomorrow)

### 1. Sorting Not Working Correctly
**Symptom**: Predictions not sorting properly when clicking column headers
**Likely Cause**: Client-side JavaScript sorting logic
**Location**: `frontend/turbomode/all_predictions.html` (JavaScript section)
**Fix Required**: Debug and update sorting function

### 2. Filter Buttons Require Page Refresh
**Symptom**: Clicking ALL/BUY/SELL filter buttons doesn't filter immediately
**Expected Behavior**: Client-side filtering without page reload
**Likely Cause**: JavaScript event handlers not properly updating DOM
**Location**: `frontend/turbomode/all_predictions.html` (filter button handlers)
**Fix Required**: Implement proper client-side filtering

### 3. Possible Enhancements:
- Add loading spinner during API fetch
- Add error handling for failed API calls
- Add timestamp display showing when data was last updated
- Add auto-refresh option
- Add export to CSV functionality

---

## Testing Checklist

✅ **Completed**:
- [x] Flask server starts without errors
- [x] Predictions API blueprint loads successfully
- [x] `/turbomode/predictions/all` endpoint returns valid JSON
- [x] Webpage loads predictions from API
- [x] Predictions display on webpage
- [x] BUY/SELL signals show from database
- [x] Confidence scores display correctly
- [x] Sector information displays

⚠️ **Issues Found**:
- [ ] Sorting functionality not working
- [ ] Filter buttons (ALL/BUY/SELL) require page refresh
- [ ] Need to test with fresh scanner run

---

## Scanner Integration

### How Scanner Saves Signals:
**File**: `backend/turbomode/core_engine/overnight_scanner.py`

**Code**:
```python
from backend.turbomode.database_schema import TurboModeDB

# In scan() method:
for signal in buy_signals:
    if self.db.add_signal(signal):
        saved_buy += 1

for signal in sell_signals:
    if self.db.add_signal(signal):
        saved_sell += 1
```

### Signal Format:
```python
signal = {
    'symbol': 'AAPL',
    'signal_type': 'BUY',  # or 'SELL'
    'confidence': 0.6234,
    'entry_price': 185.23,
    'target_price': 195.00,
    'stop_loss': 180.00,
    'sector': 'technology',
    'timestamp': datetime.now()
}
```

---

## Next Session Tasks (2026-01-23)

### High Priority:
1. **Fix Sorting**: Debug column header click handlers
2. **Fix Filtering**: Implement client-side ALL/BUY/SELL filtering without page refresh
3. **Test End-to-End**: Run scanner, verify webpage updates automatically

### Medium Priority:
4. Add loading indicators
5. Add error handling for API failures
6. Add last-updated timestamp display

### Low Priority:
7. Export to CSV functionality
8. Auto-refresh toggle
9. Advanced filtering (by sector, confidence threshold)

---

## File Locations Reference

### Modified Files:
```
C:\StockApp\backend\turbomode\predictions_api.py          (MODIFIED - fixed imports, database integration)
C:\StockApp\frontend\turbomode\all_predictions.html       (MODIFIED - updated title)
```

### Deleted Files:
```
C:\StockApp\frontend\turbomode\generate_predictions_for_web.py  (DELETED - wrong approach)
```

### Key Dependencies:
```
C:\StockApp\backend\turbomode\database_schema.py          (TurboModeDB class)
C:\StockApp\backend\turbomode\core_engine\overnight_scanner.py  (Scanner)
C:\StockApp\backend\data\turbomode.db                     (Database)
```

---

## User Constraints Honored

**Critical Constraint**: "you are not allowed to modify any of the backend training, scanning, or database logic"

**Files NOT Modified** (as required):
- ✅ `backend/turbomode/core_engine/overnight_scanner.py` (READ ONLY)
- ✅ `backend/turbomode/database_schema.py` (READ ONLY)
- ✅ `backend/turbomode/core_engine/sector_batch_trainer.py` (READ ONLY)
- ✅ `backend/turbomode/core_engine/fastmode_inference.py` (READ ONLY)

**Files Modified** (allowed):
- ✅ `backend/turbomode/predictions_api.py` (API/presentation layer)
- ✅ `frontend/turbomode/all_predictions.html` (Frontend)

---

## Success Metrics

### Achieved:
- ✅ Predictions load from database automatically
- ✅ No manual JSON file generation required
- ✅ Scanner → Database → API → Webpage integration complete
- ✅ 14-day swing semantics properly labeled
- ✅ BUY/SELL signals display with confidence scores

### Remaining:
- ⚠️ Sorting functionality
- ⚠️ Client-side filtering
- ⚠️ User experience polish

---

## Commands for Reference

### Restart Flask Server:
```bash
# Find PID
netstat -ano | findstr :5000 | findstr LISTENING

# Kill and restart
taskkill //F //PID <PID>
cd /c/StockApp/backend && python api_server.py &
```

### Test API Endpoint:
```bash
curl http://localhost:5000/turbomode/predictions/all
```

### Check Database:
```python
from backend.turbomode.database_schema import TurboModeDB
db = TurboModeDB(db_path='backend/data/turbomode.db')
signals = db.get_active_signals(limit=10)
print(f"Found {len(signals)} signals")
```

---

**End of Session Notes - 2026-01-22**
