# ORD Volume Analysis - Completely Segregated Implementation

## Overview

ORD Volume Analysis is a **completely segregated** trading analysis tool that performs volume analysis across wave patterns. This implementation has **ZERO shared code** with the existing StockApp features.

## Architecture

### Complete Segregation
- **Separate Directory**: `frontend/js/ord-volume/` (no shared directories)
- **Separate Backend**: `backend/ord-volume/` with own data storage
- **Separate Data**: `backend/data/ord-volume/` (isolated storage)
- **No Imports**: All functionality duplicated, no imports from existing code
- **Standalone**: Can be removed/modified without affecting any other features

### File Structure

```
frontend/js/ord-volume/
├── ORDVolumeAnalysis.js      # Core analysis engine
├── ORDVolumeController.js    # UI controller and modal management
├── ORDVolumeRenderer.js      # Canvas rendering for overlays
├── ord-volume-integration.js # Integration with main app
└── README.md                 # This file

backend/ord-volume/
└── ord_volume_server.py      # Standalone Flask server (optional)

backend/data/ord-volume/
└── ord_volume_{SYMBOL}.json  # Saved analyses per symbol
```

### Backend Integration

ORD Volume endpoints are added to the main api_server.py in a completely segregated section (lines 863-936):

- `POST /ord-volume/save` - Save analysis
- `GET /ord-volume/load/<symbol>` - Load analysis
- `DELETE /ord-volume/delete/<symbol>` - Delete analysis
- `GET /ord-volume/list` - List all saved analyses

All functions are prefixed with `_segregated` to prevent naming conflicts.

## Features

### Two Analysis Modes

#### 1. Draw Mode
- User manually draws 3-7 trendlines on chart
- System calculates average volume for each wave
- Compares Retest (3rd wave) volume to Initial (1st wave)

#### 2. Auto Mode
- Automatically detects swing points using 2-period fractals
- Creates zigzag pattern from price structure
- Generates 3-7 trendlines (configurable)
- Performs same volume analysis

### Volume Classification

- **Strong**: Retest volume ≥ 110% of Initial volume (green)
- **Neutral**: Retest volume 92-109% of Initial volume (yellow)
- **Weak**: Retest volume < 92% of Initial volume (red)

### Chart Overlays

- **Trendlines**: Blue lines with wave labels (Initial, Correction, Retest, Wave 4-7)
- **Volume Labels**: Draggable text showing average volume for each wave
- **Strength Indicator**: Color-coded strength assessment
- **Coordinates**: All overlays anchored to chart coordinates (persist through pan/zoom)

## Usage

### From UI

1. **Load Chart**: Load any symbol with data
2. **Click ORD Volume Button**: Top toolbar, next to "Indicators"
3. **Select Mode**:
   - **Draw**: Manually draw trendlines on chart
   - **Auto**: Specify number of lines (3-7), click Analyze
4. **View Results**: Trendlines and volume analysis rendered on chart
5. **Drag Labels**: All text labels are draggable for optimal positioning

### Programmatic Usage

```javascript
import { ORDVolumeAnalysis } from './ORDVolumeAnalysis.js';

// Prepare candle data (OHLCV format)
const candles = [
  { open: 100, high: 105, low: 98, close: 103, volume: 1000000 },
  { open: 103, high: 108, low: 102, close: 106, volume: 1200000 },
  // ... more candles
];

// Create analyzer
const analyzer = new ORDVolumeAnalysis(candles);

// Draw Mode
const userLines = [
  [0, 100, 10, 110],   // Initial: index 0 @ price 100 to index 10 @ price 110
  [10, 110, 20, 105],  // Correction: index 10 @ price 110 to index 20 @ price 105
  [20, 105, 30, 115]   // Retest: index 20 @ price 105 to index 30 @ price 115
];
const result = analyzer.analyzeDrawMode(userLines);

// Auto Mode
const result2 = analyzer.analyzeAutoMode(5); // 5 trendlines

// Access results
console.log(result.strength);     // "Strong", "Neutral", or "Weak"
console.log(result.color);        // "green", "yellow", or "red"
console.log(result.trendlines);   // Array of line objects
console.log(result.labels);       // Array of text overlay objects
console.log(result.waveData);     // Detailed wave metrics
```

## Data Persistence

### Save Analysis
```javascript
POST /ord-volume/save
Body: {
  "symbol": "BTC-USD",
  "analysis": { /* result object */ }
}
```

### Load Analysis
```javascript
GET /ord-volume/load/BTC-USD
Response: {
  "symbol": "BTC-USD",
  "analysis": { /* result object */ }
}
```

### Delete Analysis
```javascript
DELETE /ord-volume/delete/BTC-USD
```

### List All
```javascript
GET /ord-volume/list
Response: {
  "symbols": ["BTC-USD", "ETH-USD"],
  "count": 2
}
```

## Technical Details

### Fractal Detection (Auto Mode)

Uses 2-period fractal detection:
- **Swing High**: Price high is highest among 2 periods before and after
- **Swing Low**: Price low is lowest among 2 periods before and after
- **Zigzag**: Filters to alternating highs and lows

### Volume Calculation

For each trendline [x1, y1, x2, y2]:
1. Get index range: `Math.ceil(Math.min(x1, x2))` to `Math.floor(Math.max(x1, x2))`
2. Sum volume of all candles in range
3. Calculate average: `totalVolume / candleCount`

### Coordinate System

- **X coordinates**: Candle indices (0, 1, 2, ...)
- **Y coordinates**: Price values
- **Conversion**: Uses chart state's `xToIndex()`, `yToPrice()`, etc. for rendering

### Error Handling

- Validates minimum 3 trendlines
- Validates maximum 7 trendlines
- Validates candle data structure (OHLCV)
- Validates line coordinates within bounds
- Throws descriptive errors for invalid inputs

## Integration Points

### Button Location
- **HTML**: `index_tos_style.html` line 291
- **ID**: `btn-ord-volume`
- **Text**: "ORD Volume"

### Script Import
- **HTML**: `index_tos_style.html` line 464
- **Path**: `js/ord-volume/ord-volume-integration.js`
- **Type**: ES6 module

### Auto-Initialization
- Integration script auto-initializes on DOM load
- Wires up button click handler
- Extracts candle data from existing chart

## Removal Instructions

To completely remove ORD Volume feature:

1. **Delete Directories**:
   ```bash
   rm -rf frontend/js/ord-volume/
   rm -rf backend/ord-volume/
   rm -rf backend/data/ord-volume/
   ```

2. **Remove Button** (index_tos_style.html line 291):
   ```html
   <button class="tos-toolbar-btn" id="btn-ord-volume" title="ORD Volume">ORD Volume</button>
   ```

3. **Remove Script Import** (index_tos_style.html line 464):
   ```html
   <script type="module" src="js/ord-volume/ord-volume-integration.js"></script>
   ```

4. **Remove Backend Endpoints** (api_server.py lines 863-936):
   Remove the entire "ORD VOLUME ENDPOINTS" section

No other files need modification - complete segregation ensures zero dependencies.

## Development Notes

### Why Segregated?

This implementation was specifically requested to be **completely isolated** from all existing code to:
- Prevent any conflicts with existing features
- Allow independent modification without risk
- Enable easy removal if needed
- Avoid shared code dependencies
- Maintain clear separation of concerns

### Code Duplication

Yes, some code is duplicated (especially coordinate conversion logic). This is **intentional** per the segregation requirement. Do NOT refactor to share code with existing features.

### Testing

Test both modes:
1. **Draw Mode**: Draw 3+ lines on chart, verify volume calculations
2. **Auto Mode**: Test with various line counts (3-7), verify fractal detection
3. **Persistence**: Save, reload page, verify analysis restored
4. **Interaction**: Test label dragging, verify positions saved

## Future Enhancements

Potential improvements (maintain segregation):
- Support for more than 7 waves
- Custom volume ratio thresholds
- Export analysis to CSV
- Pattern detection across multiple symbols
- Alert system for strong retest signals

## License

Part of StockApp - Professional Trading Platform
