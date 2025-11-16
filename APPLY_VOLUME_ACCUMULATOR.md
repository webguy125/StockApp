# Apply VolumeAccumulator to Remaining Timeframes

The 1m timeframe has been fully updated to use the shared VolumeAccumulator service.
The following timeframes still need to be updated:

## Timeframes to Update:
- [ ] 5m.js (frontend/js/timeframes/minutes/)
- [ ] 15m.js (frontend/js/timeframes/minutes/)
- [ ] 30m.js (frontend/js/timeframes/minutes/)
- [ ] 1h.js (frontend/js/timeframes/hours/)
- [ ] 2h.js (frontend/js/timeframes/hours/)
- [ ] 4h.js (frontend/js/timeframes/hours/)
- [ ] 6h.js (frontend/js/timeframes/hours/)

## Changes Required (Per File):

### 1. Add Import
At top of file, after CanvasRenderer import:
```javascript
import { volumeAccumulator } from '../../services/VolumeAccumulator.js';
```

### 2. Update Constructor
Replace:
```javascript
this.currentCandleVolume = 0;
this.currentCandleStartTime = null;
```
With:
```javascript
this.volumeCallback = null; // Callback for volume updates from shared accumulator
```

### 3. Update initialize() Method
Replace the subscription section with:
```javascript
try {
  // Start shared volume accumulator
  volumeAccumulator.start(symbol, socket);

  // Load historical data
  await this.loadHistoricalData();

  // Initialize volume accumulator with last candle time
  if (this.data.length > 0) {
    const lastCandle = this.data[this.data.length - 1];
    volumeAccumulator.initializeCandleTimes('[INTERVAL]', lastCandle.Date);  // Replace [INTERVAL] with: 5m, 15m, 30m, 1h, 2h, 4h, or 6h
    volumeAccumulator.initializeVolume('[INTERVAL]', lastCandle.Volume);
  }

  // Register callback to receive volume updates
  this.volumeCallback = (volume) => {
    if (this.isActive && this.data.length > 0) {
      this.renderer.updateCurrentCandleVolume(volume);
    }
  };
  volumeAccumulator.registerCallback('[INTERVAL]', this.volumeCallback);  // Replace [INTERVAL]

  // Subscribe to WebSocket updates for price
  this.subscribeToLiveData();

  return true;
}
```

### 4. Update subscribeToLiveData()
Replace:
```javascript
channels: ['ticker', 'matches']
```
With:
```javascript
channels: ['ticker']  // matches handled by VolumeAccumulator
```

### 5. Remove handleTradeUpdate() Method Entirely
Delete the entire method (usually ~50 lines)

### 6. Update deactivate() Method
Replace:
```javascript
this.isActive = false;
this.currentCandleVolume = 0;
this.currentCandleStartTime = null;
```
With:
```javascript
this.isActive = false;

// Unregister volume callback
if (this.volumeCallback) {
  volumeAccumulator.unregisterCallback('[INTERVAL]', this.volumeCallback);  // Replace [INTERVAL]
  this.volumeCallback = null;
}
```

### 7. Update destroy() Method
Remove these lines:
```javascript
this.currentCandleVolume = 0;
this.currentCandleStartTime = null;
```

## Interval Codes:
- 5m.js: Use '5m'
- 15m.js: Use '15m'
- 30m.js: Use '30m'
- 1h.js: Use '1h'
- 2h.js: Use '2h'
- 4h.js: Use '4h'
- 6h.js: Use '6h'

## Implementation Status:
- ✅ VolumeAccumulator.js created
- ✅ TimeframeRegistry.js updated to route trades to accumulator
- ✅ 1m.js fully updated and tested
- ⏳ Remaining 7 timeframes pending

## Next Steps:
Apply the same pattern to each remaining timeframe file, replacing [INTERVAL] with the appropriate code.
