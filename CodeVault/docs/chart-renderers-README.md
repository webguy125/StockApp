# Chart Renderers System

## ⚠️ IMPORTANT - DO NOT MODIFY THESE FILES DIRECTLY

These files contain the isolated, stable chart rendering logic. Each timeframe has its own renderer to prevent breaking changes.

## Structure

```
chart-renderers/
├── base-renderer.js          # Base class - core rendering logic
├── volume-handler.js         # Volume display (overlay/subgraph)
├── intraday-renderer.js      # Handles: 1m, 5m, 15m, 30m, 1h
├── daily-renderer.js         # Handles: 1d, 1wk, 1mo
├── renderer-factory.js       # Auto-selects correct renderer
└── README.md                 # This file
```

## How It Works

### 1. Renderer Selection (Automatic)
```javascript
import { RendererFactory } from './chart-renderers/renderer-factory.js';

// Factory automatically picks the right renderer
await RendererFactory.renderChart(data, 'AAPL', '1m', options);
```

### 2. Supported Timeframes

**IntradayRenderer:**
- 1m (1 minute)
- 5m (5 minutes)
- 15m (15 minutes)
- 30m (30 minutes)
- 1h (1 hour)

**DailyRenderer:**
- 1d (1 day)
- 1wk (1 week)
- 1mo (1 month)

### 3. Rendering Options

```javascript
const options = {
  volumeMode: 'subgraph',  // 'overlay' or 'subgraph'
  showVolume: true         // true or false
};

await RendererFactory.renderChart(data, symbol, interval, options);
```

### 4. Live Updates

```javascript
// Update existing chart with live price
await RendererFactory.updateChart(interval, {
  price: 123.45,
  timestamp: '2025-10-22T12:30:00Z'
});
```

## File Responsibilities

### base-renderer.js
- Core Plotly rendering
- Basic candlestick trace creation
- Chart initialization/cleanup
- Common update logic

### volume-handler.js
- Creates volume traces (cyan bars)
- Manages overlay vs subgraph modes
- Volume axis configuration
- Volume updates

### intraday-renderer.js
- Intraday-specific layout (hide weekends/after-hours)
- Real-time candle updates for short timeframes
- Add new candles when interval changes
- Optimized for frequent updates

### daily-renderer.js
- Daily/weekly/monthly layout
- Date formatting for longer periods
- Weekend hiding (daily only)
- Less frequent update logic

### renderer-factory.js
- Automatic renderer selection
- Unified API for all timeframes
- Prevents wrong renderer usage

## Extending the System

### To Add a New Timeframe:

1. **Create new renderer** (if needed):
   ```javascript
   // Example: ultra-renderer.js for seconds/ticks
   export class UltraRenderer extends BaseChartRenderer {
     supports(interval) {
       return ['1s', '5s', '10s', '30s'].includes(interval);
     }
   }
   ```

2. **Register in factory**:
   ```javascript
   // In renderer-factory.js
   import { UltraRenderer } from './ultra-renderer.js';

   static renderers = [
     new UltraRenderer(),  // Add new renderer
     new IntradayRenderer(),
     new DailyRenderer()
   ];
   ```

3. **Done!** Factory auto-selects it

## Rules

1. ✅ **DO** extend BaseChartRenderer for new renderers
2. ✅ **DO** use VolumeHandler for all volume display
3. ✅ **DO** add new renderers through RendererFactory
4. ❌ **DON'T** modify existing renderer files directly
5. ❌ **DON'T** bypass the factory (use it for all rendering)
6. ❌ **DON'T** mix rendering logic with app logic

## Benefits

- **Isolation**: Each timeframe is independent
- **Stability**: Changes to one don't break others
- **Testability**: Test each renderer separately
- **Maintainability**: Clear separation of concerns
- **Extensibility**: Easy to add new timeframes

## Testing

Each renderer can be tested independently:

```javascript
import { IntradayRenderer } from './chart-renderers/intraday-renderer.js';

const renderer = new IntradayRenderer();
await renderer.render(mockData, 'AAPL', '5m', { showVolume: true });
```

## Support

If charts break:
1. Check which renderer handles that timeframe
2. Check renderer-factory.js for selection logic
3. Only modify the specific renderer file
4. Don't touch other renderers or base classes
