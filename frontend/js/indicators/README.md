# Modular Indicator System

## Overview

This is a complete rebuild of the indicator system from scratch with full modularity, customization, and extensibility. Each indicator is self-contained with its own calculation logic, settings, alerts, and rendering.

## Architecture

### Core Components

1. **IndicatorBase.js** - Abstract base class that all indicators inherit from
   - Settings management (get, set, export, import, reset)
   - Caching system for performance
   - Alert system with customizable conditions
   - Lifecycle management (enable, disable, toggle)

2. **IndicatorRegistry.js** - Central registry for all indicators
   - Register/unregister indicators
   - Enable/disable indicators
   - Calculate all enabled indicators
   - Event system for UI integration
   - Settings import/export for all indicators

### Implemented Indicators

#### 1. RSI (Relative Strength Index)
- **Location**: `indicators/RSI/RSI.js`
- **Type**: Oscillator (0-100 range)
- **Custom Logic**: Wilder's smoothing method
- **Settings**:
  - Lookback period (default: 14)
  - Overbought level (default: 70)
  - Oversold level (default: 30)
  - Line color, opacity, style
  - Level lines (show/hide, color, opacity)
- **Alerts**:
  - Overbought condition (>70)
  - Oversold condition (<30)
- **Rendering**: Line plot with overbought/oversold levels

#### 2. MACD (Moving Average Convergence Divergence)
- **Location**: `indicators/MACD/MACD.js`
- **Type**: Histogram with dual lines
- **Custom Logic**: EMA-based with signal line
- **Settings**:
  - Fast EMA period (default: 12)
  - Slow EMA period (default: 26)
  - Signal period (default: 9)
  - Line colors (MACD, Signal)
  - Histogram colors (positive/negative)
  - Opacity controls
- **Alerts**:
  - Bullish crossover (MACD crosses above Signal)
  - Bearish crossover (MACD crosses below Signal)
- **Rendering**: Histogram bars + dual line plot

## Directory Structure

```
indicators/
├── IndicatorBase.js          # Base class for all indicators
├── IndicatorRegistry.js       # Central registry
├── RSI/
│   └── RSI.js                # RSI indicator implementation
├── MACD/
│   └── MACD.js               # MACD indicator implementation
├── BollingerBands/
│   └── BollingerBands.js     # (To be implemented)
└── [More indicators...]
```

## Usage

### Registering Indicators

```javascript
import { indicatorRegistry } from './IndicatorRegistry.js';
import { RSI } from './RSI/RSI.js';
import { MACD } from './MACD/MACD.js';

// Create and register indicators
const rsi = new RSI();
const macd = new MACD();

indicatorRegistry.register(rsi);
indicatorRegistry.register(macd);
```

### Enabling/Disabling Indicators

```javascript
// Enable an indicator
indicatorRegistry.enable('RSI');

// Disable an indicator
indicatorRegistry.disable('MACD');

// Toggle an indicator
indicatorRegistry.toggle('RSI');
```

### Calculating Indicators

```javascript
// Calculate all enabled indicators for given candles
const candles = [/* OHLCV data */];
const results = indicatorRegistry.calculateAll(candles);

// Results is a Map: indicator name -> calculated data
const rsiData = results.get('RSI');
const macdData = results.get('MACD');
```

### Updating Settings

```javascript
// Update RSI settings
indicatorRegistry.updateSettings('RSI', {
  lookback_period: 20,
  overbought: 80,
  oversold: 20
});

// Update MACD settings
indicatorRegistry.updateSettings('MACD', {
  fast_period: 10,
  slow_period: 20,
  signal_period: 7
});
```

### Settings Import/Export

```javascript
// Export all indicator settings
const settings = indicatorRegistry.exportAllSettings();
// Save to localStorage or file
localStorage.setItem('indicator_settings', JSON.stringify(settings));

// Import settings
const savedSettings = JSON.parse(localStorage.getItem('indicator_settings'));
indicatorRegistry.importAllSettings(savedSettings);
```

### Alert System

```javascript
// Listen for alerts
window.addEventListener('indicator-alert', (event) => {
  const { indicator, message, data, timestamp } = event.detail;
  console.log(`Alert from ${indicator}: ${message}`);
  // Show notification to user
});
```

## Creating a New Indicator

### Step 1: Create Indicator Class

```javascript
import { IndicatorBase } from '../IndicatorBase.js';

export class MyIndicator extends IndicatorBase {
  constructor() {
    super({
      name: 'MyIndicator',
      version: '1.0.0',
      description: 'My custom indicator',
      tags: ['momentum'],
      output_type: 'line',
      default_settings: {
        period: 14,
        line_color: '#0000FF'
      },
      alerts: {
        enabled: true,
        conditions: [
          { type: 'greater_than', field: 'value', threshold: 80, message: 'Threshold exceeded' }
        ]
      },
      help_text: 'Explanation of what this indicator does'
    });
  }

  calculate(candles) {
    // Implement your calculation logic here
    // Return array of {date, value, ...} objects
    return [];
  }

  getSettingsSchema() {
    // Return schema for settings UI
    return {
      period: {
        type: 'number',
        label: 'Period',
        min: 1,
        max: 200,
        default: 14
      }
    };
  }

  render(ctx, bounds, data, visibleIndices) {
    // Implement rendering logic
  }
}
```

### Step 2: Register the Indicator

```javascript
import { MyIndicator } from './MyIndicator/MyIndicator.js';
import { indicatorRegistry } from '../IndicatorRegistry.js';

const myIndicator = new MyIndicator();
indicatorRegistry.register(myIndicator);
```

## Features

### ✅ Implemented
- Core infrastructure (Base class, Registry)
- RSI indicator with full customization
- MACD indicator with full customization
- Settings management (get, set, export, import)
- Alert system with custom conditions
- Caching for performance
- Event system for UI integration

### 🚧 In Progress
- Bollinger Bands indicator
- Settings modal UI
- Integration with main chart renderer
- Wiring up "Indicators" button

### 📋 Planned
- ATR (Average True Range)
- Ichimoku Cloud
- VWAP (Volume Weighted Average Price)
- Pivot Points
- CCI (Commodity Channel Index)
- ROC (Rate of Change)
- Williams %R
- Ultimate Oscillator
- MFI (Money Flow Index)
- OBV (On Balance Volume)
- Stochastic Oscillator
- ADX (Average Directional Index)

### 🎨 UI Features
- Settings modal for each indicator
- Color picker for line/histogram colors
- Opacity sliders
- Period adjustments
- Alert configuration
- Preset management
- Colorblind-friendly palettes
- Export/Import settings
- Multi-timeframe support

## Performance

- **Caching**: Indicators cache their calculations and only recalculate when data changes
- **Selective Calculation**: Only enabled indicators are calculated
- **Optimized Rendering**: Each indicator renders only visible data points
- **Memory Efficient**: No unnecessary data duplication

## Extensibility

The modular design makes it easy to:
1. Add new indicators without modifying existing code
2. Create indicator presets for different trading strategies
3. Combine multiple indicators
4. Export/import complete indicator configurations
5. Create custom alert conditions
6. Build indicator-based trading systems

## Next Steps

1. Complete Bollinger Bands indicator
2. Build settings modal UI with dynamic form generation
3. Integrate with canvas-renderer.js for subplot rendering
4. Wire up "Indicators" button to show/hide modal
5. Add more indicators (ATR, Ichimoku, VWAP, etc.)
6. Implement indicator presets system
7. Add backtesting support with indicator signals
8. Create documentation for each indicator

---

**Version**: 1.0.0
**Last Updated**: November 16, 2025
**Status**: Core infrastructure complete, ready for UI integration
