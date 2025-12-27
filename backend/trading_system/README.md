# ML Trading System

**Zero-cost modular ML pipeline for stock analysis and trading signals**

## Overview

This is a **completely separate** stock picking system from the multi-agent approach. It uses a modular ML pipeline with no API costs, designed to be easily extensible with proprietary indicators.

### System A vs System B

| Feature | Multi-Agent System | ML Trading System |
|---------|-------------------|-------------------|
| **Location** | `agents/` | `backend/trading_system/` |
| **Approach** | Multi-agent learning loop | Modular ML pipeline |
| **API Costs** | Variable (depends on agent complexity) | $0 (all local) |
| **Data File** | `agents/repository/fusion_output.json` | `backend/data/ml_trading_signals.json` |
| **Heat Map Page** | `/heatmap` | `/ml-trading` |
| **Extensibility** | Add new agents | Add analyzer plugins |
| **Learning** | Agent feedback loops | Ensemble ML models |

## Architecture

```
Trading System Pipeline:
1. Scanner → Filter S&P 500 to ~50 candidates
2. Analyzers → Run ALL registered analyzers (RSI, MACD, Volume, Trend)
3. Feature Extractor → Convert analyzer outputs to ML features
4. ML Model → Ensemble prediction (Random Forest)
5. Scoring → Combine analyzer + ML scores (60/40 weighted)
6. Output → Ranked signals saved to JSON
```

### Components

- **StockScanner** - Filters S&P 500 by volume, volatility, price
- **AnalyzerRegistry** - Plugin system for indicators
- **Analyzers** - RSI, MACD, Volume, Trend (easily add more!)
- **FeatureExtractor** - Converts signals to ML feature vectors
- **SimpleTradingModel** - Random Forest classifier
- **TradeTracker** - SQLite database for performance tracking
- **TradingSystem** - Orchestrates all components

## Quick Start

### 1. Run a Scan (CLI)

```bash
cd backend/trading_system
python run_scan.py

# Options:
python run_scan.py --max-stocks 100        # Analyze more stocks
python run_scan.py --no-crypto             # Exclude crypto
python run_scan.py --stats                 # Show performance stats only
```

### 2. View Results (Web UI)

```bash
# Start Flask server (if not running)
cd backend
python api_server.py

# Open browser
http://127.0.0.1:5000/ml-trading
```

### 3. Trigger Scan from UI

Click "🔍 Run Scan" button on the ML Trading page

## Adding Custom Analyzers

The system is designed for easy extension. Here's how to add your proprietary indicator:

### Step 1: Create Your Analyzer

```python
# backend/trading_system/analyzers/my_proprietary.py

from core.base_analyzer import BaseAnalyzer
from datetime import datetime
import yfinance as yf

class MyProprietaryAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__(name="my_proprietary")

    def analyze(self, symbol, start_date, end_date):
        # Fetch data
        df = yf.download(symbol, start=start_date, end=end_date, progress=False)

        # YOUR PROPRIETARY CALCULATION HERE
        signal_value = self._calculate_my_indicator(df)

        # Return standardized format
        return {
            'signal_strength': signal_value,  # 0.0 to 1.0
            'direction': 'bullish' if signal_value > 0.6 else 'bearish' if signal_value < 0.4 else 'neutral',
            'confidence': abs(signal_value - 0.5) * 2,  # 0.0 to 1.0
            'metadata': {
                'my_custom_metric': signal_value
            }
        }

    def _calculate_my_indicator(self, df):
        # Your secret sauce here
        pass
```

### Step 2: Register It

Edit `backend/trading_system/core/trading_system.py`:

```python
def _register_default_analyzers(self):
    from analyzers.rsi_analyzer import RSIAnalyzer
    from analyzers.macd_analyzer import MACDAnalyzer
    from analyzers.volume_analyzer import VolumeAnalyzer
    from analyzers.trend_analyzer import TrendAnalyzer
    from analyzers.my_proprietary import MyProprietaryAnalyzer  # Add this

    self.registry.register(RSIAnalyzer())
    self.registry.register(MACDAnalyzer())
    self.registry.register(VolumeAnalyzer())
    self.registry.register(TrendAnalyzer())
    self.registry.register(MyProprietaryAnalyzer())  # Add this
```

That's it! The system will automatically:
- Run your analyzer on all scanned stocks
- Include it in ML features
- Track its performance
- Adjust its weight based on success rate

## Signal Categories

Signals are automatically categorized by confidence and score thresholds:

- **Intraday**: Confidence ≥70%, Score ≥65 (high confidence quick trades)
- **Daily**: Confidence ≥55%, Score ≥55 (medium-term swing trades)
- **Monthly**: Confidence ≥40%, Score ≥45 (longer-term positions)

## Performance Tracking

View system performance:

```bash
python run_scan.py --stats
```

Or click "📊 Stats" on the web UI.

Metrics tracked:
- Total trades
- Win rate
- Average P/L
- Total P/L
- Profit factor (gross profit / gross loss)

## Files Created

```
backend/
  trading_system/
    __init__.py
    README.md
    run_scan.py                    # CLI interface
    core/
      __init__.py
      base_analyzer.py             # Base class for all analyzers
      analyzer_registry.py         # Plugin system
      stock_scanner.py             # S&P 500 filtering
      feature_extractor.py         # ML feature engineering
      trade_tracker.py             # SQLite database
      trading_system.py            # Main orchestrator
    analyzers/
      __init__.py
      rsi_analyzer.py              # RSI indicator
      macd_analyzer.py             # MACD indicator
      volume_analyzer.py           # Volume analysis
      trend_analyzer.py            # Moving averages
    models/
      __init__.py
      simple_trading_model.py      # Random Forest ML model
  data/
    ml_trading_signals.json        # Output signals
    trading_system.db              # SQLite database
    ml_models/
      trading_model.pkl            # Trained ML model

frontend/
  ml_trading.html                  # Web UI page
  js/
    ml-trading-page.js             # Page JavaScript
```

## API Endpoints

New endpoints added to `backend/api_server.py`:

- `GET /ml-trading` - Serve ML Trading System page
- `GET /ml-signals` - Get all ML signals
- `POST /ml-scan` - Trigger a new scan
- `GET /ml-stats` - Get performance statistics
- `GET /ml-heatmap-data` - Get heat map data

## Comparison with Agent System

Run both systems and compare results!

### Access Pages:
- **Agent System**: http://127.0.0.1:5000/heatmap
- **ML System**: http://127.0.0.1:5000/ml-trading

### Compare:
1. Which system finds better signals?
2. Which has better win rate over time?
3. Which stocks appear in both systems? (High confidence!)
4. Which system performs better on different timeframes?

## Next Steps

1. **Run initial scan** - Generate baseline signals
2. **Track some trades** - Manually record outcomes for learning
3. **Monitor performance** - Check stats weekly
4. **Add proprietary indicators** - Extend with your secret sauce
5. **Retrain models** - ML model improves with more data
6. **Compare with agents** - See which approach works best for you!

## Support

- Main project docs: `C:\StockApp\CLAUDE.md`
- Agent system: `C:\StockApp\agents\README.md`
- ML system spec: `C:\StockApp\AI_LEARNING_SYSTEM_SPEC.json`

---

**Built as a parallel system to the multi-agent approach for A/B testing different stock selection strategies.**
