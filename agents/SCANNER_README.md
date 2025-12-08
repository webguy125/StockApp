# Comprehensive Scanner - S&P 500 + Top 100 Cryptos

## Overview

The Comprehensive Scanner replaces all previous scanning functionality with a robust, multi-source data collection system that integrates seamlessly with the agent self-learning loop.

### Features

✅ **S&P 500 Stocks** via Polygon API (real-time & historical)
✅ **Top 100 Cryptocurrencies** via CoinGecko API
✅ **Comprehensive Technical Indicators** (RSI, MACD, Moving Averages, Bollinger Bands, ATR)
✅ **Volume & Volatility Filtering**
✅ **Automatic Integration** with agent learning loop
✅ **Nightly Automation** via scheduling
✅ **Fallback Support** (uses Yahoo Finance if Polygon unavailable)

---

## Quick Start

### 1. Install Dependencies

```bash
cd C:\StockApp
venv\Scripts\activate
pip install -r requirements.txt
```

New packages added:
- `polygon-api-client` - Polygon API for S&P 500 stocks
- `pycoingecko` - CoinGecko API for cryptocurrencies
- `schedule` - Task scheduling
- `ta` - Technical analysis indicators

### 2. Set Up API Keys

Edit `backend/.env` and add your Polygon API key:

```env
# Polygon API (for S&P 500 stock data)
# Get your free API key at: https://polygon.io/
POLYGON_API_KEY=your_polygon_api_key_here
```

**Get Polygon API Key:**
1. Visit https://polygon.io/
2. Sign up for free account
3. Copy your API key
4. Paste into `.env` file

**CoinGecko:** No API key needed for basic use (free tier)

### 3. Run Scanner

#### Run Immediately
```bash
cd agents
python comprehensive_scanner.py
```

#### Run on Schedule (Nightly at Midnight UTC)
```bash
cd agents
python schedule_scanner.py
```

#### Run Once Now (For Testing)
```bash
cd agents
python schedule_scanner.py --now
```

---

## How It Works

### Data Flow

```
┌─────────────────────────────────────────┐
│   COMPREHENSIVE SCANNER                 │
├─────────────────────────────────────────┤
│  1. S&P 500 Stocks (Polygon API)        │
│  2. Top 100 Cryptos (CoinGecko API)     │
│  3. Calculate Technical Indicators      │
│  4. Apply Filters (Volume/Volatility)   │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  scanner_output.json                    │
│  (Repository)                           │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  FUSION AGENT                           │
│  Combines all signals                   │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  SUPREME LEADER                         │
│  Risk management & position sizing      │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  TRACKER → EVALUATOR → ARCHIVIST        │
│  Learning Loop                          │
└─────────────────────────────────────────┘
```

### Filtering Criteria

#### Stocks (S&P 500)
- **Minimum Volume:** $50M daily
- **Minimum Price:** $1.00
- **Minimum Volatility:** 0.5%
- **Source:** Polygon API (fallback: Yahoo Finance)

#### Cryptocurrencies (Top 100)
- **Minimum Volume:** $10M daily
- **Minimum Market Cap:** $100M
- **Minimum Volatility:** 1.0%
- **Source:** CoinGecko API

### Technical Indicators Calculated

For each asset that passes filters:

1. **RSI** (Relative Strength Index) - 14 period
2. **MACD** (Moving Average Convergence Divergence)
   - MACD Line
   - Signal Line
   - Histogram
3. **Moving Averages**
   - SMA 20, SMA 50
   - EMA 12, EMA 26
4. **ATR** (Average True Range) - Volatility measure
5. **Bollinger Bands**
   - Upper, Middle, Lower bands
6. **Volume Metrics**
   - Average volume
   - Current volume ratio

---

## Output Format

### scanner_output.json

```json
{
  "timestamp": "2025-11-19T12:00:00",
  "scan_duration_seconds": 180.5,
  "total_scanned": 200,
  "total_passed": 45,
  "stocks_passed": 25,
  "cryptos_passed": 20,
  "candidates": [
    {
      "symbol": "AAPL",
      "current_price": 185.50,
      "volume_24h": 50000000,
      "price_change_24h": 2.50,
      "price_change_pct_24h": 1.37,
      "volatility_24h": 2.1,
      "market_type": "stock",
      "source": "polygon",
      "indicators": {
        "rsi": 58.5,
        "macd": 0.75,
        "macd_signal": 0.65,
        "macd_histogram": 0.10,
        "sma_20": 183.20,
        "sma_50": 180.50,
        "ema_12": 184.80,
        "ema_26": 182.30,
        "atr": 3.25,
        "bb_upper": 188.50,
        "bb_middle": 183.20,
        "bb_lower": 177.90,
        "avg_volume": 45000000,
        "volume_ratio": 1.11
      },
      "scan_score": 105.5
    }
  ],
  "thresholds": {...}
}
```

---

## Automation

### Option 1: Python Scheduler (Recommended for Testing)

Keeps a Python process running:

```bash
python schedule_scanner.py
```

### Option 2: Windows Task Scheduler (Production)

1. Open **Task Scheduler**
2. Create Basic Task
3. **Trigger:** Daily at 12:00 AM
4. **Action:** Start a program
   - Program: `C:\StockApp\venv\Scripts\python.exe`
   - Arguments: `C:\StockApp\agents\schedule_scanner.py --now`
   - Start in: `C:\StockApp\agents`

### Option 3: Linux Cron (Production)

Add to crontab:

```bash
crontab -e
```

Add line:

```cron
0 0 * * * cd /path/to/StockApp/agents && /path/to/python schedule_scanner.py --now
```

---

## Customization

### Adjust Filtering Thresholds

Edit `comprehensive_scanner.py`:

```python
self.thresholds = {
    'min_volume_usd': {
        'stock_default': 50_000_000,     # Change this
        'crypto_default': 10_000_000,    # Or this
    },
    'min_volatility_pct': {
        'stock': 0.5,    # Adjust minimum volatility
        'crypto': 1.0
    }
}
```

### Change Stock Universe

By default, scans S&P 500 (fetched from Wikipedia). To use a custom list:

```python
def get_sp500_symbols(self):
    # Return your custom list
    return ['AAPL', 'MSFT', 'GOOGL', ...]
```

### Change Crypto Count

```python
# In scan_cryptos()
cryptos = self.get_top_cryptos(100)  # Change to 50, 200, etc.
```

---

## Troubleshooting

### "No Polygon API key configured"

**Solution:** Add your Polygon API key to `backend/.env`

### "Rate limit exceeded"

**Solution:**
- Polygon free tier: 5 requests/minute
- Add delays or upgrade to paid plan
- Scanner automatically adds 0.1s delay between stocks

### "CoinGecko API error"

**Solution:**
- CoinGecko free tier: 10-50 calls/minute
- Scanner adds 0.05s delay between cryptos
- If persistent, reduce crypto count

### "Technical indicators failing"

**Solution:**
- Requires at least 14-50 days of historical data
- Scanner gracefully handles failures with defaults

---

## Integration with Learning Loop

The scanner automatically outputs to `repository/scanner_output.json`.

The existing agent pipeline picks this up:

1. **Fusion Agent** reads `scanner_output.json`
2. **Supreme Leader** applies risk management
3. **Tracker Agent** monitors positions
4. **Evaluator Agent** judges outcomes
5. **Archivist Agent** stores for learning
6. **Criteria Auditor** validates signals

**Manual trigger entire pipeline:**

```bash
cd agents
python comprehensive_scanner.py
python fusion_agent.py
python supreme_leader.py
```

---

## Performance

### Typical Scan Times

- **S&P 500 (100 symbols):** ~60-90 seconds
- **Top 100 Cryptos:** ~30-45 seconds
- **Total:** ~2-3 minutes

### API Limits

- **Polygon Free:** 5 requests/min → ~12 stocks/min
- **Polygon Basic ($29/mo):** 100 requests/min → ~600 stocks/min
- **CoinGecko Free:** 10-50 calls/min → ~200-1000 cryptos/min

---

## Next Steps

1. **Get Polygon API Key** → https://polygon.io/
2. **Install dependencies** → `pip install -r requirements.txt`
3. **Test scanner** → `python comprehensive_scanner.py`
4. **Set up automation** → `python schedule_scanner.py`
5. **Monitor output** → Check `repository/scanner_output.json`
6. **View heat maps** → http://127.0.0.1:5000/heatmap

---

## Support

For issues or questions:
- Check logs in console output
- Verify API keys in `.env`
- Test with `--now` flag first
- Review `scanner_output.json` for errors

**Happy Scanning!** 🚀
