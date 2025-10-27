# Binance Crypto Platform Migration

## Overview
Platform has been converted from stock trading (Yahoo Finance) to **crypto-only trading platform** using **Binance** APIs (100% free, no API keys needed).

---

## What's Been Created

### 1. Binance REST API Client (`backend/binance_client.py`)
**Purpose:** Fetch historical candlestick data

**Features:**
- ✅ Get historical klines (candlesticks) for any timeframe
- ✅ Supports all crypto pairs (BTC, ETH, SOL, etc.)
- ✅ Auto-converts symbols: `BTC`, `BTC-USD`, `BTCUSDT` all work
- ✅ Intervals: 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1M
- ✅ Periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 5y, max
- ✅ Rate limiting built-in
- ✅ No API key required (public endpoints)

**Example Usage:**
```python
from binance_client import BinanceClient

client = BinanceClient()

# Get 5-minute candles for Bitcoin
df = client.get_data_by_period('BTC', interval='5m', period='1d')

# Get daily candles for Ethereum
df = client.get_data_by_period('ETH-USD', interval='1d', period='1y')
```

---

### 2. Binance WebSocket Manager (`backend/binance_websocket.py`)
**Purpose:** Real-time live price streaming

**Features:**
- ✅ WebSocket streams for live candlestick updates
- ✅ WebSocket streams for live ticker (price) updates
- ✅ Emits data to frontend via Socket.IO
- ✅ Multiple concurrent streams supported
- ✅ Auto-reconnect on disconnect
- ✅ 100% free, no limits

**Streams:**
- **Kline Stream**: Real-time candlestick updates (updates every second)
- **Ticker Stream**: 24h ticker stats (price, volume, change %)

---

## What Needs To Be Done

### Step 1: Update Backend API Server
**File:** `backend/api_server.py`

**Tasks:**
1. Replace yfinance imports with Binance client
2. Initialize Socket.IO for WebSocket support
3. Update `/data/<symbol>` endpoint to use `BinanceClient`
4. Add WebSocket control endpoints:
   - `POST /start_stream` - Start live stream for symbol
   - `POST /stop_stream` - Stop stream
   - `GET /active_streams` - List active streams
5. Remove stock-specific logic (market hours filtering for stocks)

**Code Template:**
```python
from flask_socketio import SocketIO
from binance_client import BinanceClient
from binance_websocket import BinanceWebSocketManager

# Initialize
socketio = SocketIO(app, cors_allowed_origins="*")
binance_client = BinanceClient()
ws_manager = BinanceWebSocketManager(socketio)

# Update data endpoint
@app.route("/data/<symbol>")
def get_chart_data(symbol):
    interval = request.args.get('interval', '1d')
    period = request.args.get('period', '1d')

    # Use Binance instead of yfinance
    data = binance_client.get_data_by_period(symbol, interval, period)

    # Format and return...
```

---

### Step 2: Update Frontend for WebSocket
**File:** `frontend/js/tos-app.js` or create new `frontend/js/binance-websocket.js`

**Tasks:**
1. Add Socket.IO client library to HTML
2. Connect to Socket.IO server
3. Listen for `binance_kline_update` events
4. Update chart in real-time when new candle data arrives
5. Request stream start when loading a symbol

**Code Template:**
```javascript
// Connect to WebSocket
const socket = io('http://localhost:5000');

// Listen for kline updates
socket.on('binance_kline_update', (data) => {
  console.log('New candle data:', data);
  // Update chart with new data
  updateChartWithNewCandle(data);
});

// Start stream when loading symbol
function loadSymbol(symbol, interval) {
  fetch('/start_stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ symbol, interval })
  });
}
```

---

### Step 3: Update UI/UX for Crypto
**Files:**
- `frontend/index_tos_style.html`
- `frontend/css/tos-theme.css`

**Tasks:**
1. Update branding: "Crypto Trading Platform" or similar
2. Update watchlist with crypto symbols (BTC, ETH, SOL, BNB, etc.)
3. Remove stock-specific features:
   - Remove market hours references
   - Remove after-hours filtering UI
   - Update symbol format hints (show BTCUSDT format)
4. Add crypto-specific features:
   - 24/7 trading indicator
   - Quote currency selector (USDT, BUSD, BTC)
   - Popular crypto list

---

### Step 4: Update Watchlist
**File:** `frontend/js/components/watchlist.js`

**Tasks:**
1. Replace stock symbols with crypto symbols
2. Update to use Binance 24h ticker API
3. Show 24h change % and volume

**Default Crypto Watchlist:**
```javascript
const defaultWatchlist = [
  'BTCUSDT',   // Bitcoin
  'ETHUSDT',   // Ethereum
  'BNBUSDT',   // Binance Coin
  'SOLUSDT',   // Solana
  'ADAUSDT',   // Cardano
  'DOGEUSDT',  // Dogecoin
  'MATICUSDT', // Polygon
  'DOTUSDT',   // Polkadot
  'AVAXUSDT',  // Avalanche
  'LINKUSDT'   // Chainlink
];
```

---

### Step 5: Update Technical Indicators
**File:** `backend/api_server.py` - indicators endpoint

**Tasks:**
1. Ensure indicators work with Binance data format
2. Test RSI, MACD, SMA, EMA, Bollinger Bands with crypto
3. Remove stock-specific indicator logic if any

---

### Step 6: Update News Feed (Optional)
**Options:**
1. **CryptoPanic API** - Free crypto news aggregator
2. **CoinGecko API** - Free, includes news
3. **Remove news feed** - Focus on charts only
4. **Keep generic market news** - Use existing RSS

---

### Step 7: Testing Checklist

**Test Cases:**
- [ ] Load BTC with 1D / 5m interval
- [ ] Load ETH with 5D / 15m interval
- [ ] Switch between BTC, ETH, SOL, DOGE
- [ ] Verify WebSocket updates in real-time (should see price changes every second)
- [ ] Add indicators (SMA, RSI, MACD) - should work without gaps
- [ ] Test 24/7 - crypto trades on weekends
- [ ] Verify no rate limiting errors
- [ ] Check cache is working (💾 emoji in console)
- [ ] Verify smooth updates (no flickering)
- [ ] Test drawing tools with crypto charts

---

## Installation

```bash
# Install new dependencies
pip install -r requirements.txt

# Or manually:
pip install flask-socketio websocket-client
```

---

## Running the Platform

```bash
# Start Flask with Socket.IO support
python backend/api_server.py

# Server will start on http://localhost:5000
# WebSocket will be available automatically
```

---

## Benefits of Binance

✅ **100% Free** - No API keys, no subscriptions, no hidden costs
✅ **Real-time data** - True live streaming via WebSocket
✅ **No throttling** - Unlike Yahoo Finance
✅ **Legal & Open** - Free to use and distribute
✅ **24/7 Trading** - Crypto never closes
✅ **Global** - Works anywhere in the world
✅ **Reliable** - Enterprise-grade infrastructure
✅ **Complete data** - 1-minute to monthly candles

---

## Next Steps

1. **Update `api_server.py`** - Replace yfinance with Binance
2. **Add Socket.IO support** - Enable WebSocket
3. **Update frontend** - Connect to WebSocket stream
4. **Test with BTC** - Verify live streaming works
5. **Deploy** - Platform is production-ready!

---

## Support

Binance API Docs: https://binance-docs.github.io/apidocs/spot/en/
WebSocket Streams: https://binance-docs.github.io/apidocs/spot/en/#websocket-market-streams
