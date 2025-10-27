from flask import Flask, request, jsonify, send_from_directory, make_response
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import pandas as pd
import yfinance as yf
import os
import json
from datetime import datetime, timedelta
import dateutil.parser
import eventlet
import eventlet.queue
import websocket
import threading
import requests
import time

app = Flask(__name__)
CORS(app)
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='eventlet',
    logger=True,
    engineio_logger=True,
    ping_timeout=120,  # 2 minutes before considering connection dead
    ping_interval=25    # Send ping every 25 seconds to keep connection alive
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")
os.makedirs(DATA_DIR, exist_ok=True)

# =============================================================================
# COINBASE HISTORICAL DATA FETCHER (for minute-level crypto data)
# =============================================================================

def fetch_coinbase_candles(symbol, interval, period):
    """
    Fetch historical OHLC candles from Coinbase REST API.
    Only used for minute-level intervals (1m, 5m, 15m, 30m, 1h).

    Args:
        symbol: Crypto symbol (e.g., 'BTC-USD', 'ETH-USD')
        interval: Time interval ('1m', '5m', '15m', '30m', '1h')
        period: Time period ('1d', '5d', '1mo', etc.)

    Returns:
        List of candle dicts with keys: Date, Open, High, Low, Close, Volume
    """
    # Map interval to Coinbase granularity (in seconds)
    granularity_map = {
        '1m': 60,
        '5m': 300,
        '15m': 900,
        '30m': 1800,
        '1h': 3600
    }

    if interval not in granularity_map:
        raise ValueError(f"Interval {interval} not supported for Coinbase")

    granularity = granularity_map[interval]

    # Calculate time range based on period
    # Coinbase limits to 300 candles per request, so adjust time range accordingly
    end_dt = datetime.utcnow()

    # Calculate max time range based on granularity (300 candles limit)
    max_seconds = granularity * 300  # 300 candles max
    max_delta = timedelta(seconds=max_seconds)

    period_map = {
        '1d': timedelta(days=1),
        '5d': timedelta(days=5),
        '1mo': timedelta(days=30),
        '3mo': timedelta(days=90)
    }

    if period in period_map:
        requested_delta = period_map[period]
        # Use the smaller of requested period or max allowed
        start_dt = end_dt - min(requested_delta, max_delta)
    else:
        # Default to max allowed (up to 300 candles)
        start_dt = end_dt - max_delta

    # Coinbase API endpoint
    product_id = symbol  # Already in format 'BTC-USD'
    url = f"https://api.exchange.coinbase.com/products/{product_id}/candles"

    print(f"[COINBASE] Fetching {interval} candles for {symbol} from {start_dt} to {end_dt}")

    params = {
        'granularity': granularity,
        'end': end_dt.isoformat(),
        'start': start_dt.isoformat()
    }

    all_candles = []

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        all_candles = response.json()

        print(f"[COINBASE] Received {len(all_candles)} candles")

    except requests.exceptions.RequestException as e:
        print(f"[COINBASE ERROR] Failed to fetch candles: {e}")
        print(f"[COINBASE ERROR] Response: {response.text if 'response' in locals() else 'No response'}")
        return []

    # Transform Coinbase format to our format
    # Coinbase returns: [[time, low, high, open, close, volume], ...]
    # We need: [{Date, Open, High, Low, Close, Volume}, ...]

    transformed = []
    for candle in all_candles:
        timestamp, low, high, open_price, close_price, volume = candle
        dt = datetime.utcfromtimestamp(timestamp)

        transformed.append({
            'Date': dt.strftime("%Y-%m-%d %H:%M:%S"),
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close_price,
            'Volume': volume
        })

    # Sort by date (oldest first)
    transformed.sort(key=lambda x: x['Date'])

    print(f"[COINBASE] Returning {len(transformed)} candles")
    return transformed

@app.route("/")
def serve_index():
    response = make_response(send_from_directory(FRONTEND_DIR, "index_tos_style.html"))
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route("/classic")
def serve_classic():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/css/<path:filename>")
def serve_css(filename):
    response = make_response(send_from_directory(os.path.join(FRONTEND_DIR, "css"), filename))
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route("/js/<path:filename>")
def serve_js(filename):
    response = make_response(send_from_directory(os.path.join(FRONTEND_DIR, "js"), filename))
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route("/data/<symbol>")
def get_chart_data(symbol):
    symbol = symbol.upper()
    start = request.args.get('start')
    end = request.args.get('end')
    period = request.args.get('period')
    interval = request.args.get('interval', '1d')

    # Route minute-level intervals to Coinbase for crypto (proper OHLC data)
    minute_intervals = ['1m', '5m', '15m', '30m', '1h']
    with open('backend/debug.log', 'a') as f:
        f.write(f"\n[ROUTE] symbol={symbol}, interval={interval}, in_list={interval in minute_intervals}\n")

    if interval in minute_intervals:
        with open('backend/debug.log', 'a') as f:
            f.write(f"[ROUTING] Using Coinbase for {symbol} {interval}\n")
        print(f"[ROUTING] Using Coinbase for {symbol} {interval}")
        try:
            candles = fetch_coinbase_candles(symbol, interval, period or '1d')
            with open('backend/debug.log', 'a') as f:
                f.write(f"[SUCCESS] Got {len(candles)} candles\n")
            return jsonify(candles)
        except Exception as e:
            with open('backend/debug.log', 'a') as f:
                f.write(f"[ERROR] {e}\n")
            print(f"[COINBASE ERROR] {e}")
            return jsonify([])

    # Use yfinance for daily and above (unchanged)
    print(f"[ROUTING] Using yfinance for {symbol} {interval}")
    kwargs = {'interval': interval}
    if start and end:
        kwargs['start'] = start
        kwargs['end'] = end
    elif period:
        kwargs['period'] = period
    else:
        kwargs['period'] = 'max'

    data = yf.download(symbol, **kwargs)

    if data.empty:
        print(f"No data found for {symbol}.")
        return jsonify([])

    # Flatten multi-level columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.reset_index(inplace=True)
    data = data.rename(columns={'Datetime': 'Date'} if 'Datetime' in data.columns else {'Date': 'Date'})
    data['Date'] = data['Date'].dt.strftime("%Y-%m-%d %H:%M:%S") if 'h' in interval or 'm' in interval else data['Date'].dt.strftime("%Y-%m-%d")
    return jsonify(data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].fillna("").to_dict(orient="records"))

@app.route("/volume", methods=["POST"])
def calculate_volume():
    data = request.get_json()
    symbol = data["symbol"]
    start_date = data["start_date"]
    end_date = data["end_date"]
    interval = data.get("interval", "1d")

    print(f"Volume request received: {symbol} from {start_date} to {end_date} interval {interval}")

    try:
        start_dt = dateutil.parser.isoparse(start_date)
        end_dt = dateutil.parser.isoparse(end_date)
    except Exception as e:
        print("Date parsing error:", e)
        return jsonify({"avg_volume": 0})

    if start_dt > end_dt:
        start_dt, end_dt = end_dt, start_dt

    start_str = start_dt.date().strftime("%Y-%m-%d")
    end_str = (end_dt.date() + timedelta(days=1)).strftime("%Y-%m-%d")

    df = yf.download(symbol, start=start_str, end=end_str, interval=interval)

    if df.empty:
        print(f"No data found in range: {start_str} to {end_str}")
        return jsonify({"avg_volume": 0})

    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[(df.index >= start_dt) & (df.index <= end_dt)]
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    df = df.dropna(subset=["Volume"])
    avg_volume = df["Volume"].mean() if not df.empty else 0

    print(f"Avg volume for {symbol} between {start_date} and {end_date}: {avg_volume:.2f}")
    return jsonify({"avg_volume": avg_volume})

# ========================================
# TICK BAR ENDPOINTS
# ========================================

@app.route("/data/tick/<symbol>/<int:threshold>", methods=["GET"])
def get_tick_bars(symbol, threshold):
    """Load historical tick bars from file"""
    symbol = symbol.upper()
    tick_file = os.path.join(DATA_DIR, "tick_bars", f"tick_{threshold}_{symbol}.json")

    print(f"[TICK] Loading tick bars: {symbol} threshold={threshold}")

    if not os.path.exists(tick_file):
        print(f"[TICK] No tick bar file found, returning empty array")
        return jsonify([])

    try:
        with open(tick_file, 'r') as f:
            bars = json.load(f)

        # Return only last 300 bars for performance
        bars = bars[-300:] if len(bars) > 300 else bars
        print(f"[TICK] Loaded {len(bars)} tick bars for {symbol}")
        return jsonify(bars)

    except Exception as e:
        print(f"[TICK ERROR] Failed to load tick bars: {e}")
        return jsonify([])

@app.route("/data/tick/<symbol>/<int:threshold>", methods=["POST"])
def save_tick_bar(symbol, threshold):
    """Persist new tick bar to file"""
    symbol = symbol.upper()
    bar_data = request.get_json()

    if not bar_data:
        return jsonify({"success": False, "error": "No bar data provided"}), 400

    tick_dir = os.path.join(DATA_DIR, "tick_bars")
    os.makedirs(tick_dir, exist_ok=True)

    tick_file = os.path.join(tick_dir, f"tick_{threshold}_{symbol}.json")

    try:
        # Load existing bars
        if os.path.exists(tick_file):
            with open(tick_file, 'r') as f:
                bars = json.load(f)
        else:
            bars = []

        # Append new bar
        bars.append(bar_data)

        # Keep only last 1000 bars (disk limit)
        if len(bars) > 1000:
            bars = bars[-1000:]

        # Save back to file
        with open(tick_file, 'w') as f:
            json.dump(bars, f, indent=2)

        print(f"[TICK] Saved bar for {symbol} threshold={threshold}, total bars={len(bars)}")
        return jsonify({"success": True, "total_bars": len(bars)})

    except Exception as e:
        print(f"[TICK ERROR] Failed to save tick bar: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/save_line", methods=["POST"])
def save_line():
    data = request.get_json()
    symbol = data.get("symbol", "").upper()
    line = data.get("line")

    if not symbol or not line or "id" not in line:
        print("Missing symbol or line ID")
        return jsonify({"error": "Missing symbol or line ID"}), 400

    filename = os.path.join(DATA_DIR, f"lines_{symbol}.json")
    lines = []

    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                lines = json.load(f)
        except Exception as e:
            print("Error reading line file:", e)

    lines = [l for l in lines if l["id"] != line["id"]]
    lines.append(line)

    try:
        with open(filename, "w") as f:
            json.dump(lines, f)
        print(f"Saved line for {symbol}: {line['id']}")
        return jsonify({"success": True})
    except Exception as e:
        print("Error saving line:", e)
        return jsonify({"error": "Failed to save line"}), 500

@app.route("/delete_line", methods=["POST"])
def delete_line():
    data = request.get_json()
    symbol = data.get("symbol", "").upper()
    line_id = data.get("line_id")

    if not symbol or not line_id:
        print("Missing symbol or line ID for deletion")
        return jsonify({"error": "Missing symbol or line ID"}), 400

    filename = os.path.join(DATA_DIR, f"lines_{symbol}.json")
    if not os.path.exists(filename):
        print(f"No line file found for {symbol}")
        return jsonify({"success": True})

    try:
        with open(filename, "r") as f:
            lines = json.load(f)
        lines = [l for l in lines if l["id"] != line_id]
        with open(filename, "w") as f:
            json.dump(lines, f)
        print(f"Deleted line {line_id} for {symbol}")
        return jsonify({"success": True})
    except Exception as e:
        print("Error deleting line:", e)
        return jsonify({"error": "Failed to delete line"}), 500

@app.route("/clear_lines", methods=["POST"])
def clear_lines():
    data = request.get_json()
    symbol = data.get("symbol", "").upper()
    filename = os.path.join(DATA_DIR, f"lines_{symbol}.json")

    try:
        with open(filename, "w") as f:
            json.dump([], f)
        print(f"Cleared all lines for {symbol}")
        return jsonify({"success": True})
    except Exception as e:
        print("Error clearing lines:", e)
        return jsonify({"error": "Failed to clear lines"}), 500

@app.route("/lines/<symbol>")
def get_lines(symbol):
    symbol = symbol.upper()
    filename = os.path.join(DATA_DIR, f"lines_{symbol}.json")

    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                lines = json.load(f)
            print(f"Loaded {len(lines)} lines for {symbol}")
            return jsonify(lines)
        except Exception as e:
            print("Error loading lines:", e)
            return jsonify([])
    else:
        print(f"No line file found for {symbol}")
        return jsonify([])

@app.route("/news/market")
def get_market_news():
    limit = request.args.get('limit', 20, type=int)
    try:
        # Get news from Yahoo Finance for general market
        ticker = yf.Ticker("^GSPC")  # S&P 500 as market proxy
        news = ticker.news if hasattr(ticker, 'news') else []

        # Format news for frontend
        formatted_news = []
        for item in news[:limit]:
            # Yahoo Finance API now nests data under 'content' key
            content = item.get('content', {})
            formatted_news.append({
                'symbol': 'Market',
                'source': content.get('provider', {}).get('displayName', 'Yahoo Finance'),
                'title': content.get('title', ''),
                'summary': content.get('summary', '')[:200] + '...' if len(content.get('summary', '')) > 200 else content.get('summary', ''),
                'link': content.get('canonicalUrl', {}).get('url', ''),
                'published': int(datetime.fromisoformat(content.get('pubDate', datetime.now().isoformat()).replace('Z', '+00:00')).timestamp() * 1000)
            })

        return jsonify(formatted_news)
    except Exception as e:
        print(f"Error fetching market news: {e}")
        import traceback
        traceback.print_exc()
        # Return empty array on error - frontend will show "No news available"
        return jsonify([])

@app.route("/news/<symbol>")
def get_symbol_news(symbol):
    symbol = symbol.upper()
    limit = request.args.get('limit', 15, type=int)
    try:
        # Get news from Yahoo Finance for specific symbol
        ticker = yf.Ticker(symbol)
        news = ticker.news if hasattr(ticker, 'news') else []

        # Format news for frontend
        formatted_news = []
        for item in news[:limit]:
            # Yahoo Finance API now nests data under 'content' key
            content = item.get('content', {})
            formatted_news.append({
                'symbol': symbol,
                'source': content.get('provider', {}).get('displayName', 'Yahoo Finance'),
                'title': content.get('title', ''),
                'summary': content.get('summary', '')[:200] + '...' if len(content.get('summary', '')) > 200 else content.get('summary', ''),
                'link': content.get('canonicalUrl', {}).get('url', ''),
                'published': int(datetime.fromisoformat(content.get('pubDate', datetime.now().isoformat()).replace('Z', '+00:00')).timestamp() * 1000)
            })

        return jsonify(formatted_news)
    except Exception as e:
        print(f"Error fetching news for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        # Return empty array on error - frontend will show "No news available"
        return jsonify([])

@app.route("/portfolio")
def get_portfolio():
    """Stub endpoint for portfolio - returns demo account data"""
    return jsonify({
        "account": {
            "balance": 100000.00,
            "buying_power": 100000.00,
            "equity": 100000.00
        },
        "positions": []
    })

# Global set to track subscribed symbols
subscribed_symbols = set()
price_streaming_active = False
coinbase_ws = None
coinbase_thread = None
ticker_queue = eventlet.queue.Queue(maxsize=1000)  # Queue to bridge Coinbase thread and Socket.IO
ticker_worker_started = False  # Track if worker has been started
trade_queue = eventlet.queue.Queue(maxsize=1000)  # Queue for trade updates from Coinbase thread
trade_worker_started = False  # Track if trade worker has been started

@socketio.on('connect')
def handle_connect(auth=None):
    global ticker_worker_started, trade_worker_started
    print(f'[CONNECT] Client connected: {request.sid}')
    emit('connection_response', {'status': 'connected'})

    # Start ticker emit worker on first client connection
    if not ticker_worker_started:
        socketio.start_background_task(ticker_emit_worker)
        ticker_worker_started = True
        print('[TICKER WORKER] Starting background task')

    # Start trade emit worker on first client connection
    if not trade_worker_started:
        socketio.start_background_task(trade_emit_worker)
        trade_worker_started = True
        print('[TRADE WORKER] Starting background task')

@socketio.on('disconnect')
def handle_disconnect():
    print(f'[DISCONNECT] Client disconnected: {request.sid}')

@socketio.on('subscribe')
def handle_subscribe(data):
    """Subscribe to real-time updates for symbols"""
    symbols = data.get('symbols', [])
    if symbols:
        was_empty = len(subscribed_symbols) == 0
        subscribed_symbols.update(symbols)
        print(f'[SUBSCRIBE] Subscribed to symbols: {symbols}')
        print(f'[SUBSCRIBE] Total subscribed symbols: {len(subscribed_symbols)}')

        # Start Coinbase WebSocket if not already running
        if coinbase_ws is None:
            print(f'[COINBASE] Starting WebSocket (first connection)')
            start_coinbase_websocket()
        else:
            # Update existing subscription
            print(f'[COINBASE] Updating subscriptions')
            resubscribe_coinbase_ws()

@socketio.on('subscribe_ticker')
def handle_subscribe_ticker(data):
    """Subscribe to real-time updates for a single symbol"""
    symbol = data.get('symbol')
    if symbol:
        was_empty = len(subscribed_symbols) == 0
        subscribed_symbols.add(symbol)
        print(f'[SUBSCRIBE_TICKER] Subscribed to symbol: {symbol}')
        print(f'[SUBSCRIBE_TICKER] Total subscribed symbols: {len(subscribed_symbols)}')

        # Start Coinbase WebSocket if not already running
        if coinbase_ws is None:
            print(f'[COINBASE] Starting WebSocket (first connection)')
            start_coinbase_websocket()
        else:
            # Update existing subscription
            print(f'[COINBASE] Updating subscriptions')
            resubscribe_coinbase_ws()

@socketio.on('unsubscribe')
def handle_unsubscribe(data):
    """Unsubscribe from symbol updates"""
    symbols = data.get('symbols', [])
    if symbols:
        subscribed_symbols.difference_update(symbols)
        print(f'Unsubscribed from symbols: {symbols}')
        # Update Coinbase WebSocket subscriptions
        if coinbase_ws:
            resubscribe_coinbase_ws()

def on_coinbase_message(ws, message):
    """Handle incoming Coinbase WebSocket messages"""
    try:
        data = json.loads(message)

        # Debug: Log message types to see what Coinbase is sending
        msg_type = data.get('type', 'unknown')
        if msg_type not in ['ticker', 'subscriptions', 'heartbeat']:
            print(f'[COINBASE MSG TYPE] {msg_type}')

        # Handle ticker updates
        if data.get('type') == 'ticker':
            product_id = data.get('product_id', '')
            price = float(data.get('price', 0))

            if price > 0:
                ticker_data = {
                    'symbol': product_id,
                    'price': price,
                    'change': 0,  # Coinbase doesn't provide this in ticker
                    'changePercent': 0,
                    'previousClose': 0,
                    'timestamp': datetime.now().isoformat()
                }

                # Put ticker data in queue for Socket.IO background task to emit
                try:
                    ticker_queue.put_nowait(ticker_data)
                except eventlet.queue.Full:
                    pass  # Drop ticker if queue is full (not critical)

                # WORKAROUND: Coinbase doesn't send match events on public feed
                # Use ticker as synthetic trade data for tick charts
                trade_data = {
                    'symbol': product_id,
                    'product_id': product_id,
                    'price': str(price),  # Match format from real trades
                    'size': '0.001',  # Synthetic size (ticker doesn't include volume)
                    'side': 'unknown',
                    'trade_id': None,
                    'time': data.get('time', datetime.now().isoformat())
                }
                # Add to queue for trade emit worker to process
                trade_queue.put_nowait(trade_data)

                # Debug: Print every 10th ticker to avoid spam
                if price and int(price) % 10 == 0:
                    print(f'[COINBASE -> QUEUE] {product_id}: ${price:.2f} -> added to queue')

        # Handle trade/match updates (for tick charts)
        elif data.get('type') in ['match', 'last_match']:
            product_id = data.get('product_id', '')
            price = data.get('price')
            size = data.get('size')
            side = data.get('side')
            trade_id = data.get('trade_id')
            time = data.get('time')

            if price and size:
                trade_data = {
                    'symbol': product_id,
                    'product_id': product_id,
                    'price': price,
                    'size': size,
                    'side': side,
                    'trade_id': trade_id,
                    'time': time
                }

                # Add to queue for trade emit worker to process
                try:
                    trade_queue.put_nowait(trade_data)
                except eventlet.queue.Full:
                    pass  # Drop trade if queue is full (not critical)

                # Debug: Print trades to see if they're coming through
                print(f'[TRADE] {product_id}: ${price} x {size} ({side})')
    except Exception as e:
        print(f'[ERROR] Processing Coinbase message: {e}')

def on_coinbase_error(ws, error):
    """Handle Coinbase WebSocket errors"""
    print(f'[COINBASE ERROR] {error}')

def on_coinbase_close(ws, close_status_code, close_msg):
    """Handle Coinbase WebSocket close"""
    print(f'[COINBASE] Connection closed: {close_status_code} - {close_msg}')
    # Attempt to reconnect after 5 seconds
    eventlet.sleep(5)
    start_coinbase_websocket()

def on_coinbase_open(ws):
    """Handle Coinbase WebSocket open"""
    print('[COINBASE] WebSocket connected')
    # Subscribe to all symbols
    subscribe_message = {
        "type": "subscribe",
        "product_ids": list(subscribed_symbols),
        "channels": ["ticker", "matches"]
    }
    ws.send(json.dumps(subscribe_message))
    print(f'[COINBASE] Subscribed to: {list(subscribed_symbols)} (ticker + matches)')

def start_coinbase_websocket():
    """Start Coinbase Advanced Trade WebSocket connection"""
    global coinbase_ws

    if len(subscribed_symbols) == 0:
        print('[COINBASE] No symbols to subscribe, skipping WebSocket')
        return

    print('[COINBASE] Starting WebSocket connection...')

    # Coinbase Advanced Trade WebSocket URL
    coinbase_ws = websocket.WebSocketApp(
        "wss://ws-feed.exchange.coinbase.com",
        on_message=on_coinbase_message,
        on_error=on_coinbase_error,
        on_close=on_coinbase_close,
        on_open=on_coinbase_open
    )

    # Run WebSocket in separate thread (non-blocking)
    def run_ws():
        coinbase_ws.run_forever()

    ws_thread = threading.Thread(target=run_ws, daemon=True)
    ws_thread.start()

def resubscribe_coinbase_ws():
    """Update Coinbase WebSocket subscriptions"""
    if coinbase_ws and len(subscribed_symbols) > 0:
        subscribe_message = {
            "type": "subscribe",
            "product_ids": list(subscribed_symbols),
            "channels": ["ticker", "matches"]
        }
        try:
            coinbase_ws.send(json.dumps(subscribe_message))
            print(f'[COINBASE] Updated subscriptions: {list(subscribed_symbols)} (ticker + matches)')
        except Exception as e:
            print(f'[COINBASE ERROR] Failed to update subscriptions: {e}')

def ticker_emit_worker():
    """Background task that pulls ticker data from queue and emits to Socket.IO clients"""
    print('[TICKER WORKER] Started - pulling from queue and emitting to clients')
    while True:
        try:
            # Block until ticker data is available (with timeout to allow graceful shutdown)
            ticker_data = ticker_queue.get(timeout=1)

            # Emit to all connected Socket.IO clients (broadcasts by default in background task)
            socketio.emit('ticker_update', ticker_data)

            # Debug logging (reduced frequency)
            if ticker_data['price'] and int(ticker_data['price']) % 10 == 0:
                print(f'[EMIT] {ticker_data["symbol"]}: ${ticker_data["price"]:.2f}')

        except eventlet.queue.Empty:
            # No ticker data available, continue loop
            continue
        except Exception as e:
            print(f'[TICKER WORKER ERROR] {e}')
            eventlet.sleep(0.1)  # Brief pause on error

def trade_emit_worker():
    """Background task that pulls trade data from queue and emits to Socket.IO clients"""
    print('[TRADE WORKER] Started - pulling from queue and emitting to clients')
    while True:
        try:
            # Block until trade data is available (with timeout to allow graceful shutdown)
            trade_data = trade_queue.get(timeout=1)

            # Emit to all connected Socket.IO clients (broadcasts by default in background task)
            socketio.emit('trade_update', trade_data)

            # Debug logging (reduced frequency) - log every 100th trade
            if trade_data.get('trade_id') and trade_data.get('trade_id') % 100 == 0:
                print(f'[EMIT TRADE] {trade_data["symbol"]}: ${trade_data["price"]}')

        except eventlet.queue.Empty:
            # No trade data available, continue loop
            continue
        except Exception as e:
            print(f'[TRADE WORKER ERROR] {e}')
            eventlet.sleep(0.1)  # Brief pause on error

# Old yfinance polling removed - now using real Coinbase WebSocket for tick-by-tick updates

if __name__ == "__main__":
    # Use socketio.run instead of app.run for WebSocket support
    # Ticker emit worker will start automatically when first client connects
    socketio.run(app, debug=True, host='127.0.0.1', port=5000, allow_unsafe_werkzeug=True)
