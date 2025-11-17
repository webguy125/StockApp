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
import logging
from dotenv import load_dotenv
import jwt
import secrets

app = Flask(__name__)
CORS(app)

# Load environment variables from .env file
load_dotenv()
COINBASE_API_KEY = os.getenv('COINBASE_API_KEY')
COINBASE_API_SECRET = os.getenv('COINBASE_API_SECRET')

# Disable Flask's default request logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)  # Only show errors, not requests
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='eventlet',
    logger=False,  # Disabled verbose SocketIO logging
    engineio_logger=False,  # Disabled verbose EngineIO logging
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

def get_coinbase_current_ticker(symbol):
    """
    Get current ticker price from Coinbase REST API.
    Used to build the current forming candle.

    Args:
        symbol: Crypto symbol (e.g., 'BTC-USD', 'ETH-USD')

    Returns:
        float: Current price, or None if request fails
    """
    url = f"https://api.exchange.coinbase.com/products/{symbol}/ticker"

    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()

        # Get the current price from ticker
        price = float(data.get('price', 0))
        return price if price > 0 else None

    except Exception as e:
        # Silently fail - not critical, chart will just be missing current candle
        # print(f"[TICKER] Failed to get current price for {symbol}: {e}")
        return None

def generate_coinbase_jwt(request_method, request_path):
    """
    Generate JWT token for Coinbase Advanced Trade API authentication.

    Args:
        request_method: HTTP method (GET, POST, etc.)
        request_path: API path (e.g., '/api/v3/brokerage/products/BTC-USD/candles')

    Returns:
        JWT token string
    """
    if not COINBASE_API_KEY or not COINBASE_API_SECRET:
        raise ValueError("Coinbase API credentials not found in environment variables")

    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.backends import default_backend

    # Coinbase Advanced Trade API provides a PEM-formatted EC private key
    # The key may have \n escape sequences that need to be converted to actual newlines
    try:
        # Replace \n escape sequences with actual newlines
        pem_key = COINBASE_API_SECRET.replace('\\n', '\n')

        print(f"[JWT] Loading EC private key...")

        # Load the EC private key from PEM format
        private_key = serialization.load_pem_private_key(
            pem_key.encode('utf-8'),
            password=None,
            backend=default_backend()
        )

        print(f"[JWT] Successfully loaded EC private key")

    except Exception as e:
        print(f"[JWT ERROR] Failed to load private key: {e}")
        raise

    # Generate URI (method + host + path)
    uri = f"{request_method} api.coinbase.com{request_path}"

    # Create JWT payload
    payload = {
        'sub': COINBASE_API_KEY,
        'iss': 'coinbase-cloud',
        'nbf': int(time.time()),
        'exp': int(time.time()) + 120,  # Token expires in 2 minutes
        'uri': uri
    }

    # Generate random nonce
    nonce = secrets.token_hex(16)

    # Create JWT with ES256 algorithm (ECDSA with SHA-256)
    token = jwt.encode(
        payload,
        private_key,
        algorithm='ES256',
        headers={'kid': COINBASE_API_KEY, 'nonce': nonce}
    )

    return token

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
    print(f"[COINBASE] fetch_coinbase_candles called: {symbol} {interval} {period}")

    # Map interval to Coinbase Advanced Trade API granularity (string format)
    granularity_map = {
        '1m': 'ONE_MINUTE',
        '5m': 'FIVE_MINUTE',
        '15m': 'FIFTEEN_MINUTE',
        '30m': 'THIRTY_MINUTE',
        '1h': 'ONE_HOUR',
        '2h': 'TWO_HOUR',
        '4h': 'FOUR_HOUR',
        '6h': 'SIX_HOUR',
        '1d': 'ONE_DAY'
    }

    if interval not in granularity_map:
        raise ValueError(f"Interval {interval} not supported for Coinbase")

    granularity = granularity_map[interval]

    # Calculate seconds for max candle calculation
    granularity_seconds_map = {
        'ONE_MINUTE': 60,
        'FIVE_MINUTE': 300,
        'FIFTEEN_MINUTE': 900,
        'THIRTY_MINUTE': 1800,
        'ONE_HOUR': 3600,
        'TWO_HOUR': 7200,
        'FOUR_HOUR': 14400,
        'SIX_HOUR': 21600,
        'ONE_DAY': 86400
    }
    granularity_seconds = granularity_seconds_map[granularity]

    # Calculate time range based on period
    # Coinbase Advanced Trade API limits to 350 candles per request
    end_dt = datetime.utcnow()

    # Calculate max time range based on granularity (350 candles limit for Advanced Trade API)
    max_seconds = granularity_seconds * 350  # 350 candles max
    max_delta = timedelta(seconds=max_seconds)

    period_map = {
        '1d': timedelta(days=1),
        '5d': timedelta(days=5),
        '10d': timedelta(days=10),  # For 30m charts (will be capped to max allowed)
        '20d': timedelta(days=20),  # For 1h charts (will be capped to max allowed)
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

    # Coinbase Advanced Trade API endpoint
    product_id = symbol  # Already in format 'BTC-USD'
    request_path = f"/api/v3/brokerage/products/{product_id}/candles"
    url = f"https://api.coinbase.com{request_path}"

    print(f"[COINBASE] Fetching {interval} ({granularity}) candles for {symbol} from {start_dt} to {end_dt}")

    # Advanced Trade API uses Unix timestamps
    # Convert UTC datetime to Unix timestamp (calendar.timegm handles UTC properly)
    import calendar
    start_ts = int(calendar.timegm(start_dt.timetuple()))
    end_ts = int(calendar.timegm(end_dt.timetuple()))

    params = {
        'granularity': granularity,
        'start': start_ts,
        'end': end_ts
    }

    # Generate JWT token for authentication
    jwt_token = generate_coinbase_jwt('GET', request_path)

    # Set authorization header
    headers = {
        'Authorization': f'Bearer {jwt_token}'
    }

    all_candles = []

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        response_data = response.json()

        # Advanced Trade API returns: {"candles": [...]}
        all_candles = response_data.get('candles', [])

        print(f"[COINBASE] Received {len(all_candles)} candles")

    except requests.exceptions.RequestException as e:
        print(f"[COINBASE ERROR] Failed to fetch candles: {e}")
        print(f"[COINBASE ERROR] Response: {response.text if 'response' in locals() else 'No response'}")
        return []

    # Transform Coinbase Advanced Trade API format to our format
    # Advanced Trade API returns: {"candles": [{"start": "timestamp", "low": "...", "high": "...", "open": "...", "close": "...", "volume": "..."}, ...]}
    # We need: [{Date, Open, High, Low, Close, Volume}, ...]

    transformed = []
    for candle in all_candles:
        # Advanced Trade API returns string timestamps and string values
        timestamp = int(candle['start'])
        dt = datetime.utcfromtimestamp(timestamp)

        transformed.append({
            'Date': dt.strftime("%Y-%m-%d %H:%M:%S"),
            'Open': float(candle['open']),
            'High': float(candle['high']),
            'Low': float(candle['low']),
            'Close': float(candle['close']),
            'Volume': float(candle['volume'])
        })

    # Sort by date (oldest first)
    transformed.sort(key=lambda x: x['Date'])

    # =============================================================================
    # ADD CURRENT FORMING CANDLE
    # Coinbase only returns COMPLETED candles. We need to check if the last candle
    # is still the current period (forming) or if it's complete (past period)
    # =============================================================================
    if len(transformed) > 0:
        try:
            last_candle = transformed[-1]
            last_time = datetime.strptime(last_candle['Date'], "%Y-%m-%d %H:%M:%S")
            now = datetime.utcnow()

            # Calculate which period we're currently in
            # For example, if it's 22:35:09 and interval is 1m, current period is 22:35:00
            # Round down the current time to the start of the current interval period
            epoch = datetime(1970, 1, 1)
            now_seconds = int((now - epoch).total_seconds())
            current_period_seconds = (now_seconds // granularity) * granularity
            current_period_time = epoch + timedelta(seconds=current_period_seconds)

            # print(f"[COINBASE] Check: last={last_candle['Date']}, current_period={current_period_time}, now={now}")

            # If the last candle is NOT the current period, we need to add the current forming candle
            if last_time < current_period_time:
                # Get current ticker price from Coinbase
                print(f"[COINBASE] Last candle is old. Fetching current ticker price for {symbol}...")
                current_price = get_coinbase_current_ticker(symbol)
                print(f"[COINBASE] Ticker price received: ${current_price}")

                if current_price is not None:
                    # Build current forming candle
                    current_candle = {
                        'Date': current_period_time.strftime("%Y-%m-%d %H:%M:%S"),
                        'Open': last_candle['Close'],  # Open at previous close
                        'High': current_price,
                        'Low': current_price,
                        'Close': current_price,
                        'Volume': 0  # Will be updated by WebSocket ticker
                    }

                    transformed.append(current_candle)
                    print(f"[COINBASE] ✅ Added current forming candle at {current_candle['Date']}, price=${current_price}")
                else:
                    print(f"[COINBASE] ❌ Ticker price was None, cannot add current candle")
            else:
                pass  # print(f"[COINBASE] Last candle IS the current period - already have current candle")

        except Exception as e:
            # If anything fails, just return the historical data without current candle
            # This ensures we don't break existing functionality
            print(f"[COINBASE] Failed to add current candle: {e}")
            pass

    # print(f"[COINBASE] Returning {len(transformed)} candles for {symbol} {interval}")
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

def detect_market_type(symbol):
    """
    Detect market type based on symbol format.

    Args:
        symbol: Symbol string (e.g., 'BTC-USD', 'AAPL', 'ETH-USDT')

    Returns:
        str: 'crypto' or 'stock'
    """
    # Crypto symbols typically have a hyphen (BTC-USD, ETH-USDT)
    if '-' in symbol:
        return 'crypto'
    # Stock symbols are typically plain tickers (AAPL, TSLA, GE)
    return 'stock'

@app.route("/data/<symbol>")
def get_chart_data(symbol):
    symbol = symbol.upper()
    start = request.args.get('start')
    end = request.args.get('end')
    period = request.args.get('period')
    interval = request.args.get('interval', '1d')

    # Get market type from query param or auto-detect from symbol
    market_type = request.args.get('market_type', detect_market_type(symbol))

    # CRYPTO: Route to Coinbase for intraday data
    coinbase_intervals = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '1d']

    if market_type == 'crypto' and interval in coinbase_intervals:
        # print(f"[ROUTING] Using Coinbase for crypto {symbol} {interval} period={period}")
        try:
            candles = fetch_coinbase_candles(symbol, interval, period or '1d')
            return jsonify(candles)
        except Exception as e:
            print(f"[COINBASE ERROR] {e}")
            return jsonify([])

    # STOCKS: Always use yfinance
    # CRYPTO: Use yfinance for weekly/monthly intervals (fallback)
    # print(f"[ROUTING] Using yfinance for {market_type} {symbol} {interval}")
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
        # print(f"No data found for {symbol}.")
        return jsonify([])

    # Flatten multi-level columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.reset_index(inplace=True)
    data = data.rename(columns={'Datetime': 'Date'} if 'Datetime' in data.columns else {'Date': 'Date'})
    data['Date'] = data['Date'].dt.strftime("%Y-%m-%d %H:%M:%S") if 'h' in interval or 'm' in interval else data['Date'].dt.strftime("%Y-%m-%d")
    return jsonify(data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].fillna("").to_dict(orient="records"))

def get_current_candle_volume(symbol, interval):
    """
    Get the accumulated volume for the current forming candle by fetching the actual current candle from Coinbase.
    This ensures accurate volume display and uses Coinbase's authoritative candle start time.

    Args:
        symbol: Crypto symbol (e.g., 'BTC', 'ETH' - will be converted to 'BTC-USD')
        interval: Time interval ('1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '1d')

    Returns:
        dict: {'volume': float, 'candle_start_time': ISO8601 string}
    """
    print(f"[CURRENT_CANDLE] Getting current candle volume for {symbol} {interval}")

    # Ensure symbol is in Coinbase format (BTC-USD)
    if not symbol.endswith('-USD'):
        symbol = f"{symbol}-USD"

    # Map interval to Coinbase granularity
    interval_to_granularity = {
        '1m': 'ONE_MINUTE',
        '5m': 'FIVE_MINUTE',
        '15m': 'FIFTEEN_MINUTE',
        '30m': 'THIRTY_MINUTE',
        '1h': 'ONE_HOUR',
        '2h': 'TWO_HOUR',
        '4h': 'FOUR_HOUR',
        '6h': 'SIX_HOUR',
        '1d': 'ONE_DAY'
    }

    if interval not in interval_to_granularity:
        return {'volume': 0, 'candle_start_time': datetime.utcnow().isoformat() + 'Z'}

    granularity = interval_to_granularity[interval]

    # Fetch the most recent candles from Coinbase (including the current forming candle)
    # We'll fetch 2 candles to ensure we get the current one
    now = datetime.utcnow()
    # Go back enough time to get at least 2 candles
    granularity_seconds_map = {
        'ONE_MINUTE': 60,
        'FIVE_MINUTE': 300,
        'FIFTEEN_MINUTE': 900,
        'THIRTY_MINUTE': 1800,
        'ONE_HOUR': 3600,
        'TWO_HOUR': 7200,
        'FOUR_HOUR': 14400,
        'SIX_HOUR': 21600,
        'ONE_DAY': 86400
    }
    granularity_seconds = granularity_seconds_map[granularity]
    start_dt = now - timedelta(seconds=granularity_seconds * 3)  # Go back 3 candle periods

    # Convert to Unix timestamps for API call
    import calendar
    start_ts = int(calendar.timegm(start_dt.timetuple()))
    end_ts = int(calendar.timegm(now.timetuple()))

    # Fetch candles from Coinbase to get the authoritative candle start time
    request_path = f"/api/v3/brokerage/products/{symbol}/candles"
    url = f"https://api.coinbase.com{request_path}"

    params = {
        'granularity': granularity,
        'start': start_ts,
        'end': end_ts
    }

    try:
        # Generate JWT token for authentication
        token = generate_coinbase_jwt('GET', request_path)

        headers = {
            'Authorization': f'Bearer {token}'
        }

        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Get the candles array
        candles = data.get('candles', [])

        if not candles:
            print(f"[CURRENT_CANDLE] No candles returned from Coinbase")
            return {'volume': 0, 'candle_start_time': now.isoformat() + 'Z'}

        # The most recent candle (first in the array for Coinbase API) is the current forming candle
        current_candle = candles[0]
        candle_start_timestamp = int(current_candle['start'])
        candle_volume = float(current_candle['volume'])

        # Convert timestamp to datetime for ISO format
        candle_start_dt = datetime.utcfromtimestamp(candle_start_timestamp)

        print(f"[CURRENT_CANDLE] Coinbase candle: start={candle_start_dt.isoformat()}Z, volume={candle_volume:.4f} BTC")

        return {
            'volume': candle_volume,
            'candle_start_time': candle_start_dt.isoformat() + 'Z'
        }

    except Exception as e:
        print(f"[CURRENT_CANDLE ERROR] Failed to fetch current candle: {e}")
        import traceback
        traceback.print_exc()
        # Return 0 volume on error - better than crashing
        return {
            'volume': 0,
            'candle_start_time': now.isoformat() + 'Z'
        }

@app.route("/current-candle-volume/<symbol>")
def current_candle_volume_endpoint(symbol):
    """
    Flask endpoint to get current candle's accumulated volume.

    Query params:
        interval: Time interval (default: '1m')

    Returns:
        JSON: {'volume': float, 'candle_start_time': ISO8601 string}
    """
    interval = request.args.get('interval', '1m')
    result = get_current_candle_volume(symbol, interval)
    return jsonify(result)

@app.route("/volume", methods=["POST"])
def calculate_volume():
    data = request.get_json()
    symbol = data["symbol"]
    start_date = data["start_date"]
    end_date = data["end_date"]
    interval = data.get("interval", "1d")

    # print(f"Volume request received: {symbol} from {start_date} to {end_date} interval {interval}")

    try:
        start_dt = dateutil.parser.isoparse(start_date)
        end_dt = dateutil.parser.isoparse(end_date)
    except Exception as e:
        # print("Date parsing error:", e)
        return jsonify({"avg_volume": 0})

    if start_dt > end_dt:
        start_dt, end_dt = end_dt, start_dt

    start_str = start_dt.date().strftime("%Y-%m-%d")
    end_str = (end_dt.date() + timedelta(days=1)).strftime("%Y-%m-%d")

    df = yf.download(symbol, start=start_str, end=end_str, interval=interval)

    if df.empty:
        # print(f"No data found in range: {start_str} to {end_str}")
        return jsonify({"avg_volume": 0})

    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[(df.index >= start_dt) & (df.index <= end_dt)]
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    df = df.dropna(subset=["Volume"])
    avg_volume = df["Volume"].mean() if not df.empty else 0

    # print(f"Avg volume for {symbol} between {start_date} and {end_date}: {avg_volume:.2f}")
    return jsonify({"avg_volume": avg_volume})

# ========================================
# TICK BAR ENDPOINTS
# ========================================

@app.route("/data/tick/<symbol>/<int:threshold>", methods=["GET"])
def get_tick_bars(symbol, threshold):
    """Load historical tick bars from file"""
    symbol = symbol.upper()
    tick_file = os.path.join(DATA_DIR, "tick_bars", f"tick_{threshold}_{symbol}.json")

    # print(f"[TICK] Loading tick bars: {symbol} threshold={threshold}")

    if not os.path.exists(tick_file):
        # print(f"[TICK] No tick bar file found, returning empty array")
        return jsonify([])

    try:
        with open(tick_file, 'r') as f:
            bars = json.load(f)

        # Return only last 300 bars for performance
        bars = bars[-300:] if len(bars) > 300 else bars
        # print(f"[TICK] Loaded {len(bars)} tick bars for {symbol}")
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

        # print(f"[TICK] Saved bar for {symbol} threshold={threshold}, total bars={len(bars)}")
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
        # print("Missing symbol or line ID")
        return jsonify({"error": "Missing symbol or line ID"}), 400

    filename = os.path.join(DATA_DIR, f"lines_{symbol}.json")
    lines = []

    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                lines = json.load(f)
        except Exception as e:
            pass  # print("Error reading line file:", e)

    lines = [l for l in lines if l["id"] != line["id"]]
    lines.append(line)

    try:
        with open(filename, "w") as f:
            json.dump(lines, f)
        # print(f"Saved line for {symbol}: {line['id']}")
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
        # print("Missing symbol or line ID for deletion")
        return jsonify({"error": "Missing symbol or line ID"}), 400

    filename = os.path.join(DATA_DIR, f"lines_{symbol}.json")
    if not os.path.exists(filename):
        # print(f"No line file found for {symbol}")
        return jsonify({"success": True})

    try:
        with open(filename, "r") as f:
            lines = json.load(f)
        lines = [l for l in lines if l["id"] != line_id]
        with open(filename, "w") as f:
            json.dump(lines, f)
        # print(f"Deleted line {line_id} for {symbol}")
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
        # print(f"Cleared all lines for {symbol}")
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
            # print(f"Loaded {len(lines)} lines for {symbol}")
            return jsonify(lines)
        except Exception as e:
            print("Error loading lines:", e)
            return jsonify([])
    else:
        # print(f"No line file found for {symbol}")
        return jsonify([])

# ==================== DRAWING PERSISTENCE ENDPOINTS ====================
# Universal endpoints for all drawing types (trend lines, dots, arrows, etc.)

@app.route("/save_drawing", methods=["POST"])
def save_drawing():
    """Save a single drawing (trend line, dot, arrow, etc.)"""
    data = request.get_json()
    symbol = data.get("symbol", "").upper()
    drawing = data.get("drawing")

    if not symbol or not drawing or "id" not in drawing:
        # print("Missing symbol or drawing ID")
        return jsonify({"error": "Missing symbol or drawing ID"}), 400

    filename = os.path.join(DATA_DIR, f"drawings_{symbol}.json")
    drawings = []

    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                drawings = json.load(f)
        except Exception as e:
            pass  # print("Error reading drawings file:", e)

    # Remove old version of this drawing (if exists) and add new version
    drawings = [d for d in drawings if d["id"] != drawing["id"]]
    drawings.append(drawing)

    try:
        with open(filename, "w") as f:
            json.dump(drawings, f, indent=2)
        # print(f"Saved drawing for {symbol}: {drawing['id']} (type: {drawing.get('action', 'unknown')})")
        return jsonify({"success": True})
    except Exception as e:
        print("Error saving drawing:", e)
        return jsonify({"error": "Failed to save drawing"}), 500

@app.route("/drawings/<symbol>")
def get_drawings(symbol):
    """Load all drawings for a symbol"""
    symbol = symbol.upper()
    filename = os.path.join(DATA_DIR, f"drawings_{symbol}.json")

    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                drawings = json.load(f)
            # print(f"Loaded {len(drawings)} drawings for {symbol}")
            return jsonify(drawings)
        except Exception as e:
            print("Error loading drawings:", e)
            return jsonify([])
    else:
        # print(f"No drawings file found for {symbol}")
        return jsonify([])

@app.route("/delete_drawing", methods=["POST"])
def delete_drawing():
    """Delete a single drawing by ID"""
    data = request.get_json()
    symbol = data.get("symbol", "").upper()
    drawing_id = data.get("drawing_id")

    if not symbol or not drawing_id:
        # print("Missing symbol or drawing ID for deletion")
        return jsonify({"error": "Missing symbol or drawing ID"}), 400

    filename = os.path.join(DATA_DIR, f"drawings_{symbol}.json")
    if not os.path.exists(filename):
        # print(f"No drawings file found for {symbol}")
        return jsonify({"success": True})

    try:
        with open(filename, "r") as f:
            drawings = json.load(f)
        drawings = [d for d in drawings if d["id"] != drawing_id]
        with open(filename, "w") as f:
            json.dump(drawings, f, indent=2)
        # print(f"Deleted drawing {drawing_id} for {symbol}")
        return jsonify({"success": True})
    except Exception as e:
        print("Error deleting drawing:", e)
        return jsonify({"error": "Failed to delete drawing"}), 500

@app.route("/save_all_drawings", methods=["POST"])
def save_all_drawings():
    """Replace all drawings for a symbol (used for cleanup)"""
    data = request.get_json()
    symbol = data.get("symbol", "").upper()
    drawings = data.get("drawings", [])

    if not symbol:
        # print("Missing symbol for save_all_drawings")
        return jsonify({"error": "Missing symbol"}), 400

    filename = os.path.join(DATA_DIR, f"drawings_{symbol}.json")

    try:
        with open(filename, "w") as f:
            json.dump(drawings, f, indent=2)
        # print(f"Saved {len(drawings)} drawings for {symbol} (full replace)")
        return jsonify({"success": True})
    except Exception as e:
        print("Error saving all drawings:", e)
        return jsonify({"error": "Failed to save drawings"}), 500

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
    # print(f'[CONNECT] Client connected: {request.sid}')
    emit('connection_response', {'status': 'connected'})

    # Start ticker emit worker on first client connection
    if not ticker_worker_started:
        socketio.start_background_task(ticker_emit_worker)
        ticker_worker_started = True
        # print('[TICKER WORKER] Starting background task')

    # Start trade emit worker on first client connection
    if not trade_worker_started:
        socketio.start_background_task(trade_emit_worker)
        trade_worker_started = True
        # print('[TRADE WORKER] Starting background task')

@socketio.on('disconnect')
def handle_disconnect():
    pass  # print(f'[DISCONNECT] Client disconnected: {request.sid}')

@socketio.on('subscribe')
def handle_subscribe(data):
    """Subscribe to real-time updates for symbols"""
    symbols = data.get('symbols', [])
    if symbols:
        was_empty = len(subscribed_symbols) == 0
        subscribed_symbols.update(symbols)
        # print(f'[SUBSCRIBE] Subscribed to symbols: {symbols}')
        # print(f'[SUBSCRIBE] Total subscribed symbols: {len(subscribed_symbols)}')

        # Start Coinbase WebSocket if not already running
        if coinbase_ws is None:
            # print(f'[COINBASE] Starting WebSocket (first connection)')
            start_coinbase_websocket()
        else:
            # Update existing subscription
            # print(f'[COINBASE] Updating subscriptions')
            resubscribe_coinbase_ws()

@socketio.on('subscribe_ticker')
def handle_subscribe_ticker(data):
    """Subscribe to real-time updates for a single symbol"""
    symbol = data.get('symbol')
    if symbol:
        was_empty = len(subscribed_symbols) == 0
        subscribed_symbols.add(symbol)
        # print(f'[SUBSCRIBE_TICKER] Subscribed to symbol: {symbol}')
        # print(f'[SUBSCRIBE_TICKER] Total subscribed symbols: {len(subscribed_symbols)}')

        # Start Coinbase WebSocket if not already running
        if coinbase_ws is None:
            # print(f'[COINBASE] Starting WebSocket (first connection)')
            start_coinbase_websocket()
        else:
            # Update existing subscription
            # print(f'[COINBASE] Updating subscriptions')
            resubscribe_coinbase_ws()

@socketio.on('unsubscribe')
def handle_unsubscribe(data):
    """Unsubscribe from symbol updates"""
    symbols = data.get('symbols', [])
    if symbols:
        subscribed_symbols.difference_update(symbols)
        # print(f'Unsubscribed from symbols: {symbols}')
        # Update Coinbase WebSocket subscriptions
        if coinbase_ws:
            resubscribe_coinbase_ws()

def on_coinbase_message(ws, message):
    """Handle incoming Coinbase WebSocket messages"""
    try:
        data = json.loads(message)

        # Debug: Log message types to see what Coinbase is sending
        # msg_type = data.get('type', 'unknown')
        # if msg_type not in ['ticker', 'subscriptions', 'heartbeat']:
        #     print(f'[COINBASE MSG TYPE] {msg_type}')

        # Handle ticker updates
        if data.get('type') == 'ticker':
            product_id = data.get('product_id', '')
            price = float(data.get('price', 0))
            best_bid = float(data.get('best_bid', 0))
            best_ask = float(data.get('best_ask', 0))

            if price > 0:
                ticker_data = {
                    'symbol': product_id,
                    'price': price,
                    'bid': best_bid,
                    'ask': best_ask,
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
                # if price and int(price) % 10 == 0:
                #     print(f'[COINBASE -> QUEUE] {product_id}: ${price:.2f} -> added to queue')

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
                # print(f'[TRADE] {product_id}: ${price} x {size} ({side})')
    except Exception as e:
        print(f'[ERROR] Processing Coinbase message: {e}')

def on_coinbase_error(ws, error):
    """Handle Coinbase WebSocket errors"""
    print(f'[COINBASE ERROR] {error}')

def on_coinbase_close(ws, close_status_code, close_msg):
    """Handle Coinbase WebSocket close"""
    # print(f'[COINBASE] Connection closed: {close_status_code} - {close_msg}')
    # Attempt to reconnect after 5 seconds
    eventlet.sleep(5)
    start_coinbase_websocket()

def on_coinbase_open(ws):
    """Handle Coinbase WebSocket open"""
    # print('[COINBASE] WebSocket connected')
    # Subscribe to all symbols
    subscribe_message = {
        "type": "subscribe",
        "product_ids": list(subscribed_symbols),
        "channels": ["ticker", "matches"]
    }
    ws.send(json.dumps(subscribe_message))
    # print(f'[COINBASE] Subscribed to: {list(subscribed_symbols)} (ticker + matches)')

def start_coinbase_websocket():
    """Start Coinbase Advanced Trade WebSocket connection"""
    global coinbase_ws

    if len(subscribed_symbols) == 0:
        # print('[COINBASE] No symbols to subscribe, skipping WebSocket')
        return

    # print('[COINBASE] Starting WebSocket connection...')

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
        # Check if WebSocket is still connected before trying to send
        if not hasattr(coinbase_ws, 'sock') or coinbase_ws.sock is None or not coinbase_ws.sock.connected:
            # print('[COINBASE] WebSocket not connected, skipping resubscribe')
            return

        subscribe_message = {
            "type": "subscribe",
            "product_ids": list(subscribed_symbols),
            "channels": ["ticker", "matches"]
        }
        try:
            coinbase_ws.send(json.dumps(subscribe_message))
            # print(f'[COINBASE] Updated subscriptions: {list(subscribed_symbols)} (ticker + matches)')
        except Exception as e:
            # Silently ignore errors when socket is closed (expected during reconnection)
            if 'already closed' not in str(e).lower():
                print(f'[COINBASE ERROR] Failed to update subscriptions: {e}')

def ticker_emit_worker():
    """Background task that pulls ticker data from queue and emits to Socket.IO clients"""
    # print('[TICKER WORKER] Started - pulling from queue and emitting to clients')
    while True:
        try:
            # Block until ticker data is available (with timeout to allow graceful shutdown)
            ticker_data = ticker_queue.get(timeout=1)

            # Emit to all connected Socket.IO clients (broadcasts by default in background task)
            socketio.emit('ticker_update', ticker_data)

            # Debug logging (reduced frequency)
            # if ticker_data['price'] and int(ticker_data['price']) % 10 == 0:
            #     print(f'[EMIT] {ticker_data["symbol"]}: ${ticker_data["price"]:.2f}')

        except eventlet.queue.Empty:
            # No ticker data available, continue loop
            continue
        except Exception as e:
            print(f'[TICKER WORKER ERROR] {e}')
            eventlet.sleep(0.1)  # Brief pause on error

def trade_emit_worker():
    """Background task that pulls trade data from queue and emits to Socket.IO clients"""
    # print('[TRADE WORKER] Started - pulling from queue and emitting to clients')
    while True:
        try:
            # Block until trade data is available (with timeout to allow graceful shutdown)
            trade_data = trade_queue.get(timeout=1)

            # Emit to all connected Socket.IO clients (broadcasts by default in background task)
            socketio.emit('trade_update', trade_data)

            # Debug logging (reduced frequency) - log every 100th trade
            # if trade_data.get('trade_id') and trade_data.get('trade_id') % 100 == 0:
            #     print(f'[EMIT TRADE] {trade_data["symbol"]}: ${trade_data["price"]}')

        except eventlet.queue.Empty:
            # No trade data available, continue loop
            continue
        except Exception as e:
            print(f'[TRADE WORKER ERROR] {e}')
            eventlet.sleep(0.1)  # Brief pause on error

# Old yfinance polling removed - now using real Coinbase WebSocket for tick-by-tick updates

# =============================================================================
# ORD VOLUME ENDPOINTS (COMPLETELY SEGREGATED - NO SHARED CODE)
# =============================================================================

# Segregated data directory for ORD Volume
ORD_VOLUME_DATA_DIR = os.path.join(DATA_DIR, 'ord-volume')
os.makedirs(ORD_VOLUME_DATA_DIR, exist_ok=True)

def get_ord_volume_file_path_segregated(symbol):
    """Get file path for ORD Volume analysis (segregated function)"""
    filename = f"ord_volume_{symbol}.json"
    return os.path.join(ORD_VOLUME_DATA_DIR, filename)

@app.route('/ord-volume/save', methods=['POST'])
def save_ord_volume_segregated():
    """Save ORD Volume analysis (completely segregated endpoint)"""
    try:
        data = request.get_json()
        if not data or 'symbol' not in data or 'analysis' not in data:
            return jsonify({'error': 'Missing required fields'}), 400

        symbol = data['symbol']
        analysis = data['analysis']
        file_path = get_ord_volume_file_path_segregated(symbol)

        with open(file_path, 'w') as f:
            json.dump({'symbol': symbol, 'analysis': analysis}, f, indent=2)

        return jsonify({'success': True, 'message': f'ORD Volume analysis saved for {symbol}'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ord-volume/load/<symbol>', methods=['GET'])
def load_ord_volume_segregated(symbol):
    """Load ORD Volume analysis (completely segregated endpoint)"""
    try:
        file_path = get_ord_volume_file_path_segregated(symbol)
        if not os.path.exists(file_path):
            return jsonify({'error': f'No ORD Volume analysis found for {symbol}'}), 404

        with open(file_path, 'r') as f:
            data = json.load(f)

        return jsonify(data), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ord-volume/delete/<symbol>', methods=['DELETE'])
def delete_ord_volume_segregated(symbol):
    """Delete ORD Volume analysis (completely segregated endpoint)"""
    try:
        file_path = get_ord_volume_file_path_segregated(symbol)
        if not os.path.exists(file_path):
            return jsonify({'error': f'No ORD Volume analysis found for {symbol}'}), 404

        os.remove(file_path)
        return jsonify({'success': True, 'message': f'ORD Volume analysis deleted for {symbol}'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ord-volume/list', methods=['GET'])
def list_ord_volume_segregated():
    """List all symbols with ORD Volume analyses (completely segregated endpoint)"""
    try:
        files = os.listdir(ORD_VOLUME_DATA_DIR)
        symbols = []
        for filename in files:
            if filename.startswith('ord_volume_') and filename.endswith('.json'):
                symbol = filename.replace('ord_volume_', '').replace('.json', '')
                symbols.append(symbol)
        return jsonify({'symbols': symbols, 'count': len(symbols)}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =============================================================================
# END OF ORD VOLUME ENDPOINTS
# =============================================================================

# =============================================================================
# ML ENDPOINTS - Triad Trend Pulse Pivot Reliability
# =============================================================================

# Lazy import ML module to avoid errors if PyTorch not installed
_ml_model = None

def get_ml_model():
    """Get singleton ML model instance"""
    global _ml_model
    if _ml_model is None:
        try:
            from ml.pivot_model import get_model
            _ml_model = get_model()
            print("[ML] Model loaded successfully")
        except Exception as e:
            print(f"[ML] Model not available: {e}")
            _ml_model = False  # Mark as unavailable
    return _ml_model if _ml_model is not False else None


@app.route("/ml/pivot-reliability", methods=["POST"])
def ml_pivot_reliability():
    """
    ML endpoint for pivot reliability prediction

    Request body:
    {
        "features": [9 float values],  # Single pivot
        OR
        "features": [[9 values], [9 values], ...]  # Multiple pivots
    }

    Response:
    {
        "scores": [0.85, 0.72, ...],  # Probability scores
        "count": 2
    }
    """
    try:
        # Get ML model
        model = get_ml_model()

        if model is None:
            return jsonify({
                'error': 'ML model not available. Train model first using train_pivot_model.py'
            }), 503

        # Parse request
        data = request.get_json()

        if 'features' not in data:
            return jsonify({'error': 'Missing "features" in request body'}), 400

        features = data['features']

        # Convert to numpy array
        import numpy as np
        features = np.array(features, dtype=np.float32)

        # Validate shape
        if features.ndim == 1:
            # Single pivot
            if features.shape[0] != 9:
                return jsonify({'error': f'Expected 9 features, got {features.shape[0]}'}), 400
            features = features.reshape(1, -1)
        elif features.ndim == 2:
            # Multiple pivots
            if features.shape[1] != 9:
                return jsonify({'error': f'Expected 9 features per pivot, got {features.shape[1]}'}), 400
        else:
            return jsonify({'error': 'Invalid feature array shape'}), 400

        # Run inference
        scores = model.predict_proba(features)

        return jsonify({
            'scores': scores.tolist(),
            'count': len(scores)
        }), 200

    except Exception as e:
        print(f"[ML ERROR] Pivot reliability prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route("/ml/model-info", methods=["GET"])
def ml_model_info():
    """Get information about the loaded ML model"""
    try:
        model = get_ml_model()

        if model is None:
            return jsonify({
                'loaded': False,
                'message': 'ML model not available'
            }), 200

        return jsonify({
            'loaded': True,
            'architecture': '9 -> 32 -> 16 -> 1',
            'input_features': 9,
            'output': 'pivot_reliability_score',
            'model_type': 'PyTorch MLP'
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =============================================================================
# END OF ML ENDPOINTS
# =============================================================================

if __name__ == "__main__":
    # Use socketio.run instead of app.run for WebSocket support
    # Ticker emit worker will start automatically when first client connects
    print("Flask server starting on http://127.0.0.1:5000")
    print("Logging: QUIET MODE (errors only)")
    socketio.run(app, debug=False, host='127.0.0.1', port=5000, allow_unsafe_werkzeug=True, log_output=False)
