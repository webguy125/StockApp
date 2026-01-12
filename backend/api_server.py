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

# Initialize ML Automation Service (built-in scheduler)
try:
    from trading_system.ml_automation_service import (
        init_automation_service,
        start_automation,
        stop_automation,
        get_automation_status,
        run_daily_cycle
    )
    ML_AUTOMATION_AVAILABLE = True
except ImportError as e:
    print(f"[TURBO AUTOMATION] Not available: {e}")
    ML_AUTOMATION_AVAILABLE = False

# Initialize ML Model Manager (multi-model configuration system)
try:
    from trading_system.ml_model_manager import MLModelManager
    ml_model_manager = MLModelManager()
    ML_MODEL_MANAGER_AVAILABLE = True
except ImportError as e:
    print(f"[TURBO MODEL MANAGER] Not available: {e}")
    ML_MODEL_MANAGER_AVAILABLE = False

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

@app.route("/heatmap")
def serve_heatmap():
    response = make_response(send_from_directory(FRONTEND_DIR, "heatmap.html"))
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route("/turbomode.html")
@app.route("/turbomode")
def serve_turbomode():
    response = make_response(send_from_directory(FRONTEND_DIR, "turbomode.html"))
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route("/turbomode/<path:filename>")
def serve_turbomode_pages(filename):
    response = make_response(send_from_directory(os.path.join(FRONTEND_DIR, "turbomode"), filename))
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

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
    # Yahoo Finance doesn't support 2h, 6h, 45m, 2m, 3m, 10m, 3h intervals for stocks, so we aggregate from smaller intervals
    unsupported_intervals = {'2h': '1h', '3h': '1h', '6h': '1h', '2m': '1m', '3m': '1m', '10m': '5m', '45m': '15m'}
    source_interval = unsupported_intervals.get(interval, interval)

    # print(f"[ROUTING] Using yfinance for {market_type} {symbol} {interval}")
    kwargs = {'interval': source_interval}
    if start and end:
        kwargs['start'] = start
        kwargs['end'] = end
    elif period:
        kwargs['period'] = period
    else:
        kwargs['period'] = 'max'

    try:
        data = yf.download(symbol, **kwargs)
    except Exception as e:
        print(f"[YFINANCE ERROR] Failed to download {symbol} {interval}: {e}")
        return jsonify([])

    if data.empty:
        # print(f"No data found for {symbol}.")
        return jsonify([])

    # Flatten multi-level columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Aggregate data if we fetched a smaller interval than requested
    if interval in unsupported_intervals:
        print(f"[AGGREGATION] Building {interval} candles from {source_interval} data for {symbol}")

        # Determine resampling rule for pandas
        # Hours: '2h' -> '2h', '6h' -> '6h'
        # Minutes: '45m' -> '45min' (min for minutes in pandas 2.0+)
        if interval.endswith('m'):
            # Extract number and use 'min' for minutes
            minutes = interval[:-1]  # '45m' -> '45'
            resample_rule = f"{minutes}min"  # '45min'
        else:
            # Hours: lowercase 'h'
            resample_rule = interval  # '2h' stays '2h'

        print(f"[AGGREGATION] Resample rule: {resample_rule}")

        # Ensure datetime is in the index for resampling
        if 'Date' in data.columns or 'Datetime' in data.columns:
            date_col = 'Datetime' if 'Datetime' in data.columns else 'Date'
            data = data.set_index(date_col)

        # Resample to create larger candles
        # No need for 'on' parameter when datetime is already the index
        data_resampled = data.resample(resample_rule).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()

        data = data_resampled.reset_index()

        # Ensure the datetime column is named 'Date' for consistency
        if 'Datetime' in data.columns:
            data = data.rename(columns={'Datetime': 'Date'})

    # Ensure we have a Date column from the index
    if 'Date' not in data.columns:
        if data.index.name in ['Date', 'Datetime'] or isinstance(data.index, pd.DatetimeIndex):
            data = data.reset_index()
            # Rename the index column to 'Date'
            if 'index' in data.columns:
                data = data.rename(columns={'index': 'Date'})
            elif 'Datetime' in data.columns:
                data = data.rename(columns={'Datetime': 'Date'})
            elif data.columns[0] not in ['Open', 'High', 'Low', 'Close', 'Volume']:
                # First column is probably the datetime
                data = data.rename(columns={data.columns[0]: 'Date'})

    # Format dates based on interval type - intraday intervals need time
    intraday_intervals = ['1m', '2m', '3m', '5m', '10m', '15m', '30m', '45m', '60m', '90m', '1h', '2h', '3h', '4h', '6h']
    if interval in intraday_intervals:
        data['Date'] = pd.to_datetime(data['Date']).dt.strftime("%Y-%m-%d %H:%M:%S")
    else:
        data['Date'] = pd.to_datetime(data['Date']).dt.strftime("%Y-%m-%d")

    return jsonify(data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].fillna("").to_dict(orient="records"))

def get_current_candle_volume(symbol, interval):
    """
    Get the accumulated volume for the current forming candle.
    For crypto: fetches from Coinbase API
    For stocks: fetches from Yahoo Finance

    Args:
        symbol: Stock or crypto symbol (e.g., 'AAPL', 'BTC', 'BTC-USD')
        interval: Time interval ('1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '1d')

    Returns:
        dict: {'volume': float, 'open': float, 'high': float, 'low': float, 'close': float, 'candle_start_time': ISO8601 string}
    """
    print(f"[CURRENT_CANDLE] Getting current candle data for {symbol} {interval}")

    # Determine if this is a crypto symbol
    # Crypto symbols either end with -USD or are in our known crypto list
    known_crypto = ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'ADA', 'AVAX', 'DOT', 'LINK', 'LTC',
                   'MATIC', 'UNI', 'BNB', 'USDT', 'USDC']

    is_crypto = symbol.endswith('-USD') or symbol in known_crypto

    # For stocks, use Yahoo Finance to get current volume
    if not is_crypto:
        try:
            ticker = yf.Ticker(symbol)
            # Get today's data with the specified interval
            if interval == '1d':
                hist = ticker.history(period='1d')
            else:
                hist = ticker.history(period='1d', interval=interval)

            if not hist.empty:
                # Get the last row which is the current candle
                last_row = hist.iloc[-1]
                current_time = hist.index[-1]

                print(f"[CURRENT_CANDLE] Yahoo Finance data for {symbol}: O={last_row['Open']:.2f} H={last_row['High']:.2f} L={last_row['Low']:.2f} C={last_row['Close']:.2f} V={last_row['Volume']}")

                return {
                    'volume': float(last_row['Volume']),
                    'open': float(last_row['Open']),
                    'high': float(last_row['High']),
                    'low': float(last_row['Low']),
                    'close': float(last_row['Close']),
                    'candle_start_time': current_time.isoformat() + 'Z'
                }
            else:
                print(f"[CURRENT_CANDLE ERROR] No data returned from Yahoo Finance for {symbol}")
                return {'volume': 0, 'candle_start_time': datetime.utcnow().isoformat() + 'Z'}
        except Exception as e:
            print(f"[CURRENT_CANDLE ERROR] Failed to fetch from Yahoo Finance: {e}")
            return {'volume': 0, 'candle_start_time': datetime.utcnow().isoformat() + 'Z'}

    # For crypto, use Coinbase API
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
        candle_open = float(current_candle['open'])
        candle_high = float(current_candle['high'])
        candle_low = float(current_candle['low'])
        candle_close = float(current_candle['close'])

        # Convert timestamp to datetime for ISO format
        candle_start_dt = datetime.utcfromtimestamp(candle_start_timestamp)

        print(f"[CURRENT_CANDLE] Coinbase data for {symbol}: O={candle_open:.2f} H={candle_high:.2f} L={candle_low:.2f} C={candle_close:.2f} V={candle_volume:.4f}")

        return {
            'volume': candle_volume,
            'open': candle_open,
            'high': candle_high,
            'low': candle_low,
            'close': candle_close,
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
_ml_model_stock = None
_ml_model_crypto = None

def get_ml_model(symbol=None):
    """
    Get appropriate ML model instance based on symbol type

    Args:
        symbol: Trading symbol (optional). If provided, will auto-detect stock vs crypto

    Returns:
        tuple: (model, model_type) or (None, None) if unavailable
    """
    global _ml_model_stock, _ml_model_crypto

    try:
        from ml.pivot_model import get_model, is_crypto_symbol

        # Detect model type
        if symbol and is_crypto_symbol(symbol):
            model_type = 'crypto'
            if _ml_model_crypto is None:
                _ml_model_crypto = get_model(model_type='crypto')
                print(f"[ML] Crypto model loaded successfully")
            return _ml_model_crypto, 'crypto'
        else:
            model_type = 'stock'
            if _ml_model_stock is None:
                _ml_model_stock = get_model(model_type='stock')
                print(f"[ML] Stock model loaded successfully")
            return _ml_model_stock, 'stock'

    except Exception as e:
        print(f"[ML] Model not available: {e}")
        return None, None


@app.route("/ml/pivot-reliability", methods=["POST"])
def ml_pivot_reliability():
    """
    ML endpoint for pivot reliability prediction

    Request body:
    {
        "symbol": "AAPL" or "BTC-USD",  # Required for model selection
        "features": [9 float values],  # Single pivot
        OR
        "features": [[9 values], [9 values], ...]  # Multiple pivots
    }

    Response:
    {
        "scores": [0.85, 0.72, ...],  # Probability scores
        "count": 2,
        "model_type": "stock" or "crypto"
    }
    """
    try:
        # Parse request
        data = request.get_json()

        if 'features' not in data:
            return jsonify({'error': 'Missing "features" in request body'}), 400

        if 'symbol' not in data:
            return jsonify({'error': 'Missing "symbol" in request body'}), 400

        symbol = data['symbol']
        features = data['features']

        # Get appropriate ML model based on symbol
        model, model_type = get_ml_model(symbol=symbol)

        if model is None:
            return jsonify({
                'error': f'ML model not available for {model_type}. Train model first using train_pivot_model.py or train_pivot_model_crypto.py'
            }), 503

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
            'count': len(scores),
            'model_type': model_type
        }), 200

    except Exception as e:
        print(f"[TURBO ERROR] Pivot reliability prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route("/ml/model-info", methods=["GET"])
def ml_model_info():
    """Get information about the loaded ML models"""
    try:
        # Check both models
        stock_model, _ = get_ml_model(symbol='AAPL')  # Test with stock symbol
        crypto_model, _ = get_ml_model(symbol='BTC-USD')  # Test with crypto symbol

        return jsonify({
            'stock_model': {
                'loaded': stock_model is not None,
                'architecture': '9 -> 32 -> 16 -> 1',
                'input_features': 9,
                'output': 'pivot_reliability_score',
                'model_type': 'PyTorch MLP',
                'trained_on': ['AAPL', 'TSLA', 'NVDA', 'SPY', 'QQQ']
            },
            'crypto_model': {
                'loaded': crypto_model is not None,
                'architecture': '9 -> 32 -> 16 -> 1',
                'input_features': 9,
                'output': 'pivot_reliability_score',
                'model_type': 'PyTorch MLP',
                'trained_on': ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'DOGE-USD', 'MATIC-USD']
            }
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =============================================================================
# END OF ML ENDPOINTS
# =============================================================================

# =============================================================================
# AGENT SIGNALS API (for multi-agent trading system)
# =============================================================================

@app.route("/agent-signals", methods=["GET"])
def get_agent_signals():
    """
    Get the latest signals from the multi-agent trading system.
    Returns fusion output with buy/sell recommendations for all symbols.
    """
    try:
        # Path to the agent repository
        agent_repo = os.path.join(BASE_DIR, "..", "agents", "repository")

        # Load fusion output (contains all agent signals)
        fusion_file = os.path.join(agent_repo, "fusion_output.json")
        if not os.path.exists(fusion_file):
            return jsonify({
                "status": "no_data",
                "message": "No agent signals available. Run scanner and fusion agents first.",
                "signals": []
            }), 200

        with open(fusion_file, 'r', encoding='utf-8') as f:
            fusion_data = json.load(f)

        # Extract top opportunities and format for heatmap
        signals = []

        # Process all fusions (not just top opportunities)
        all_fusions = fusion_data.get('all_fusions', [])

        for fusion in all_fusions:
            signal = {
                'symbol': fusion['symbol'],
                'score': fusion['total_score'],
                'confidence': fusion['confidence'],
                'recommendation': fusion['recommendation'],
                'emoji': fusion.get('emoji_string', ''),
                'timestamp': fusion.get('timestamp', ''),

                # Color coding for heatmap
                'color': get_signal_color(fusion['recommendation'], fusion['total_score']),
                'intensity': min(100, int(fusion['confidence'] * 100))  # 0-100 intensity
            }
            signals.append(signal)

        # Sort by score * confidence (strongest signals first)
        signals.sort(key=lambda x: x['score'] * x['confidence'], reverse=True)

        return jsonify({
            "status": "success",
            "timestamp": fusion_data.get('timestamp', ''),
            "total_signals": len(signals),
            "signals": signals,
            "summary": fusion_data.get('recommendations', {})
        }), 200

    except Exception as e:
        app.logger.error(f"Error getting agent signals: {e}")
        return jsonify({"error": str(e)}), 500


def get_signal_color(recommendation, score):
    """Convert recommendation to color for heatmap"""
    if recommendation == 'strong_buy':
        return '#00ff00'  # Bright green
    elif recommendation == 'buy':
        return '#90ee90'  # Light green
    elif recommendation == 'hold':
        return '#ffff00'  # Yellow
    elif recommendation == 'sell':
        return '#ffb6c1'  # Light red
    elif recommendation == 'strong_sell':
        return '#ff0000'  # Bright red
    else:
        return '#808080'  # Gray for unknown


@app.route("/agent-signals/refresh", methods=["POST"])
def refresh_agent_signals():
    """
    Trigger the agent system to refresh signals
    (Runs scanner -> fusion pipeline)
    """
    try:
        import subprocess
        import sys

        # Path to agents
        agent_path = os.path.join(BASE_DIR, "..", "agents")

        # Run scanner agent
        scanner_result = subprocess.run(
            [sys.executable, os.path.join(agent_path, "scanner_agent.py")],
            capture_output=True,
            text=True,
            cwd=agent_path
        )

        if scanner_result.returncode != 0:
            return jsonify({
                "status": "error",
                "message": f"Scanner failed: {scanner_result.stderr}"
            }), 500

        # Run fusion agent
        fusion_result = subprocess.run(
            [sys.executable, os.path.join(agent_path, "fusion_agent.py")],
            capture_output=True,
            text=True,
            cwd=agent_path
        )

        if fusion_result.returncode != 0:
            return jsonify({
                "status": "error",
                "message": f"Fusion failed: {fusion_result.stderr}"
            }), 500

        return jsonify({
            "status": "success",
            "message": "Agent signals refreshed successfully"
        }), 200

    except Exception as e:
        app.logger.error(f"Error refreshing agent signals: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/heatmap-data", methods=["GET"])
def get_heatmap_data():
    """
    Get heat map data categorized by timeframe (intraday, daily, monthly).
    Returns dynamic heat maps from the agent learning loop.

    Query params:
        - timeframe: Optional filter ('intraday', 'daily', 'monthly', 'all')
    """
    try:
        timeframe_filter = request.args.get('timeframe', 'all').lower()

        # Path to the agent repository
        agent_repo = os.path.join(BASE_DIR, "..", "agents", "repository")

        # Load fusion output (contains all agent signals)
        fusion_file = os.path.join(agent_repo, "fusion_output.json")
        if not os.path.exists(fusion_file):
            return jsonify({
                "status": "no_data",
                "message": "No heat map data available. Run scanner and fusion agents first.",
                "heatmaps": {
                    'intraday': [],
                    'daily': [],
                    'monthly': []
                }
            }), 200

        with open(fusion_file, 'r', encoding='utf-8') as f:
            fusion_data = json.load(f)

        # Get all fusions
        all_fusions = fusion_data.get('all_fusions', [])

        # Categorize signals by timeframe
        # Based on signal characteristics and confidence levels
        heatmaps = {
            'intraday': [],
            'daily': [],
            'monthly': [],
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'source': 'agent_learning_loop',
                'version': '1.0',
                'total_signals': len(all_fusions)
            }
        }

        for fusion in all_fusions:
            signal_data = {
                'symbol': fusion['symbol'],
                'score': fusion['total_score'],
                'confidence': fusion['confidence'],
                'recommendation': fusion['recommendation'],
                'emoji': fusion.get('emoji_string', ''),
                'timestamp': fusion.get('timestamp', ''),
                'color': get_signal_color(fusion['recommendation'], fusion['total_score']),
                'intensity': min(100, int(fusion['confidence'] * 100)),

                # Add agent breakdown for detailed view
                'agent_scores': fusion.get('weighted_scores', {}),
                'components': fusion.get('components', {}),
                'metadata': fusion.get('metadata', {})
            }

            # Categorize based on signal characteristics
            # Intraday: High confidence (>=0.7), high score (>=65), suitable for quick trades
            if fusion['confidence'] >= 0.7 and fusion['total_score'] >= 65:
                heatmaps['intraday'].append(signal_data.copy())

            # Daily: Medium-to-high confidence (>=0.55), good score (>=55), suitable for swing trades
            if fusion['confidence'] >= 0.55 and fusion['total_score'] >= 55:
                heatmaps['daily'].append(signal_data.copy())

            # Monthly: All signals with reasonable confidence (>=0.4), suitable for position trades
            if fusion['confidence'] >= 0.4 and fusion['total_score'] >= 45:
                heatmaps['monthly'].append(signal_data.copy())

        # Sort each timeframe by score * confidence (strongest signals first)
        for timeframe in ['intraday', 'daily', 'monthly']:
            heatmaps[timeframe].sort(
                key=lambda x: x['score'] * x['confidence'],
                reverse=True
            )

        # Filter if specific timeframe requested
        if timeframe_filter in ['intraday', 'daily', 'monthly']:
            filtered_data = {
                timeframe_filter: heatmaps[timeframe_filter],
                'metadata': heatmaps['metadata']
            }
            return jsonify({
                "status": "success",
                "timeframe": timeframe_filter,
                "heatmaps": filtered_data,
                "count": len(heatmaps[timeframe_filter])
            }), 200

        # Return all timeframes
        return jsonify({
            "status": "success",
            "timeframe": "all",
            "heatmaps": heatmaps,
            "summary": {
                'intraday_count': len(heatmaps['intraday']),
                'daily_count': len(heatmaps['daily']),
                'monthly_count': len(heatmaps['monthly']),
                'total_unique_symbols': len(set(f['symbol'] for f in all_fusions))
            }
        }), 200

    except Exception as e:
        app.logger.error(f"Error getting heatmap data: {e}")
        return jsonify({"error": str(e)}), 500


# =============================================================================
# ML TRADING SYSTEM ENDPOINTS (Separate from Multi-Agent System)
# =============================================================================

@app.route("/ml-trading")
def serve_ml_trading():
    """Serve ML Trading System page"""
    response = make_response(send_from_directory(FRONTEND_DIR, "ml_trading.html"))
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@app.route('/ml-signals', methods=['GET'])
def get_ml_signals():
    """Get ML trading signals (separate from agent system)"""
    try:
        signals_file = os.path.join(DATA_DIR, 'ml_trading_signals.json')

        if not os.path.exists(signals_file):
            return jsonify({
                "status": "no_data",
                "message": "No ML signals found. Run a scan first.",
                "signals": []
            }), 200

        with open(signals_file, 'r') as f:
            data = json.load(f)

        return jsonify({
            "status": "success",
            "signals": data.get('all_signals', []),
            "intraday": data.get('intraday', []),
            "daily": data.get('daily', []),
            "monthly": data.get('monthly', []),
            "timestamp": data.get('timestamp'),
            "total_count": data.get('total_count', 0),
            "scan_metadata": data.get('scan_metadata', {})
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/ml-scan', methods=['POST'])
def run_ml_scan():
    """Trigger ML system scan in background"""
    try:
        import sys
        # Add trading_system to path
        trading_system_path = os.path.join(BASE_DIR, 'trading_system')
        if trading_system_path not in sys.path:
            sys.path.insert(0, BASE_DIR)

        from trading_system.core.trading_system import TradingSystem
        from trading_system.core.scan_job_manager import get_job_manager

        # Get parameters
        data = request.get_json() or {}
        max_stocks = data.get('max_stocks', 500)  # Default to full S&P 500
        include_crypto = data.get('include_crypto', True)

        # Calculate total items to scan (S&P 500 + cryptos)
        total_items = max_stocks + (100 if include_crypto else 0)

        # Create job
        job_manager = get_job_manager()
        job_id = job_manager.create_job(total_items)

        # Define background scan function
        def run_scan_background():
            try:
                system = TradingSystem()

                # Progress callback
                def progress_callback(current, total, symbol):
                    job_manager.update_progress(job_id, current, symbol)

                # Run scan with progress tracking
                signals = system.run_daily_scan(
                    max_stocks=max_stocks,
                    include_crypto=include_crypto,
                    progress_callback=progress_callback
                )

                # Mark job as complete
                job_manager.complete_job(job_id, len(signals))

            except Exception as e:
                # Mark job as failed
                job_manager.fail_job(job_id, str(e))
                print(f"[TURBO SCAN ERROR] {e}")

        # Start background thread
        scan_thread = threading.Thread(target=run_scan_background, daemon=True)
        scan_thread.start()

        return jsonify({
            "status": "started",
            "message": "Scan started in background",
            "job_id": job_id,
            "total_items": total_items
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/ml-scan-progress/<job_id>', methods=['GET'])
def get_ml_scan_progress(job_id):
    """Get progress for a specific scan job"""
    try:
        import sys
        if BASE_DIR not in sys.path:
            sys.path.insert(0, BASE_DIR)

        from trading_system.core.scan_job_manager import get_job_manager

        job_manager = get_job_manager()
        job = job_manager.get_job(job_id)

        if not job:
            return jsonify({
                "status": "error",
                "message": f"Job {job_id} not found"
            }), 404

        return jsonify({
            "status": "success",
            "job": job
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/ml-scan-active', methods=['GET'])
def get_ml_scan_active():
    """Check if any scan is currently running"""
    try:
        import sys
        if BASE_DIR not in sys.path:
            sys.path.insert(0, BASE_DIR)

        from trading_system.core.scan_job_manager import get_job_manager

        job_manager = get_job_manager()
        active_job = job_manager.get_active_job()

        if active_job:
            return jsonify({
                "status": "success",
                "active": True,
                "job": active_job
            }), 200
        else:
            return jsonify({
                "status": "success",
                "active": False,
                "job": None
            }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/ml-stats', methods=['GET'])
def get_ml_stats():
    """Get ML system performance statistics"""
    try:
        import sys
        if BASE_DIR not in sys.path:
            sys.path.insert(0, BASE_DIR)

        from trading_system.core.trading_system import TradingSystem

        system = TradingSystem()
        stats = system.get_performance_stats()

        return jsonify({
            "status": "success",
            "stats": stats
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/ml-training-progress', methods=['GET'])
def get_training_progress():
    """Get ML training progress for countdown timer"""
    try:
        import sys
        if BASE_DIR not in sys.path:
            sys.path.insert(0, BASE_DIR)

        from trading_system.core.trade_tracker import TradeTracker

        tracker = TradeTracker()
        stats = tracker.get_performance_stats()

        # Calculate training progress
        total_trades = stats.get('total_trades', 0)
        wins = stats.get('wins', 0)
        losses = stats.get('losses', 0)
        completed_trades = wins + losses
        win_rate = (wins / completed_trades * 100) if completed_trades > 0 else 0

        # Determine training level
        if total_trades >= 840:
            level = 'expert'
            level_name = '🧠 Expert'
        elif total_trades >= 420:
            level = 'well_trained'
            level_name = '📈 Well Trained'
        elif total_trades >= 100:
            level = 'trained'
            level_name = '🎓 Trained'
        elif total_trades >= 30:
            level = 'learning'
            level_name = '📊 Learning'
        else:
            level = 'untrained'
            level_name = 'Untrained'

        return jsonify({
            "status": "success",
            "total_trades": total_trades,
            "completed_trades": completed_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "level": level,
            "level_name": level_name,
            "expert_target": 840,
            "progress_pct": min((total_trades / 840) * 100, 100)
        }), 200

    except Exception as e:
        # Return default values if database doesn't exist yet
        return jsonify({
            "status": "success",
            "total_trades": 0,
            "completed_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0,
            "level": "untrained",
            "level_name": "Untrained",
            "expert_target": 840,
            "progress_pct": 0
        }), 200


@app.route('/ml-heatmap-data', methods=['GET'])
def get_ml_heatmap_data():
    """Get ML system heat map data (separate from agent heatmaps)"""
    try:
        signals_file = os.path.join(DATA_DIR, 'ml_trading_signals.json')

        if not os.path.exists(signals_file):
            return jsonify({
                "status": "no_data",
                "message": "No ML signals found",
                "heatmaps": {'intraday': [], 'daily': [], 'monthly': []}
            }), 200

        with open(signals_file, 'r') as f:
            data = json.load(f)

        # Transform to heat map format
        heatmaps = {
            'intraday': data.get('intraday', []),
            'daily': data.get('daily', []),
            'monthly': data.get('monthly', []),
            'metadata': {
                'timestamp': data.get('timestamp'),
                'system': 'ML Trading System',
                'total_signals': data.get('total_count', 0)
            }
        }

        return jsonify({
            "status": "success",
            "heatmaps": heatmaps,
            "summary": {
                'intraday_count': len(heatmaps['intraday']),
                'daily_count': len(heatmaps['daily']),
                'monthly_count': len(heatmaps['monthly'])
            }
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/ml-run-full-cycle', methods=['POST'])
def run_ml_full_cycle():
    """
    Trigger a complete ML automated cycle (scan + learner + retrain)
    Runs immediately without waiting for scheduled time
    """
    try:
        import subprocess
        import sys

        # Path to automated scheduler
        scheduler_path = os.path.join(BASE_DIR, 'trading_system', 'automated_scheduler.py')

        if not os.path.exists(scheduler_path):
            return jsonify({
                "status": "error",
                "message": "Automated scheduler not found"
            }), 404

        # Run with --now flag to execute immediately
        process = subprocess.Popen(
            [sys.executable, scheduler_path, '--now'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=os.path.dirname(scheduler_path)
        )

        # Wait for completion
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            return jsonify({
                "status": "error",
                "message": f"Full cycle failed: {stderr}"
            }), 500

        return jsonify({
            "status": "success",
            "message": "Full ML cycle completed successfully",
            "output": stdout
        }), 200

    except Exception as e:
        app.logger.error(f"Error running full ML cycle: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/ml-run-scan-only', methods=['POST'])
def run_ml_scan_only():
    """Trigger just the ML scan (no learner)"""
    try:
        import sys
        if BASE_DIR not in sys.path:
            sys.path.insert(0, BASE_DIR)

        from trading_system.core.trading_system import TradingSystem

        # Get parameters
        data = request.get_json() or {}
        max_stocks = data.get('max_stocks', 500)  # Default: scan ENTIRE S&P 500
        include_crypto = data.get('include_crypto', True)

        # Run scan
        system = TradingSystem()
        signals = system.run_daily_scan(max_stocks=max_stocks, include_crypto=include_crypto)

        return jsonify({
            "status": "success",
            "message": f"Scan complete. Generated {len(signals)} signals.",
            "signals_count": len(signals)
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/ml-run-learner-only', methods=['POST'])
def run_ml_learner_only():
    """Trigger just the automated learner (no scan)"""
    try:
        import subprocess
        import sys

        # Path to automated learner
        learner_path = os.path.join(BASE_DIR, 'trading_system', 'automated_learner.py')

        if not os.path.exists(learner_path):
            return jsonify({
                "status": "error",
                "message": "Automated learner not found"
            }), 404

        # Run learner
        process = subprocess.Popen(
            [sys.executable, learner_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=os.path.dirname(learner_path)
        )

        # Wait for completion
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            return jsonify({
                "status": "error",
                "message": f"Learner failed: {stderr}"
            }), 500

        return jsonify({
            "status": "success",
            "message": "Automated learner completed successfully",
            "output": stdout
        }), 200

    except Exception as e:
        app.logger.error(f"Error running automated learner: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/ml-automation-status', methods=['GET'])
def get_ml_automation_status_api():
    """
    Get ML automation status (built-in scheduler)
    Works on any platform - no Windows dependencies
    """
    try:
        if not ML_AUTOMATION_AVAILABLE:
            return jsonify({
                "status": "unavailable",
                "message": "ML Automation service not available. Install APScheduler."
            }), 503

        status = get_automation_status()

        if status['enabled']:
            return jsonify({
                "status": "enabled",
                "enabled": True,
                "schedule_time": status['schedule_time'],
                "last_run": status['last_run'],
                "next_run": status['next_run'],
                "scheduler_running": status['scheduler_running']
            }), 200
        else:
            return jsonify({
                "status": "disabled",
                "enabled": False,
                "message": "Automation is disabled. Click 'Start Automation' to enable."
            }), 200

    except Exception as e:
        app.logger.error(f"Error checking automation status: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/ml-automation-start', methods=['POST'])
def start_ml_automation():
    """
    Start ML automation (built-in scheduler)
    Runs daily at specified time inside Flask application
    """
    try:
        if not ML_AUTOMATION_AVAILABLE:
            return jsonify({
                "status": "error",
                "message": "ML Automation service not available"
            }), 503

        # Get schedule time from request (default 18:00)
        data = request.get_json() or {}
        schedule_time = data.get('schedule_time', '18:00')

        # Validate time format
        try:
            hour, minute = map(int, schedule_time.split(':'))
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                raise ValueError()
        except:
            return jsonify({
                "status": "error",
                "message": "Invalid time format. Use HH:MM (24-hour format)"
            }), 400

        # Start automation
        success = start_automation(schedule_time=schedule_time)

        if success:
            status = get_automation_status()
            return jsonify({
                "status": "success",
                "message": f"Automation started! Will run daily at {schedule_time}",
                "enabled": True,
                "schedule_time": schedule_time,
                "next_run": status['next_run']
            }), 200
        else:
            return jsonify({
                "status": "error",
                "message": "Automation already running"
            }), 400

    except Exception as e:
        app.logger.error(f"Error starting automation: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/ml-automation-stop', methods=['POST'])
def stop_ml_automation():
    """Stop ML automation (built-in scheduler)"""
    try:
        if not ML_AUTOMATION_AVAILABLE:
            return jsonify({
                "status": "error",
                "message": "ML Automation service not available"
            }), 503

        success = stop_automation()

        if success:
            return jsonify({
                "status": "success",
                "message": "Automation stopped",
                "enabled": False
            }), 200
        else:
            return jsonify({
                "status": "error",
                "message": "Automation not running"
            }), 400

    except Exception as e:
        app.logger.error(f"Error stopping automation: {e}")
        return jsonify({"error": str(e)}), 500


# ==================== ML MODEL CONFIGURATION ENDPOINTS ====================

@app.route('/ml-settings')
def serve_ml_settings():
    """Serve ML Settings page"""
    return send_from_directory('../frontend', 'ml_settings.html')


@app.route('/ml-configs', methods=['GET'])
def get_ml_configs():
    """Get all model configurations"""
    try:
        if not ML_MODEL_MANAGER_AVAILABLE:
            return jsonify({
                "status": "error",
                "message": "ML Model Manager not available"
            }), 503

        configs = ml_model_manager.get_all_configurations()

        return jsonify({
            "status": "success",
            "configurations": configs
        }), 200

    except Exception as e:
        app.logger.error(f"Error getting configurations: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/ml-config-create', methods=['POST'])
def create_ml_config():
    """Create new model configuration"""
    try:
        if not ML_MODEL_MANAGER_AVAILABLE:
            return jsonify({
                "status": "error",
                "message": "ML Model Manager not available"
            }), 503

        data = request.get_json()

        if not data or 'name' not in data:
            return jsonify({
                "status": "error",
                "message": "Model name is required"
            }), 400

        name = data['name']
        config = {
            'analysis_type': data.get('analysis_type', 'price_action'),
            'philosophy': data.get('philosophy', []),
            'win_criteria': data.get('win_criteria', 'price_movement'),
            'hold_period_days': data.get('hold_period_days', 14),
            'win_threshold_pct': data.get('win_threshold_pct', 10.0),
            'loss_threshold_pct': data.get('loss_threshold_pct', -5.0)
        }

        success = ml_model_manager.create_configuration(name, config)

        if success:
            return jsonify({
                "status": "success",
                "message": f"Model '{name}' created successfully",
                "name": name
            }), 200
        else:
            return jsonify({
                "status": "error",
                "message": f"Model '{name}' already exists"
            }), 400

    except Exception as e:
        app.logger.error(f"Error creating configuration: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/ml-config-activate', methods=['POST'])
def activate_ml_config():
    """Activate a model configuration"""
    try:
        if not ML_MODEL_MANAGER_AVAILABLE:
            return jsonify({
                "status": "error",
                "message": "ML Model Manager not available"
            }), 503

        data = request.get_json()

        if not data or 'name' not in data:
            return jsonify({
                "status": "error",
                "message": "Model name is required"
            }), 400

        name = data['name']
        success = ml_model_manager.activate_model(name)

        if success:
            return jsonify({
                "status": "success",
                "message": f"Model '{name}' activated",
                "active_model": name
            }), 200
        else:
            return jsonify({
                "status": "error",
                "message": f"Model '{name}' not found"
            }), 404

    except Exception as e:
        app.logger.error(f"Error activating configuration: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/ml-config-delete/<name>', methods=['DELETE'])
def delete_ml_config(name):
    """Delete a model configuration"""
    try:
        if not ML_MODEL_MANAGER_AVAILABLE:
            return jsonify({
                "status": "error",
                "message": "ML Model Manager not available"
            }), 503

        success = ml_model_manager.delete_configuration(name)

        if success:
            return jsonify({
                "status": "success",
                "message": f"Model '{name}' deleted"
            }), 200
        else:
            return jsonify({
                "status": "error",
                "message": f"Model '{name}' not found or is active (cannot delete active model)"
            }), 400

    except Exception as e:
        app.logger.error(f"Error deleting configuration: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/ml-config-performance/<name>', methods=['GET'])
def get_ml_config_performance(name):
    """Get performance stats for a model"""
    try:
        if not ML_MODEL_MANAGER_AVAILABLE:
            return jsonify({
                "status": "error",
                "message": "ML Model Manager not available"
            }), 503

        config = ml_model_manager.get_configuration(name)

        if not config:
            return jsonify({
                "status": "error",
                "message": f"Model '{name}' not found"
            }), 404

        return jsonify({
            "status": "success",
            "configuration": config
        }), 200

    except Exception as e:
        app.logger.error(f"Error getting performance: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# ML SIGNALS ENDPOINTS (Advanced ML Trading System)
# ============================================================================

@app.route('/ml-signals')
def ml_signals_page():
    """Serve the ML Signals frontend page"""
    return send_from_directory('../frontend', 'ml_signals.html')


@app.route('/ml/models', methods=['GET'])
def get_ml_models():
    """Get list of available ML models with their status"""
    try:
        import glob
        import json
        from pathlib import Path

        models = []

        # Define available model configurations
        model_configs = [
            {
                'id': 'quick_14d',
                'name': 'Quick Training (14-day)',
                'hold_days': 14,
                'profit_target': 10.0,
                'loss_limit': 5.0,
                'symbols_count': 20,
                'model_path': 'backend/data/ml_models/xgboost',
                'results_file': 'backend/data/quick_training_results.json'
            },
            {
                'id': 'options_5d',
                'name': 'Options 5-Day',
                'hold_days': 5,
                'profit_target': 25.0,
                'loss_limit': 15.0,
                'symbols_count': 20,
                'model_path': 'backend/data/ml_models/xgboost',
                'results_file': 'backend/data/options_training_5d_results.json'
            },
            {
                'id': 'options_7d',
                'name': 'Options 7-Day',
                'hold_days': 7,
                'profit_target': 30.0,
                'loss_limit': 20.0,
                'symbols_count': 20,
                'model_path': 'backend/data/ml_models/xgboost',
                'results_file': 'backend/data/options_training_7d_results.json'
            },
            {
                'id': 'full_14d',
                'name': 'Full Training (14-day)',
                'hold_days': 14,
                'profit_target': 10.0,
                'loss_limit': 5.0,
                'symbols_count': 82,
                'model_path': 'backend/data/ml_models/xgboost',
                'results_file': 'backend/data/training_results_checkpoint.json'
            }
        ]

        for config in model_configs:
            model_info = {
                'id': config['id'],
                'name': config['name'],
                'hold_days': config['hold_days'],
                'profit_target': config['profit_target'],
                'loss_limit': config['loss_limit'],
                'symbols_count': config['symbols_count'],
                'status': 'not_trained',
                'performance': None
            }

            # Check if model is trained
            if Path(config['model_path']).exists():
                model_info['status'] = 'trained'

                # Try to load performance metrics
                if Path(config['results_file']).exists():
                    try:
                        with open(config['results_file'], 'r') as f:
                            results = json.load(f)

                            # Extract best model performance
                            if 'results' in results:
                                best_acc = 0
                                best_gap = 0
                                for model_name, metrics in results['results'].items():
                                    if metrics.get('test_accuracy', 0) > best_acc:
                                        best_acc = metrics['test_accuracy']
                                        best_gap = metrics.get('gap', 0)

                                model_info['performance'] = {
                                    'test_accuracy': best_acc,
                                    'gap': best_gap,
                                    'last_trained': results.get('timestamp', 'Unknown')
                                }
                    except Exception as e:
                        print(f"Error loading results for {config['id']}: {e}")

            models.append(model_info)

        return jsonify({
            'success': True,
            'models': models
        })

    except Exception as e:
        print(f"Error getting ML models: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/ml/predict', methods=['POST'])
def get_ml_prediction():
    """Get ML prediction for a symbol"""
    try:
        data = request.json
        model_id = data.get('model_id')
        symbol = data.get('symbol')

        if not model_id or not symbol:
            return jsonify({'error': 'model_id and symbol required'}), 400

        # DEPRECATED: Old ML pipeline removed - use TurboMode API instead
        return jsonify({'error': 'This endpoint is deprecated. Use /turbomode/predict instead.'}), 410

        # Import ML components
        # import sys
        # sys.path.insert(0, 'backend')
        # from advanced_ml.training.training_pipeline import TrainingPipeline

        # Initialize pipeline
        # pipeline = TrainingPipeline()

        # Get current market data for the symbol
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='1y')

        if len(hist) == 0:
            return jsonify({'error': f'No data available for {symbol}'}), 404

        current_price = hist['Close'].iloc[-1]

        # Extract features for latest data point
        # (This is simplified - in production, use the full feature extraction)
        features = pipeline.feature_engineer.extract_features(hist, symbol)

        if features is None or len(features) == 0:
            return jsonify({'error': f'Could not extract features for {symbol}'}), 500

        # Get the latest feature vector
        latest_features = features.iloc[-1].to_dict()

        # Make prediction using XGBoost model (best performing)
        prediction = pipeline.xgb_model.predict(latest_features)

        # Map prediction to action
        label_map = {0: 'buy', 1: 'hold', 2: 'sell'}
        predicted_label = prediction['prediction']

        if isinstance(predicted_label, (int, float)):
            action = label_map.get(int(predicted_label), 'hold')
        else:
            action = predicted_label

        # Get confidence (if available)
        confidence = prediction.get('confidence', 0.5)

        return jsonify({
            'success': True,
            'symbol': symbol,
            'prediction': action,
            'confidence': float(confidence),
            'current_price': float(current_price),
            'model': model_id,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/ml/train', methods=['POST'])
def start_ml_training():
    """Start ML model training"""
    try:
        data = request.json
        training_type = data.get('training_type', 'quick')

        import subprocess
        import sys

        # Map training type to script
        script_map = {
            'quick': 'run_quick_training.py',
            'options_5d': 'run_options_training.py --hold-days 5',
            'options_7d': 'run_options_training.py --hold-days 7',
            'full': 'run_training_with_checkpoints.py'
        }

        script = script_map.get(training_type)
        if not script:
            return jsonify({'error': f'Unknown training type: {training_type}'}), 400

        # Start training in background
        cmd = f'{sys.executable} {script}'
        subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        return jsonify({
            'success': True,
            'message': f'Training started: {training_type}',
            'estimated_time': '2-3 hours' if 'quick' in training_type or 'options' in training_type else '8-11 hours'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# TURBOMODE API ENDPOINTS
# ============================================================================

# Initialize TurboMode database
try:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from backend.turbomode.database_schema import TurboModeDB
    turbomode_db = TurboModeDB(db_path="backend/data/turbomode.db")
    TURBOMODE_AVAILABLE = True
except ImportError as e:
    print(f"[TURBOMODE] Not available: {e}")
    TURBOMODE_AVAILABLE = False

# Initialize TurboMode Scheduler (separate from ML automation)
try:
    from backend.turbomode.turbomode_scheduler import (
        init_turbomode_scheduler,
        start_scheduler as start_turbomode,
        stop_scheduler as stop_turbomode,
        get_status as get_turbomode_status
    )
    TURBOMODE_SCHEDULER_AVAILABLE = True
except ImportError as e:
    print(f"[TURBOMODE SCHEDULER] Not available: {e}")
    TURBOMODE_SCHEDULER_AVAILABLE = False

# Initialize Stock Ranking API Blueprint (adaptive top 10 stock selection)
try:
    from backend.turbomode.stock_ranking_api import ranking_bp, init_stock_ranking_scheduler
    app.register_blueprint(ranking_bp)
    STOCK_RANKING_AVAILABLE = True
except ImportError as e:
    print(f"[STOCK RANKING] Not available: {e}")
    STOCK_RANKING_AVAILABLE = False

# Initialize Predictions API Blueprint (all model predictions with confidence levels)
try:
    from backend.turbomode.predictions_api import predictions_bp
    app.register_blueprint(predictions_bp)
    print("[PREDICTIONS API] Registered - /turbomode/predictions/*")
    PREDICTIONS_API_AVAILABLE = True
except ImportError as e:
    print(f"[PREDICTIONS API] Not available: {e}")
    PREDICTIONS_API_AVAILABLE = False

# Initialize Options API Blueprint (options analysis with Greeks and ML strike selection)
try:
    from backend.turbomode.options_api import options_bp
    app.register_blueprint(options_bp)
    print("[OPTIONS API] Registered - /api/options/*")
    OPTIONS_API_AVAILABLE = True
except Exception as e:
    print(f"[OPTIONS API] Not available: {e}")
    import traceback
    traceback.print_exc()
    OPTIONS_API_AVAILABLE = False


@app.route('/turbomode/signals', methods=['GET'])
def get_turbomode_signals():
    """
    Get active TurboMode signals with filters

    Query params:
        - market_cap: 'large_cap', 'mid_cap', or 'small_cap' (optional)
        - signal_type: 'BUY' or 'SELL' (optional)
        - limit: max number of results (default 20)
    """
    if not TURBOMODE_AVAILABLE:
        return jsonify({'error': 'TurboMode not available'}), 503

    try:
        market_cap = request.args.get('market_cap')
        signal_type = request.args.get('signal_type')
        limit = int(request.args.get('limit', 20))

        signals = turbomode_db.get_active_signals(
            market_cap=market_cap,
            signal_type=signal_type,
            limit=limit
        )

        # Calculate age color and effective confidence for each signal
        for signal in signals:
            age_days = signal.get('age_days', 0)
            confidence = signal.get('confidence', 0)

            # Age color
            if age_days <= 3:
                signal['age_color'] = 'hot'  # Red/hot
            elif age_days <= 7:
                signal['age_color'] = 'warm'  # Orange/warm
            elif age_days <= 10:
                signal['age_color'] = 'cool'  # Yellow/cool
            else:
                signal['age_color'] = 'cold'  # Blue/cold

            # Effective confidence (time-decayed)
            signal['effective_confidence'] = turbomode_db.calculate_effective_confidence(
                confidence, age_days
            )

        return jsonify({
            'success': True,
            'signals': signals,
            'count': len(signals)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/turbomode/sectors', methods=['GET'])
def get_turbomode_sectors():
    """
    Get sector statistics for Sectors Overview page

    Query params:
        - date: YYYY-MM-DD (optional, defaults to latest)
    """
    if not TURBOMODE_AVAILABLE:
        return jsonify({'error': 'TurboMode not available'}), 503

    try:
        date = request.args.get('date')  # None = latest

        sector_stats = turbomode_db.get_sector_stats(date=date)

        # Separate into bullish and bearish
        bullish_sectors = [s for s in sector_stats if s['sentiment'] == 'BULLISH']
        bearish_sectors = [s for s in sector_stats if s['sentiment'] == 'BEARISH']
        neutral_sectors = [s for s in sector_stats if s['sentiment'] == 'NEUTRAL']

        return jsonify({
            'success': True,
            'bullish': bullish_sectors,
            'bearish': bearish_sectors,
            'neutral': neutral_sectors,
            'total_sectors': len(sector_stats)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/turbomode/stats', methods=['GET'])
def get_turbomode_stats():
    """
    Get overall TurboMode database statistics
    """
    if not TURBOMODE_AVAILABLE:
        return jsonify({'error': 'TurboMode not available'}), 503

    try:
        stats = turbomode_db.get_stats()

        return jsonify({
            'success': True,
            'stats': stats
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/turbomode/scan', methods=['POST'])
def run_turbomode_scan():
    """
    Trigger overnight TurboMode scan manually

    WARNING: This takes 20-30 minutes for full S&P 500 scan
    """
    if not TURBOMODE_AVAILABLE:
        return jsonify({'error': 'TurboMode not available'}), 503

    try:
        import subprocess
        import sys

        # Start scanner in background
        cmd = [sys.executable, 'backend/turbomode/overnight_scanner.py']
        subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        return jsonify({
            'success': True,
            'message': 'TurboMode scan started',
            'estimated_time': '20-30 minutes for full S&P 500 scan'
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/turbomode/scheduler/status', methods=['GET'])
def get_turbomode_scheduler_status():
    """Get TurboMode scheduler status"""
    if not TURBOMODE_SCHEDULER_AVAILABLE:
        return jsonify({'error': 'TurboMode scheduler not available'}), 503

    try:
        status = get_turbomode_status()
        return jsonify({
            'success': True,
            'status': status
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/turbomode/scheduler/start', methods=['POST'])
def start_turbomode_scheduler():
    """Start TurboMode scheduler"""
    if not TURBOMODE_SCHEDULER_AVAILABLE:
        return jsonify({'error': 'TurboMode scheduler not available'}), 503

    try:
        data = request.json or {}
        schedule_time = data.get('schedule_time', '23:00')

        success = start_turbomode(schedule_time=schedule_time)

        if success:
            status = get_turbomode_status()
            return jsonify({
                'success': True,
                'message': f'TurboMode scheduler started at {schedule_time}',
                'status': status
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Scheduler already running or failed to start'
            })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/turbomode/scheduler/stop', methods=['POST'])
def stop_turbomode_scheduler():
    """Stop TurboMode scheduler"""
    if not TURBOMODE_SCHEDULER_AVAILABLE:
        return jsonify({'error': 'TurboMode scheduler not available'}), 503

    try:
        success = stop_turbomode()

        if success:
            return jsonify({
                'success': True,
                'message': 'TurboMode scheduler stopped'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Scheduler not running or failed to stop'
            })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    # ML Automation Service DISABLED - Using TurboMode instead
    # if ML_AUTOMATION_AVAILABLE:
    #     print("[TURBO AUTOMATION] Initializing built-in scheduler...")
    #     init_automation_service()
    #     print("[TURBO AUTOMATION] Ready")
    # else:
    #     print("[TURBO AUTOMATION] Not available - install APScheduler")

    # Initialize TurboMode Scheduler (runs at 11 PM nightly on 82 curated stocks)
    if TURBOMODE_SCHEDULER_AVAILABLE:
        print("[TURBOMODE SCHEDULER] Initializing overnight scan scheduler...")
        init_turbomode_scheduler()
        status = get_turbomode_status()
        if status['enabled']:
            print(f"[TURBOMODE SCHEDULER] Ready - Next scan at {status['next_run']}")
        else:
            print("[TURBOMODE SCHEDULER] Disabled")
    else:
        print("[TURBOMODE SCHEDULER] Not available")

    # Initialize Stock Ranking Scheduler (runs monthly on 1st at 2 AM)
    if STOCK_RANKING_AVAILABLE:
        print("[STOCK RANKING] Initializing monthly adaptive stock ranking scheduler...")
        init_stock_ranking_scheduler()
        print("[STOCK RANKING] Ready - Runs monthly on 1st at 2:00 AM")
    else:
        print("[STOCK RANKING] Not available")

    # Initialize Unified Scheduler (all 6 scheduled tasks - config-driven)
    try:
        from backend.unified_scheduler_api import init_unified_scheduler_api
        app = init_unified_scheduler_api(app)
    except Exception as e:
        print(f"[UNIFIED SCHEDULER] Failed to initialize: {e}")

    # Use socketio.run instead of app.run for WebSocket support
    # Ticker emit worker will start automatically when first client connects
    print("Flask server starting on http://127.0.0.1:5000")
    print("Logging: QUIET MODE (errors only)")
    socketio.run(app, debug=False, host='127.0.0.1', port=5000, allow_unsafe_werkzeug=True, log_output=False)
