"""
Tradier WebSocket Client for Real-Time Stock Quotes
Production service for streaming real-time equity prices via Tradier WebSocket API

Architecture:
- WebSocket connection to Tradier (wss://ws.tradier.com/v1/markets/events)
- Session management with auto-renewal every 55 minutes (sessions expire at 60 min)
- Subscribes to quote and trade events for stock symbols
- Thread-safe callback system for distributing updates to multiple consumers
- Automatic reconnection on connection loss

Usage:
    from tradier_websocket_client import TradierWebSocketClient

    client = TradierWebSocketClient(api_key="your_key")

    def on_quote(symbol, data):
        print(f"{symbol}: ${data['last']:.2f}")

    client.register_callback("AAPL", on_quote)
    client.start()  # Runs in background thread
"""
import os
import json
import time
import asyncio
import websockets
import requests
import threading
from datetime import datetime, timedelta
from typing import Dict, Callable, Optional, Any
from collections import defaultdict
import pytz

# Tradier configuration
TRADIER_API_URL = os.environ.get("TRADIER_API_URL", "https://api.tradier.com")
TRADIER_WS_URL = "wss://ws.tradier.com/v1/markets/events"
TRADIER_SESSION_ENDPOINT = "/v1/markets/events/session"

# Session expires after ~60 minutes, renew at 55 min
SESSION_RENEWAL_SECONDS = 55 * 60

# Reconnection limits
MAX_RECONNECT_ATTEMPTS = 5  # Max retries before checking market hours
MARKET_CLOSED_WAIT_SECONDS = 300  # 5 minutes between retries when market is closed


def is_market_hours() -> bool:
    """
    Check if US stock market is currently open
    Market hours in Central Time: 8:30 AM - 3:00 PM CT (9:30 AM - 4:00 PM ET)

    Returns:
        True if market is open, False otherwise
    """
    try:
        central = pytz.timezone('US/Central')
        now = datetime.now(central)

        # Check if weekend (Saturday=5, Sunday=6)
        if now.weekday() >= 5:
            return False

        # Market hours: 8:30 AM - 3:00 PM CT (9:30 AM - 4:00 PM ET)
        market_open = now.replace(hour=8, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=0, second=0, microsecond=0)

        return market_open <= now <= market_close
    except Exception:
        # If we can't determine, assume market is open to allow connection attempt
        return True


def seconds_until_market_open() -> int:
    """
    Calculate seconds until 30 seconds before market opens

    Returns:
        Number of seconds until 30 seconds before market opens (8:29:30 AM CT)
    """
    try:
        central = pytz.timezone('US/Central')
        now = datetime.now(central)

        # Pre-market connection time: 8:29:30 AM CT (30 seconds before open)
        pre_market_time = now.replace(hour=8, minute=29, second=30, microsecond=0)

        # If today is a weekday and before pre-market time, wait until pre-market time today
        if now.weekday() < 5 and now < pre_market_time:
            seconds = (pre_market_time - now).total_seconds()
            return int(seconds)

        # Otherwise, calculate next weekday at pre-market time
        days_ahead = 1
        while True:
            next_day = now + timedelta(days=days_ahead)
            # If it's a weekday (Monday=0 through Friday=4)
            if next_day.weekday() < 5:
                next_open = next_day.replace(hour=8, minute=29, second=30, microsecond=0)
                seconds = (next_open - now).total_seconds()
                return int(seconds)
            days_ahead += 1

            # Safety limit: don't calculate more than 7 days ahead
            if days_ahead > 7:
                return 3600  # Default to 1 hour if calculation fails

    except Exception:
        # Default to 1 hour if calculation fails
        return 3600


class TradierWebSocketClient:
    """
    Production-ready Tradier WebSocket client with automatic session management
    """

    def __init__(self, api_key: str):
        """
        Initialize Tradier WebSocket client

        Args:
            api_key: Tradier API key (Bearer token)
        """
        self.api_key = api_key
        self.session_id = None
        self.session_created_at = None

        # Symbol subscriptions and callbacks
        self.symbols = set()  # {symbol1, symbol2, ...}
        self.callbacks = defaultdict(list)  # {symbol: [callback1, callback2, ...]}
        self.lock = threading.Lock()

        # WebSocket state
        self.ws = None
        self.running = False
        self.thread = None
        self.loop = None

        # Last known prices (cache for immediate callback on new subscriptions)
        self.last_prices = {}  # {symbol: {last, bid, ask, volume, timestamp}}

        print(f"[TRADIER WS] Client initialized")

    def create_session(self) -> Optional[str]:
        """
        Create a new Tradier streaming session

        Returns:
            Session ID string, or None on failure
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json"
        }

        try:
            print(f"[TRADIER WS] Creating new streaming session...")
            response = requests.post(
                TRADIER_API_URL + TRADIER_SESSION_ENDPOINT,
                headers=headers,
                timeout=10
            )

            if response.status_code != 200:
                print(f"[TRADIER WS ERROR] Session creation failed: HTTP {response.status_code}")
                print(f"[TRADIER WS ERROR] Response: {response.text}")
                return None

            session_data = response.json()

            if "stream" in session_data and "sessionid" in session_data["stream"]:
                session_id = session_data["stream"]["sessionid"]
                self.session_created_at = datetime.now()
                print(f"[TRADIER WS] Session created: {session_id[:8]}... (expires in ~60 min)")
                return session_id

            print(f"[TRADIER WS ERROR] No session ID in response: {session_data}")
            return None

        except Exception as e:
            print(f"[TRADIER WS ERROR] Session creation failed: {e}")
            return None

    def register_callback(self, symbol: str, callback: Callable[[str, Dict], None]):
        """
        Register a callback for symbol updates

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            callback: Function that takes (symbol, data_dict)
                     data_dict contains: {last, bid, ask, volume, change, change_percentage, timestamp}
        """
        symbol = symbol.upper()

        with self.lock:
            self.callbacks[symbol].append(callback)

            # If we have cached data, immediately call the callback
            if symbol in self.last_prices:
                try:
                    callback(symbol, self.last_prices[symbol])
                except Exception as e:
                    print(f"[TRADIER WS ERROR] Callback error for {symbol}: {e}")

        print(f"[TRADIER WS] Registered callback for {symbol}")

    def unregister_callback(self, symbol: str, callback: Callable):
        """
        Unregister a specific callback for a symbol

        Args:
            symbol: Stock symbol
            callback: The callback function to remove
        """
        symbol = symbol.upper()

        with self.lock:
            if symbol in self.callbacks:
                try:
                    self.callbacks[symbol].remove(callback)
                    if not self.callbacks[symbol]:
                        del self.callbacks[symbol]
                    print(f"[TRADIER WS] Unregistered callback for {symbol}")
                except ValueError:
                    pass

    def subscribe(self, symbol: str):
        """
        Subscribe to real-time updates for a symbol

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
        """
        symbol = symbol.upper()

        with self.lock:
            if symbol not in self.symbols:
                self.symbols.add(symbol)
                print(f"[TRADIER WS] Added {symbol} to subscription list (total: {len(self.symbols)})")

                # If websocket is already running, update subscription
                if self.running and self.loop:
                    asyncio.run_coroutine_threadsafe(
                        self._update_subscription(),
                        self.loop
                    )

    def unsubscribe(self, symbol: str):
        """
        Unsubscribe from a symbol

        Args:
            symbol: Stock symbol
        """
        symbol = symbol.upper()

        with self.lock:
            if symbol in self.symbols:
                self.symbols.remove(symbol)
                print(f"[TRADIER WS] Removed {symbol} from subscription list (total: {len(self.symbols)})")

                # Clean up callbacks
                if symbol in self.callbacks:
                    del self.callbacks[symbol]

                # Update websocket subscription if running
                if self.running and self.loop:
                    asyncio.run_coroutine_threadsafe(
                        self._update_subscription(),
                        self.loop
                    )

    async def _update_subscription(self):
        """
        Update WebSocket subscription with current symbol list
        Internal method called when symbols are added/removed while streaming
        """
        if not self.ws or not self.session_id:
            return

        with self.lock:
            symbols_list = list(self.symbols)

        if not symbols_list:
            print(f"[TRADIER WS] No symbols to subscribe to")
            return

        subscribe_message = {
            "symbols": symbols_list,
            "sessionid": self.session_id,
            "filter": ["quote", "trade"],
            "linebreak": True
        }

        try:
            await self.ws.send(json.dumps(subscribe_message))
            print(f"[TRADIER WS] Updated subscription: {len(symbols_list)} symbols")
        except Exception as e:
            print(f"[TRADIER WS ERROR] Failed to update subscription: {e}")

    async def _stream_loop(self):
        """
        Main WebSocket streaming loop with auto-reconnection
        """
        reconnect_delay = 1  # Start with 1 second
        max_reconnect_delay = 60  # Cap at 60 seconds
        reconnect_attempts = 0  # Track consecutive failures

        while self.running:
            try:
                # Check if session needs renewal (after 55 minutes)
                if self.session_created_at:
                    elapsed = (datetime.now() - self.session_created_at).total_seconds()
                    if elapsed >= SESSION_RENEWAL_SECONDS:
                        print(f"[TRADIER WS] Session expired after {elapsed:.0f}s, renewing...")
                        self.session_id = self.create_session()
                        if not self.session_id:
                            print(f"[TRADIER WS ERROR] Session renewal failed, retrying in {reconnect_delay}s...")
                            await asyncio.sleep(reconnect_delay)
                            reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
                            continue

                # Create session if needed
                if not self.session_id:
                    self.session_id = self.create_session()
                    if not self.session_id:
                        print(f"[TRADIER WS ERROR] Session creation failed, retrying in {reconnect_delay}s...")
                        await asyncio.sleep(reconnect_delay)
                        reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
                        continue

                # Connect to WebSocket
                print(f"[TRADIER WS] Connecting to {TRADIER_WS_URL}...")
                async with websockets.connect(TRADIER_WS_URL, ssl=True, compression=None, ping_interval=20, ping_timeout=10) as websocket:
                    self.ws = websocket
                    print(f"[TRADIER WS] [OK] Connected!")

                    # Subscribe to symbols
                    await self._update_subscription()

                    # Reset reconnect delay and attempts on successful connection
                    reconnect_delay = 1
                    reconnect_attempts = 0

                    # Process messages
                    while self.running:
                        try:
                            message = await asyncio.wait_for(websocket.recv(), timeout=70)
                            self._handle_message(message)

                        except asyncio.TimeoutError:
                            # No message in 70 seconds, check session expiration
                            if self.session_created_at:
                                elapsed = (datetime.now() - self.session_created_at).total_seconds()
                                if elapsed >= SESSION_RENEWAL_SECONDS:
                                    print(f"[TRADIER WS] Session renewal needed, reconnecting...")
                                    break
                            continue

                        except websockets.exceptions.ConnectionClosed:
                            print(f"[TRADIER WS] Connection closed by server")
                            reconnect_attempts += 1
                            break

            except Exception as e:
                print(f"[TRADIER WS ERROR] Stream error: {e}")
                reconnect_attempts += 1

            finally:
                self.ws = None

            if self.running:
                # Check if we've hit the retry limit - if so, check market hours
                if reconnect_attempts >= MAX_RECONNECT_ATTEMPTS:
                    if not is_market_hours():
                        wait_seconds = seconds_until_market_open()
                        wait_hours = wait_seconds / 3600
                        print(f"[TRADIER WS] Market is closed. Waiting {wait_hours:.1f} hours until 30 seconds before market opens...")
                        await asyncio.sleep(wait_seconds)
                        reconnect_attempts = 0  # Reset counter after waiting for market open
                        reconnect_delay = 1  # Reset delay
                        continue
                    else:
                        # Market is open but connection failing - reset attempts and continue
                        print(f"[TRADIER WS] Market is open but connection failing, resetting retry counter")
                        reconnect_attempts = 0

                print(f"[TRADIER WS] Reconnecting in {reconnect_delay}s... (attempt {reconnect_attempts}/{MAX_RECONNECT_ATTEMPTS})")
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)

    def _handle_message(self, message: str):
        """
        Process incoming WebSocket message

        Args:
            message: JSON string from Tradier WebSocket
        """
        try:
            data = json.loads(message)
            msg_type = data.get('type', '')
            symbol = data.get('symbol', '').upper()

            if not symbol:
                return

            # Extract price data based on message type
            price_data = None

            if msg_type == 'trade':
                # Trade execution: use last price
                price_data = {
                    'last': data.get('price'),
                    'bid': None,
                    'ask': None,
                    'volume': data.get('cvol'),  # Cumulative volume
                    'timestamp': datetime.now()
                }

            elif msg_type == 'quote':
                # Quote update: has bid/ask
                bid = data.get('bid')
                ask = data.get('ask')

                # Calculate mid price if both bid and ask available
                if bid is not None and ask is not None:
                    last = (bid + ask) / 2
                else:
                    last = bid or ask or None

                price_data = {
                    'last': last,
                    'bid': bid,
                    'ask': ask,
                    'volume': None,  # Quotes don't have volume
                    'timestamp': datetime.now()
                }

            if not price_data or price_data['last'] is None:
                return

            # Update cache
            with self.lock:
                if symbol not in self.last_prices:
                    self.last_prices[symbol] = {}

                # Merge new data with cached data
                self.last_prices[symbol].update({k: v for k, v in price_data.items() if v is not None})

                # Call all registered callbacks for this symbol
                if symbol in self.callbacks:
                    cached_data = self.last_prices[symbol].copy()
                    for callback in self.callbacks[symbol]:
                        try:
                            callback(symbol, cached_data)
                        except Exception as e:
                            print(f"[TRADIER WS ERROR] Callback error for {symbol}: {e}")

        except json.JSONDecodeError:
            # Ignore non-JSON messages (e.g., heartbeats)
            pass
        except Exception as e:
            print(f"[TRADIER WS ERROR] Message handling error: {e}")

    def start(self):
        """
        Start the WebSocket client in a background thread
        """
        if self.running:
            print(f"[TRADIER WS] Already running")
            return

        self.running = True

        def run_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self._stream_loop())
            self.loop.close()

        self.thread = threading.Thread(target=run_loop, daemon=True)
        self.thread.start()

        print(f"[TRADIER WS] Started in background thread")

    def stop(self):
        """
        Stop the WebSocket client
        """
        print(f"[TRADIER WS] Stopping...")
        self.running = False

        if self.thread:
            self.thread.join(timeout=5)

        print(f"[TRADIER WS] Stopped")

    def get_last_price(self, symbol: str) -> Optional[Dict]:
        """
        Get the last known price for a symbol

        Args:
            symbol: Stock symbol

        Returns:
            Dict with {last, bid, ask, volume, timestamp} or None
        """
        symbol = symbol.upper()
        with self.lock:
            return self.last_prices.get(symbol, None)


# Singleton instance for global use
_tradier_ws_client = None
_client_lock = threading.Lock()


def get_tradier_ws_client() -> TradierWebSocketClient:
    """
    Get singleton Tradier WebSocket client

    Returns:
        TradierWebSocketClient instance
    """
    global _tradier_ws_client

    with _client_lock:
        if _tradier_ws_client is None:
            # Load API key from environment
            api_key = os.environ.get("TRADIER_API_KEY")
            if not api_key:
                raise ValueError("TRADIER_API_KEY environment variable not set")

            _tradier_ws_client = TradierWebSocketClient(api_key)
            _tradier_ws_client.start()

        return _tradier_ws_client
