"""
Tradier REST API Client for Unified Scheduler
==============================================

This module provides REST API methods for fetching equity data on-demand.
Unlike the WebSocket client (used for real-time chart streaming), this client
uses Tradier's REST endpoints for scheduler tasks that need historical/current data.

Architecture:
- Primary data source for scheduler tasks
- Fallback chain: Tradier → IB Gateway → Yahoo Finance
- Session management with auto-renewal for 24/7 operation
- Pull model (on-demand) vs WebSocket push model

Session Management:
- Sessions expire after 60 minutes
- Auto-renewal at 55 minutes (5-minute buffer)
- Thread-safe session creation

Endpoints Used:
- /v1/markets/quotes - Current quotes
- /v1/markets/history - Historical OHLCV data
- /v1/markets/timesales - Intraday tick data
"""

import os
import requests
import time
from datetime import datetime, timedelta
from threading import Lock
from typing import Dict, List, Optional, Any
import pandas as pd

# Tradier API configuration
TRADIER_API_URL = "https://api.tradier.com"
TRADIER_SESSION_ENDPOINT = "/v1/markets/events/session"
TRADIER_QUOTES_ENDPOINT = "/v1/markets/quotes"
TRADIER_HISTORY_ENDPOINT = "/v1/markets/history"
TRADIER_TIMESALES_ENDPOINT = "/v1/markets/timesales"

# Session renewal configuration (same as WebSocket client)
SESSION_RENEWAL_SECONDS = 55 * 60  # Renew at 55 minutes (expire at 60)


class TradierSchedulerClient:
    """
    REST API client for Tradier - designed for scheduler tasks.

    Features:
    - On-demand data fetching (pull model)
    - Session auto-renewal for 24/7 operation
    - Thread-safe session management
    - Supports quotes, historical data, and timesales
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Tradier REST client.

        Args:
            api_key: Tradier API key (defaults to environment variable)
        """
        self.api_key = api_key or os.getenv("TRADIER_API_KEY")
        if not self.api_key:
            raise ValueError("TRADIER_API_KEY environment variable not set")

        # Session management
        self.session_id: Optional[str] = None
        self.session_created_at: Optional[datetime] = None
        self.session_lock = Lock()

        print("[TRADIER REST] Client initialized")

    def _create_session(self) -> str:
        """
        Create a new Tradier session.

        Sessions are required for streaming endpoints but also recommended
        for REST endpoints to avoid rate limits.

        Returns:
            Session ID string

        Raises:
            Exception: If session creation fails
        """
        try:
            response = requests.post(
                TRADIER_API_URL + TRADIER_SESSION_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Accept": "application/json"
                },
                timeout=10
            )
            response.raise_for_status()

            data = response.json()
            session_id = data["stream"]["sessionid"]

            print(f"[TRADIER REST] Session created: {session_id[:8]}...")
            return session_id

        except Exception as e:
            print(f"[TRADIER REST] Session creation failed: {e}")
            raise

    def _ensure_valid_session(self) -> str:
        """
        Ensure we have a valid session, creating/renewing as needed.

        Thread-safe session management with auto-renewal at 55 minutes.

        Returns:
            Valid session ID
        """
        with self.session_lock:
            # Check if we need to create/renew session
            needs_renewal = False

            if not self.session_id or not self.session_created_at:
                needs_renewal = True
            else:
                elapsed = (datetime.now() - self.session_created_at).total_seconds()
                if elapsed >= SESSION_RENEWAL_SECONDS:
                    needs_renewal = True
                    print(f"[TRADIER REST] Session expired ({elapsed:.0f}s), renewing...")

            if needs_renewal:
                self.session_id = self._create_session()
                self.session_created_at = datetime.now()

            return self.session_id

    def get_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get current quotes for multiple symbols.

        Args:
            symbols: List of ticker symbols (e.g., ['AAPL', 'MSFT', 'TSLA'])

        Returns:
            Dictionary mapping symbol -> quote data
            Quote data contains: last, bid, ask, volume, open, high, low, close, etc.

        Example:
            {
                'AAPL': {
                    'symbol': 'AAPL',
                    'last': 150.25,
                    'bid': 150.24,
                    'ask': 150.26,
                    'volume': 50000000,
                    'open': 149.50,
                    'high': 150.75,
                    'low': 149.25,
                    'close': 150.25,
                    'prevclose': 149.00
                }
            }
        """
        if not symbols:
            return {}

        try:
            # Ensure valid session
            session_id = self._ensure_valid_session()

            # Join symbols with comma
            symbols_str = ",".join(symbols)

            # Make request
            response = requests.get(
                TRADIER_API_URL + TRADIER_QUOTES_ENDPOINT,
                params={"symbols": symbols_str, "sessionid": session_id},
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Accept": "application/json"
                },
                timeout=10
            )
            response.raise_for_status()

            data = response.json()

            # Parse response
            quotes = {}
            if "quotes" in data and "quote" in data["quotes"]:
                quote_data = data["quotes"]["quote"]

                # Handle single vs multiple quotes
                if isinstance(quote_data, dict):
                    # Single symbol
                    symbol = quote_data.get("symbol")
                    if symbol:
                        quotes[symbol] = quote_data
                elif isinstance(quote_data, list):
                    # Multiple symbols
                    for quote in quote_data:
                        symbol = quote.get("symbol")
                        if symbol:
                            quotes[symbol] = quote

            print(f"[TRADIER REST] Fetched quotes for {len(quotes)} symbols")
            return quotes

        except Exception as e:
            print(f"[TRADIER REST] Failed to fetch quotes: {e}")
            return {}

    def get_historical_data(
        self,
        symbol: str,
        interval: str = "daily",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data for a symbol.

        Args:
            symbol: Ticker symbol (e.g., 'AAPL')
            interval: Data interval - 'daily', 'weekly', 'monthly'
            start_date: Start date in YYYY-MM-DD format (defaults to 1 year ago)
            end_date: End date in YYYY-MM-DD format (defaults to today)

        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Volume

        Example:
            df = client.get_historical_data('AAPL', interval='daily')
            # Returns DataFrame with OHLCV data
        """
        try:
            # Default date range: 1 year
            if not end_date:
                end_date = datetime.now().strftime("%Y-%m-%d")
            if not start_date:
                start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

            # Ensure valid session
            session_id = self._ensure_valid_session()

            # Make request
            response = requests.get(
                TRADIER_API_URL + TRADIER_HISTORY_ENDPOINT,
                params={
                    "symbol": symbol,
                    "interval": interval,
                    "start": start_date,
                    "end": end_date,
                    "sessionid": session_id
                },
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Accept": "application/json"
                },
                timeout=10
            )
            response.raise_for_status()

            data = response.json()

            # Parse response
            if "history" in data and data["history"] and "day" in data["history"]:
                day_data = data["history"]["day"]

                # Handle single vs multiple days
                if isinstance(day_data, dict):
                    day_data = [day_data]

                # Convert to DataFrame
                df = pd.DataFrame(day_data)

                # Rename columns to match expected format
                if "date" in df.columns:
                    df.rename(columns={
                        "date": "Date",
                        "open": "Open",
                        "high": "High",
                        "low": "Low",
                        "close": "Close",
                        "volume": "Volume"
                    }, inplace=True)

                # Drop rows with null OHLC values (database requires NOT NULL)
                df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])

                print(f"[TRADIER REST] Fetched {len(df)} rows for {symbol} ({interval})")
                return df
            else:
                print(f"[TRADIER REST] No historical data for {symbol}")
                return pd.DataFrame()

        except Exception as e:
            print(f"[TRADIER REST] Failed to fetch historical data: {e}")
            return pd.DataFrame()

    def get_intraday_data(
        self,
        symbol: str,
        interval: str = "5min",
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get intraday tick/timesales data for a symbol.

        Args:
            symbol: Ticker symbol (e.g., 'AAPL')
            interval: Time interval - '1min', '5min', '15min'
            start_time: Start timestamp in YYYY-MM-DD HH:MM format
            end_time: End timestamp in YYYY-MM-DD HH:MM format

        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Volume

        Note:
            Tradier's timesales endpoint returns tick data.
            For candlestick data, we need to resample ticks into OHLCV candles.
        """
        try:
            # Default time range: today's market hours
            if not end_time:
                end_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            if not start_time:
                # Start at market open (9:30 AM ET)
                today = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
                start_time = today.strftime("%Y-%m-%d %H:%M")

            # Ensure valid session
            session_id = self._ensure_valid_session()

            # Make request
            response = requests.get(
                TRADIER_API_URL + TRADIER_TIMESALES_ENDPOINT,
                params={
                    "symbol": symbol,
                    "interval": interval,
                    "start": start_time,
                    "end": end_time,
                    "sessionid": session_id
                },
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Accept": "application/json"
                },
                timeout=10
            )
            response.raise_for_status()

            data = response.json()

            # Parse response
            if "series" in data and data["series"] and "data" in data["series"]:
                tick_data = data["series"]["data"]

                # Handle single vs multiple ticks
                if isinstance(tick_data, dict):
                    tick_data = [tick_data]

                # Convert to DataFrame
                df = pd.DataFrame(tick_data)

                # Rename columns
                if "time" in df.columns:
                    df.rename(columns={
                        "time": "Date",
                        "price": "Close"
                    }, inplace=True)

                # Resample ticks into OHLCV candles based on interval
                if len(df) > 0 and "Date" in df.columns:
                    df["Date"] = pd.to_datetime(df["Date"])
                    df = df.set_index("Date")

                    # Resample based on interval
                    interval_map = {
                        "1min": "1T",
                        "5min": "5T",
                        "15min": "15T"
                    }
                    resample_rule = interval_map.get(interval, "5T")

                    ohlc = df["Close"].resample(resample_rule).ohlc()
                    volume = df["volume"].resample(resample_rule).sum() if "volume" in df.columns else 0

                    result = ohlc.copy()
                    result["Volume"] = volume
                    result = result.reset_index()
                    result.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]

                    print(f"[TRADIER REST] Fetched {len(result)} intraday candles for {symbol} ({interval})")
                    return result

                return df
            else:
                print(f"[TRADIER REST] No intraday data for {symbol}")
                return pd.DataFrame()

        except Exception as e:
            print(f"[TRADIER REST] Failed to fetch intraday data: {e}")
            return pd.DataFrame()

    def test_connection(self) -> bool:
        """
        Test connection to Tradier API.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            quotes = self.get_quotes(["AAPL"])
            if quotes and "AAPL" in quotes:
                print("[TRADIER REST] Connection test successful")
                return True
            else:
                print("[TRADIER REST] Connection test failed - no data returned")
                return False
        except Exception as e:
            print(f"[TRADIER REST] Connection test failed: {e}")
            return False


# Global singleton instance
_tradier_scheduler_client: Optional[TradierSchedulerClient] = None


def get_tradier_scheduler_client() -> TradierSchedulerClient:
    """
    Get singleton instance of TradierSchedulerClient.

    Returns:
        Global TradierSchedulerClient instance
    """
    global _tradier_scheduler_client

    if _tradier_scheduler_client is None:
        _tradier_scheduler_client = TradierSchedulerClient()

    return _tradier_scheduler_client


if __name__ == "__main__":
    # Test the client
    print("=" * 60)
    print("TRADIER REST CLIENT TEST")
    print("=" * 60)

    # Load API key from environment
    from dotenv import load_dotenv
    load_dotenv()

    client = get_tradier_scheduler_client()

    # Test 1: Connection test
    print("\n[TEST 1] Connection Test")
    print("-" * 60)
    client.test_connection()

    # Test 2: Get quotes
    print("\n[TEST 2] Get Quotes")
    print("-" * 60)
    symbols = ["AAPL", "MSFT", "TSLA"]
    quotes = client.get_quotes(symbols)
    for symbol, quote in quotes.items():
        print(f"{symbol}: ${quote.get('last', 0):.2f} (Bid: ${quote.get('bid', 0):.2f}, Ask: ${quote.get('ask', 0):.2f})")

    # Test 3: Get historical data
    print("\n[TEST 3] Get Historical Data")
    print("-" * 60)
    df = client.get_historical_data("AAPL", interval="daily")
    if len(df) > 0:
        print(f"Last 5 days of AAPL:")
        print(df.tail())

    # Test 4: Session renewal
    print("\n[TEST 4] Session Info")
    print("-" * 60)
    if client.session_created_at:
        elapsed = (datetime.now() - client.session_created_at).total_seconds()
        print(f"Session age: {elapsed:.0f} seconds")
        print(f"Session will renew in: {SESSION_RENEWAL_SECONDS - elapsed:.0f} seconds")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
