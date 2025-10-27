"""
Coinbase API Client for Crypto Data
- REST API for historical candlestick data
- WebSocket streams handled separately
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time

class CoinbaseClient:
    def __init__(self):
        self.base_url = "https://api.exchange.coinbase.com"
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests

    def _rate_limit(self):
        """Ensure we don't spam the API"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def _make_request(self, endpoint, params=None):
        """Make API request with rate limiting"""
        self._rate_limit()

        url = f"{self.base_url}{endpoint}"

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json'
        }

        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f">> Coinbase API Error: {e}")
            raise

    def normalize_symbol(self, symbol):
        """
        Normalize symbol format to Coinbase format (BTC-USD)
        Examples:
            'BTC' -> 'BTC-USD'
            'BTCUSD' -> 'BTC-USD'
            'BTC-USD' -> 'BTC-USD'
            'ETH' -> 'ETH-USD'
        """
        symbol = symbol.upper().replace('/', '-')

        # If already has dash, return as-is
        if '-' in symbol:
            return symbol

        # If ends with USD/USDT, add dash
        if symbol.endswith('USD'):
            base = symbol[:-3]
            return f"{base}-USD"
        elif symbol.endswith('USDT'):
            base = symbol[:-4]
            return f"{base}-USD"
        else:
            # Add -USD as default quote currency
            return f"{symbol}-USD"

    def get_candles(self, symbol, granularity=300, start=None, end=None):
        """
        Get candlestick data from Coinbase

        Args:
            symbol: Trading pair (e.g., 'BTC-USD', 'BTC', 'ETH-USD')
            granularity: Candle duration in seconds (60, 300, 900, 3600, 21600, 86400)
            start: Start time ISO 8601 or unix timestamp
            end: End time ISO 8601 or unix timestamp

        Returns:
            DataFrame with Date, Open, High, Low, Close, Volume
        """
        symbol = self.normalize_symbol(symbol)

        params = {
            'granularity': granularity
        }

        if start:
            params['start'] = start
        if end:
            params['end'] = end

        print(f">> Fetching Coinbase data: {symbol} granularity={granularity}s")

        data = self._make_request(f'/products/{symbol}/candles', params)

        if not data:
            raise Exception(f"No data found for {symbol}")

        # Coinbase returns: [time, low, high, open, close, volume]
        df = pd.DataFrame(data, columns=['timestamp', 'low', 'high', 'open', 'close', 'volume'])

        # Convert to correct types
        df['Date'] = pd.to_datetime(df['timestamp'], unit='s')
        df['Open'] = df['open'].astype(float)
        df['High'] = df['high'].astype(float)
        df['Low'] = df['low'].astype(float)
        df['Close'] = df['close'].astype(float)
        df['Volume'] = df['volume'].astype(float)

        # Sort by date (Coinbase returns newest first)
        df = df.sort_values('Date')

        # Keep only needed columns
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

        print(f">> Fetched {len(df)} candles for {symbol}")
        if not df.empty:
            first_candle = df.iloc[0]['Date']
            last_candle = df.iloc[-1]['Date']
            print(f">> Candle range: {first_candle} to {last_candle}")

        return df

    def get_data_by_period(self, symbol, interval='5m', period='1d'):
        """
        Get data for a specific period (matches yfinance interface)

        Args:
            symbol: Trading pair
            interval: '1m', '5m', '15m', '1h', '6h', '1d'
            period: '1d', '5d', '1mo', '3mo', '6mo', '1y', '5y', 'max'

        Returns:
            DataFrame with candlestick data
        """
        # Map intervals to Coinbase granularity (in seconds)
        interval_map = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '1h': 3600,
            '6h': 21600,
            '1d': 86400
        }

        granularity = interval_map.get(interval, 300)

        # Calculate date range based on period
        # Coinbase max is 300 candles, so we need to handle this intelligently
        end_time = datetime.now()

        # Map period to timedelta
        period_map = {
            '1d': timedelta(days=1),
            '5d': timedelta(days=5),
            '1mo': timedelta(days=30),
            '3mo': timedelta(days=90),
            '6mo': timedelta(days=180),
            '1y': timedelta(days=365),
            '5y': timedelta(days=1825),
            'max': timedelta(days=3650)  # ~10 years
        }

        period_delta = period_map.get(period, timedelta(days=1))
        start_time = end_time - period_delta

        # For periods that would exceed 300 candles, we need to make multiple requests
        # Calculate number of candles requested
        total_seconds = period_delta.total_seconds()
        num_candles = int(total_seconds / granularity)

        print(f">> DEBUG: period={period}, granularity={granularity}s, num_candles={num_candles}")

        if num_candles <= 300:
            # Single request is enough
            return self.get_candles(symbol, granularity,
                                  start=start_time.isoformat(),
                                  end=end_time.isoformat())
        else:
            # Need multiple requests - fetch in chunks and combine
            print(f">> Period {period} requires {num_candles} candles, fetching in chunks...")

            all_data = []
            chunk_duration = timedelta(seconds=300 * granularity)
            current_end = end_time

            # For 1-minute intervals, limit to last 300 candles (5 hours) to get most recent data
            # Coinbase API has a delay, so request up to NOW
            # The API will return the most recent complete candles it has
            if granularity == 60:  # 1-minute interval
                print(f">> 1-minute interval: fetching last 300 candles only for recent data")
                # Don't set an end time - let Coinbase return the most recent data it has
                return self.get_candles(symbol, granularity,
                                      start=(end_time - timedelta(seconds=300 * granularity)).isoformat())

            while current_end > start_time:
                current_start = max(current_end - chunk_duration, start_time)

                df_chunk = self.get_candles(symbol, granularity,
                                          start=current_start.isoformat(),
                                          end=current_end.isoformat())

                if not df_chunk.empty:
                    all_data.append(df_chunk)

                current_end = current_start - timedelta(seconds=granularity)

                # Limit to prevent excessive requests (max 10 chunks = 3000 candles)
                if len(all_data) >= 10:
                    break

            if not all_data:
                raise Exception(f"No data fetched for {symbol}")

            # Combine all chunks
            result = pd.concat(all_data, ignore_index=True)
            result = result.sort_values('Date').drop_duplicates(subset=['Date'])
            result = result.reset_index(drop=True)

            print(f">> Combined {len(result)} candles from {len(all_data)} chunks")

            return result

    def get_24h_ticker(self, symbol):
        """Get 24-hour ticker statistics"""
        symbol = self.normalize_symbol(symbol)

        data = self._make_request(f'/products/{symbol}/ticker')

        return {
            'symbol': symbol,
            'price': float(data.get('price', 0)),
            'volume': float(data.get('volume', 0)),
            'time': data.get('time')
        }

    def get_all_symbols(self):
        """Get all available trading pairs"""
        data = self._make_request('/products')

        # Filter for USD pairs that are trading
        symbols = []
        for product in data:
            if product['quote_currency'] == 'USD' and product['status'] == 'online':
                symbols.append({
                    'symbol': product['id'],
                    'base': product['base_currency'],
                    'quote': product['quote_currency']
                })

        return symbols
