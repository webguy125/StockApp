"""
Binance API Client for Crypto Data
- REST API for historical candlestick data
- WebSocket streams handled separately
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time

class BinanceClient:
    def __init__(self):
        self.base_url = "https://api.binance.us"  # Use Binance US for US-based access
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests (to be safe)

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
            print(f"❌ Binance API Error: {e}")
            raise

    def normalize_symbol(self, symbol):
        """
        Normalize symbol format to Binance format
        Examples:
            'BTC-USD' -> 'BTCUSDT'
            'BTC' -> 'BTCUSDT'
            'ETHUSDT' -> 'ETHUSDT'
            'ETH-USDT' -> 'ETHUSDT'
        """
        symbol = symbol.upper().replace('-', '').replace('/', '')

        # Handle USD -> USDT conversion first
        if symbol.endswith('USD') and not symbol.endswith('USDT') and not symbol.endswith('BUSD'):
            symbol = symbol[:-3] + 'USDT'

        # If symbol doesn't end with a quote currency, add USDT
        # Only check for trading pair quote currencies (not base currencies like BTC, ETH)
        quote_currencies = ['USDT', 'BUSD']
        has_quote = any(symbol.endswith(q) for q in quote_currencies)

        if not has_quote:
            # Add USDT as default quote currency
            symbol = f"{symbol}USDT"

        return symbol

    def get_klines(self, symbol, interval='5m', limit=500, start_time=None, end_time=None):
        """
        Get candlestick data (klines)

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT', 'BTC-USD', 'BTC')
            interval: '1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1M'
            limit: Number of candles (max 1000)
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds

        Returns:
            DataFrame with Date, Open, High, Low, Close, Volume
        """
        symbol = self.normalize_symbol(symbol)

        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': min(limit, 1000)  # Binance max is 1000
        }

        if start_time:
            params['startTime'] = int(start_time)
        if end_time:
            params['endTime'] = int(end_time)

        print(f"📊 Fetching Binance data: {symbol} {interval} (limit={params['limit']})")

        data = self._make_request('/api/v3/klines', params)

        if not data:
            raise Exception(f"No data found for {symbol}")

        # Parse klines data
        # Format: [Open time, Open, High, Low, Close, Volume, Close time, Quote volume, Trades, Taker buy base, Taker buy quote, Ignore]
        df = pd.DataFrame(data, columns=[
            'Open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Close_time', 'Quote_volume', 'Trades', 'Taker_buy_base',
            'Taker_buy_quote', 'Ignore'
        ])

        # Convert to correct types
        df['Date'] = pd.to_datetime(df['Open_time'], unit='ms')
        df['Open'] = df['Open'].astype(float)
        df['High'] = df['High'].astype(float)
        df['Low'] = df['Low'].astype(float)
        df['Close'] = df['Close'].astype(float)
        df['Volume'] = df['Volume'].astype(float)

        # Keep only needed columns
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

        print(f"✅ Fetched {len(df)} candles for {symbol}")

        return df

    def get_data_by_period(self, symbol, interval='5m', period='1d'):
        """
        Get data for a specific period (matches yfinance interface)

        Args:
            symbol: Trading pair
            interval: '1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w'
            period: '1d', '5d', '1mo', '3mo', '6mo', '1y', '5y', 'max'

        Returns:
            DataFrame with candlestick data
        """
        # Calculate limit based on period and interval
        period_to_candles = {
            '1d': {'1m': 1440, '5m': 288, '15m': 96, '30m': 48, '1h': 24, '4h': 6, '1d': 1},
            '5d': {'1m': 1000, '5m': 1000, '15m': 480, '30m': 240, '1h': 120, '4h': 30, '1d': 5},
            '1mo': {'5m': 1000, '15m': 1000, '30m': 1000, '1h': 720, '4h': 180, '1d': 30},
            '3mo': {'15m': 1000, '30m': 1000, '1h': 1000, '4h': 540, '1d': 90},
            '6mo': {'1h': 1000, '4h': 1000, '1d': 180},
            '1y': {'4h': 1000, '1d': 365, '1w': 52},
            '5y': {'1d': 1000, '1w': 260, '1M': 60},
            'max': {'1d': 1000, '1w': 1000, '1M': 1000}
        }

        # Map intervals
        interval_map = {
            '1m': '1m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '1h',
            '4h': '4h',
            '1d': '1d',
            '1wk': '1w',
            '1w': '1w',
            '1mo': '1M',
            '1M': '1M'
        }

        binance_interval = interval_map.get(interval, interval)

        # Get limit for this period/interval combo
        limit = 500  # default
        if period in period_to_candles and binance_interval in period_to_candles[period]:
            limit = min(period_to_candles[period][binance_interval], 1000)

        return self.get_klines(symbol, binance_interval, limit)

    def get_24h_ticker(self, symbol):
        """Get 24-hour ticker statistics"""
        symbol = self.normalize_symbol(symbol)

        params = {'symbol': symbol}
        data = self._make_request('/api/v3/ticker/24hr', params)

        return {
            'symbol': data['symbol'],
            'price': float(data['lastPrice']),
            'change': float(data['priceChange']),
            'change_percent': float(data['priceChangePercent']),
            'high': float(data['highPrice']),
            'low': float(data['lowPrice']),
            'volume': float(data['volume']),
            'quote_volume': float(data['quoteVolume'])
        }

    def get_all_symbols(self):
        """Get all available trading pairs"""
        data = self._make_request('/api/v3/exchangeInfo')

        # Filter for USDT pairs that are trading
        symbols = []
        for symbol_info in data['symbols']:
            if symbol_info['status'] == 'TRADING' and symbol_info['symbol'].endswith('USDT'):
                symbols.append({
                    'symbol': symbol_info['symbol'],
                    'base': symbol_info['baseAsset'],
                    'quote': symbol_info['quoteAsset']
                })

        return symbols

    def get_server_time(self):
        """Get Binance server time"""
        data = self._make_request('/api/v3/time')
        return data['serverTime']
