"""
Alpha Vantage API Client
Provides stock data, crypto data, and technical indicators
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time

class AlphaVantageClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.last_request_time = 0
        self.min_request_interval = 0.2  # 200ms between requests (5 per second for paid plans)

    def _rate_limit(self):
        """Ensure we don't exceed rate limits"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def _make_request(self, params):
        """Make API request with rate limiting and error handling"""
        self._rate_limit()

        params['apikey'] = self.api_key

        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Check for API errors
            if 'Error Message' in data:
                raise Exception(f"Alpha Vantage Error: {data['Error Message']}")
            if 'Note' in data:
                raise Exception(f"Alpha Vantage Rate Limit: {data['Note']}")

            return data
        except Exception as e:
            print(f"❌ Alpha Vantage API Error: {e}")
            raise

    def get_intraday_data(self, symbol, interval='5min', outputsize='compact'):
        """
        Get intraday stock data

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            interval: '1min', '5min', '15min', '30min', '60min'
            outputsize: 'compact' (last 100 points) or 'full' (full history)

        Returns:
            DataFrame with Date, Open, High, Low, Close, Volume
        """
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': symbol,
            'interval': interval,
            'outputsize': outputsize
        }

        data = self._make_request(params)

        # Parse time series data
        time_series_key = f'Time Series ({interval})'
        if time_series_key not in data:
            raise Exception(f"No intraday data found for {symbol}")

        time_series = data[time_series_key]

        # Convert to DataFrame
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Rename columns
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df.astype(float)

        # Reset index to Date column
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Date'}, inplace=True)

        return df

    def get_daily_data(self, symbol, outputsize='compact'):
        """
        Get daily stock data

        Args:
            symbol: Stock symbol
            outputsize: 'compact' (last 100 days) or 'full' (20+ years)

        Returns:
            DataFrame with Date, Open, High, Low, Close, Volume
        """
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'outputsize': outputsize
        }

        data = self._make_request(params)

        if 'Time Series (Daily)' not in data:
            raise Exception(f"No daily data found for {symbol}")

        time_series = data['Time Series (Daily)']

        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df.astype(float)

        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Date'}, inplace=True)

        return df

    def get_weekly_data(self, symbol):
        """Get weekly stock data"""
        params = {
            'function': 'TIME_SERIES_WEEKLY',
            'symbol': symbol
        }

        data = self._make_request(params)

        if 'Weekly Time Series' not in data:
            raise Exception(f"No weekly data found for {symbol}")

        time_series = data['Weekly Time Series']

        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df.astype(float)

        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Date'}, inplace=True)

        return df

    def get_monthly_data(self, symbol):
        """Get monthly stock data"""
        params = {
            'function': 'TIME_SERIES_MONTHLY',
            'symbol': symbol
        }

        data = self._make_request(params)

        if 'Monthly Time Series' not in data:
            raise Exception(f"No monthly data found for {symbol}")

        time_series = data['Monthly Time Series']

        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df.astype(float)

        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Date'}, inplace=True)

        return df

    def get_crypto_intraday(self, symbol, market='USD', interval='5min'):
        """
        Get crypto intraday data

        Args:
            symbol: Crypto symbol (e.g., 'BTC', 'ETH')
            market: Market currency (e.g., 'USD')
            interval: '1min', '5min', '15min', '30min', '60min'

        Returns:
            DataFrame with Date, Open, High, Low, Close, Volume
        """
        params = {
            'function': 'CRYPTO_INTRADAY',
            'symbol': symbol,
            'market': market,
            'interval': interval
        }

        data = self._make_request(params)

        time_series_key = f'Time Series Crypto ({interval})'
        if time_series_key not in data:
            raise Exception(f"No crypto data found for {symbol}/{market}")

        time_series = data[time_series_key]

        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Crypto data has different column names
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df.astype(float)

        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Date'}, inplace=True)

        return df

    def get_crypto_daily(self, symbol, market='USD'):
        """Get daily crypto data"""
        params = {
            'function': 'DIGITAL_CURRENCY_DAILY',
            'symbol': symbol,
            'market': market
        }

        data = self._make_request(params)

        if 'Time Series (Digital Currency Daily)' not in data:
            raise Exception(f"No crypto daily data found for {symbol}/{market}")

        time_series = data['Time Series (Digital Currency Daily)']

        # Parse and extract USD values
        records = []
        for date_str, values in time_series.items():
            records.append({
                'Date': date_str,
                'Open': float(values[f'1a. open ({market})']),
                'High': float(values[f'2a. high ({market})']),
                'Low': float(values[f'3a. low ({market})']),
                'Close': float(values[f'4a. close ({market})']),
                'Volume': float(values['5. volume'])
            })

        df = pd.DataFrame(records)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        df.reset_index(drop=True, inplace=True)

        return df

    def get_stock_data(self, symbol, interval='1d', period=None, outputsize='compact'):
        """
        Unified method to get stock data (matches yfinance interface)

        Args:
            symbol: Stock symbol
            interval: '1m', '5m', '15m', '30m', '1h', '1d', '1wk', '1mo'
            period: Period string (e.g., '1d', '5d', '1mo', '1y') - determines outputsize
            outputsize: 'compact' or 'full'

        Returns:
            DataFrame compatible with existing code
        """
        # Determine outputsize based on period
        if period in ['1y', '5y', 'max'] or outputsize == 'full':
            outputsize = 'full'

        # Map intervals to Alpha Vantage format
        interval_map = {
            '1m': '1min',
            '5m': '5min',
            '15m': '15min',
            '30m': '30min',
            '1h': '60min',
            '60m': '60min'
        }

        # Check if crypto
        is_crypto = any(symbol.endswith(suffix) for suffix in ['-USD', '-USDT', '-BTC', '-ETH'])
        crypto_symbols = ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'SOL', 'DOGE', 'MATIC', 'DOT', 'AVAX']

        if is_crypto or symbol in crypto_symbols:
            # Handle crypto
            base_symbol = symbol.split('-')[0] if '-' in symbol else symbol
            market = symbol.split('-')[1] if '-' in symbol else 'USD'

            if interval in interval_map:
                return self.get_crypto_intraday(base_symbol, market, interval_map[interval])
            else:
                return self.get_crypto_daily(base_symbol, market)
        else:
            # Handle stocks
            if interval in interval_map:
                return self.get_intraday_data(symbol, interval_map[interval], outputsize)
            elif interval == '1d':
                return self.get_daily_data(symbol, outputsize)
            elif interval == '1wk':
                return self.get_weekly_data(symbol)
            elif interval == '1mo':
                return self.get_monthly_data(symbol)
            else:
                # Default to daily
                return self.get_daily_data(symbol, outputsize)
