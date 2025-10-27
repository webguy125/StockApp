"""
Yahoo Finance Client for Cryptocurrency Historical Data
- Full historical data back to 2014 for major cryptos
- Best for daily, weekly, monthly candles
- Free and reliable
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

class YahooFinanceClient:
    def __init__(self):
        self.supported_symbols = {
            'BTC-USD': 'BTC-USD',
            'ETH-USD': 'ETH-USD',
            'SOL-USD': 'SOL-USD',
            'XRP-USD': 'XRP-USD',
            'DOGE-USD': 'DOGE-USD',
            'BNB-USD': 'BNB-USD',
            'ADA-USD': 'ADA-USD',
            'AVAX-USD': 'AVAX-USD',
            'DOT-USD': 'DOT1-USD',  # Polkadot has different ticker
            'MATIC-USD': 'MATIC-USD',
            'LINK-USD': 'LINK-USD',
            'UNI-USD': 'UNI7083-USD',  # Uniswap different ticker
            'AAVE-USD': 'AAVE-USD',
        }

    def normalize_symbol(self, symbol):
        """Convert to Yahoo Finance ticker format"""
        symbol = symbol.upper()
        return self.supported_symbols.get(symbol, symbol)

    def get_data_by_period(self, symbol, interval='1d', period='1d'):
        """
        Get historical data from Yahoo Finance

        Args:
            symbol: Trading pair (e.g., 'BTC-USD')
            interval: '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'
            period: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'

        Returns:
            DataFrame with Date, Open, High, Low, Close, Volume
        """
        yf_symbol = self.normalize_symbol(symbol)

        # Yahoo Finance interval mapping
        yf_interval_map = {
            '1m': '1m',
            '2m': '2m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '45m': '1h',   # YF doesn't have 45m, use 1h
            '1h': '1h',
            '2h': '1h',    # Will aggregate
            '4h': '1h',    # Will aggregate
            '12h': '1h',   # Will aggregate
            '1d': '1d',
            '3d': '1d',    # Will aggregate
            '5d': '5d',
            '1wk': '1wk',
            '2wk': '1wk',  # Will aggregate
            '1mo': '1mo',
        }

        yf_interval = yf_interval_map.get(interval, '1d')

        # For maximum historical data on daily+, use 'max'
        # Yahoo Finance 'max' period gives us data back to 2014 for BTC!
        if interval in ['1d', '3d', '5d', '1wk', '2wk', '1mo']:
            use_period = 'max'  # Get ALL available history
        else:
            # For intraday, respect the period limits (YF has restrictions)
            period_map = {
                '1d': '1d',
                '5d': '5d',
                '1mo': '1mo',
                '3mo': '3mo',
                '6mo': '6mo',
                '1y': '1y',
                '5y': '5y',
                'ytd': 'ytd',
                'max': 'max'
            }
            use_period = period_map.get(period, '1mo')

        print(f">> Yahoo Finance: Fetching {yf_symbol} interval={yf_interval} period={use_period}")

        try:
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(period=use_period, interval=yf_interval)

            if df.empty:
                print(f">> WARNING: No data returned from Yahoo Finance for {yf_symbol}")
                return pd.DataFrame()

            # Reset index to make Date a column
            df = df.reset_index()

            # Rename columns to match our format
            df = df.rename(columns={
                'Date': 'Date',
                'Datetime': 'Date',  # Intraday uses 'Datetime'
            })

            # Keep only needed columns
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

            # print(f">> Yahoo Finance: Fetched {len(df)} candles from {df.iloc[0]['Date']} to {df.iloc[-1]['Date']}")  # Commented out - too verbose

            # Aggregate if needed (for 2h, 3d, 12h, etc.)
            if interval in ['2h', '4h', '12h', '3d', '2wk']:
                df = self._aggregate_candles(df, interval)

            return df

        except Exception as e:
            print(f">> ERROR: Yahoo Finance fetch failed for {yf_symbol}: {e}")
            return pd.DataFrame()

    def _aggregate_candles(self, df, target_interval):
        """Aggregate candles for intervals not natively supported"""
        if df.empty:
            return df

        aggregation_rules = {
            '2h': '2H',
            '4h': '4H',
            '12h': '12H',
            '3d': '3D',
            '2wk': '2W',
        }

        freq = aggregation_rules.get(target_interval)
        if not freq:
            return df

        # Set Date as index for resampling
        df = df.set_index('Date')

        # Resample and aggregate
        agg_df = df.resample(freq).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()

        agg_df = agg_df.reset_index()

        # print(f">> Aggregated {len(df)} candles to {len(agg_df)} {target_interval} candles")  # Commented out - too verbose

        return agg_df
