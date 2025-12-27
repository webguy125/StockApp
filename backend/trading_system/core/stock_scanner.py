"""
Stock Scanner
Filters S&P 500 to promising candidates based on volume, volatility, and price action
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any
import requests
from io import StringIO


class StockScanner:
    """
    Scans S&P 500 stocks and filters to promising candidates
    """

    def __init__(self, min_volume: float = 1_000_000, min_price: float = 5.0, max_price: float = 500.0):
        """
        Initialize scanner with filter criteria

        Args:
            min_volume: Minimum average daily volume
            min_price: Minimum stock price (avoid penny stocks)
            max_price: Maximum stock price
        """
        self.min_volume = min_volume
        self.min_price = min_price
        self.max_price = max_price
        self.sp500_symbols = []

    def get_sp500_list(self) -> List[str]:
        """Fetch S&P 500 ticker list from Wikipedia"""
        try:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

            # Add User-Agent header to avoid 403 Forbidden error
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            # Fetch page with requests first (pandas.read_html doesn't support headers)
            response = requests.get(url, headers=headers)
            response.raise_for_status()

            # Parse HTML with pandas (use StringIO to avoid FutureWarning)
            tables = pd.read_html(StringIO(response.text))
            df = tables[0]
            symbols = df['Symbol'].tolist()

            # Clean symbols (fix special characters)
            symbols = [s.replace('.', '-') for s in symbols]

            self.sp500_symbols = symbols
            print(f"[OK] Loaded {len(symbols)} S&P 500 symbols from Wikipedia")
            return symbols

        except Exception as e:
            print(f"[ERROR] Error fetching S&P 500 list: {e}")
            # Fallback to a subset of major stocks
            self.sp500_symbols = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B',
                'JPM', 'JNJ', 'V', 'PG', 'XOM', 'UNH', 'MA', 'HD', 'CVX', 'MRK',
                'ABBV', 'PEP', 'KO', 'AVGO', 'COST', 'LLY', 'WMT', 'MCD', 'CSCO',
                'ACN', 'ABT', 'TMO', 'DIS', 'ADBE', 'VZ', 'NKE', 'CMCSA', 'NFLX',
                'CRM', 'INTC', 'PM', 'WFC', 'AMD', 'BA', 'GE', 'CAT', 'IBM'
            ]
            print(f"[WARNING] Using fallback list: {len(self.sp500_symbols)} symbols")
            return self.sp500_symbols

    def calculate_volatility(self, df: pd.DataFrame) -> float:
        """Calculate ATR-based volatility as percentage"""
        if len(df) < 14:
            return 0.0

        # Calculate True Range
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift())
        low_close = abs(df['Low'] - df['Close'].shift())

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean().iloc[-1]

        # Convert to percentage
        current_price = df['Close'].iloc[-1]
        if current_price > 0:
            return (atr / current_price) * 100
        return 0.0

    def scan(self, lookback_days: int = 90, max_results: int = 500) -> List[Dict[str, Any]]:
        """
        Scan ENTIRE S&P 500 - NO FILTERS (analyze all stocks)

        Args:
            lookback_days: Days of historical data to analyze
            max_results: Maximum number of stocks to return (default 500 = all S&P 500)

        Returns:
            List of dicts with stock info and metrics
        """
        if not self.sp500_symbols:
            self.get_sp500_list()

        candidates = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days + 10)  # Extra buffer

        print(f"\n[SCAN] Scanning ENTIRE S&P 500 ({len(self.sp500_symbols)} stocks) - NO FILTERS...")

        for i, symbol in enumerate(self.sp500_symbols):
            if i % 50 == 0:
                print(f"Progress: {i}/{len(self.sp500_symbols)}")

            try:
                # Fetch data
                df = yf.download(symbol, start=start_date, end=end_date, progress=False)

                if df.empty or len(df) < 20:
                    continue

                # Flatten multi-index columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                # Calculate metrics
                current_price = float(df['Close'].iloc[-1])
                avg_volume = float(df['Volume'].tail(20).mean())
                volatility = self.calculate_volatility(df)

                # NO FILTERS - include ALL stocks regardless of price/volume

                # Calculate additional signals
                sma_20 = df['Close'].rolling(window=20).mean().iloc[-1]
                price_change_pct = ((current_price - df['Close'].iloc[-20]) / df['Close'].iloc[-20]) * 100

                # Volume spike detection
                recent_volume = df['Volume'].iloc[-1]
                volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0

                candidates.append({
                    'symbol': symbol,
                    'price': current_price,
                    'volume': avg_volume,
                    'volatility': volatility,
                    'price_change_20d': price_change_pct,
                    'above_sma20': current_price > sma_20,
                    'volume_spike': volume_ratio > 1.5,
                    'score': 0.0  # Will be filled by analyzers
                })

            except Exception as e:
                # Silently skip errors for individual stocks
                continue

        # Sort by volatility (most volatile = most opportunity)
        candidates.sort(key=lambda x: x['volatility'], reverse=True)

        # Limit results
        candidates = candidates[:max_results]

        print(f"\n[OK] Scan complete: {len(candidates)} candidates found from {len(self.sp500_symbols)} S&P 500 stocks")
        return candidates

    def scan_crypto(self, top_n: int = 100) -> List[Dict[str, Any]]:
        """
        Scan top 100 cryptocurrencies

        Args:
            top_n: Number of top cryptos to scan (default: 100)

        Returns:
            List of dicts with crypto info
        """
        # Top 100 cryptocurrencies by market cap
        crypto_symbols = [
            'BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'DOGE-USD',
            'SOL-USD', 'MATIC-USD', 'DOT-USD', 'AVAX-USD', 'LINK-USD', 'UNI-USD',
            'LTC-USD', 'ATOM-USD', 'XLM-USD', 'ALGO-USD', 'VET-USD', 'FIL-USD',
            'TRX-USD', 'ETC-USD', 'NEAR-USD', 'HBAR-USD', 'APT-USD', 'QNT-USD',
            'ICP-USD', 'ARB-USD', 'STX-USD', 'IMX-USD', 'INJ-USD', 'MKR-USD',
            'GRT-USD', 'RNDR-USD', 'RUNE-USD', 'FTM-USD', 'AAVE-USD', 'SNX-USD',
            'SAND-USD', 'MANA-USD', 'AXS-USD', 'EGLD-USD', 'XTZ-USD', 'THETA-USD',
            'EOS-USD', 'KAVA-USD', 'ZEC-USD', 'DASH-USD', 'COMP-USD', 'YFI-USD',
            'CRV-USD', 'BAT-USD', 'ENJ-USD', 'ZRX-USD', 'LRC-USD', 'SUSHI-USD',
            'ANKR-USD', 'STORJ-USD', 'SKL-USD', 'REN-USD', 'KNC-USD', '1INCH-USD',
            'OMG-USD', 'BNT-USD', 'NMR-USD', 'UMA-USD', 'BAL-USD', 'BAND-USD',
            'RSR-USD', 'COTI-USD', 'OCEAN-USD', 'SRM-USD', 'AUDIO-USD', 'PAXG-USD',
            'CHZ-USD', 'HOT-USD', 'ICX-USD', 'ZIL-USD', 'ONT-USD', 'QTUM-USD',
            'WAVES-USD', 'IOTA-USD', 'NEO-USD', 'LSK-USD', 'SC-USD', 'DGB-USD',
            'STEEM-USD', 'DCR-USD', 'KMD-USD', 'ARDR-USD', 'STRAT-USD', 'NXT-USD',
            'XEM-USD', 'MAID-USD', 'GAS-USD', 'POWR-USD', 'REQ-USD', 'RLC-USD',
            'POLY-USD', 'DENT-USD', 'FUN-USD', 'CVC-USD'
        ]

        candidates = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=40)

        print(f"\n[CRYPTO] Scanning top {min(top_n, len(crypto_symbols))} cryptocurrencies...")

        for symbol in crypto_symbols[:top_n]:
            try:
                df = yf.download(symbol, start=start_date, end=end_date, progress=False)

                if df.empty or len(df) < 20:
                    continue

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                current_price = float(df['Close'].iloc[-1])
                avg_volume = float(df['Volume'].tail(20).mean())
                volatility = self.calculate_volatility(df)

                candidates.append({
                    'symbol': symbol,
                    'price': current_price,
                    'volume': avg_volume,
                    'volatility': volatility,
                    'asset_type': 'crypto',
                    'score': 0.0
                })

            except Exception as e:
                continue

        candidates.sort(key=lambda x: x['volatility'], reverse=True)

        print(f"[OK] Crypto scan complete: {len(candidates)} candidates found")
        return candidates

    def __repr__(self):
        return f"<StockScanner symbols={len(self.sp500_symbols)} filters=(vol>{self.min_volume}, ${self.min_price}-${self.max_price})>"
