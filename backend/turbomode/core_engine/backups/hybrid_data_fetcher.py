
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

"""
Hybrid Data Fetcher for Master Market Data DB
Primary: IBKR Gateway (300x faster, higher quality)
Fallback: yfinance (for when IBKR unavailable)

Architecture: MASTER_MARKET_DATA_ARCHITECTURE.json v1.1
Author: Master Data System
Date: 2026-01-06
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ib_insync import IB, Stock, util
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple
import logging
import time

logger = logging.getLogger('hybrid_fetcher')


class HybridDataFetcher:
    """
    Hybrid data fetcher with IBKR primary + yfinance fallback

    Features:
    - IBKR Gateway for 300x faster fetching (50 req/sec vs yfinance's ~1/sec)
    - Automatic fallback to yfinance if IBKR unavailable
    - Unified interface - returns same format regardless of source
    - Connection pooling and rate limiting
    """

    def __init__(self, ibkr_host='127.0.0.1', ibkr_port=4002, use_ibkr=True):
        """
        Initialize hybrid data fetcher

        Args:
            ibkr_host: IBKR Gateway host
            ibkr_port: IBKR Gateway port (4002 = paper, 7496 = live)
            use_ibkr: Whether to attempt IBKR connection
        """
        self.use_ibkr = use_ibkr
        self.ibkr_available = False
        self.ib = None
        self.ibkr_host = ibkr_host
        self.ibkr_port = ibkr_port
        self.last_ibkr_check = None
        self.ibkr_check_interval = 60  # Recheck IBKR every 60 seconds if previously failed

        if use_ibkr:
            try:
                self.ib = IB()
                self.ib.connect(ibkr_host, ibkr_port, clientId=999, readonly=True)
                self.ibkr_available = self._ibkr_connection_is_fully_ready()
                self.last_ibkr_check = time.time()
                logger.info(f"[HYBRID] Connected to IBKR Gateway on port {ibkr_port}")
            except Exception as e:
                logger.warning(f"[HYBRID] IBKR unavailable ({e}), using yfinance only - will retry in 60s")
                self.ibkr_available = False
                self.last_ibkr_check = time.time()

    def _ibkr_connection_is_fully_ready(self) -> bool:
        """
        Test if IBKR connection is fully operational (including event loop availability)

        Returns:
            True if IBKR is ready for data fetching, False otherwise
        """
        if not self.ib or not self.ib.isConnected():
            return False

        try:
            # Test a simple operation that requires event loop
            # Use a contract that should always work
            test_contract = Stock('AAPL', 'SMART', 'USD')
            contracts = self.ib.qualifyContracts(test_contract)

            if contracts:
                logger.info("[HYBRID] IBKR connection fully validated (event loop OK)")
                return True
            else:
                logger.warning("[HYBRID] IBKR connected but contract qualification failed")
                return False

        except Exception as e:
            logger.warning(f"[HYBRID] IBKR connected but not fully operational ({e}), using yfinance only")
            return False

    def _try_reconnect_ibkr(self) -> bool:
        """
        Attempt to reconnect to IBKR if it was previously unavailable

        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Close old connection if exists
            if self.ib:
                try:
                    self.ib.disconnect()
                except:
                    pass

            # Create new connection
            self.ib = IB()
            self.ib.connect(self.ibkr_host, self.ibkr_port, clientId=999, readonly=True)

            # Test if fully ready
            if self._ibkr_connection_is_fully_ready():
                logger.info("[HYBRID] Successfully reconnected to IBKR Gateway - switching from Yahoo to IBKR")
                return True
            else:
                logger.warning("[HYBRID] IBKR connection established but not fully ready")
                return False

        except Exception as e:
            logger.debug(f"[HYBRID] IBKR reconnection attempt failed: {e}")
            return False

    def disconnect(self):
        """Disconnect from IBKR Gateway"""
        if self.ib and self.ibkr_available:
            self.ib.disconnect()
            logger.info("[HYBRID] Disconnected from IBKR")

    def fetch_candles_ibkr(self, symbol: str, duration: str = '10 Y', bar_size: str = '1 day') -> Optional[pd.DataFrame]:
        """
        Fetch historical candles from IBKR

        Args:
            symbol: Stock ticker
            duration: Data duration (e.g., '10 Y', '1 M', '5 D')
            bar_size: Bar size (e.g., '1 day', '1 hour', '5 mins')

        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            # Handle crypto symbols
            if symbol.endswith('-USD'):
                crypto_symbol = symbol.replace('-USD', 'USD')
                contract = Stock(crypto_symbol, 'PAXOS', 'USD')
            else:
                contract = Stock(symbol, 'SMART', 'USD')

            # Qualify contract
            contracts = self.ib.qualifyContracts(contract)
            if not contracts:
                logger.warning(f"[IBKR] Failed to qualify {symbol}")
                return None

            contract = contracts[0]

            # Request historical data
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1
            )

            if not bars:
                logger.warning(f"[IBKR] No data returned for {symbol}")
                return None

            # Convert to DataFrame
            df = util.df(bars)

            # Rename columns to match standard format
            df.rename(columns={
                'date': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }, inplace=True)

            # Set Date as index
            df.set_index('Date', inplace=True)

            logger.info(f"[IBKR] Fetched {len(df)} bars for {symbol}")

            # Rate limiting (50 req/sec max)
            time.sleep(0.02)

            return df

        except Exception as e:
            logger.error(f"[IBKR] Error fetching {symbol}: {e}")
            return None

    def fetch_candles_yfinance(self, symbol: str, period: str = '10y', interval: str = '1d') -> Optional[pd.DataFrame]:
        """
        Fetch historical candles from yfinance (fallback)

        Args:
            symbol: Stock ticker
            period: Data period (e.g., '10y', '1mo', '5d')
            interval: Bar interval (e.g., '1d', '1h', '5m')

        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)

            if df.empty:
                logger.warning(f"[YFINANCE] No data returned for {symbol}")
                return None

            logger.info(f"[YFINANCE] Fetched {len(df)} bars for {symbol}")

            # Rate limiting (be nice to yfinance)
            time.sleep(1)

            return df

        except Exception as e:
            logger.error(f"[YFINANCE] Error fetching {symbol}: {e}")
            return None

    def fetch_candles(self, symbol: str, period: str = '10y', interval: str = '1d') -> Optional[pd.DataFrame]:
        """
        Fetch candles using best available source

        Args:
            symbol: Stock ticker
            period: Data period (e.g., '10y', '1mo')
            interval: Bar interval (e.g., '1d', '1h')

        Returns:
            DataFrame with OHLCV data or None if both sources fail
        """
        # Check if we should retry IBKR connection (if previously failed and 60+ seconds have passed)
        if self.use_ibkr and not self.ibkr_available and self.last_ibkr_check:
            time_since_last_check = time.time() - self.last_ibkr_check
            if time_since_last_check >= self.ibkr_check_interval:
                logger.info(f"[HYBRID] Attempting to reconnect to IBKR ({int(time_since_last_check)}s since last check)...")
                self.ibkr_available = self._try_reconnect_ibkr()
                self.last_ibkr_check = time.time()

        # Try IBKR first if available
        if self.ibkr_available:
            try:
                # Convert period/interval to IBKR format
                duration_map = {
                    '1d': '1 D', '5d': '5 D', '1mo': '1 M', '3mo': '3 M',
                    '6mo': '6 M', '1y': '1 Y', '2y': '2 Y', '5y': '5 Y',
                    '10y': '10 Y', 'max': '20 Y'
                }

                bar_size_map = {
                    '1m': '1 min', '5m': '5 mins', '15m': '15 mins',
                    '30m': '30 mins', '1h': '1 hour', '1d': '1 day',
                    '1wk': '1 week', '1mo': '1 month'
                }

                duration = duration_map.get(period, '10 Y')
                bar_size = bar_size_map.get(interval, '1 day')

                df = self.fetch_candles_ibkr(symbol, duration, bar_size)

                if df is not None and not df.empty:
                    logger.info(f"[HYBRID] [OK] {symbol} fetched from IBKR")
                    return df
                else:
                    logger.warning(f"[HYBRID] IBKR returned no data for {symbol}, falling back to yfinance")

            except Exception as e:
                logger.warning(f"[HYBRID] IBKR exception for {symbol} ({e}), falling back to yfinance")

        # Fallback to yfinance
        df = self.fetch_candles_yfinance(symbol, period, interval)

        if df is not None and not df.empty:
            logger.info(f"[HYBRID] ✓ {symbol} fetched from yfinance")
            return df

        logger.error(f"[HYBRID] ✗ Failed to fetch {symbol} from both sources")
        return None

    def fetch_fundamentals(self, symbol: str) -> Dict:
        """
        Fetch fundamental data (always uses yfinance for now)
        IBKR fundamental data requires additional subscriptions

        Args:
            symbol: Stock ticker

        Returns:
            Dictionary of fundamental metrics
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info
        except Exception as e:
            logger.error(f"[YFINANCE] Error fetching fundamentals for {symbol}: {e}")
            return {}

    def fetch_splits_and_dividends(self, symbol: str) -> Tuple[pd.Series, pd.Series]:
        """
        Fetch stock splits and dividends (uses yfinance)

        Args:
            symbol: Stock ticker

        Returns:
            Tuple of (splits Series, dividends Series)
        """
        try:
            ticker = yf.Ticker(symbol)
            splits = ticker.splits
            dividends = ticker.dividends
            return (splits, dividends)
        except Exception as e:
            logger.error(f"[YFINANCE] Error fetching splits/dividends for {symbol}: {e}")
            return (pd.Series(), pd.Series())


if __name__ == '__main__':
    # Test hybrid fetcher
    print("=" * 80)
    print("HYBRID DATA FETCHER - TEST")
    print("=" * 80)

    print("\nAttempting to connect to IBKR Gateway...")
    print("(If this fails, we'll automatically fall back to yfinance)\n")

    fetcher = HybridDataFetcher(use_ibkr=True)

    test_symbols = ['AAPL', 'TSLA', 'BTC-USD']

    for symbol in test_symbols:
        print(f"\n[TEST] Fetching {symbol}...")
        df = fetcher.fetch_candles(symbol, period='1y', interval='1d')

        if df is not None:
            print(f"  ✓ Success! Retrieved {len(df)} bars")
            print(f"    Date range: {df.index[0]} to {df.index[-1]}")
            print(f"    Latest close: ${df['Close'].iloc[-1]:.2f}")
        else:
            print(f"  ✗ Failed to fetch {symbol}")

    fetcher.disconnect()

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
