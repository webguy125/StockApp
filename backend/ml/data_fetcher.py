"""
Data Fetcher Module
Fetches historical OHLCV data from yfinance (stocks) and ccxt (crypto)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# Lazy imports to avoid import errors if libraries not installed
_yfinance = None
_ccxt = None


def _import_yfinance():
    """Lazy import yfinance"""
    global _yfinance
    if _yfinance is None:
        try:
            import yfinance as yf
            _yfinance = yf
        except ImportError:
            logger.error("❌ yfinance not installed. Install with: pip install yfinance")
            raise
    return _yfinance


def _import_ccxt():
    """Lazy import ccxt"""
    global _ccxt
    if _ccxt is None:
        try:
            import ccxt
            _ccxt = ccxt
        except ImportError:
            logger.error("❌ ccxt not installed. Install with: pip install ccxt")
            raise
    return _ccxt


def detect_asset_type(symbol):
    """
    Detect if symbol is crypto or stock based on format

    Args:
        symbol: Asset symbol (e.g., 'BTC/USD', 'AAPL', 'BTC-USD')

    Returns:
        str: 'crypto' or 'stock'
    """
    # Crypto typically has / or - separator
    if '/' in symbol or '-' in symbol:
        return 'crypto'
    return 'stock'


def fetch_stock_ohlcv(symbol, days=1095):
    """
    Fetch historical stock data using yfinance

    Args:
        symbol: Stock ticker (e.g., 'AAPL', 'TSLA')
        days: Number of days of history (default: 1095 = 3 years)

    Returns:
        pandas DataFrame with columns [timestamp, open, high, low, close, volume]
    """
    try:
        yf = _import_yfinance()

        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        logger.info(f"📊 Fetching {days} days of stock data for {symbol}...")

        # Fetch data
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval='1d')

        if df.empty:
            logger.warning(f"⚠️ No data returned for {symbol}")
            return pd.DataFrame()

        # Normalize column names
        df.columns = [col.lower() for col in df.columns]

        # Reset index to make date a column
        df.reset_index(inplace=True)

        # Rename columns - handle various index names
        rename_map = {}
        if 'date' in df.columns:
            rename_map['date'] = 'timestamp'
        if 'datetime' in df.columns:
            rename_map['datetime'] = 'timestamp'
        if df.index.name and df.index.name.lower() in ['date', 'datetime']:
            df.index.name = 'timestamp'

        df.rename(columns=rename_map, inplace=True)

        # If still no timestamp column, use index
        if 'timestamp' not in df.columns:
            df['timestamp'] = df.index

        # Ensure required columns
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = df[[col for col in required_cols if col in df.columns]]

        # Convert to float32
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = df[col].astype(np.float32)

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        logger.info(f"✅ Fetched {len(df)} bars for {symbol}")
        return df

    except Exception as e:
        logger.error(f"❌ Stock fetch failed for {symbol}: {e}")
        return pd.DataFrame()


def fetch_crypto_ohlcv(symbol, days=30, timeframe='1m'):
    """
    Fetch historical crypto data using ccxt (Coinbase)

    Args:
        symbol: Crypto pair (e.g., 'BTC/USD', 'ETH/USD')
        days: Number of days of history (default: 30)
        timeframe: Timeframe ('1m', '5m', '15m', '1h', '1d', etc.)

    Returns:
        pandas DataFrame with columns [timestamp, open, high, low, close, volume]
    """
    try:
        ccxt_lib = _import_ccxt()

        # Normalize symbol format for Coinbase
        if '-' in symbol:
            symbol = symbol.replace('-', '/')

        logger.info(f"📊 Fetching {days} days of crypto data for {symbol} ({timeframe})...")

        # Initialize Coinbase Pro exchange
        exchange = ccxt_lib.coinbasepro()

        # Calculate time range
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

        # Fetch OHLCV data
        # Note: ccxt returns [[timestamp, open, high, low, close, volume], ...]
        ohlcv_data = []

        # Fetch in chunks (exchanges limit requests)
        current_time = start_time
        chunk_size = 300  # Most exchanges limit to 300-500 candles per request

        while current_time < end_time:
            try:
                candles = exchange.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    since=current_time,
                    limit=chunk_size
                )

                if not candles:
                    break

                ohlcv_data.extend(candles)

                # Move to next chunk
                current_time = candles[-1][0] + 1

                # Rate limiting
                import time
                time.sleep(exchange.rateLimit / 1000)

            except Exception as e:
                logger.warning(f"⚠️ Chunk fetch failed: {e}")
                break

        if not ohlcv_data:
            logger.warning(f"⚠️ No data returned for {symbol}")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        # Convert timestamp from milliseconds
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Convert to float32
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(np.float32)

        # Remove duplicates
        df.drop_duplicates(subset=['timestamp'], inplace=True)

        logger.info(f"✅ Fetched {len(df)} bars for {symbol}")
        return df

    except Exception as e:
        logger.error(f"❌ Crypto fetch failed for {symbol}: {e}")
        return pd.DataFrame()


def fetch_historical_data(symbol, days=None, timeframe='1d'):
    """
    Unified function to fetch historical data for any asset

    Args:
        symbol: Asset symbol (e.g., 'AAPL', 'BTC/USD')
        days: Number of days (default: auto-detect based on asset type)
        timeframe: Timeframe for crypto ('1m', '1d', etc.)

    Returns:
        pandas DataFrame with OHLCV data
    """
    try:
        asset_type = detect_asset_type(symbol)

        if asset_type == 'crypto':
            # Default to 30 days for crypto
            days = days or 30
            return fetch_crypto_ohlcv(symbol, days=days, timeframe=timeframe)
        else:
            # Default to 3 years for stocks
            days = days or 1095
            return fetch_stock_ohlcv(symbol, days=days)

    except Exception as e:
        logger.error(f"❌ Data fetch failed for {symbol}: {e}")
        return pd.DataFrame()


def fetch_intraday_stock(symbol, interval='5m', days=7):
    """
    Fetch intraday stock data (for near-real-time updates)

    Args:
        symbol: Stock ticker
        interval: Interval ('1m', '5m', '15m', '30m', '1h')
        days: Number of days (max 60 for yfinance)

    Returns:
        pandas DataFrame
    """
    try:
        yf = _import_yfinance()

        # yfinance limits intraday data
        if days > 60:
            days = 60

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        logger.info(f"📊 Fetching intraday {interval} data for {symbol}...")

        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval=interval)

        if df.empty:
            return pd.DataFrame()

        # Normalize
        df.columns = [col.lower() for col in df.columns]
        df.reset_index(inplace=True)
        df.rename(columns={'date': 'timestamp', 'datetime': 'timestamp'}, inplace=True)

        # Convert types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = df[col].astype(np.float32)

        df['timestamp'] = pd.to_datetime(df['timestamp'])

        logger.info(f"✅ Fetched {len(df)} intraday bars for {symbol}")
        return df

    except Exception as e:
        logger.error(f"❌ Intraday fetch failed: {e}")
        return pd.DataFrame()


def append_realtime_data(df_existing, new_bar):
    """
    Append a new real-time bar to existing DataFrame

    Args:
        df_existing: Existing DataFrame with OHLCV data
        new_bar: dict with keys {timestamp, open, high, low, close, volume}

    Returns:
        pandas DataFrame with new bar appended
    """
    try:
        # Create new row
        new_row = pd.DataFrame([new_bar])

        # Ensure timestamp is datetime
        new_row['timestamp'] = pd.to_datetime(new_row['timestamp'])

        # Convert to float32
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in new_row.columns:
                new_row[col] = new_row[col].astype(np.float32)

        # Append
        df_updated = pd.concat([df_existing, new_row], ignore_index=True)

        # Remove duplicates (keep last)
        df_updated.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)

        # Sort by timestamp
        df_updated.sort_values('timestamp', inplace=True)
        df_updated.reset_index(drop=True, inplace=True)

        return df_updated

    except Exception as e:
        logger.error(f"❌ Append failed: {e}")
        return df_existing


# Timeframe mapping for aggregation
TIMEFRAME_MAP = {
    '1m': '1T',
    '5m': '5T',
    '15m': '15T',
    '30m': '30T',
    '1h': '1H',
    '2h': '2H',
    '4h': '4H',
    '6h': '6H',
    '1d': '1D',
    '1w': '1W',
    '1mo': '1M',
    '3mo': '3M'
}
