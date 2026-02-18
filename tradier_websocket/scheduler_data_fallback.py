"""
3-Tier Fallback Logic for Unified Scheduler
============================================

This module implements the fallback chain for scheduler data fetching:
1. Primary: Tradier REST API
2. Secondary: IB Gateway (IBKR)
3. Last Resort: Yahoo Finance

Architecture:
- Try each source in order until successful
- Log which source was used for monitoring
- Return data in unified format regardless of source
- Handle errors gracefully at each tier

Usage:
    from scheduler_data_fallback import get_market_data_with_fallback

    # Fetch quotes
    quotes = get_market_data_with_fallback(
        symbols=['AAPL', 'MSFT'],
        data_type='quotes'
    )

    # Fetch historical data
    historical = get_market_data_with_fallback(
        symbol='AAPL',
        data_type='historical',
        interval='daily',
        period='1y'
    )
"""

import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import pandas as pd

# Add paths for imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "..", "backend"))

# Import Tradier client
from tradier_unified_scheduler_rest_client import get_tradier_scheduler_client

# Import IBKR client (placeholder - you already have this working)
# from your_ibkr_module import fetch_from_ibkr

# Import Yahoo client (placeholder - you already have this working)
# from your_yahoo_module import fetch_from_yahoo


class DataFallbackLogger:
    """Simple logger to track which data source was used."""

    def __init__(self):
        self.usage_stats = {
            "tradier": 0,
            "ibkr": 0,
            "yahoo": 0,
            "failed": 0
        }

    def log_success(self, source: str, symbol: str, data_type: str):
        """Log successful data fetch."""
        self.usage_stats[source] += 1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[FALLBACK] {timestamp} - {source.upper()} SUCCESS - {symbol} ({data_type})")

    def log_failure(self, source: str, symbol: str, error: str):
        """Log failed data fetch."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[FALLBACK] {timestamp} - {source.upper()} FAILED - {symbol} - {error}")

    def log_all_failed(self, symbol: str, data_type: str):
        """Log when all sources failed."""
        self.usage_stats["failed"] += 1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[FALLBACK] {timestamp} - ALL SOURCES FAILED - {symbol} ({data_type})")

    def get_stats(self) -> Dict[str, int]:
        """Get usage statistics."""
        return self.usage_stats.copy()


# Global logger instance
_fallback_logger = DataFallbackLogger()


def get_fallback_logger() -> DataFallbackLogger:
    """Get global fallback logger instance."""
    return _fallback_logger


# ============================================================================
# TIER 1: TRADIER
# ============================================================================

def fetch_quotes_tradier(symbols: List[str]) -> Optional[Dict[str, Dict[str, Any]]]:
    """
    Fetch quotes from Tradier.

    Args:
        symbols: List of ticker symbols

    Returns:
        Dictionary mapping symbol -> quote data, or None if failed
    """
    try:
        client = get_tradier_scheduler_client()
        quotes = client.get_quotes(symbols)

        if quotes and len(quotes) > 0:
            return quotes
        else:
            return None

    except Exception as e:
        print(f"[TRADIER] Quote fetch error: {e}")
        return None


def fetch_historical_tradier(
    symbol: str,
    interval: str = "daily",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """
    Fetch historical data from Tradier.

    Args:
        symbol: Ticker symbol
        interval: Data interval (daily, weekly, monthly)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with OHLCV data, or None if failed
    """
    try:
        client = get_tradier_scheduler_client()
        df = client.get_historical_data(symbol, interval, start_date, end_date)

        if df is not None and len(df) > 0:
            return df
        else:
            return None

    except Exception as e:
        print(f"[TRADIER] Historical fetch error: {e}")
        return None


def fetch_intraday_tradier(
    symbol: str,
    interval: str = "5min",
    start_time: Optional[str] = None,
    end_time: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """
    Fetch intraday data from Tradier.

    Args:
        symbol: Ticker symbol
        interval: Time interval (1min, 5min, 15min)
        start_time: Start timestamp
        end_time: End timestamp

    Returns:
        DataFrame with OHLCV data, or None if failed
    """
    try:
        client = get_tradier_scheduler_client()
        df = client.get_intraday_data(symbol, interval, start_time, end_time)

        if df is not None and len(df) > 0:
            return df
        else:
            return None

    except Exception as e:
        print(f"[TRADIER] Intraday fetch error: {e}")
        return None


# ============================================================================
# TIER 2: IB GATEWAY (IBKR)
# ============================================================================

def fetch_quotes_ibkr(symbols: List[str]) -> Optional[Dict[str, Dict[str, Any]]]:
    """
    Fetch quotes from IB Gateway.

    NOTE: This is a placeholder. Your existing IBKR integration should be used here.

    Args:
        symbols: List of ticker symbols

    Returns:
        Dictionary mapping symbol -> quote data, or None if failed
    """
    try:
        # TODO: Call your existing IBKR quote fetching code
        # Example:
        # from your_ibkr_module import get_ibkr_quotes
        # return get_ibkr_quotes(symbols)

        print("[IBKR] Quote fetching not yet implemented (placeholder)")
        return None

    except Exception as e:
        print(f"[IBKR] Quote fetch error: {e}")
        return None


def fetch_historical_ibkr(
    symbol: str,
    interval: str = "daily",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """
    Fetch historical data from IB Gateway.

    NOTE: This is a placeholder. Your existing IBKR integration should be used here.

    Args:
        symbol: Ticker symbol
        interval: Data interval
        start_date: Start date
        end_date: End date

    Returns:
        DataFrame with OHLCV data, or None if failed
    """
    try:
        # TODO: Call your existing IBKR historical data code
        # Example:
        # from your_ibkr_module import get_ibkr_historical
        # return get_ibkr_historical(symbol, interval, start_date, end_date)

        print("[IBKR] Historical fetching not yet implemented (placeholder)")
        return None

    except Exception as e:
        print(f"[IBKR] Historical fetch error: {e}")
        return None


# ============================================================================
# TIER 3: YAHOO FINANCE
# ============================================================================

def fetch_quotes_yahoo(symbols: List[str]) -> Optional[Dict[str, Dict[str, Any]]]:
    """
    Fetch quotes from Yahoo Finance.

    NOTE: This is a placeholder. Your existing Yahoo integration should be used here.

    Args:
        symbols: List of ticker symbols

    Returns:
        Dictionary mapping symbol -> quote data, or None if failed
    """
    try:
        # TODO: Call your existing Yahoo quote fetching code
        # Example:
        # from your_yahoo_module import get_yahoo_quotes
        # return get_yahoo_quotes(symbols)

        print("[YAHOO] Quote fetching not yet implemented (placeholder)")
        return None

    except Exception as e:
        print(f"[YAHOO] Quote fetch error: {e}")
        return None


def fetch_historical_yahoo(
    symbol: str,
    interval: str = "daily",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """
    Fetch historical data from Yahoo Finance.

    NOTE: This is a placeholder. Your existing Yahoo integration should be used here.

    Args:
        symbol: Ticker symbol
        interval: Data interval
        start_date: Start date
        end_date: End date

    Returns:
        DataFrame with OHLCV data, or None if failed
    """
    try:
        # TODO: Call your existing Yahoo historical data code
        # Example:
        # from your_yahoo_module import get_yahoo_historical
        # return get_yahoo_historical(symbol, interval, start_date, end_date)

        print("[YAHOO] Historical fetching not yet implemented (placeholder)")
        return None

    except Exception as e:
        print(f"[YAHOO] Historical fetch error: {e}")
        return None


# ============================================================================
# MAIN FALLBACK LOGIC
# ============================================================================

def get_quotes_with_fallback(symbols: List[str]) -> Optional[Dict[str, Dict[str, Any]]]:
    """
    Get quotes with 3-tier fallback: Tradier -> IBKR -> Yahoo.

    Args:
        symbols: List of ticker symbols

    Returns:
        Dictionary mapping symbol -> quote data, or None if all failed
    """
    logger = get_fallback_logger()
    symbols_str = ", ".join(symbols)

    # Tier 1: Tradier
    result = fetch_quotes_tradier(symbols)
    if result:
        logger.log_success("tradier", symbols_str, "quotes")
        return result
    else:
        logger.log_failure("tradier", symbols_str, "No data returned")

    # Tier 2: IBKR
    result = fetch_quotes_ibkr(symbols)
    if result:
        logger.log_success("ibkr", symbols_str, "quotes")
        return result
    else:
        logger.log_failure("ibkr", symbols_str, "No data returned")

    # Tier 3: Yahoo
    result = fetch_quotes_yahoo(symbols)
    if result:
        logger.log_success("yahoo", symbols_str, "quotes")
        return result
    else:
        logger.log_failure("yahoo", symbols_str, "No data returned")

    # All failed
    logger.log_all_failed(symbols_str, "quotes")
    return None


def get_historical_with_fallback(
    symbol: str,
    interval: str = "daily",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """
    Get historical data with 3-tier fallback: Tradier -> IBKR -> Yahoo.

    Args:
        symbol: Ticker symbol
        interval: Data interval
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with OHLCV data, or None if all failed
    """
    logger = get_fallback_logger()

    # Tier 1: Tradier
    result = fetch_historical_tradier(symbol, interval, start_date, end_date)
    if result is not None and len(result) > 0:
        logger.log_success("tradier", symbol, f"historical-{interval}")
        return result
    else:
        logger.log_failure("tradier", symbol, "No data returned")

    # Tier 2: IBKR
    result = fetch_historical_ibkr(symbol, interval, start_date, end_date)
    if result is not None and len(result) > 0:
        logger.log_success("ibkr", symbol, f"historical-{interval}")
        return result
    else:
        logger.log_failure("ibkr", symbol, "No data returned")

    # Tier 3: Yahoo
    result = fetch_historical_yahoo(symbol, interval, start_date, end_date)
    if result is not None and len(result) > 0:
        logger.log_success("yahoo", symbol, f"historical-{interval}")
        return result
    else:
        logger.log_failure("yahoo", symbol, "No data returned")

    # All failed
    logger.log_all_failed(symbol, f"historical-{interval}")
    return None


def get_intraday_with_fallback(
    symbol: str,
    interval: str = "5min",
    start_time: Optional[str] = None,
    end_time: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """
    Get intraday data with fallback (Tradier only for now).

    NOTE: IBKR and Yahoo intraday support can be added later if needed.

    Args:
        symbol: Ticker symbol
        interval: Time interval (1min, 5min, 15min)
        start_time: Start timestamp
        end_time: End timestamp

    Returns:
        DataFrame with OHLCV data, or None if failed
    """
    logger = get_fallback_logger()

    # Tier 1: Tradier
    result = fetch_intraday_tradier(symbol, interval, start_time, end_time)
    if result is not None and len(result) > 0:
        logger.log_success("tradier", symbol, f"intraday-{interval}")
        return result
    else:
        logger.log_failure("tradier", symbol, "No data returned")
        logger.log_all_failed(symbol, f"intraday-{interval}")

    return None


def get_market_data_with_fallback(
    symbols: Optional[Union[str, List[str]]] = None,
    symbol: Optional[str] = None,
    data_type: str = "quotes",
    **kwargs
) -> Optional[Union[Dict, pd.DataFrame]]:
    """
    Universal market data fetcher with 3-tier fallback.

    This is the main function you should call from scheduler tasks.

    Args:
        symbols: List of symbols (for quotes)
        symbol: Single symbol (for historical/intraday)
        data_type: Type of data - 'quotes', 'historical', 'intraday'
        **kwargs: Additional arguments (interval, start_date, end_date, etc.)

    Returns:
        Data in appropriate format, or None if all sources failed

    Examples:
        # Fetch quotes
        quotes = get_market_data_with_fallback(
            symbols=['AAPL', 'MSFT'],
            data_type='quotes'
        )

        # Fetch historical data
        df = get_market_data_with_fallback(
            symbol='AAPL',
            data_type='historical',
            interval='daily',
            start_date='2024-01-01',
            end_date='2024-12-31'
        )

        # Fetch intraday data
        df = get_market_data_with_fallback(
            symbol='TSLA',
            data_type='intraday',
            interval='5min'
        )
    """
    if data_type == "quotes":
        # Handle both single symbol and list of symbols
        if symbol and not symbols:
            symbols = [symbol]
        elif isinstance(symbols, str):
            symbols = [symbols]

        return get_quotes_with_fallback(symbols)

    elif data_type == "historical":
        if not symbol:
            raise ValueError("symbol required for historical data")

        interval = kwargs.get("interval", "daily")
        start_date = kwargs.get("start_date")
        end_date = kwargs.get("end_date")

        return get_historical_with_fallback(symbol, interval, start_date, end_date)

    elif data_type == "intraday":
        if not symbol:
            raise ValueError("symbol required for intraday data")

        interval = kwargs.get("interval", "5min")
        start_time = kwargs.get("start_time")
        end_time = kwargs.get("end_time")

        return get_intraday_with_fallback(symbol, interval, start_time, end_time)

    else:
        raise ValueError(f"Unknown data_type: {data_type}")


if __name__ == "__main__":
    # Test the fallback logic
    print("=" * 80)
    print("3-TIER FALLBACK LOGIC TEST")
    print("=" * 80)

    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()

    # Test 1: Quotes
    print("\n[TEST 1] Fetch Quotes with Fallback")
    print("-" * 80)
    quotes = get_market_data_with_fallback(
        symbols=["AAPL", "MSFT", "TSLA"],
        data_type="quotes"
    )
    if quotes:
        for symbol, quote in quotes.items():
            print(f"{symbol}: ${quote.get('last', 0):.2f}")
    else:
        print("Failed to fetch quotes from all sources")

    # Test 2: Historical
    print("\n[TEST 2] Fetch Historical Data with Fallback")
    print("-" * 80)
    df = get_market_data_with_fallback(
        symbol="AAPL",
        data_type="historical",
        interval="daily"
    )
    if df is not None and len(df) > 0:
        print(f"Fetched {len(df)} rows")
        print(df.tail())
    else:
        print("Failed to fetch historical data from all sources")

    # Test 3: Statistics
    print("\n[TEST 3] Fallback Statistics")
    print("-" * 80)
    logger = get_fallback_logger()
    stats = logger.get_stats()
    print(f"Tradier: {stats['tradier']} requests")
    print(f"IBKR: {stats['ibkr']} requests")
    print(f"Yahoo: {stats['yahoo']} requests")
    print(f"Failed: {stats['failed']} requests")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
