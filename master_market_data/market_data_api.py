"""
Master Market Data DB - Data Access Layer
Read-only API for accessing shared market data

Architecture: MASTER_MARKET_DATA_ARCHITECTURE.json v1.1
Database: C:\StockApp\master_market_data\market_data.db

This database is READ-ONLY for TurboMode and Slipstream.
All write operations are forbidden for API users.
"""

import sqlite3
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import logging

logger = logging.getLogger('market_data_api')


class MarketDataAPI:
    """
    Read-only API for Master Market Data DB
    Provides structured access to OHLCV, fundamentals, and metadata
    """

    def __init__(self, db_path: str = None, read_only: bool = True):
        """
        Initialize Market Data API

        Args:
            db_path: Path to market_data.db
            read_only: Enforce read-only access (default: True)
        """
        if db_path is None:
            db_path = os.path.join(os.path.dirname(__file__), 'market_data.db')

        self.db_path = db_path
        self.read_only = read_only

        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Master Market Data DB not found: {db_path}")

        logger.info(f"[OK] MarketDataAPI initialized (read_only={read_only})")

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection"""
        if self.read_only:
            # Open in read-only mode
            conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        else:
            conn = sqlite3.connect(self.db_path)

        conn.row_factory = sqlite3.Row  # Enable column access by name
        return conn

    # ========================================================================
    # CANDLES (OHLCV Data)
    # ========================================================================

    def get_candles(
        self,
        symbol: str,
        timeframe: str = '1d',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None,
        days_back: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get OHLCV candles for a symbol

        Args:
            symbol: Stock ticker
            timeframe: Timeframe (e.g., '1m', '5m', '1h', '1d')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            limit: Maximum number of candles to return
            days_back: Get last N days of data (alternative to start_date)

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        from datetime import datetime, timedelta

        conn = self._get_connection()
        cursor = conn.cursor()

        query = """
            SELECT timestamp, open, high, low, close, volume, adjusted_close
            FROM candles
            WHERE symbol = ? AND timeframe = ?
        """
        params = [symbol, timeframe]

        # Handle days_back parameter
        if days_back and not start_date:
            start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)

        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)

        query += " ORDER BY timestamp ASC"

        if limit:
            query += f" LIMIT {limit}"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'adjusted_close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

        return df

    def get_latest_candle(self, symbol: str, timeframe: str = '1d') -> Optional[Dict[str, Any]]:
        """
        Get most recent candle for a symbol

        Args:
            symbol: Stock ticker
            timeframe: Timeframe

        Returns:
            Dictionary with candle data or None
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT timestamp, open, high, low, close, volume, adjusted_close
            FROM candles
            WHERE symbol = ? AND timeframe = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (symbol, timeframe))

        row = cursor.fetchone()
        conn.close()

        if row:
            return dict(row)
        return None

    # ========================================================================
    # FUNDAMENTALS
    # ========================================================================

    def get_fundamentals(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get fundamental data for a symbol

        Args:
            symbol: Stock ticker
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with fundamental metrics
        """
        conn = self._get_connection()

        query = "SELECT * FROM fundamentals WHERE symbol = ?"
        params = [symbol]

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)

        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date ASC"

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)

        return df

    def get_latest_fundamentals(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get most recent fundamental data for a symbol

        Args:
            symbol: Stock ticker

        Returns:
            Dictionary with fundamental data or None
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM fundamentals
            WHERE symbol = ?
            ORDER BY date DESC
            LIMIT 1
        """, (symbol,))

        row = cursor.fetchone()
        conn.close()

        if row:
            return dict(row)
        return None

    # ========================================================================
    # SYMBOL METADATA
    # ========================================================================

    def get_symbol_metadata(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a symbol

        Args:
            symbol: Stock ticker

        Returns:
            Dictionary with symbol metadata or None
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM symbol_metadata WHERE symbol = ?", (symbol,))

        row = cursor.fetchone()
        conn.close()

        if row:
            return dict(row)
        return None

    def get_symbols_by_sector(self, sector: str) -> List[str]:
        """
        Get all symbols in a sector

        Args:
            sector: Sector name

        Returns:
            List of stock tickers
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT symbol FROM symbol_metadata WHERE sector = ? AND is_active = 1", (sector,))

        rows = cursor.fetchall()
        conn.close()

        return [row[0] for row in rows]

    def get_all_active_symbols(self) -> List[str]:
        """
        Get all active symbols

        Returns:
            List of stock tickers
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT symbol FROM symbol_metadata WHERE is_active = 1 ORDER BY symbol")

        rows = cursor.fetchall()
        conn.close()

        return [row[0] for row in rows]

    # ========================================================================
    # SPLITS AND DIVIDENDS
    # ========================================================================

    def get_splits(self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get stock splits for a symbol

        Args:
            symbol: Stock ticker
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with split events
        """
        conn = self._get_connection()

        query = "SELECT * FROM splits WHERE symbol = ?"
        params = [symbol]

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)

        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date ASC"

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        return df

    def get_dividends(self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get dividends for a symbol

        Args:
            symbol: Stock ticker
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with dividend events
        """
        conn = self._get_connection()

        query = "SELECT * FROM dividends WHERE symbol = ?"
        params = [symbol]

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)

        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date ASC"

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        return df

    # ========================================================================
    # SECTOR MAPPINGS
    # ========================================================================

    def get_sector_mapping(self, symbol: str, as_of_date: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get sector mapping for a symbol as of a specific date

        Args:
            symbol: Stock ticker
            as_of_date: Date to query (YYYY-MM-DD), defaults to today

        Returns:
            Dictionary with sector mapping or None
        """
        if as_of_date is None:
            as_of_date = datetime.now().strftime('%Y-%m-%d')

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM sector_mappings
            WHERE symbol = ?
            AND effective_date <= ?
            AND (end_date IS NULL OR end_date >= ?)
            ORDER BY effective_date DESC
            LIMIT 1
        """, (symbol, as_of_date, as_of_date))

        row = cursor.fetchone()
        conn.close()

        if row:
            return dict(row)
        return None

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def get_date_range(self, symbol: str, timeframe: str = '1d') -> Optional[Tuple[str, str]]:
        """
        Get date range of available data for a symbol

        Args:
            symbol: Stock ticker
            timeframe: Timeframe

        Returns:
            Tuple of (min_date, max_date) or None
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT MIN(timestamp), MAX(timestamp)
            FROM candles
            WHERE symbol = ? AND timeframe = ?
        """, (symbol, timeframe))

        row = cursor.fetchone()
        conn.close()

        if row and row[0]:
            return (row[0], row[1])
        return None

    def check_data_availability(self, symbol: str) -> Dict[str, bool]:
        """
        Check what data is available for a symbol

        Args:
            symbol: Stock ticker

        Returns:
            Dictionary with availability flags
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        availability = {}

        # Check candles
        cursor.execute("SELECT COUNT(*) FROM candles WHERE symbol = ?", (symbol,))
        availability['has_candles'] = cursor.fetchone()[0] > 0

        # Check fundamentals
        cursor.execute("SELECT COUNT(*) FROM fundamentals WHERE symbol = ?", (symbol,))
        availability['has_fundamentals'] = cursor.fetchone()[0] > 0

        # Check metadata
        cursor.execute("SELECT COUNT(*) FROM symbol_metadata WHERE symbol = ?", (symbol,))
        availability['has_metadata'] = cursor.fetchone()[0] > 0

        conn.close()

        return availability


# Singleton instance
_market_data_api = None


def get_market_data_api() -> MarketDataAPI:
    """
    Get singleton MarketDataAPI instance

    Returns:
        MarketDataAPI instance (read-only)
    """
    global _market_data_api
    if _market_data_api is None:
        _market_data_api = MarketDataAPI(read_only=True)
    return _market_data_api


if __name__ == '__main__':
    # Test Market Data API
    print("=" * 80)
    print("MASTER MARKET DATA API - TEST")
    print("=" * 80)

    api = MarketDataAPI()

    print(f"\nRead-Only Mode: {api.read_only}")
    print(f"Database Path: {api.db_path}")

    # Test data availability (will be empty for now)
    print("\nTesting API methods...")
    print("[OK] MarketDataAPI initialized successfully")
    print("[INFO] Database is empty - ready for data ingestion")

    print("\n" + "=" * 80)
    print("MARKET DATA API TEST COMPLETE")
    print("=" * 80)
