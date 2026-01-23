"""
Canonical OHLCV Data Loader for 1D/2D Horizon Training Pipeline

This module provides read-only access to the Master Market Data DB candles table.
NO schema changes. NO data duplication. Uses ONLY absolute paths.

Database: C:\\StockApp\\master_market_data\\market_data.db
Table: candles
Schema:
  - symbol (TEXT)
  - date (TEXT, YYYY-MM-DD)
  - open (REAL)
  - high (REAL)
  - low (REAL)
  - close (REAL)
  - adjusted_close (REAL)
  - volume (INTEGER)
Primary Key: (symbol, date)

Author: Auto-generated for 1D/2D horizon integration
Date: 2026-01-18
"""

import os
import sqlite3
import pandas as pd
from typing import Dict, List
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Canonical database path (absolute)
CANONICAL_DB_PATH = r"C:\StockApp\master_market_data\market_data.db"

# Required candles table columns
REQUIRED_CANDLES_COLUMNS = {'symbol', 'timestamp', 'timeframe', 'open', 'high', 'low', 'close', 'adjusted_close', 'volume'}


def _validate_absolute_path(db_path: str) -> None:
    """Validate that db_path is absolute."""
    if not os.path.isabs(db_path):
        raise ValueError(f"db_path must be an absolute path, got: {db_path}")
    logger.info("Validated absolute DB path")


def _validate_date_format(date_str: str, param_name: str) -> None:
    """Validate date format is YYYY-MM-DD."""
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        raise ValueError(f"Invalid date format for {param_name}: expected YYYY-MM-DD, got: {date_str}")


def _validate_candles_schema(conn: sqlite3.Connection) -> None:
    """Validate that candles table has required columns."""
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(candles)")
    rows = cursor.fetchall()

    actual_columns = {row[1] for row in rows}  # row[1] is column name

    missing_columns = REQUIRED_CANDLES_COLUMNS - actual_columns
    if missing_columns:
        raise RuntimeError(f"candles table schema mismatch: missing columns {missing_columns}")

    logger.info("Validated candles schema: OK")


def load_ohlcv_for_symbols(
    db_path: str,
    symbols: List[str],
    start_date: str,
    end_date: str
) -> Dict[str, pd.DataFrame]:
    """
    Load OHLCV data from the canonical candles table.

    Args:
        db_path: Absolute path to market_data.db
        symbols: List of stock tickers
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        Dict mapping symbol -> DataFrame with columns:
          - timestamp (YYYY-MM-DD)
          - open
          - high
          - low
          - close
          - adjusted_close
          - volume
        Sorted by timestamp ascending.
        Returns empty DataFrame for symbols with no data.

    Behavior:
        - Read-only access (SELECT queries only)
        - Deterministic: same inputs -> same outputs
        - Logs warnings for symbols with zero rows
        - Never modifies database
        - Never fabricates or forward-fills data
    """
    # Validation: absolute path
    _validate_absolute_path(db_path)

    # Validation: date formats
    _validate_date_format(start_date, 'start_date')
    _validate_date_format(end_date, 'end_date')

    logger.info(f"Loading OHLCV data from: {db_path}")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Symbols: {len(symbols)} total")

    # Connect to database
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Return rows as dict-like objects
    cursor = conn.cursor()

    # Validation: schema
    _validate_candles_schema(conn)

    result = {}

    for symbol in symbols:
        # Read-only SELECT query (filter by timeframe = '1d' for daily candles)
        query = """
            SELECT
                symbol,
                timestamp,
                open,
                high,
                low,
                close,
                adjusted_close,
                volume
            FROM candles
            WHERE symbol = ?
              AND timeframe = '1d'
              AND DATE(timestamp) BETWEEN ? AND ?
            ORDER BY timestamp ASC
        """

        cursor.execute(query, (symbol, start_date, end_date))
        rows = cursor.fetchall()

        if len(rows) == 0:
            logger.warning(f"[{symbol}] Zero rows in candles table for date range {start_date} to {end_date}")
            result[symbol] = pd.DataFrame(columns=[
                'symbol', 'timestamp', 'open', 'high', 'low', 'close', 'adjusted_close', 'volume'
            ])
        else:
            # Convert sqlite3.Row objects to DataFrame
            df = pd.DataFrame([dict(row) for row in rows])
            result[symbol] = df
            logger.info(f"[{symbol}] Loaded {len(df)} rows (min: {df['timestamp'].min()}, max: {df['timestamp'].max()})")

    conn.close()

    logger.info(f"OHLCV load complete: {len(result)} symbols processed")
    return result


def load_ohlcv_for_trades(
    db_path: str,
    trades: List[Dict],
    horizon_days: int
) -> Dict[str, pd.DataFrame]:
    """
    Load OHLCV data covering all trade entry dates + horizon windows.

    Args:
        db_path: Absolute path to market_data.db
        trades: List of trade dicts with 'symbol' and 'entry_date' keys
        horizon_days: Number of days to look forward (1 or 2)

    Returns:
        Dict mapping symbol -> DataFrame with OHLCV data
        Date range auto-computed from trades

    Behavior:
        - Computes min/max dates from trades
        - Adds horizon_days buffer to max date
        - Calls load_ohlcv_for_symbols() with computed range
    """
    if not trades:
        logger.warning("No trades provided to load_ohlcv_for_trades()")
        return {}

    # Extract unique symbols
    symbols = sorted(set(t['symbol'] for t in trades))

    # Compute date range from trades
    entry_dates = [t['entry_date'] for t in trades]
    min_date = min(entry_dates)
    max_date = max(entry_dates)

    # Add buffer for horizon window
    max_dt = datetime.strptime(max_date, '%Y-%m-%d')
    end_dt = max_dt + timedelta(days=horizon_days + 5)  # +5 for weekend/holiday buffer
    end_date = end_dt.strftime('%Y-%m-%d')

    logger.info(f"Loading OHLCV for {len(trades)} trades across {len(symbols)} symbols")
    logger.info(f"Trade date range: {min_date} to {max_date}")
    logger.info(f"OHLCV date range: {min_date} to {end_date} (horizon: {horizon_days}d + 5d buffer)")

    return load_ohlcv_for_symbols(db_path, symbols, min_date, end_date)


if __name__ == '__main__':
    # Test the loader
    print("Canonical OHLCV Loader Test")
    print()

    # Test with a few symbols
    test_symbols = ['AAPL', 'MSFT', 'GOOGL']
    test_start = '2024-01-01'
    test_end = '2024-12-31'

    print(f"Testing load_ohlcv_for_symbols():")
    print(f"  Symbols: {test_symbols}")
    print(f"  Date range: {test_start} to {test_end}")
    print()

    data = load_ohlcv_for_symbols(
        CANONICAL_DB_PATH,
        test_symbols,
        test_start,
        test_end
    )

    print()
    print("Results:")
    for symbol, df in data.items():
        if len(df) > 0:
            print(f"  [{symbol}] {len(df)} rows, dates: {df['date'].min()} to {df['date'].max()}")
        else:
            print(f"  [{symbol}] EMPTY")

    print()
    print("Test complete!")
