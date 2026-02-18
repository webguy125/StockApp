#!/usr/bin/env python
"""
Unified master market data ingestion script (v1.1)

Single source of truth for:
    - CORE_230 symbol universe
    - master_market_data.db
    - ohlcv table
    - sector_mappings table
    - incremental ingestion via --period
"""

import argparse
import datetime as dt
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

# Load environment variables (needed for TRADIER_API_KEY)
from dotenv import load_dotenv
backend_env_path = project_root / "backend" / ".env"
load_dotenv(backend_env_path)

# Import hybrid data fetcher (3-tier: Tradier → IBKR → Yahoo)
from backend.turbomode.core_engine.hybrid_data_fetcher import HybridDataFetcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ingest_master_market_data')

# ---------------------------------------------------------------------
# CONFIG: PATHS AND CONSTANTS
# ---------------------------------------------------------------------

APP_ROOT = str(project_root)
CONFIG_DIR = os.path.join(APP_ROOT, "config")
SYMBOLS_DIR = os.path.join(CONFIG_DIR, "symbols")
CORE_230_PATH = os.path.join(SYMBOLS_DIR, "CORE_230.json")

MASTER_DB_DIR = os.path.join(APP_ROOT, "master_market_data")
MASTER_DB_PATH = os.path.join(MASTER_DB_DIR, "master_market_data.db")

DEFAULT_PERIOD = "10y"  # used if not provided (10 years for full historical data)


# ---------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------

def parse_period(period_str: str) -> dt.datetime:
    """
    Convert a period string like '5d', '10y' into a start datetime.
    Supports: '5d' (days), '10y' (years), '3mo' (months)
    End datetime is 'now' (UTC).
    """
    period_str = period_str.strip().lower()

    if period_str.endswith("d"):
        days = int(period_str[:-1])
    elif period_str.endswith("y"):
        years = int(period_str[:-1])
        days = years * 365
    elif period_str.endswith("mo"):
        months = int(period_str[:-2])
        days = months * 30
    else:
        raise ValueError(f"Unsupported period format: {period_str} (expected like '5d', '10y', '3mo')")

    end = dt.datetime.utcnow()
    start = end - dt.timedelta(days=days)
    return start


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------
# SYMBOL UNIVERSE / SECTOR MAPPINGS
# ---------------------------------------------------------------------

def load_core_230() -> List[Dict]:
    """
    Load CORE_230.json.

    Expected structure:
    [
        {
            "symbol": "AAPL",
            "sector": "technology",
            "market_cap": "large_cap",
            "name": "Apple Inc."
        },
        ...
    ]
    """
    if not os.path.exists(CORE_230_PATH):
        raise FileNotFoundError(f"CORE_230 symbol file not found at: {CORE_230_PATH}")

    with open(CORE_230_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("CORE_230.json must contain a list of symbol records")

    return data


def extract_symbols_and_sectors(core_230: List[Dict]) -> Tuple[List[str], Dict[str, str]]:
    """
    Returns:
        symbols: list of ticker strings
        sector_map: dict[symbol] = sector
    """
    symbols = []
    sector_map = {}
    for entry in core_230:
        symbol = entry.get("ticker")  # Fixed: CORE_230.json uses 'ticker' not 'symbol'
        sector = entry.get("sector")
        if not symbol:
            continue
        symbols.append(symbol)
        if sector:
            sector_map[symbol] = sector
    return symbols, sector_map


# ---------------------------------------------------------------------
# DB SCHEMA
# ---------------------------------------------------------------------

def init_db_schema(conn: sqlite3.Connection) -> None:
    """
    Ensure ohlcv and sector_mappings tables exist with the correct schema.
    """
    cur = conn.cursor()

    # OHLCV table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ohlcv (
            symbol TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            PRIMARY KEY (symbol, timestamp)
        )
        """
    )

    # Sector mappings table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sector_mappings (
            symbol TEXT PRIMARY KEY,
            sector TEXT NOT NULL
        )
        """
    )

    conn.commit()


def upsert_sector_mappings(conn: sqlite3.Connection, sector_map: Dict[str, str]) -> None:
    """
    Upsert sector mappings for all CORE_230 symbols.
    """
    cur = conn.cursor()
    for symbol, sector in sector_map.items():
        cur.execute(
            """
            INSERT INTO sector_mappings (symbol, sector)
            VALUES (?, ?)
            ON CONFLICT(symbol) DO UPDATE SET sector=excluded.sector
            """,
            (symbol, sector),
        )
    conn.commit()


# ---------------------------------------------------------------------
# DATA PROVIDER (IBKR + YAHOO FALLBACK)
# ---------------------------------------------------------------------

def fetch_ohlcv_for_symbol(
    fetcher: HybridDataFetcher,
    symbol: str,
    start: dt.datetime,
    end: dt.datetime,
) -> List[Dict]:
    """
    Fetch OHLCV data for a single symbol between [start, end].
    Uses HybridDataFetcher (IBKR with Yahoo fallback).

    Returns list of dicts with:
        [
            {
                "timestamp": int (unix epoch seconds),
                "open": float,
                "high": float,
                "low": float,
                "close": float,
                "volume": float
            },
            ...
        ]
    """
    try:
        # Calculate period in days
        days = (end - start).days
        period = f"{days}d"

        # Fetch using hybrid fetcher
        df = fetcher.fetch_candles(symbol, period=period, interval='1d')

        if df is None or df.empty:
            logger.warning(f"[SKIP] {symbol}: No data available")
            return []

        # Convert DataFrame to list of dicts
        rows = []
        for timestamp, row in df.iterrows():
            # Handle both datetime.datetime and datetime.date objects from IBKR/yfinance
            if hasattr(timestamp, 'timestamp'):
                ts_int = int(timestamp.timestamp())
            else:
                # datetime.date - convert to datetime at midnight UTC
                import datetime
                ts_int = int(datetime.datetime.combine(timestamp, datetime.time()).timestamp())

            rows.append({
                "timestamp": ts_int,
                "open": float(row['Open']),
                "high": float(row['High']),
                "low": float(row['Low']),
                "close": float(row['Close']),
                "volume": float(row['Volume'])
            })

        return rows

    except Exception as e:
        logger.error(f"[ERROR] {symbol}: Failed to fetch data - {e}")
        return []


# ---------------------------------------------------------------------
# INGESTION CORE
# ---------------------------------------------------------------------

def ingest_symbol_ohlcv(
    conn: sqlite3.Connection,
    fetcher: HybridDataFetcher,
    symbol: str,
    start: dt.datetime,
    end: dt.datetime,
) -> int:
    """
    Fetch and upsert OHLCV data for a single symbol.
    Returns number of rows ingested.

    Strategy: Try Tradier first, then use Yahoo to fill gaps if Tradier has missing bars.
    """
    import math

    rows = fetch_ohlcv_for_symbol(fetcher, symbol, start, end)
    if not rows:
        return 0

    # Track skipped timestamps for gap-filling with Yahoo
    skipped_timestamps = []

    cur = conn.cursor()
    inserted = 0
    for row in rows:
        # Normalize Tradier keys (Open/High/Low/Close/Volume -> open/high/low/close/volume)
        row = {k.lower(): v for k, v in row.items()}

        # Skip invalid or incomplete bars (check for None and NaN)
        if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in [row.get('open'), row.get('high'), row.get('low'), row.get('close'), row.get('volume')]):
            logger.warning(f"[SKIP] Invalid bar for {symbol} on {row.get('timestamp')} (missing OHLCV) - will try Yahoo")
            skipped_timestamps.append(row.get('timestamp'))
            continue

        cur.execute(
            """
            INSERT OR REPLACE INTO ohlcv (
                symbol, timestamp, open, high, low, close, volume
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                symbol,
                int(row["timestamp"]),
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                float(row["volume"]),
            ),
        )
        inserted += 1

    # Gap-filling: If we skipped any bars, try Yahoo to fill them in
    if skipped_timestamps:
        logger.info(f"[GAP-FILL] {symbol}: Trying Yahoo for {len(skipped_timestamps)} missing bars from Tradier")

        # Temporarily disable Tradier to force Yahoo fallback
        original_use_tradier = fetcher.use_tradier
        fetcher.use_tradier = False

        try:
            yahoo_rows = fetch_ohlcv_for_symbol(fetcher, symbol, start, end)
            if yahoo_rows:
                filled = 0
                for row in yahoo_rows:
                    row = {k.lower(): v for k, v in row.items()}

                    # Only insert if this timestamp was skipped from Tradier
                    if row.get('timestamp') in skipped_timestamps:
                        # Check if Yahoo data is valid
                        if not any(v is None or (isinstance(v, float) and math.isnan(v)) for v in [row.get('open'), row.get('high'), row.get('low'), row.get('close'), row.get('volume')]):
                            cur.execute(
                                """
                                INSERT OR REPLACE INTO ohlcv (
                                    symbol, timestamp, open, high, low, close, volume
                                )
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    symbol,
                                    int(row["timestamp"]),
                                    float(row["open"]),
                                    float(row["high"]),
                                    float(row["low"]),
                                    float(row["close"]),
                                    float(row["volume"]),
                                ),
                            )
                            filled += 1
                            inserted += 1

                if filled > 0:
                    logger.info(f"[GAP-FILL] {symbol}: Yahoo filled {filled} missing bars")
                else:
                    logger.warning(f"[GAP-FILL] {symbol}: Yahoo could not fill any missing bars")
        finally:
            # Restore Tradier setting
            fetcher.use_tradier = original_use_tradier

    conn.commit()
    return inserted


def run_ingestion(
    period: str,
    full_refresh: bool = False,
    symbols_override: Optional[List[str]] = None,
    use_ibkr: bool = True,
) -> None:
    """
    Main ingestion routine.

    Args:
        period: e.g. '5d' for incremental window.
        full_refresh: if True, ignore period and re-ingest full history (provider-dependent).
        symbols_override: if provided, restrict ingestion to this subset of symbols.
        use_ibkr: If True, attempt IBKR connection, else use Yahoo only.
    """
    ensure_dir(MASTER_DB_DIR)
    conn = sqlite3.connect(MASTER_DB_PATH)

    # Initialize hybrid fetcher (IBKR + Yahoo fallback)
    fetcher = HybridDataFetcher(use_ibkr=use_ibkr)

    try:
        init_db_schema(conn)

        core_230 = load_core_230()
        all_symbols, sector_map = extract_symbols_and_sectors(core_230)

        # Determine symbol universe
        if symbols_override:
            symbols = [s for s in symbols_override if s in all_symbols]
        else:
            symbols = all_symbols

        # Upsert sector mappings
        upsert_sector_mappings(conn, sector_map)
        logger.info(f"[OK] Updated sector mappings for {len(sector_map)} symbols")

        # Determine time window
        if full_refresh:
            # For full refresh, use 10 years
            start = dt.datetime.utcnow() - dt.timedelta(days=3650)
            end = dt.datetime.utcnow()
            logger.info("[FULL REFRESH] Fetching 10 years of data")
        else:
            start = parse_period(period or DEFAULT_PERIOD)
            end = dt.datetime.utcnow()
            logger.info(f"[INCREMENTAL] Fetching {period} of data")

        total_inserted = 0
        for i, symbol in enumerate(symbols, 1):
            inserted = ingest_symbol_ohlcv(conn, fetcher, symbol, start, end)
            total_inserted += inserted
            logger.info(f"[{i}/{len(symbols)}] {symbol}: {inserted} rows")

        logger.info("=" * 80)
        logger.info(f"[INGEST] Completed. Total rows inserted/updated: {total_inserted}")
        logger.info(f"[INGEST] DB path: {MASTER_DB_PATH}")
        logger.info(f"[INGEST] Symbols: {len(symbols)} (CORE_230-aligned)")
        logger.info("=" * 80)

    finally:
        fetcher.disconnect()
        conn.close()


# ---------------------------------------------------------------------
# LEGACY-COMPATIBLE WRAPPER FOR PIPELINE STAGE 0
# ---------------------------------------------------------------------

def run_full_ingestion(period: str = DEFAULT_PERIOD) -> None:
    """
    Legacy-style entrypoint so pipeline Stage 0 can call:

        from backend.turbomode.core_engine.ingest_master_market_data import run_full_ingestion

    This keeps the pipeline happy while using the unified ingestion logic.
    """
    run_ingestion(period=period, full_refresh=False, use_ibkr=True)


# ---------------------------------------------------------------------
# CLI ENTRYPOINT
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Unified master market data ingestion (CORE_230 → master_market_data.db)"
    )
    parser.add_argument(
        "--period",
        type=str,
        default=DEFAULT_PERIOD,
        help="Incremental window, e.g. '5d'. Ignored if --full-refresh is set.",
    )
    parser.add_argument(
        "--full-refresh",
        action="store_true",
        help="Re-ingest full history (10 years).",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="*",
        default=None,
        help="Optional subset of symbols to ingest (must be in CORE_230).",
    )
    parser.add_argument(
        "--no-ibkr",
        action="store_true",
        help="Skip IBKR and use Yahoo only.",
    )

    args = parser.parse_args()

    run_ingestion(
        period=args.period,
        full_refresh=args.full_refresh,
        symbols_override=args.symbols,
        use_ibkr=not args.no_ibkr,
    )


if __name__ == "__main__":
    main()
