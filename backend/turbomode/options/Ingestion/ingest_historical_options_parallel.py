"""
OPTIMIZED EODHD HISTORICAL OPTIONS INGESTION (PARALLEL)

Simplified approach: Process by date (not by expiration), use symbol-level parallelism

Key optimizations:
- Symbol-level parallelism: 4-8 workers processing symbols concurrently
- Batched writes: Insert 100-2000 records per transaction
- Rate limiting: Global token bucket with 5 req/sec
- Retry logic: Exponential backoff on 5xx errors
- SQLite WAL mode: Concurrent writes
- Date-driven: Fetch full chain for each date, insert all options

Usage:
    python ingest_historical_options_parallel.py [--workers 4] [--test]
"""
import os
import sys
import json
import sqlite3
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading

# Add backend to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OPTIONS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
BACKEND_DIR = os.path.abspath(os.path.join(OPTIONS_DIR, '..', '..'))
STOCKAPP_DIR = os.path.abspath(os.path.join(BACKEND_DIR, '..'))
sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, STOCKAPP_DIR)

from backend.eodhd_client import get_eodhd_client

# Configuration
OPTIONS_UNIVERSE_DB = os.path.join(OPTIONS_DIR, 'Data', 'options_universe.db')
CORE_230_PATH = os.path.join(BACKEND_DIR, '..', 'config', 'symbols', 'CORE_230.json')

# Ingestion settings
DEFAULT_LOOKBACK_TRADING_DAYS = 30  # 1 month of trading days (excludes weekends)
BATCH_SIZE_MAX = 2000
MIN_WORKERS = 4
MAX_WORKERS = 8

# Rate limiting
RATE_LIMIT_RPS = 5  # 5 requests per second

# Retry configuration
MAX_RETRIES = 3
BACKOFF_SEQUENCE = [1, 2, 4]  # seconds

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [%(processName)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class RateLimiter:
    """Global token bucket rate limiter"""

    def __init__(self, rate_per_second: float):
        self.rate = rate_per_second
        self.tokens = rate_per_second
        self.last_update = time.time()
        self.lock = threading.Lock()

    def acquire(self):
        """Acquire a token, blocking if necessary"""
        with self.lock:
            now = time.time()
            # Refill tokens based on time elapsed
            elapsed = now - self.last_update
            self.tokens = min(self.rate, self.tokens + elapsed * self.rate)
            self.last_update = now

            if self.tokens >= 1.0:
                self.tokens -= 1.0
            else:
                # Need to wait
                wait_time = (1.0 - self.tokens) / self.rate
                time.sleep(wait_time)
                self.tokens = 0.0
                self.last_update = time.time()


# Global rate limiter
rate_limiter = RateLimiter(RATE_LIMIT_RPS)


def initialize_database():
    """Initialize database with WAL mode"""
    logger.info('[INIT] Initializing database with WAL mode')

    conn = sqlite3.connect(OPTIONS_UNIVERSE_DB)

    # Enable WAL mode for concurrent access
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA synchronous=NORMAL')
    conn.execute('PRAGMA cache_size=-64000')  # 64MB cache
    conn.execute('PRAGMA temp_store=MEMORY')

    # Verify WAL mode
    journal_mode = conn.execute('PRAGMA journal_mode').fetchone()[0]
    logger.info(f'[INIT] Database journal mode: {journal_mode}')

    conn.close()


def get_trading_days_range(end_date_str: str, num_trading_days: int) -> Tuple[str, str]:
    """
    Calculate date range for N trading days (excludes weekends and holidays)

    Args:
        end_date_str: End date in YYYY-MM-DD format (e.g., '2026-02-13')
        num_trading_days: Number of trading days to go back

    Returns:
        Tuple of (start_date, end_date) in YYYY-MM-DD format
    """
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

    # Common US market holidays for 2025-2026
    holidays = [
        datetime(2025, 1, 1),   # New Year's Day
        datetime(2025, 1, 20),  # MLK Day
        datetime(2025, 2, 17),  # Presidents Day
        datetime(2025, 4, 18),  # Good Friday
        datetime(2025, 5, 26),  # Memorial Day
        datetime(2025, 7, 4),   # Independence Day
        datetime(2025, 9, 1),   # Labor Day
        datetime(2025, 11, 27), # Thanksgiving
        datetime(2025, 12, 25), # Christmas
        datetime(2026, 1, 1),   # New Year's Day
        datetime(2026, 1, 19),  # MLK Day
        datetime(2026, 2, 16),  # Presidents Day
    ]

    trading_days_count = 0
    current = end_date

    # Go backwards counting trading days
    while trading_days_count < num_trading_days:
        # Skip weekends (Saturday=5, Sunday=6)
        if current.weekday() < 5:
            # Skip holidays
            if current not in holidays:
                trading_days_count += 1

        if trading_days_count < num_trading_days:
            current -= timedelta(days=1)

    start_date = current

    logger.info(f'[DATE_CALC] {num_trading_days} trading days: {start_date.strftime("%Y-%m-%d")} to {end_date_str}')

    return start_date.strftime('%Y-%m-%d'), end_date_str


def load_symbol_universe() -> List[Dict]:
    """Load and sort symbol universe"""
    try:
        with open(CORE_230_PATH, 'r') as f:
            data = json.load(f)

        # Sort symbols for deterministic ordering
        data.sort(key=lambda x: x['ticker'])

        logger.info(f'[SYMBOLS] Loaded {len(data)} symbols (sorted)')
        return data
    except Exception as e:
        logger.error(f'[SYMBOLS] Failed to load CORE_230.json: {e}')
        return []


def fetch_with_retry(fetch_func, *args, **kwargs):
    """Fetch with exponential backoff retry logic"""
    for attempt in range(MAX_RETRIES + 1):
        try:
            # Rate limiting
            rate_limiter.acquire()

            # Make request
            result = fetch_func(*args, **kwargs)

            if result is not None:
                return result

            # None result - treat as retryable
            if attempt < MAX_RETRIES:
                wait = BACKOFF_SEQUENCE[min(attempt, len(BACKOFF_SEQUENCE) - 1)]
                time.sleep(wait)

        except Exception as e:
            error_msg = str(e)

            # Check for hard fail status codes
            if any(str(code) in error_msg for code in [400, 401, 403, 404]):
                logger.error(f'[HARD_FAIL] {error_msg} - skipping')
                return None

            # Retry on 5xx or timeout
            if attempt < MAX_RETRIES:
                wait = BACKOFF_SEQUENCE[min(attempt, len(BACKOFF_SEQUENCE) - 1)]
                logger.warning(f'[RETRY] Attempt {attempt + 1} error: {error_msg}, retrying in {wait}s')
                time.sleep(wait)
            else:
                logger.error(f'[RETRY] Max retries exceeded: {error_msg}')
                return None

    return None


def build_insert_rows(symbol: str, snapshot_date: str, options_data: List[Dict],
                     underlying_price: float) -> List[Tuple]:
    """Build rows for batched insert"""
    rows = []

    for option in options_data:
        try:
            # Parse option data
            contract_name = option.get('contractName', '')
            expiration = option.get('expirationDate', '')
            strike = float(option.get('strike', 0))
            option_type = option.get('type', '').lower()

            last_price = option.get('lastTradePrice')
            bid = option.get('bid')
            ask = option.get('ask')
            volume = option.get('volume')
            open_interest = option.get('openInterest')
            iv = option.get('impliedVolatility')
            delta = option.get('delta')
            gamma = option.get('gamma')
            theta = option.get('theta')
            vega = option.get('vega')

            # Calculate days to expiration
            days_to_exp = None
            if expiration:
                try:
                    exp_date = datetime.strptime(expiration, '%Y-%m-%d')
                    snap_date = datetime.strptime(snapshot_date, '%Y-%m-%d')
                    days_to_exp = (exp_date - snap_date).days
                except:
                    pass

            # Calculate moneyness
            moneyness = None
            if underlying_price and strike:
                moneyness = strike / underlying_price

            rows.append((
                symbol, snapshot_date, contract_name, expiration, strike, option_type,
                last_price, bid, ask, volume, open_interest, iv,
                delta, gamma, theta, vega, underlying_price, days_to_exp, moneyness
            ))

        except Exception as e:
            continue

    return rows


def batched_insert(conn, rows: List[Tuple]) -> int:
    """Insert rows in batches"""
    cursor = conn.cursor()
    inserted_count = 0

    for i in range(0, len(rows), BATCH_SIZE_MAX):
        batch = rows[i:i + BATCH_SIZE_MAX]

        try:
            cursor.executemany('''
                INSERT OR IGNORE INTO historical_options_chains
                (symbol, snapshot_date, contract_name, expiration_date, strike, option_type,
                 last_price, bid, ask, volume, open_interest, implied_volatility,
                 delta, gamma, theta, vega, underlying_price, days_to_expiration, moneyness)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', batch)

            inserted_count += cursor.rowcount
            conn.commit()

        except Exception as e:
            logger.error(f'[BATCHED_INSERT] Error: {e}')
            conn.rollback()

    return inserted_count


def ingest_symbol_parallel(symbol: str, start_date: str, end_date: str,
                           worker_id: int) -> Dict:
    """
    Ingest historical options for a symbol using date-driven approach

    Args:
        symbol: Stock ticker
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        worker_id: Worker process ID

    Returns:
        Dictionary with ingestion statistics
    """
    logger.info(f'[WORKER_{worker_id}] Starting {symbol} ({start_date} to {end_date})')

    stats = {
        'symbol': symbol,
        'worker_id': worker_id,
        'total_records': 0,
        'dates_processed': 0,
        'success': False
    }

    try:
        # Initialize EODHD client (per-worker)
        client = get_eodhd_client()

        # Connect to database (WAL mode allows concurrent writes)
        conn = sqlite3.connect(OPTIONS_UNIVERSE_DB, timeout=30.0)

        # Process each date in range
        current = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')

        while current <= end:
            date_str = current.strftime('%Y-%m-%d')

            # Skip weekends
            if current.weekday() >= 5:
                current += timedelta(days=1)
                continue

            # Fetch options chain for this date
            data = fetch_with_retry(client.get_options_data, symbol, date_str)

            if data and data.get('data'):
                # Extract underlying price
                underlying_price = data['data'][0].get('underlyingPrice') if data['data'] else None

                # Build rows
                rows = build_insert_rows(symbol, date_str, data['data'], underlying_price)

                # Batched insert
                if rows:
                    inserted = batched_insert(conn, rows)
                    stats['total_records'] += inserted
                    stats['dates_processed'] += 1

                    if inserted > 0:
                        logger.info(
                            f'[WORKER_{worker_id}] {symbol} {date_str}: '
                            f'{inserted} records (cumulative: {stats["total_records"]:,})'
                        )

            current += timedelta(days=1)

        conn.close()
        stats['success'] = True

        logger.info(
            f'[WORKER_{worker_id}] {symbol} COMPLETE: '
            f'{stats["total_records"]:,} records from {stats["dates_processed"]} dates'
        )

    except Exception as e:
        logger.error(f'[WORKER_{worker_id}] {symbol} FAILED: {e}')
        stats['error'] = str(e)

    return stats


def run_parallel_ingestion(lookback_trading_days: int = DEFAULT_LOOKBACK_TRADING_DAYS,
                           end_date: str = '2026-02-13',
                           symbols_limit: Optional[int] = None,
                           num_workers: int = MIN_WORKERS):
    """
    Run parallel ingestion with symbol-level parallelism

    Args:
        lookback_trading_days: Number of trading days to look back (default 30)
        end_date: End date in YYYY-MM-DD format (default '2026-02-13' = last Friday)
        symbols_limit: Limit to first N symbols (for testing)
        num_workers: Number of parallel workers (4-8)
    """
    start_time = time.time()

    logger.info('=' * 80)
    logger.info('PARALLEL EODHD HISTORICAL OPTIONS INGESTION')
    logger.info('=' * 80)

    # Initialize database
    initialize_database()

    # Load symbols (sorted for determinism)
    symbol_data = load_symbol_universe()
    if not symbol_data:
        logger.error('[ERROR] No symbols loaded - exiting')
        return

    if symbols_limit:
        symbol_data = symbol_data[:symbols_limit]
        logger.info(f'[CONFIG] Limited to first {symbols_limit} symbols (testing mode)')

    # Calculate date range using trading days (excludes weekends and holidays)
    start_date, end_date = get_trading_days_range(end_date, lookback_trading_days)

    # Clamp workers
    num_workers = max(MIN_WORKERS, min(num_workers, MAX_WORKERS))

    logger.info(f'[CONFIG] Date range: {start_date} to {end_date} ({lookback_trading_days} trading days)')
    logger.info(f'[CONFIG] Symbols: {len(symbol_data)} (sorted)')
    logger.info(f'[CONFIG] Workers: {num_workers}')
    logger.info(f'[CONFIG] Rate limit: {RATE_LIMIT_RPS} req/sec')
    logger.info(f'[CONFIG] Batch size: {BATCH_SIZE_MAX}')
    logger.info(f'[CONFIG] Database: {OPTIONS_UNIVERSE_DB}')
    logger.info('=' * 80)

    # Run parallel ingestion
    all_stats = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all symbols
        futures = {}
        for idx, symbol_info in enumerate(symbol_data):
            symbol = symbol_info['ticker']
            worker_id = idx % num_workers

            future = executor.submit(
                ingest_symbol_parallel,
                symbol, start_date, end_date, worker_id
            )
            futures[future] = symbol

        # Collect results as they complete
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                stats = future.result()
                all_stats.append(stats)

                logger.info(
                    f'[PROGRESS] {len(all_stats)}/{len(symbol_data)}: '
                    f'{symbol} - {stats["total_records"]:,} records'
                )

            except Exception as e:
                logger.error(f'[PROGRESS] {symbol} failed with exception: {e}')

    # Summary
    end_time = time.time()
    runtime = end_time - start_time

    success_count = sum(1 for s in all_stats if s['success'])
    error_count = len(all_stats) - success_count
    total_records = sum(s['total_records'] for s in all_stats)

    logger.info('')
    logger.info('=' * 80)
    logger.info('INGESTION COMPLETE')
    logger.info('=' * 80)
    logger.info(f'Runtime: {runtime:.2f} seconds ({runtime/60:.2f} minutes)')
    logger.info(f'Symbols processed: {len(all_stats)}')
    logger.info(f'Success: {success_count}')
    logger.info(f'Errors: {error_count}')
    logger.info(f'Total records: {total_records:,}')
    logger.info(f'Records/second: {total_records/runtime:.1f}')
    logger.info('=' * 80)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Parallel EODHD historical options ingestion')
    parser.add_argument('--workers', type=int, default=MIN_WORKERS,
                       help=f'Number of workers ({MIN_WORKERS}-{MAX_WORKERS})')
    parser.add_argument('--test', action='store_true',
                       help='Test mode: 3 symbols only')
    parser.add_argument('--trading-days', type=int, default=DEFAULT_LOOKBACK_TRADING_DAYS,
                       help='Number of trading days to look back (default: 30)')
    parser.add_argument('--end-date', type=str, default='2026-02-13',
                       help='End date in YYYY-MM-DD format (default: 2026-02-13)')

    args = parser.parse_args()

    symbols_limit = 3 if args.test else None

    run_parallel_ingestion(
        lookback_trading_days=args.trading_days,
        end_date=args.end_date,
        symbols_limit=symbols_limit,
        num_workers=args.workers
    )
