"""
Daily Append-Only Options Ingestion using Tradier (TURBO MODE)
===============================================================

Version 4.0.0 - High-Performance Parallel Architecture

Features:
- Symbol-level parallelism (16 workers)
- Concurrent expiration fetching (4 per symbol)
- Streaming batch inserts (50K contracts)
- Vectorized parsing where possible
- Target runtime: <60 seconds for 226 symbols

Performance:
- Sequential: ~11 minutes (226 × 3s)
- Turbo: ~14-42 seconds (16x-48x speedup)
"""

import sys
import os
import json
import sqlite3
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OPTIONS_DIR = SCRIPT_DIR
BACKEND_DIR = os.path.abspath(os.path.join(OPTIONS_DIR, '..', '..'))

# Import Tradier client
from backend.turbomode.options.tradier_options_client import TradierOptionsClient

# File paths
FILTERED_UNIVERSE_PATH = os.path.join(OPTIONS_DIR, 'options_universe_filtered.json')
OPTIONS_UNIVERSE_DB = os.path.join(OPTIONS_DIR, 'Data', 'options_universe.db')

# Turbo Mode Configuration
MAX_WORKERS = 16  # Symbol-level parallelism
EXPIRATION_WORKERS = 4  # Concurrent chain fetching per symbol
BATCH_INSERT_SIZE = 50000  # Insert every N contracts
BATCH_SIZE = 5000  # SQLite executemany batch size
LOG_INTERVAL = 10000

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_filtered_universe() -> List[str]:
    """
    Load filtered symbol universe

    Returns:
        List of ticker symbols
    """
    logger.info(f'[LOAD] Loading filtered universe from: {FILTERED_UNIVERSE_PATH}')

    with open(FILTERED_UNIVERSE_PATH, 'r') as f:
        data = json.load(f)

    # Extract tickers
    tickers = [entry['ticker'] for entry in data]

    logger.info(f'[LOAD] Loaded {len(tickers)} symbols')
    return tickers


def get_most_recent_date_in_db(conn: sqlite3.Connection) -> Optional[str]:
    """
    Get the most recent snapshot_date from historical_options_chains

    Args:
        conn: Database connection

    Returns:
        Most recent date string (YYYY-MM-DD) or None if empty
    """
    cursor = conn.cursor()

    cursor.execute('''
        SELECT MAX(snapshot_date) FROM historical_options_chains
    ''')

    result = cursor.fetchone()
    most_recent_date = result[0] if result else None

    if most_recent_date:
        logger.info(f'[DB] Most recent date in database: {most_recent_date}')
    else:
        logger.info('[DB] Database is empty - no historical data found')

    return most_recent_date


def get_current_trading_date() -> str:
    """
    Get current date in YYYY-MM-DD format

    Returns:
        Today's date string
    """
    return datetime.now().strftime('%Y-%m-%d')


def fetch_symbol_options_turbo(
    symbol: str,
    snapshot_date: str,
    api_key: str,
    worker_id: int
) -> Tuple[str, List[Dict], float]:
    """
    TURBO: Fetch all options for a single symbol with concurrent expiration fetching

    Args:
        symbol: Stock ticker
        snapshot_date: Date to assign to snapshot_date field
        api_key: Tradier API key
        worker_id: Worker thread ID for logging

    Returns:
        Tuple of (symbol, contracts list, elapsed_time)
    """
    start_time = time.time()
    contracts = []

    try:
        # Create per-worker Tradier client
        client = TradierOptionsClient(api_key=api_key)

        # Get underlying quote
        quote = client.get_underlying_quote(symbol)
        if not quote:
            logger.warning(f'[W{worker_id}] {symbol}: Failed to get underlying quote')
            return (symbol, [], time.time() - start_time)

        underlying_price = quote.get('last', 0)
        if underlying_price <= 0:
            logger.warning(f'[W{worker_id}] {symbol}: Invalid underlying price: {underlying_price}')
            return (symbol, [], time.time() - start_time)

        # Get expirations
        expirations = client.get_expirations(symbol)
        if not expirations:
            logger.warning(f'[W{worker_id}] {symbol}: No expirations found')
            return (symbol, [], time.time() - start_time)

        # TURBO: Fetch ALL option chains concurrently
        with ThreadPoolExecutor(max_workers=EXPIRATION_WORKERS) as exp_pool:
            chain_futures = {}
            for exp in expirations:
                expiration_date = exp.get('date')
                if expiration_date:
                    future = exp_pool.submit(
                        client.get_option_chain,
                        symbol,
                        expiration_date,
                        True  # greeks=True
                    )
                    chain_futures[future] = expiration_date

            # Collect chain results as they complete
            for future in as_completed(chain_futures):
                expiration_date = chain_futures[future]
                try:
                    chain = future.result()
                    if not chain:
                        continue

                    # Process calls
                    calls = chain.get('calls', {})
                    for strike, contract_data in calls.items():
                        contract = build_option_dict(
                            symbol=symbol,
                            snapshot_date=snapshot_date,
                            expiration_date=expiration_date,
                            strike=strike,
                            option_type='call',
                            contract_data=contract_data,
                            underlying_price=underlying_price
                        )
                        if contract:
                            contracts.append(contract)

                    # Process puts
                    puts = chain.get('puts', {})
                    for strike, contract_data in puts.items():
                        contract = build_option_dict(
                            symbol=symbol,
                            snapshot_date=snapshot_date,
                            expiration_date=expiration_date,
                            strike=strike,
                            option_type='put',
                            contract_data=contract_data,
                            underlying_price=underlying_price
                        )
                        if contract:
                            contracts.append(contract)

                except Exception as e:
                    logger.error(f'[W{worker_id}] {symbol} exp {expiration_date}: Chain fetch error: {e}')
                    continue

        elapsed = time.time() - start_time
        logger.info(f'[W{worker_id}] {symbol}: {len(contracts)} contracts in {elapsed:.2f}s')
        return (symbol, contracts, elapsed)

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f'[W{worker_id}] {symbol}: Fatal error: {e}')
        return (symbol, [], elapsed)


def build_option_dict(
    symbol: str,
    snapshot_date: str,
    expiration_date: str,
    strike: float,
    option_type: str,
    contract_data: Dict,
    underlying_price: float
) -> Optional[Dict]:
    """
    Build option dictionary matching historical_options_chains schema

    Returns:
        Dictionary with all required fields or None if invalid
    """
    try:
        # Extract market data
        bid = contract_data.get('bid')
        ask = contract_data.get('ask')
        last_price = contract_data.get('last')
        volume = contract_data.get('volume')
        open_interest = contract_data.get('open_interest')

        # Extract greeks
        delta = contract_data.get('delta')
        gamma = contract_data.get('gamma')
        theta = contract_data.get('theta')
        vega = contract_data.get('vega')
        iv = contract_data.get('implied_volatility')

        # Calculate contract name (OCC format)
        contract_symbol = contract_data.get('contract_symbol', '')

        # Calculate days to expiration
        days_to_exp = None
        try:
            exp_date = datetime.strptime(expiration_date, '%Y-%m-%d')
            snap_date = datetime.strptime(snapshot_date, '%Y-%m-%d')
            days_to_exp = (exp_date - snap_date).days
        except:
            pass

        # Calculate moneyness
        moneyness = None
        if underlying_price and strike:
            moneyness = strike / underlying_price

        return {
            'symbol': symbol,
            'snapshot_date': snapshot_date,
            'contract_name': contract_symbol,
            'expiration_date': expiration_date,
            'strike': strike,
            'option_type': option_type,
            'last_price': last_price,
            'bid': bid,
            'ask': ask,
            'volume': volume,
            'open_interest': open_interest,
            'implied_volatility': iv,
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'underlying_price': underlying_price,
            'days_to_expiration': days_to_exp,
            'moneyness': moneyness
        }

    except Exception as e:
        logger.error(f'[BUILD] Error building option dict: {e}')
        return None


def insert_batch(conn: sqlite3.Connection, batch: List[Dict]) -> int:
    """
    Insert a batch of contracts into database

    Args:
        conn: Database connection
        batch: List of contract dictionaries

    Returns:
        Number of rows inserted
    """
    if not batch:
        return 0

    cursor = conn.cursor()

    # Prepare rows
    rows = []
    for opt in batch:
        rows.append((
            opt['symbol'],
            opt['snapshot_date'],
            opt['contract_name'],
            opt['expiration_date'],
            opt['strike'],
            opt['option_type'],
            opt['last_price'],
            opt['bid'],
            opt['ask'],
            opt['volume'],
            opt['open_interest'],
            opt['implied_volatility'],
            opt['delta'],
            opt['gamma'],
            opt['theta'],
            opt['vega'],
            opt['underlying_price'],
            opt['days_to_expiration'],
            opt['moneyness']
        ))

    # Insert with OR IGNORE for idempotency
    try:
        cursor.executemany('''
            INSERT OR IGNORE INTO historical_options_chains
            (symbol, snapshot_date, contract_name, expiration_date, strike, option_type,
             last_price, bid, ask, volume, open_interest, implied_volatility,
             delta, gamma, theta, vega, underlying_price, days_to_expiration, moneyness)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', rows)

        conn.commit()
        return cursor.rowcount

    except Exception as e:
        logger.error(f'[INSERT] Batch insert error: {e}')
        conn.rollback()
        return 0


def run_daily_ingestion():
    """
    Main function: TURBO daily append-only options ingestion from Tradier
    """
    overall_start = time.time()

    logger.info('=' * 80)
    logger.info('DAILY OPTIONS INGESTION - TRADIER TURBO MODE')
    logger.info('=' * 80)
    logger.info(f'[CONFIG] Max workers: {MAX_WORKERS}')
    logger.info(f'[CONFIG] Expiration workers: {EXPIRATION_WORKERS}')
    logger.info(f'[CONFIG] Batch insert size: {BATCH_INSERT_SIZE:,}')

    # Load filtered universe
    symbols = load_filtered_universe()
    total_symbols = len(symbols)

    # Connect to database
    logger.info(f'[DB] Connecting to: {OPTIONS_UNIVERSE_DB}')
    conn = sqlite3.connect(OPTIONS_UNIVERSE_DB)

    # Get most recent date in database
    most_recent_db_date = get_most_recent_date_in_db(conn)

    # Get current trading date (today)
    current_date = get_current_trading_date()
    logger.info(f'[DATE] Current date: {current_date}')

    # Check if we need to ingest
    if most_recent_db_date and current_date <= most_recent_db_date:
        logger.info('[DECISION] No new trading day detected - skipping inserts')
        logger.info(f'[DECISION] DB date: {most_recent_db_date}, Current date: {current_date}')
        logger.info('=' * 80)
        conn.close()
        return

    logger.info(f'[DECISION] New trading day detected: {current_date}')
    logger.info('[DECISION] Proceeding with TURBO ingestion...')

    # Get API key (try environment first, then config file)
    api_key = os.getenv('TRADIER_API_KEY')
    if not api_key:
        try:
            config_path = os.path.join(BACKEND_DIR, '..', 'config', 'api_keys.json')
            with open(config_path, 'r') as f:
                api_keys = json.load(f)
                api_key = api_keys.get('TRADIER_API_KEY')
        except Exception as e:
            logger.error(f'[ERROR] Failed to load API key: {e}')
            conn.close()
            return

    if not api_key:
        logger.error('[ERROR] TRADIER_API_KEY not found in environment or config')
        conn.close()
        return

    logger.info(f'[TURBO] Starting parallel ingestion with {MAX_WORKERS} workers...')

    # TURBO: Parallel symbol processing
    batch_buffer = []
    completed_count = 0
    total_contracts = 0
    worker_times = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all symbol tasks
        futures = {}
        for idx, symbol in enumerate(symbols):
            worker_id = idx % MAX_WORKERS
            future = executor.submit(
                fetch_symbol_options_turbo,
                symbol,
                current_date,
                api_key,
                worker_id
            )
            futures[future] = symbol

        # Process results as they complete (streaming)
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                symbol_name, contracts, elapsed = future.result()
                completed_count += 1
                worker_times.append(elapsed)

                if contracts:
                    batch_buffer.extend(contracts)
                    total_contracts += len(contracts)

                # Progress logging
                percent = (completed_count / total_symbols) * 100
                logger.info(f'[PROGRESS] {completed_count}/{total_symbols} ({percent:.1f}%) - {symbol_name}: {len(contracts)} contracts')

                # Streaming batch insert
                if len(batch_buffer) >= BATCH_INSERT_SIZE:
                    inserted = insert_batch(conn, batch_buffer)
                    logger.info(f'[INSERT] Batch: {inserted:,} rows inserted')
                    batch_buffer = []

            except Exception as e:
                logger.error(f'[ERROR] Failed to process {symbol}: {e}')
                completed_count += 1

    # Insert remaining contracts
    if batch_buffer:
        inserted = insert_batch(conn, batch_buffer)
        logger.info(f'[INSERT] Final batch: {inserted:,} rows inserted')

    # Calculate statistics
    overall_elapsed = time.time() - overall_start
    avg_worker_time = sum(worker_times) / len(worker_times) if worker_times else 0
    contracts_per_sec = total_contracts / overall_elapsed if overall_elapsed > 0 else 0

    # Final summary
    logger.info('=' * 80)
    logger.info('TURBO DAILY INGESTION COMPLETE')
    logger.info('=' * 80)
    logger.info(f'Date: {current_date}')
    logger.info(f'Symbols processed: {completed_count}/{total_symbols}')
    logger.info(f'Total contracts: {total_contracts:,}')
    logger.info(f'Total runtime: {overall_elapsed:.2f}s')
    logger.info(f'Avg per-symbol time: {avg_worker_time:.2f}s')
    logger.info(f'Throughput: {contracts_per_sec:,.0f} contracts/sec')
    logger.info(f'Speedup estimate: ~{678/overall_elapsed:.1f}x vs sequential')
    logger.info('=' * 80)

    # Close connection
    conn.close()


if __name__ == '__main__':
    try:
        run_daily_ingestion()
    except Exception as e:
        logger.error(f'[FATAL] Daily ingestion failed: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
