"""
Step 16: Populate options_training_history table with contract-level outcomes
Created per task specification version 3.0.1
"""
import sqlite3
from datetime import datetime, timedelta
from collections import defaultdict

# Database paths
UNIVERSE_DB = 'C:/StockApp/backend/turbomode/Options/Data/options_universe.db'
TRAINING_DB = 'C:/StockApp/backend/turbomode/Options/Data/options_training_history.db'

# Processing parameters
BATCH_SIZE = 5000
LOG_INTERVAL = 10000
TRADING_DAYS_FORWARD = 14

# US Market Holidays 2026
HOLIDAYS = [
    '2026-01-01',  # New Year's Day
    '2026-01-19',  # MLK Day
    '2026-02-16',  # Presidents Day
]


def get_next_trading_date(date_str, days_forward):
    """
    Get the date N trading days forward (skip weekends and holidays)

    Args:
        date_str: Starting date YYYY-MM-DD
        days_forward: Number of trading days to go forward

    Returns:
        Date string YYYY-MM-DD or None if not found
    """
    current = datetime.strptime(date_str, '%Y-%m-%d')
    trading_days_counted = 0

    while trading_days_counted < days_forward:
        current += timedelta(days=1)

        # Skip weekends (Saturday=5, Sunday=6)
        if current.weekday() >= 5:
            continue

        # Skip holidays
        if current.strftime('%Y-%m-%d') in HOLIDAYS:
            continue

        trading_days_counted += 1

    return current.strftime('%Y-%m-%d')


def load_historical_data(conn_universe):
    """
    Load all historical options data organized by (symbol, date, expiration, strike, option_type)

    Returns:
        dict: Nested dict structure for fast lookups
    """
    print('[LOAD] Loading historical options chains from options_universe.db...')

    cursor = conn_universe.cursor()
    cursor.execute('''
        SELECT
            symbol,
            snapshot_date,
            expiration_date,
            strike,
            option_type,
            bid,
            ask,
            underlying_price,
            implied_volatility
        FROM historical_options_chains
        ORDER BY symbol, snapshot_date, expiration_date, strike, option_type
    ''')

    # Build nested structure: data[symbol][date][expiration][strike][option_type]
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    row_count = 0

    for row in cursor:
        symbol, date, expiration, strike, opt_type, bid, ask, underlying, iv = row

        # Calculate midprice
        midprice = None
        if bid is not None and ask is not None:
            midprice = (bid + ask) / 2.0

        data[symbol][date][expiration][strike][opt_type] = {
            'midprice': midprice,
            'underlying_price': underlying,
            'iv': iv
        }

        row_count += 1
        if row_count % 100000 == 0:
            print(f'[LOAD] Loaded {row_count:,} rows...')

    print(f'[LOAD] Loaded {row_count:,} total rows')
    return data


def compute_outcomes(data):
    """
    Compute 14-day outcomes for all contracts

    Args:
        data: Nested dict from load_historical_data()

    Yields:
        Tuple of values ready for INSERT
    """
    print('[COMPUTE] Computing 14-day outcomes...')

    total_processed = 0
    total_with_outcomes = 0

    for symbol in sorted(data.keys()):
        symbol_data = data[symbol]
        dates = sorted(symbol_data.keys())

        for date in dates:
            # Get t+14 trading date
            future_date = get_next_trading_date(date, TRADING_DAYS_FORWARD)

            date_data = symbol_data[date]

            for expiration in date_data:
                for strike in date_data[expiration]:
                    for option_type in date_data[expiration][strike]:
                        contract = date_data[expiration][strike][option_type]

                        # Extract t=0 values
                        midprice_t0 = contract['midprice']
                        underlying_t0 = contract['underlying_price']
                        iv_t0 = contract['iv']

                        # Initialize outcome fields
                        return_14d = None
                        direction_14d = None
                        underlying_return_14d = None
                        iv_change_14d = None

                        # Check if future date exists and contract still exists
                        if future_date and future_date in symbol_data:
                            future_data = symbol_data[future_date]

                            # Check if same contract exists at t+14
                            if expiration in future_data:
                                if strike in future_data[expiration]:
                                    if option_type in future_data[expiration][strike]:
                                        contract_t14 = future_data[expiration][strike][option_type]

                                        midprice_t14 = contract_t14['midprice']
                                        underlying_t14 = contract_t14['underlying_price']
                                        iv_t14 = contract_t14['iv']

                                        # Compute return_14d
                                        if midprice_t0 and midprice_t14 and midprice_t0 > 0:
                                            return_14d = ((midprice_t14 - midprice_t0) / midprice_t0) * 100.0
                                            direction_14d = 1 if return_14d > 0 else 0
                                            total_with_outcomes += 1

                                        # Compute underlying_return_14d
                                        if underlying_t0 and underlying_t14 and underlying_t0 > 0:
                                            underlying_return_14d = ((underlying_t14 - underlying_t0) / underlying_t0) * 100.0

                                        # Compute iv_change_14d
                                        if iv_t0 is not None and iv_t14 is not None:
                                            iv_change_14d = iv_t14 - iv_t0

                        # Yield row for insertion
                        yield (
                            symbol,
                            date,
                            expiration,
                            strike,
                            option_type,
                            midprice_t0,
                            underlying_t0,
                            iv_t0,
                            return_14d,
                            direction_14d,
                            underlying_return_14d,
                            iv_change_14d
                        )

                        total_processed += 1
                        if total_processed % LOG_INTERVAL == 0:
                            print(f'[COMPUTE] Processed {total_processed:,} contracts ({total_with_outcomes:,} with outcomes)...')

    print(f'[COMPUTE] Complete: {total_processed:,} contracts processed, {total_with_outcomes:,} with outcomes')


def populate_training_history():
    """
    Main function to populate options_training_history table
    """
    print('=' * 80)
    print('STEP 16: POPULATE OPTIONS_TRAINING_HISTORY')
    print('=' * 80)

    # Connect to databases
    print('[CONNECT] Opening database connections...')
    conn_universe = sqlite3.connect(UNIVERSE_DB)
    conn_training = sqlite3.connect(TRAINING_DB)
    cursor_training = conn_training.cursor()

    # Check for existing rows
    cursor_training.execute('SELECT COUNT(*) FROM options_training_history')
    existing_count = cursor_training.fetchone()[0]

    if existing_count > 0:
        print(f'[WARNING] Found {existing_count:,} existing rows - skipping to ensure idempotency')
        print('[INFO] Delete existing rows manually if you want to repopulate')
        conn_universe.close()
        conn_training.close()
        return

    # Load historical data
    data = load_historical_data(conn_universe)

    # Insert in batches
    print('[INSERT] Inserting outcomes into options_training_history...')
    batch = []
    total_inserted = 0

    for row in compute_outcomes(data):
        batch.append(row)

        if len(batch) >= BATCH_SIZE:
            cursor_training.executemany('''
                INSERT INTO options_training_history
                (symbol, date, expiration, strike, option_type, midprice,
                 underlying_price, iv, return_14d, direction_14d,
                 underlying_return_14d, iv_change_14d)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', batch)
            conn_training.commit()
            total_inserted += len(batch)
            print(f'[INSERT] Committed {total_inserted:,} rows...')
            batch = []

    # Insert remaining batch
    if batch:
        cursor_training.executemany('''
            INSERT INTO options_training_history
            (symbol, date, expiration, strike, option_type, midprice,
             underlying_price, iv, return_14d, direction_14d,
             underlying_return_14d, iv_change_14d)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', batch)
        conn_training.commit()
        total_inserted += len(batch)
        print(f'[INSERT] Committed final batch - {total_inserted:,} total rows')

    # Verify final count
    cursor_training.execute('SELECT COUNT(*) FROM options_training_history')
    final_count = cursor_training.fetchone()[0]

    cursor_training.execute('SELECT COUNT(*) FROM options_training_history WHERE return_14d IS NOT NULL')
    with_outcomes = cursor_training.fetchone()[0]

    # Close connections
    conn_universe.close()
    conn_training.close()

    print('=' * 80)
    print('STEP 16: COMPLETE')
    print('=' * 80)
    print(f'Total rows inserted: {final_count:,}')
    print(f'Rows with outcomes: {with_outcomes:,}')
    print(f'Rows without outcomes: {final_count - with_outcomes:,}')
    print('=' * 80)


if __name__ == '__main__':
    populate_training_history()
