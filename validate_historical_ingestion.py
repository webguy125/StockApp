"""
Step 17: Validate historical ingestion integrity
Created per task specification version 3.0.1
"""
import sqlite3
import json
from datetime import datetime, timedelta
from collections import defaultdict

# File paths
UNIVERSE_DB = 'C:/StockApp/backend/turbomode/Options/Data/options_universe.db'
CORE_230_PATH = 'C:/StockApp/config/symbols/CORE_230.json'
REPORT_PATH = 'C:/StockApp/backend/turbomode/Options/Reports/historical_validation_report.json'

# US Market Holidays 2026
HOLIDAYS = [
    '2026-01-01',  # New Year's Day
    '2026-01-19',  # MLK Day
    '2026-02-16',  # Presidents Day
]


def get_trading_days(start_date_str, end_date_str):
    """
    Get list of trading days between start and end (excludes weekends and holidays)

    Args:
        start_date_str: Start date YYYY-MM-DD
        end_date_str: End date YYYY-MM-DD

    Returns:
        List of date strings
    """
    trading_days = []
    current = datetime.strptime(start_date_str, '%Y-%m-%d')
    end = datetime.strptime(end_date_str, '%Y-%m-%d')

    while current <= end:
        date_str = current.strftime('%Y-%m-%d')

        # Skip weekends
        if current.weekday() < 5:
            # Skip holidays
            if date_str not in HOLIDAYS:
                trading_days.append(date_str)

        current += timedelta(days=1)

    return trading_days


def validate_historical_ingestion():
    """
    Perform full integrity check on historical ingestion
    """
    print('=' * 80)
    print('STEP 17: VALIDATE HISTORICAL INGESTION')
    print('=' * 80)

    # Initialize report
    report = {
        'symbols_missing': [],
        'dates_missing': {},
        'symbols_with_incomplete_chains': [],
        'symbols_with_zero_contracts': [],
        'duplicate_rows_found': 0,
        'null_field_violations': [],
        'expiration_ladder_anomalies': [],
        'strike_ladder_anomalies': [],
        'summary_pass_fail': 'PASS'
    }

    # Load CORE_230 symbols
    print('[VALIDATE] Loading CORE_230 symbols...')
    with open(CORE_230_PATH, 'r') as f:
        core_230_data = json.load(f)

    expected_symbols = set([entry['ticker'] for entry in core_230_data])
    print(f'[VALIDATE] Expected symbols: {len(expected_symbols)}')

    # Connect to database
    print('[VALIDATE] Connecting to options_universe.db...')
    conn = sqlite3.connect(UNIVERSE_DB)
    cursor = conn.cursor()

    # Get actual symbols in database
    cursor.execute('SELECT DISTINCT symbol FROM historical_options_chains ORDER BY symbol')
    actual_symbols = set([row[0] for row in cursor.fetchall()])
    print(f'[VALIDATE] Actual symbols in DB: {len(actual_symbols)}')

    # Check 1: Verify all symbols exist
    print('[CHECK 1] Verifying all symbols exist...')
    missing_symbols = expected_symbols - actual_symbols
    if missing_symbols:
        report['symbols_missing'] = sorted(list(missing_symbols))
        report['summary_pass_fail'] = 'FAIL'
        print(f'[CHECK 1] FAIL: {len(missing_symbols)} missing symbols')
    else:
        print('[CHECK 1] PASS: All symbols present')

    # Get date range from database
    cursor.execute('SELECT MIN(snapshot_date), MAX(snapshot_date) FROM historical_options_chains')
    min_date, max_date = cursor.fetchone()
    print(f'[INFO] Date range in DB: {min_date} to {max_date}')

    # Get expected trading days
    expected_trading_days = get_trading_days(min_date, max_date)
    print(f'[INFO] Expected trading days: {len(expected_trading_days)}')

    # Check 2: Verify complete coverage for each symbol
    print('[CHECK 2] Verifying complete date coverage per symbol...')
    for symbol in sorted(actual_symbols):
        cursor.execute('''
            SELECT DISTINCT snapshot_date
            FROM historical_options_chains
            WHERE symbol = ?
            ORDER BY snapshot_date
        ''', (symbol,))

        symbol_dates = set([row[0] for row in cursor.fetchall()])
        missing_dates = set(expected_trading_days) - symbol_dates

        if missing_dates:
            report['dates_missing'][symbol] = sorted(list(missing_dates))
            if symbol not in report['symbols_with_incomplete_chains']:
                report['symbols_with_incomplete_chains'].append(symbol)
            report['summary_pass_fail'] = 'FAIL'

    if report['symbols_with_incomplete_chains']:
        print(f'[CHECK 2] FAIL: {len(report["symbols_with_incomplete_chains"])} symbols with incomplete coverage')
    else:
        print('[CHECK 2] PASS: All symbols have complete date coverage')

    # Check 3: Verify each (symbol, date) has at least one contract
    print('[CHECK 3] Verifying all (symbol, date) pairs have contracts...')
    for symbol in sorted(actual_symbols):
        cursor.execute('''
            SELECT snapshot_date, COUNT(*)
            FROM historical_options_chains
            WHERE symbol = ?
            GROUP BY snapshot_date
            HAVING COUNT(*) = 0
        ''', (symbol,))

        zero_contract_dates = cursor.fetchall()
        if zero_contract_dates:
            if symbol not in report['symbols_with_zero_contracts']:
                report['symbols_with_zero_contracts'].append(symbol)
            report['summary_pass_fail'] = 'FAIL'

    if report['symbols_with_zero_contracts']:
        print(f'[CHECK 3] FAIL: {len(report["symbols_with_zero_contracts"])} symbols with zero-contract dates')
    else:
        print('[CHECK 3] PASS: All (symbol, date) pairs have contracts')

    # Check 4: Verify expiration ladders are consistent
    print('[CHECK 4] Verifying expiration ladder consistency...')
    for symbol in sorted(actual_symbols):
        cursor.execute('''
            SELECT snapshot_date, COUNT(DISTINCT expiration_date) as exp_count
            FROM historical_options_chains
            WHERE symbol = ?
            GROUP BY snapshot_date
            ORDER BY snapshot_date
        ''', (symbol,))

        exp_counts = [row[1] for row in cursor.fetchall()]

        # Check if any date has 0 expirations (anomaly)
        if any(count == 0 for count in exp_counts):
            if symbol not in report['expiration_ladder_anomalies']:
                report['expiration_ladder_anomalies'].append(symbol)
            report['summary_pass_fail'] = 'FAIL'

    if report['expiration_ladder_anomalies']:
        print(f'[CHECK 4] FAIL: {len(report["expiration_ladder_anomalies"])} symbols with expiration anomalies')
    else:
        print('[CHECK 4] PASS: Expiration ladders consistent')

    # Check 5: Verify strike ladders are non-empty
    print('[CHECK 5] Verifying strike ladders are non-empty...')
    for symbol in sorted(actual_symbols):
        cursor.execute('''
            SELECT snapshot_date, expiration_date, COUNT(DISTINCT strike) as strike_count
            FROM historical_options_chains
            WHERE symbol = ?
            GROUP BY snapshot_date, expiration_date
            HAVING strike_count = 0
        ''', (symbol,))

        zero_strike_results = cursor.fetchall()
        if zero_strike_results:
            if symbol not in report['strike_ladder_anomalies']:
                report['strike_ladder_anomalies'].append(symbol)
            report['summary_pass_fail'] = 'FAIL'

    if report['strike_ladder_anomalies']:
        print(f'[CHECK 5] FAIL: {len(report["strike_ladder_anomalies"])} symbols with strike anomalies')
    else:
        print('[CHECK 5] PASS: Strike ladders non-empty')

    # Check 6: Verify no duplicate rows
    print('[CHECK 6] Checking for duplicate rows...')
    cursor.execute('''
        SELECT symbol, snapshot_date, contract_name, COUNT(*) as cnt
        FROM historical_options_chains
        GROUP BY symbol, snapshot_date, contract_name
        HAVING cnt > 1
    ''')

    duplicates = cursor.fetchall()
    report['duplicate_rows_found'] = len(duplicates)

    if duplicates:
        print(f'[CHECK 6] FAIL: {len(duplicates)} duplicate row groups found')
        report['summary_pass_fail'] = 'FAIL'
    else:
        print('[CHECK 6] PASS: No duplicate rows')

    # Check 7: Verify no NULL critical fields
    print('[CHECK 7] Checking for NULL critical fields...')
    cursor.execute('''
        SELECT
            SUM(CASE WHEN symbol IS NULL THEN 1 ELSE 0 END) as null_symbol,
            SUM(CASE WHEN expiration_date IS NULL THEN 1 ELSE 0 END) as null_expiration,
            SUM(CASE WHEN strike IS NULL THEN 1 ELSE 0 END) as null_strike,
            SUM(CASE WHEN option_type IS NULL THEN 1 ELSE 0 END) as null_type
        FROM historical_options_chains
    ''')

    null_counts = cursor.fetchone()
    null_symbol, null_expiration, null_strike, null_type = null_counts

    if any(null_counts):
        report['null_field_violations'] = [
            {'field': 'symbol', 'count': null_symbol},
            {'field': 'expiration_date', 'count': null_expiration},
            {'field': 'strike', 'count': null_strike},
            {'field': 'option_type', 'count': null_type}
        ]
        report['summary_pass_fail'] = 'FAIL'
        print(f'[CHECK 7] FAIL: NULL violations found (symbol={null_symbol}, exp={null_expiration}, strike={null_strike}, type={null_type})')
    else:
        print('[CHECK 7] PASS: No NULL critical fields')

    # Close connection
    conn.close()

    # Write report
    print('[REPORT] Writing validation report...')
    with open(REPORT_PATH, 'w') as f:
        json.dump(report, f, indent=2)

    print(f'[REPORT] Report written to: {REPORT_PATH}')

    # Print summary
    print('=' * 80)
    print('STEP 17: COMPLETE')
    print('=' * 80)
    print(f'Validation Result: {report["summary_pass_fail"]}')
    print('')
    print('Summary:')
    print(f'  - Missing symbols: {len(report["symbols_missing"])}')
    print(f'  - Symbols with incomplete chains: {len(report["symbols_with_incomplete_chains"])}')
    print(f'  - Symbols with zero contracts: {len(report["symbols_with_zero_contracts"])}')
    print(f'  - Duplicate rows: {report["duplicate_rows_found"]}')
    print(f'  - NULL field violations: {len([v for v in report["null_field_violations"] if v["count"] > 0]) if report["null_field_violations"] else 0}')
    print(f'  - Expiration ladder anomalies: {len(report["expiration_ladder_anomalies"])}')
    print(f'  - Strike ladder anomalies: {len(report["strike_ladder_anomalies"])}')
    print('=' * 80)

    if report['summary_pass_fail'] == 'FAIL':
        print('[WARNING] Validation FAILED - Issues detected (see report for details)')
        print('[WARNING] Do NOT attempt to fix - report only as per requirements')
    else:
        print('[SUCCESS] Validation PASSED - All integrity checks passed')

    print('=' * 80)


if __name__ == '__main__':
    validate_historical_ingestion()
