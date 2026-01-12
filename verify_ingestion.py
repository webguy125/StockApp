import sqlite3
import pandas as pd

db_path = 'master_market_data/market_data.db'
conn = sqlite3.connect(db_path)

print('=' * 80)
print('PHASE 1 INGESTION VERIFICATION - 43 SYMBOL UNIVERSE')
print('=' * 80)
print()

# 1. List all tables first
print('1. DATABASE SCHEMA (ALL TABLES):')
print('-' * 80)
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
all_tables = [row[0] for row in cursor.fetchall()]
for table in all_tables:
    cursor.execute(f'SELECT COUNT(*) FROM "{table}"')
    count = cursor.fetchone()[0]
    print(f'  {table:30s}: {count:>8,} rows')
print()

# 2. Symbol coverage
print('2. SYMBOL METADATA:')
print('-' * 80)
cursor.execute('SELECT COUNT(DISTINCT symbol) FROM symbol_metadata')
total_symbols = cursor.fetchone()[0]
print(f'  Total symbols in metadata: {total_symbols}')

cursor.execute('SELECT symbol FROM symbol_metadata ORDER BY symbol')
symbols = [row[0] for row in cursor.fetchall()]
print(f'  Symbols: {", ".join(symbols)}')
print()

# 3. Expected vs Actual
print('3. EXPECTED VS ACTUAL:')
print('-' * 80)
from backend.advanced_ml.config import get_all_core_symbols, CRYPTO_SYMBOLS

expected_stocks = get_all_core_symbols()
expected_all = sorted(expected_stocks + CRYPTO_SYMBOLS)

print(f'  Expected: {len(expected_all)} symbols (40 stocks + 3 crypto)')
print(f'  Actual:   {total_symbols} symbols in database')

if len(expected_all) != total_symbols:
    print(f'  [WARNING] MISMATCH: Database has {total_symbols - len(expected_all)} extra symbols')

    # Find which symbols are extra
    extra_symbols = set(symbols) - set(expected_all)
    missing_symbols = set(expected_all) - set(symbols)

    if extra_symbols:
        print(f'  Extra symbols in DB (not in config): {sorted(extra_symbols)}')
    if missing_symbols:
        print(f'  Missing symbols (in config but not DB): {sorted(missing_symbols)}')
else:
    print('  [OK] Symbol count matches!')
print()

# 4. Candle data by timeframe (find all candles_* tables)
print('4. CANDLE DATA BY TIMEFRAME:')
print('-' * 80)
candle_tables = [t for t in all_tables if t.startswith('candles_')]
if candle_tables:
    for table in sorted(candle_tables):
        cursor.execute(f'SELECT COUNT(*) FROM "{table}"')
        count = cursor.fetchone()[0]
        cursor.execute(f'SELECT COUNT(DISTINCT symbol) FROM "{table}"')
        symbol_count = cursor.fetchone()[0]
        print(f'  {table:20s}: {count:>8,} rows, {symbol_count:>3} symbols')
else:
    print('  [WARNING] No candles_* tables found')
print()

# 5. Date ranges (use first candle table found)
if candle_tables:
    first_candle_table = candle_tables[0]
    print(f'5. DATE RANGES (using {first_candle_table}):')
    print('-' * 80)
    cursor.execute(f'''
        SELECT MIN(date) as min_date, MAX(date) as max_date, COUNT(DISTINCT date) as total_days
        FROM "{first_candle_table}"
    ''')
    min_date, max_date, total_days = cursor.fetchone()
    print(f'  Earliest date: {min_date}')
    print(f'  Latest date:   {max_date}')
    print(f'  Total days:    {total_days:,}')
    print()

# 6. Fundamentals
print('6. FUNDAMENTALS DATA:')
print('-' * 80)
if 'fundamentals' in all_tables:
    cursor.execute('SELECT COUNT(*) FROM fundamentals')
    fund_count = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(DISTINCT symbol) FROM fundamentals')
    fund_symbols = cursor.fetchone()[0]
    print(f'  Total fundamentals records: {fund_count}')
    print(f'  Symbols with fundamentals:  {fund_symbols}')
else:
    print('  [WARNING] No fundamentals table found')
print()

# 7. Corporate actions
print('7. CORPORATE ACTIONS:')
print('-' * 80)
if 'splits' in all_tables:
    cursor.execute('SELECT COUNT(*) FROM splits')
    splits = cursor.fetchone()[0]
    print(f'  Total splits:    {splits}')
else:
    print('  No splits table')

if 'dividends' in all_tables:
    cursor.execute('SELECT COUNT(*) FROM dividends')
    divs = cursor.fetchone()[0]
    print(f'  Total dividends: {divs}')
else:
    print('  No dividends table')
print()

# 8. Data quality - check for symbols with no data
if candle_tables:
    print('8. DATA QUALITY CHECKS:')
    print('-' * 80)
    first_candle_table = candle_tables[0]
    cursor.execute(f'''
        SELECT m.symbol, COALESCE(c.count, 0) as count
        FROM symbol_metadata m
        LEFT JOIN (SELECT symbol, COUNT(*) as count FROM "{first_candle_table}" GROUP BY symbol) c
            ON m.symbol = c.symbol
        WHERE c.count IS NULL OR c.count = 0
        ORDER BY m.symbol
    ''')
    failed = cursor.fetchall()
    if failed:
        print(f'  [WARNING] Symbols with no candle data ({len(failed)}):')
        for symbol, count in failed:
            print(f'    - {symbol}')
    else:
        print('  [OK] All symbols have candle data')
    print()

print('=' * 80)
print('VERIFICATION COMPLETE')
print('=' * 80)

conn.close()
