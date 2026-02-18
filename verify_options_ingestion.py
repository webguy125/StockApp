"""
Verify EODHD Historical Options Ingestion
"""
import sqlite3

DB_PATH = 'C:\\StockApp\\backend\\turbomode\\Options\\Data\\options_universe.db'

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

print("\n" + "="*80)
print("EODHD HISTORICAL OPTIONS INGESTION - VERIFICATION")
print("="*80)

# Summary by symbol
cursor.execute('''
    SELECT symbol,
           COUNT(*) as records,
           COUNT(DISTINCT snapshot_date) as days,
           COUNT(DISTINCT expiration_date) as expirations,
           MIN(snapshot_date) as first_date,
           MAX(snapshot_date) as last_date
    FROM historical_options_chains
    GROUP BY symbol
    ORDER BY symbol
''')

print(f"\n{'Symbol':<10} {'Records':>10} {'Days':>6} {'Expirations':>12}  {'Date Range':<30}")
print("-"*80)

for row in cursor.fetchall():
    symbol, records, days, expirations, first_date, last_date = row
    date_range = f"{first_date} to {last_date}"
    print(f"{symbol:<10} {records:>10,} {days:>6} {expirations:>12}  {date_range:<30}")

# Overall totals
cursor.execute('''
    SELECT COUNT(*) as total_records,
           COUNT(DISTINCT symbol) as total_symbols,
           COUNT(DISTINCT snapshot_date) as total_dates,
           MIN(snapshot_date) as earliest_date,
           MAX(snapshot_date) as latest_date
    FROM historical_options_chains
''')

total_records, total_symbols, total_dates, earliest_date, latest_date = cursor.fetchone()

print("-"*80)
print(f"\nTOTAL: {total_records:,} records")
print(f"Symbols: {total_symbols}")
print(f"Trading days: {total_dates}")
print(f"Date range: {earliest_date} to {latest_date}")

# Sample records
print("\n" + "="*80)
print("SAMPLE RECORDS (AAPL, most recent date)")
print("="*80)

cursor.execute('''
    SELECT contract_name, expiration_date, strike, option_type,
           last_price, bid, ask, volume, open_interest, implied_volatility,
           delta, underlying_price, days_to_expiration
    FROM historical_options_chains
    WHERE symbol = 'AAPL'
      AND snapshot_date = (
          SELECT MAX(snapshot_date) FROM historical_options_chains WHERE symbol = 'AAPL'
      )
    LIMIT 5
''')

print(f"\n{'Contract':<22} {'Exp':<12} {'Strike':>8} {'Type':<4} {'Last':>8} {'Bid':>8} {'Ask':>8} {'IV':>8} {'Delta':>8}")
print("-"*80)

for row in cursor.fetchall():
    contract, exp, strike, opt_type, last, bid, ask, vol, oi, iv, delta, underlying, dte = row
    last_str = f"${last:.2f}" if last else "N/A"
    bid_str = f"${bid:.2f}" if bid else "N/A"
    ask_str = f"${ask:.2f}" if ask else "N/A"
    iv_str = f"{iv*100:.1f}%" if iv else "N/A"
    delta_str = f"{delta:.3f}" if delta else "N/A"

    print(f"{contract:<22} {exp:<12} ${strike:>7.2f} {opt_type:<4} {last_str:>8} {bid_str:>8} {ask_str:>8} {iv_str:>8} {delta_str:>8}")

print("\n" + "="*80)

conn.close()
