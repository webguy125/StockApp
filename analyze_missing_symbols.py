"""
Analyze which training symbols have no historical data in database
"""
import sqlite3
import sys
import os

# Add backend to path
backend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend')
sys.path.insert(0, backend_path)

from turbomode.training_symbols import get_training_symbols, get_symbol_metadata

# Connect to database
db_path = os.path.join(backend_path, 'data', 'turbomode.db')
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get all training symbols
training_symbols = get_training_symbols()

print("=" * 80)
print("ANALYZING TRAINING SYMBOL DATA COVERAGE")
print("=" * 80)
print(f"\nTotal training symbols: {len(training_symbols)}")

# Check which symbols have data
symbols_with_data = []
symbols_without_data = []

for symbol in training_symbols:
    cursor.execute("""
        SELECT COUNT(*) FROM trades
        WHERE symbol = ? AND trade_type = 'backtest'
    """, (symbol,))
    count = cursor.fetchone()[0]

    if count > 0:
        symbols_with_data.append((symbol, count))
    else:
        symbols_without_data.append(symbol)

conn.close()

print(f"\nSymbols WITH data: {len(symbols_with_data)}")
print(f"Symbols WITHOUT data: {len(symbols_without_data)}")
print(f"Coverage: {len(symbols_with_data) / len(training_symbols) * 100:.1f}%")

# Print symbols without data
if symbols_without_data:
    print(f"\n{'=' * 80}")
    print(f"SYMBOLS WITHOUT HISTORICAL DATA ({len(symbols_without_data)} total)")
    print("=" * 80)

    # Group by sector
    by_sector = {}
    for symbol in symbols_without_data:
        metadata = get_symbol_metadata(symbol)
        sector = metadata['sector']
        cap = metadata['market_cap_category']

        if sector not in by_sector:
            by_sector[sector] = {'large_cap': [], 'mid_cap': [], 'small_cap': []}

        by_sector[sector][cap].append(symbol)

    # Print by sector
    for sector in sorted(by_sector.keys()):
        caps = by_sector[sector]
        total_missing = len(caps['large_cap']) + len(caps['mid_cap']) + len(caps['small_cap'])

        if total_missing > 0:
            print(f"\n{sector.upper().replace('_', ' ')} ({total_missing} missing):")
            if caps['large_cap']:
                print(f"  Large cap: {', '.join(sorted(caps['large_cap']))}")
            if caps['mid_cap']:
                print(f"  Mid cap: {', '.join(sorted(caps['mid_cap']))}")
            if caps['small_cap']:
                print(f"  Small cap: {', '.join(sorted(caps['small_cap']))}")

# Print top 10 symbols with most data
print(f"\n{'=' * 80}")
print("TOP 10 SYMBOLS BY SAMPLE COUNT")
print("=" * 80)

symbols_with_data.sort(key=lambda x: x[1], reverse=True)
for i, (symbol, count) in enumerate(symbols_with_data[:10], 1):
    metadata = get_symbol_metadata(symbol)
    print(f"{i:2d}. {symbol:6s} {count:,} samples  ({metadata['sector'].replace('_', ' ').title()}, {metadata['market_cap_category'].replace('_', ' ')})")

print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)

if len(symbols_without_data) == 0:
    print("\nAll symbols have data - ready to train!")
elif len(symbols_without_data) < 20:
    print(f"\nOnly {len(symbols_without_data)} symbols missing data ({len(symbols_without_data)/len(training_symbols)*100:.1f}%)")
    print("RECOMMENDATION: Proceed with training using available data")
    print("               Remove missing symbols from training_symbols.py")
elif len(symbols_without_data) < 50:
    print(f"\n{len(symbols_without_data)} symbols missing data ({len(symbols_without_data)/len(training_symbols)*100:.1f}%)")
    print("RECOMMENDATION: Check sector balance before proceeding")
    print("               May need to ingest missing symbols or find replacements")
else:
    print(f"\nWARNING: {len(symbols_without_data)} symbols missing data ({len(symbols_without_data)/len(training_symbols)*100:.1f}%)")
    print("RECOMMENDATION: Investigate Master Market Data ingestion")
    print("               Too many missing to proceed with training")
