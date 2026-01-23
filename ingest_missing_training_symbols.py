"""
Ingest the 63 missing training symbols into master_market_data DB
"""
import sys
import os
import sqlite3

backend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend')
sys.path.insert(0, backend_path)

from turbomode.training_symbols import get_training_symbols_with_crypto

# Get symbols in master_market_data
master_db = os.path.join('master_market_data', 'market_data.db')
conn = sqlite3.connect(master_db)
cursor = conn.cursor()
cursor.execute('SELECT DISTINCT symbol FROM candles')
symbols_in_master = set(row[0] for row in cursor.fetchall())
conn.close()

# Get training symbols (230 stocks + 3 crypto = 233)
training_symbols = get_training_symbols_with_crypto()

# Find missing symbols
missing_symbols = sorted([s for s in training_symbols if s not in symbols_in_master])

print("=" * 80)
print("MISSING TRAINING SYMBOLS FOR INGESTION")
print("=" * 80)
print(f"\nTotal training symbols needed: {len(training_symbols)} (230 stocks + 3 crypto)")
print(f"Symbols in master_market_data: {len(symbols_in_master)}")
print(f"Missing symbols to ingest: {len(missing_symbols)}")
print()

if missing_symbols:
    # Write to file for ingestion
    output_file = 'missing_symbols_to_ingest.txt'
    with open(output_file, 'w') as f:
        for symbol in missing_symbols:
            f.write(f"{symbol}\n")

    print(f"Missing symbols saved to: {output_file}")
    print()
    print("Missing symbols list:")
    for i, symbol in enumerate(missing_symbols, 1):
        print(f"  {i:2d}. {symbol}")

    print()
    print("=" * 80)
    print("NEXT STEP")
    print("=" * 80)
    print()
    print("Run the ingestion command:")
    print()
    print(f"  python master_market_data/ingest_market_data.py --symbols-file {output_file}")
    print()
    print(f"Estimated time: ~15-20 minutes for {len(missing_symbols)} symbols (10 years each)")
    print()
else:
    print("All training symbols are already in master_market_data DB!")
    print("Ready to proceed with backtest generation.")
