"""
Check the mismatch between training_symbols.py and master_market_data DB
"""
import sys
import os
import sqlite3

# Add backend to path
backend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend')
sys.path.insert(0, backend_path)

from turbomode.training_symbols import get_training_symbols_with_crypto

# Get symbols from master_market_data DB
conn = sqlite3.connect('master_market_data/market_data.db')
cursor = conn.cursor()
cursor.execute('SELECT DISTINCT symbol FROM candles ORDER BY symbol')
db_symbols = set(row[0] for row in cursor.fetchall())
conn.close()

# Get symbols from training list (230 stocks + 3 crypto = 233)
training_symbols = set(get_training_symbols_with_crypto())

print("=" * 80)
print("SYMBOL MISMATCH ANALYSIS")
print("=" * 80)
print(f"\nSymbols in training_symbols.py: {len(training_symbols)}")
print(f"Symbols in master_market_data DB: {len(db_symbols)}")
print()

# In training but NOT in DB (need to ingest)
missing_from_db = sorted(training_symbols - db_symbols)
print(f"Symbols in TRAINING but NOT in DB ({len(missing_from_db)}):")
print(f"[NEED TO INGEST THESE]")
if missing_from_db:
    for i in range(0, len(missing_from_db), 10):
        print(", ".join(missing_from_db[i:i+10]))
else:
    print("None")
print()

# In DB but NOT in training (already ingested, not used for training)
extra_in_db = sorted(db_symbols - training_symbols)
print(f"Symbols in DB but NOT in TRAINING ({len(extra_in_db)}):")
print(f"[ALREADY IN DB, NOT USED FOR TRAINING]")
if extra_in_db:
    for i in range(0, len(extra_in_db), 10):
        print(", ".join(extra_in_db[i:i+10]))
else:
    print("None")
print()

# In both
in_both = sorted(training_symbols & db_symbols)
print(f"Symbols in BOTH training and DB ({len(in_both)}):")
print(f"[READY TO USE]")
for i in range(0, len(in_both), 10):
    print(", ".join(in_both[i:i+10]))
print()

print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Need to ingest: {len(missing_from_db)} symbols")
print(f"Already have (not in training): {len(extra_in_db)} symbols")
print(f"Ready to use: {len(in_both)} symbols")
print()

if missing_from_db:
    print("ACTION REQUIRED:")
    print(f"Run ingest for {len(missing_from_db)} missing symbols")
    print()
    print("Missing symbols list:")
    print(missing_from_db)
