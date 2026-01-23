"""
Compare training_symbols.py vs master_market_data DB symbols
Show both lists side by side for manual verification
"""
import sys
import os
import sqlite3

# Add backend to path
backend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend')
sys.path.insert(0, backend_path)

from turbomode.training_symbols import get_training_symbols_with_crypto

# Get training symbols
training_symbols = sorted(get_training_symbols_with_crypto())

# Get DB symbols
conn = sqlite3.connect('master_market_data/market_data.db')
cursor = conn.cursor()
cursor.execute('SELECT DISTINCT symbol FROM candles ORDER BY symbol')
db_symbols = sorted([row[0] for row in cursor.fetchall()])
conn.close()

print("=" * 120)
print("TRAINING SYMBOLS vs DATABASE SYMBOLS")
print("=" * 120)
print()
print(f"Training symbols: {len(training_symbols)}")
print(f"Database symbols: {len(db_symbols)}")
print()

# Print side by side
print("=" * 120)
print(f"{'TRAINING_SYMBOLS.PY':<60} {'MASTER_MARKET_DATA DB':<60}")
print("=" * 120)

max_len = max(len(training_symbols), len(db_symbols))
for i in range(max_len):
    left = training_symbols[i] if i < len(training_symbols) else ""
    right = db_symbols[i] if i < len(db_symbols) else ""

    # Mark mismatches
    if left != right:
        marker = " <-- MISMATCH"
    else:
        marker = ""

    print(f"{left:<60} {right:<60}{marker}")

print()
print("=" * 120)

# Check for differences
only_in_training = set(training_symbols) - set(db_symbols)
only_in_db = set(db_symbols) - set(training_symbols)

if only_in_training:
    print(f"\nSymbols ONLY in training_symbols.py ({len(only_in_training)}):")
    for symbol in sorted(only_in_training):
        print(f"  {symbol}")

if only_in_db:
    print(f"\nSymbols ONLY in database ({len(only_in_db)}):")
    for symbol in sorted(only_in_db):
        print(f"  {symbol}")

if not only_in_training and not only_in_db:
    print("\n[SUCCESS] Lists match exactly!")
    print("All symbols are identical.")
