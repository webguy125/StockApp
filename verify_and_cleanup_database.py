"""
Verify and cleanup master_market_data DB to match training_symbols.py exactly

This script:
1. Identifies symbols in DB that shouldn't be there (old symbols)
2. Removes those old symbols from all tables
3. Verifies final count matches training_symbols.py (233 symbols)
"""
import sys
import os
import sqlite3

# Add backend to path
backend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend')
sys.path.insert(0, backend_path)

from turbomode.training_symbols import get_training_symbols_with_crypto

# Get symbols from master_market_data DB
master_db = 'master_market_data/market_data.db'
conn = sqlite3.connect(master_db)
cursor = conn.cursor()
cursor.execute('SELECT DISTINCT symbol FROM candles ORDER BY symbol')
db_symbols = set(row[0] for row in cursor.fetchall())
conn.close()

# Get training symbols (230 stocks + 3 crypto = 233)
training_symbols = set(get_training_symbols_with_crypto())

print("=" * 80)
print("DATABASE VERIFICATION AND CLEANUP")
print("=" * 80)
print()
print(f"Training symbols needed: {len(training_symbols)}")
print(f"Symbols in database:     {len(db_symbols)}")
print()

# Find symbols to remove (in DB but not in training list)
symbols_to_remove = sorted(db_symbols - training_symbols)
symbols_missing = sorted(training_symbols - db_symbols)

if symbols_missing:
    print(f"[WARNING] Missing symbols ({len(symbols_missing)}):")
    print(f"These should have been ingested!")
    for i in range(0, len(symbols_missing), 10):
        print("  " + ", ".join(symbols_missing[i:i+10]))
    print()

if symbols_to_remove:
    print(f"[CLEANUP] Symbols to remove ({len(symbols_to_remove)}):")
    print(f"These are old symbols not in training list:")
    for i in range(0, len(symbols_to_remove), 10):
        print("  " + ", ".join(symbols_to_remove[i:i+10]))
    print()

    response = input(f"Remove {len(symbols_to_remove)} old symbols from database? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Cancelled. Database not modified.")
        sys.exit(0)

    print()
    print("Removing old symbols from database...")
    print()

    conn = sqlite3.connect(master_db)
    cursor = conn.cursor()

    for symbol in symbols_to_remove:
        print(f"  Removing {symbol}...")

        # Remove from all tables
        cursor.execute('DELETE FROM candles WHERE symbol = ?', (symbol,))
        cursor.execute('DELETE FROM symbol_metadata WHERE symbol = ?', (symbol,))
        cursor.execute('DELETE FROM fundamentals WHERE symbol = ?', (symbol,))
        cursor.execute('DELETE FROM splits WHERE symbol = ?', (symbol,))
        cursor.execute('DELETE FROM dividends WHERE symbol = ?', (symbol,))
        cursor.execute('DELETE FROM sector_mappings WHERE symbol = ?', (symbol,))

    conn.commit()
    conn.close()

    print()
    print(f"[OK] Removed {len(symbols_to_remove)} symbols from database")
    print()
else:
    print("[OK] No old symbols to remove - database is clean!")
    print()

# Final verification
conn = sqlite3.connect(master_db)
cursor = conn.cursor()
cursor.execute('SELECT DISTINCT symbol FROM candles ORDER BY symbol')
final_db_symbols = set(row[0] for row in cursor.fetchall())
conn.close()

final_missing = sorted(training_symbols - final_db_symbols)
final_extra = sorted(final_db_symbols - training_symbols)

print("=" * 80)
print("FINAL VERIFICATION")
print("=" * 80)
print()
print(f"Expected symbols: {len(training_symbols)}")
print(f"Database symbols: {len(final_db_symbols)}")
print()

if final_missing:
    print(f"[ERROR] Missing symbols ({len(final_missing)}):")
    for i in range(0, len(final_missing), 10):
        print("  " + ", ".join(final_missing[i:i+10]))
    print()
    print("ACTION: Run ingestion for missing symbols")
    print()

if final_extra:
    print(f"[ERROR] Extra symbols ({len(final_extra)}):")
    for i in range(0, len(final_extra), 10):
        print("  " + ", ".join(final_extra[i:i+10]))
    print()
    print("ACTION: Run cleanup again to remove extra symbols")
    print()

if not final_missing and not final_extra:
    print("[SUCCESS] Database matches training_symbols.py exactly!")
    print(f"All {len(training_symbols)} symbols present (230 stocks + 3 crypto)")
    print()
    print("=" * 80)
    print("READY FOR TRAINING DATA GENERATION")
    print("=" * 80)
    print()
    print("Next step:")
    print("  python backend/turbomode/generate_backtest_data.py")
    print()
else:
    print("[FAILED] Database does not match training_symbols.py")
    print("Fix the issues above before proceeding.")
    print()
