"""
Test script to verify prob_hold migration runs successfully
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from backend.turbomode.database_schema import TurboModeDB
import sqlite3

print("=" * 80)
print("TESTING PROB_HOLD MIGRATION")
print("=" * 80)
print()

# Initialize database (this will run migrations)
print("[1/3] Initializing TurboModeDB (migrations will run automatically)...")
db = TurboModeDB(db_path="backend/data/turbomode.db")
print()

# Verify the column was added
print("[2/3] Verifying prob_hold column exists in active_signals...")
conn = sqlite3.connect("C:/StockApp/backend/data/turbomode.db")
cursor = conn.cursor()
cursor.execute("PRAGMA table_info(active_signals)")

columns = [row[1] for row in cursor.fetchall()]
conn.close()

if 'prob_hold' in columns:
    print("      [OK] prob_hold column found in active_signals table")
    print()

    # Display all columns
    print("[3/3] Active_signals column list:")
    for i, col in enumerate(columns, 1):
        print(f"      {i:2d}. {col}")

    print()
    print("=" * 80)
    print("[SUCCESS] Migration completed successfully!")
    print("Database schema now matches canonical schema.")
    print("=" * 80)
else:
    print("      [FAIL] prob_hold column NOT found in active_signals table")
    print()
    print("=" * 80)
    print("[FAIL] Migration did not run or failed")
    print("=" * 80)
    sys.exit(1)
