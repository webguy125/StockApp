"""
Inspect the REAL TurboMode database schema.
This version forces the correct DB path so we never inspect the wrong file.
"""

import os
import sqlite3

# FORCE the correct database path (the 79‑GB active DB)
DB_PATH = r"C:\StockApp\backend\data\turbomode.db"

print("=" * 80)
print("INSPECTING TURBOMODE DATABASE")
print("=" * 80)
print("Resolved DB path:", os.path.abspath(DB_PATH))
print()

# Validate DB exists
if not os.path.exists(DB_PATH):
    print("[ERROR] Database not found at this path!")
    raise SystemExit(1)

# Open in READ-ONLY mode (safe even for corrupted DBs)
conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
cursor = conn.cursor()

# Fetch all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
tables = cursor.fetchall()

print(f"Found {len(tables)} tables")
print()

for (table_name,) in tables:
    # Get column info
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()

    # Get row count
    try:
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
    except Exception as e:
        count = f"ERROR: {e}"

    print(f"  - {table_name:30s} ({len(columns):2d} columns, {count} rows)")

conn.close()

print()
print("=" * 80)
print("DATABASE INSPECTION COMPLETE")
print("=" * 80)