"""
Rebuild TurboMode Database from Schema
This script uses database_schema.py to create a fresh turbomode.db with correct schema
"""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from backend.turbomode.database_schema import TurboModeDB

# Database path
db_path = os.path.join(project_root, "backend", "data", "turbomode.db")

print("=" * 80)
print("REBUILD TURBOMODE DATABASE FROM SCHEMA")
print("=" * 80)
print(f"Database path: {db_path}")
print()

# Check if database exists
if os.path.exists(db_path):
    print("[WARNING] Database file already exists!")
    response = input("Do you want to DELETE and rebuild? (yes/no): ")
    if response.lower() != 'yes':
        print("[ABORT] Database rebuild cancelled")
        sys.exit(0)

    # Delete existing database
    os.remove(db_path)
    print("[OK] Existing database deleted")
    print()

# Create fresh database using TurboModeDB class
print("Creating fresh database with schema from database_schema.py...")
db = TurboModeDB(db_path=db_path)

print()
print("=" * 80)
print("DATABASE REBUILD COMPLETE")
print("=" * 80)
print(f"Database: {db_path}")
print()

# Show created tables
import sqlite3
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
tables = cursor.fetchall()

print("Tables created:")
for (table_name,) in tables:
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cursor.fetchone()[0]
    print(f"  - {table_name:30s} ({count} records)")

conn.close()

print()
print("[OK] TurboMode database rebuilt successfully!")
print("=" * 80)
