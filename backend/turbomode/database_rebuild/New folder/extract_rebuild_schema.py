"""
Extract Schema from ANY turbomode.db (Read-Only)
Safely reads CREATE TABLE and CREATE INDEX statements from a corrupted DB.
Does NOT modify the database.
"""

import os
import sys
import sqlite3
from pathlib import Path

db_path = input("Enter path to turbomode.db to EXTRACT schema from: ").strip().strip('"')

print("=" * 80)
print("EXTRACTING SCHEMA (READ-ONLY)")
print("=" * 80)
print(f"Source DB: {db_path}")
print()

if not os.path.exists(db_path):
    print(f"[ERROR] Database not found: {db_path}")
    sys.exit(1)

# Open in READ-ONLY mode
conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
cursor = conn.cursor()

# Read tables
cursor.execute("""
    SELECT name, sql FROM sqlite_master
    WHERE type='table' AND sql IS NOT NULL
    ORDER BY name
""")
tables = cursor.fetchall()

# Read indices
cursor.execute("""
    SELECT name, sql FROM sqlite_master
    WHERE type='index' AND sql IS NOT NULL
    ORDER BY name
""")
indices = cursor.fetchall()

conn.close()

print(f"[OK] Found {len(tables)} tables")
print(f"[OK] Found {len(indices)} indices")
print()

# Output file
output_path = Path(__file__).resolve().parent / "extracted_schema.sql"

with open(output_path, "w", encoding="utf-8") as f:
    f.write("-- Extracted TurboMode Schema\n\n")

    for name, sql in tables:
        f.write(sql + ";\n\n")

    for name, sql in indices:
        f.write(sql + ";\n\n")

print("=" * 80)
print("SCHEMA EXTRACTION COMPLETE")
print("=" * 80)
print(f"Schema saved to: {output_path}")
print()