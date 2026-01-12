"""
Check complete schema for turbomode.db
Shows all tables and their columns
"""
import sqlite3
import os

db_path = os.path.join('backend', 'data', 'turbomode.db')

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("=" * 80)
print("TURBOMODE.DB - COMPLETE SCHEMA")
print("=" * 80)

# Get all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
tables = cursor.fetchall()

print(f"\nTotal Tables: {len(tables)}\n")

for (table_name,) in tables:
    print("=" * 80)
    print(f"TABLE: {table_name}")
    print("=" * 80)

    # Get table info
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()

    print(f"\nColumns ({len(columns)}):")
    print("-" * 80)
    for col in columns:
        col_id, col_name, col_type, not_null, default_val, pk = col

        # Format output
        null_str = "NOT NULL" if not_null else "NULL"
        pk_str = "PRIMARY KEY" if pk else ""
        default_str = f"DEFAULT {default_val}" if default_val else ""

        extras = " ".join(filter(None, [null_str, default_str, pk_str]))

        print(f"  {col_name:<30} {col_type:<15} {extras}")

    # Get row count
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cursor.fetchone()[0]
    print(f"\nRow Count: {count}")

    # Get indexes
    cursor.execute(f"PRAGMA index_list({table_name})")
    indexes = cursor.fetchall()
    if indexes:
        print(f"\nIndexes ({len(indexes)}):")
        for idx in indexes:
            print(f"  {idx[1]} (unique={idx[2]})")

    print()

conn.close()

print("=" * 80)
print("SCHEMA EXPORT COMPLETE")
print("=" * 80)
