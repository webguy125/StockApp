"""
Extract schema from advanced_ml_system.db for tables used by TurboMode training
"""
import sqlite3

conn = sqlite3.connect('backend/data/advanced_ml_system.db')
cursor = conn.cursor()

# Tables needed for TurboMode training
tables_needed = ['trades', 'feature_store', 'price_data']

print("=" * 80)
print("SCHEMA EXTRACTION FOR TURBOMODE TABLES")
print("=" * 80)

for table in tables_needed:
    print(f"\n-- {table.upper()} TABLE")
    print("-" * 80)

    cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table}'")
    result = cursor.fetchone()

    if result:
        print(result[0] + ";")

        # Show sample record count
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"-- Current records: {count:,}")
    else:
        print(f"-- Table '{table}' not found")

conn.close()

print("\n" + "=" * 80)
print("Schema extraction complete")
print("=" * 80)
