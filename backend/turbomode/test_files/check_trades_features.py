"""
Check which features are in the trades table
"""
import sys
import os
import sqlite3

# Add backend to path
backend_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

# Database path
db_path = os.path.join(backend_path, "data", "advanced_ml_system.db")

# Connect and get column names
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get column names from trades table
cursor.execute("PRAGMA table_info(trades)")
columns = cursor.fetchall()

print("Columns in trades table:")
print("=" * 80)

feature_cols = []
other_cols = []

for col in columns:
    col_name = col[1]
    col_type = col[2]
    if 'feature' in col_name.lower() or col_name.startswith('f_'):
        feature_cols.append(col_name)
    else:
        other_cols.append(col_name)

print(f"\nNon-feature columns ({len(other_cols)}):")
for col in other_cols[:20]:  # Show first 20
    print(f"  {col}")
if len(other_cols) > 20:
    print(f"  ... and {len(other_cols) - 20} more")

print(f"\nFeature-related columns ({len(feature_cols)}):")
for i, col in enumerate(sorted(feature_cols)[:10], 1):  # Show first 10
    print(f"  {i}. {col}")
if len(feature_cols) > 10:
    print(f"  ... and {len(feature_cols) - 10} more")

# Check if features are stored as JSON
cursor.execute("SELECT COUNT(*) FROM trades")
count = cursor.fetchone()[0]
print(f"\nTotal trades in database: {count}")

# Sample one row to see structure
cursor.execute("SELECT * FROM trades LIMIT 1")
sample = cursor.fetchone()
cursor.execute("PRAGMA table_info(trades)")
col_info = cursor.fetchall()

print(f"\nSample row structure:")
for i, col in enumerate(col_info[:15]):  # First 15 columns
    col_name = col[1]
    value = sample[i] if sample and i < len(sample) else "N/A"
    value_str = str(value)[:50] if value else "NULL"
    print(f"  {col_name}: {value_str}")

print(f"\n{'=' * 80}")
print(f"Total columns in trades table: {len(columns)}")

conn.close()
