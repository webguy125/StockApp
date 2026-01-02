"""
Check which features are in the database
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

# Get column names from backtest_results table
cursor.execute("PRAGMA table_info(backtest_results)")
columns = cursor.fetchall()

print("Columns in backtest_results table:")
print("=" * 80)

feature_cols = []
other_cols = []

for col in columns:
    col_name = col[1]
    col_type = col[2]
    if col_name.startswith('feature_'):
        feature_cols.append(col_name)
    else:
        other_cols.append(col_name)

print(f"\nNon-feature columns ({len(other_cols)}):")
for col in other_cols:
    print(f"  {col}")

print(f"\nFeature columns ({len(feature_cols)}):")
for i, col in enumerate(sorted(feature_cols), 1):
    print(f"  {i}. {col}")

print(f"\n{'=' * 80}")
print(f"Total feature columns: {len(feature_cols)}")
print(f"Expected: 179")
print(f"Missing: {179 - len(feature_cols)}")

conn.close()
